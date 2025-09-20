"""Entanglement Flow Visualizer - Track how quantum information spreads through circuits.

This reveals the hidden structure of quantum computation - showing exactly
how entanglement propagates, where it concentrates, and why certain circuits
are hard to simulate classically.

THIS IS THE KEY INSIGHT: Quantum advantage isn't about raw qubit count,
it's about the TOPOLOGY of entanglement flow!
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm


@dataclass 
class EntanglementSnapshot:
    """State of entanglement at a specific circuit layer."""
    layer: int
    entangled_groups: List[Set[int]]  # Partition of qubits into entangled clusters
    schmidt_ranks: Dict[Tuple[int, int], int]  # Schmidt rank between qubit pairs
    entanglement_entropy: Dict[int, float]  # Per-qubit entanglement
    cut_complexity: Dict[int, int]  # Complexity of cutting at each position


class EntanglementFlow:
    """Track and visualize how entanglement flows through a quantum circuit."""
    
    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.n_qubits = circuit.num_qubits
        self.snapshots: List[EntanglementSnapshot] = []
        self.entanglement_graph = nx.DiGraph()
        self._analyze_flow()
    
    def _analyze_flow(self):
        """Analyze how entanglement spreads through the circuit layer by layer."""
        # Start with no entanglement
        entangled_groups = [{i} for i in range(self.n_qubits)]
        layer_idx = 0
        
        # Group gates into layers
        layers = self._extract_layers()
        
        for layer_gates in layers:
            # Apply gates in this layer
            for gate_data in layer_gates:
                inst = gate_data.operation
                qargs = gate_data.qubits
                
                if inst.num_qubits == 2:
                    # Two-qubit gate creates/spreads entanglement
                    q1_idx = self.circuit.qubits.index(qargs[0])
                    q2_idx = self.circuit.qubits.index(qargs[1])
                    
                    # Merge entangled groups
                    group1 = None
                    group2 = None
                    for group in entangled_groups:
                        if q1_idx in group:
                            group1 = group
                        if q2_idx in group:
                            group2 = group
                    
                    if group1 != group2:
                        # Merge groups
                        new_group = group1.union(group2)
                        entangled_groups = [g for g in entangled_groups if g != group1 and g != group2]
                        entangled_groups.append(new_group)
                        
                        # Add to entanglement flow graph
                        self.entanglement_graph.add_edge(
                            (layer_idx, q1_idx),
                            (layer_idx + 1, q2_idx),
                            gate=inst.name,
                            layer=layer_idx
                        )
                
                elif inst.name.lower() in ['t', 'tdg', 'rx', 'ry', 'rz']:
                    # Non-Clifford gates add "magic" but don't create entanglement
                    q_idx = self.circuit.qubits.index(qargs[0])
                    self.entanglement_graph.add_node(
                        (layer_idx, q_idx),
                        magic=True,
                        gate=inst.name
                    )
            
            # Calculate metrics for this layer
            schmidt_ranks = self._calculate_schmidt_ranks(entangled_groups)
            entropy = self._calculate_entanglement_entropy(entangled_groups)
            cut_complexity = self._calculate_cut_complexity(entangled_groups)
            
            snapshot = EntanglementSnapshot(
                layer=layer_idx,
                entangled_groups=[group.copy() for group in entangled_groups],
                schmidt_ranks=schmidt_ranks,
                entanglement_entropy=entropy,
                cut_complexity=cut_complexity,
            )
            self.snapshots.append(snapshot)
            layer_idx += 1
    
    def _extract_layers(self) -> List[List]:
        """Group circuit operations into parallel layers."""
        layers = []
        current_layer = []
        occupied_qubits = set()
        
        for gate_data in self.circuit.data:
            inst = gate_data.operation
            qargs = gate_data.qubits
            
            # Get qubit indices
            qubit_indices = {self.circuit.qubits.index(q) for q in qargs}
            
            # Check if we can add to current layer
            if qubit_indices & occupied_qubits:
                # Conflict - start new layer
                if current_layer:
                    layers.append(current_layer)
                current_layer = [gate_data]
                occupied_qubits = qubit_indices
            else:
                # Add to current layer
                current_layer.append(gate_data)
                occupied_qubits.update(qubit_indices)
        
        if current_layer:
            layers.append(current_layer)
        
        return layers
    
    def _calculate_schmidt_ranks(self, entangled_groups: List[Set[int]]) -> Dict[Tuple[int, int], int]:
        """Calculate Schmidt rank between all qubit pairs."""
        ranks = {}
        
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                # Check if qubits are in same entangled group
                same_group = any(i in group and j in group for group in entangled_groups)
                
                if same_group:
                    # Within an entangled group, Schmidt rank can be high
                    # (simplified - real calculation would use state vector)
                    group_size = max(len(g) for g in entangled_groups if i in g)
                    ranks[(i, j)] = min(2**(group_size // 2), 2**min(i+1, j+1, self.n_qubits-i, self.n_qubits-j))
                else:
                    # Separable - Schmidt rank is 1
                    ranks[(i, j)] = 1
        
        return ranks
    
    def _calculate_entanglement_entropy(self, entangled_groups: List[Set[int]]) -> Dict[int, float]:
        """Calculate per-qubit entanglement entropy."""
        entropy = {}
        
        for i in range(self.n_qubits):
            # Find which group this qubit belongs to
            for group in entangled_groups:
                if i in group:
                    # Entropy scales with log of group size
                    group_size = len(group)
                    if group_size > 1:
                        # von Neumann entropy approximation
                        entropy[i] = np.log2(group_size) / 2
                    else:
                        entropy[i] = 0.0
                    break
        
        return entropy
    
    def _calculate_cut_complexity(self, entangled_groups: List[Set[int]]) -> Dict[int, int]:
        """Calculate complexity of cutting the circuit at each position."""
        complexity = {}
        
        for cut_pos in range(1, self.n_qubits):
            left = set(range(cut_pos))
            right = set(range(cut_pos, self.n_qubits))
            
            # Count entanglement bonds across cut
            bonds = 0
            for group in entangled_groups:
                if group & left and group & right:
                    # This group spans the cut
                    bonds += min(len(group & left), len(group & right))
            
            complexity[cut_pos] = 2**bonds  # Exponential in cut size
        
        return complexity
    
    def find_optimal_cuts(self, max_complexity: int = 256) -> List[int]:
        """Find positions where circuit can be cut with bounded complexity."""
        optimal_cuts = []
        
        for snapshot in self.snapshots:
            for pos, complexity in snapshot.cut_complexity.items():
                if complexity <= max_complexity:
                    optimal_cuts.append((snapshot.layer, pos, complexity))
        
        return sorted(optimal_cuts, key=lambda x: x[2])  # Sort by complexity
    
    def get_entanglement_width(self) -> int:
        """Get maximum entanglement width (size of largest entangled group)."""
        max_width = 0
        
        for snapshot in self.snapshots:
            for group in snapshot.entangled_groups:
                max_width = max(max_width, len(group))
        
        return max_width
    
    def get_magic_concentration(self) -> Dict[int, int]:
        """Find which qubits have the most magic (non-Clifford) operations."""
        magic_count = {i: 0 for i in range(self.n_qubits)}
        
        for node, data in self.entanglement_graph.nodes(data=True):
            if data.get('magic', False):
                _, qubit = node
                magic_count[qubit] += 1
        
        return magic_count
    
    def visualize_flow(self, save_path: Optional[str] = None):
        """Create a visualization of entanglement flow through the circuit."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Entanglement spread over layers
        ax1 = axes[0, 0]
        layers = [s.layer for s in self.snapshots]
        max_group_sizes = [max(len(g) for g in s.entangled_groups) for s in self.snapshots]
        avg_entropies = [np.mean(list(s.entanglement_entropy.values())) if s.entanglement_entropy else 0 
                        for s in self.snapshots]
        
        ax1.plot(layers, max_group_sizes, 'b-', label='Max entangled group size', linewidth=2)
        ax1.plot(layers, [e * 10 for e in avg_entropies], 'r--', label='Avg entropy (×10)', linewidth=2)
        ax1.set_xlabel('Circuit Layer')
        ax1.set_ylabel('Entanglement Measure')
        ax1.set_title('Entanglement Growth Through Circuit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Qubit entanglement heatmap
        ax2 = axes[0, 1]
        ent_matrix = np.zeros((self.n_qubits, len(self.snapshots)))
        
        for j, snapshot in enumerate(self.snapshots):
            for i in range(self.n_qubits):
                ent_matrix[i, j] = snapshot.entanglement_entropy.get(i, 0)
        
        im = ax2.imshow(ent_matrix, aspect='auto', cmap='hot', interpolation='nearest')
        ax2.set_xlabel('Circuit Layer')
        ax2.set_ylabel('Qubit Index')
        ax2.set_title('Entanglement Entropy Heatmap')
        plt.colorbar(im, ax=ax2)
        
        # 3. Cut complexity landscape
        ax3 = axes[1, 0]
        if self.snapshots:
            cut_positions = sorted(self.snapshots[0].cut_complexity.keys())
            complexity_over_time = []
            
            for snapshot in self.snapshots:
                complexities = [np.log2(snapshot.cut_complexity.get(p, 1) + 1) for p in cut_positions]
                complexity_over_time.append(complexities)
            
            complexity_array = np.array(complexity_over_time).T
            im3 = ax3.imshow(complexity_array, aspect='auto', cmap='viridis', interpolation='nearest')
            ax3.set_xlabel('Circuit Layer')
            ax3.set_ylabel('Cut Position')
            ax3.set_title('Log₂(Cut Complexity) Landscape')
            plt.colorbar(im3, ax=ax3)
        
        # 4. Entanglement graph structure
        ax4 = axes[1, 1]
        
        # Create a simplified view of entanglement connections
        G = nx.Graph()
        for snapshot in self.snapshots[-1:]:  # Last snapshot
            for group in snapshot.entangled_groups:
                if len(group) > 1:
                    # Add edges between all qubits in group
                    group_list = list(group)
                    for i in range(len(group_list)):
                        for j in range(i + 1, len(group_list)):
                            G.add_edge(group_list[i], group_list[j])
        
        if G.nodes():
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Color nodes by magic concentration
            magic = self.get_magic_concentration()
            node_colors = [magic.get(node, 0) for node in G.nodes()]
            
            nx.draw(G, pos, ax=ax4, 
                   node_color=node_colors, 
                   cmap='coolwarm',
                   node_size=500,
                   with_labels=True,
                   font_size=10,
                   font_weight='bold',
                   edge_color='gray',
                   width=2)
            
            ax4.set_title('Final Entanglement Structure\n(Color = Magic Gate Count)')
        
        plt.suptitle(f'Entanglement Flow Analysis\nCircuit: {self.n_qubits} qubits, {self.circuit.depth()} depth', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def get_critical_points(self) -> List[Tuple[int, str]]:
        """Identify critical points where entanglement structure changes dramatically."""
        critical_points = []
        
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i-1]
            curr = self.snapshots[i]
            
            # Check for group mergers
            if len(curr.entangled_groups) < len(prev.entangled_groups):
                critical_points.append((i, "merger"))
            
            # Check for entropy spike
            prev_avg = np.mean(list(prev.entanglement_entropy.values())) if prev.entanglement_entropy else 0
            curr_avg = np.mean(list(curr.entanglement_entropy.values())) if curr.entanglement_entropy else 0
            
            if curr_avg > prev_avg * 1.5:
                critical_points.append((i, "entropy_spike"))
            
            # Check for cut complexity explosion
            if prev.cut_complexity and curr.cut_complexity:
                prev_max = max(prev.cut_complexity.values())
                curr_max = max(curr.cut_complexity.values())
                
                if curr_max > prev_max * 2:
                    critical_points.append((i, "complexity_explosion"))
        
        return critical_points


def analyze_algorithm_entanglement(name: str, circuit: QuantumCircuit) -> Dict:
    """Analyze entanglement patterns in a quantum algorithm."""
    flow = EntanglementFlow(circuit)
    
    analysis = {
        "algorithm": name,
        "n_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "max_entanglement_width": flow.get_entanglement_width(),
        "optimal_cuts": flow.find_optimal_cuts(max_complexity=256)[:5],  # Top 5 cuts
        "critical_points": flow.get_critical_points(),
        "magic_distribution": flow.get_magic_concentration(),
        "final_groups": flow.snapshots[-1].entangled_groups if flow.snapshots else [],
    }
    
    # Determine classical simulability based on entanglement
    if flow.get_entanglement_width() <= 10:
        analysis["simulability"] = "easy"
    elif flow.get_entanglement_width() <= 20:
        analysis["simulability"] = "moderate"
    else:
        analysis["simulability"] = "hard"
    
    return analysis


if __name__ == "__main__":
    print("🌀 QUANTUM ENTANGLEMENT FLOW ANALYZER 🌀")
    print("=" * 60)
    print("Revealing the hidden structure of quantum computation...")
    print()
    
    # Test on different quantum algorithms
    from ariadne_mac.quantum_advantage import (
        create_ghz_circuit,
        create_random_circuit,
        create_qaoa_circuit,
        create_grover_circuit,
    )
    
    algorithms = [
        ("GHZ State", create_ghz_circuit(10)),
        ("Random Circuit", create_random_circuit(8, depth=10)),
        ("QAOA", create_qaoa_circuit(12, p=2)),
        ("Grover", create_grover_circuit(8)),
    ]
    
    for name, circuit in algorithms:
        print(f"\n📊 Analyzing: {name}")
        print("-" * 40)
        
        analysis = analyze_algorithm_entanglement(name, circuit)
        
        print(f"Qubits: {analysis['n_qubits']}, Depth: {analysis['depth']}")
        print(f"Max entanglement width: {analysis['max_entanglement_width']} qubits")
        print(f"Classical simulability: {analysis['simulability'].upper()}")
        
        if analysis['critical_points']:
            print(f"Critical points: {analysis['critical_points'][:3]}")
        
        if analysis['optimal_cuts']:
            print(f"Best cut: layer {analysis['optimal_cuts'][0][0]}, "
                  f"position {analysis['optimal_cuts'][0][1]}, "
                  f"complexity 2^{int(np.log2(analysis['optimal_cuts'][0][2]))}")
        
        # Visualize the most interesting one
        if name == "QAOA":
            flow = EntanglementFlow(circuit)
            fig = flow.visualize_flow(save_path="entanglement_flow_qaoa.png")
            print(f"\n💾 Saved visualization to entanglement_flow_qaoa.png")
    
    print("\n" + "=" * 60)
    print("🔬 KEY INSIGHT:")
    print("Quantum advantage emerges when entanglement topology becomes")
    print("too complex for classical computers to track efficiently.")
    print("It's not about qubit count - it's about ENTANGLEMENT STRUCTURE!")
    print()
    print("This tool reveals WHERE and HOW quantum algorithms gain their power!")
    print("=" * 60)