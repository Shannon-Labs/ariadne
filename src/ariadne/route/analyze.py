from __future__ import annotations

import math
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from typing import Dict, List, Tuple, Any

CLIFFORD_ONE_Q = {"i", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg"}
CLIFFORD_TWO_Q = {"cx", "cz", "swap"}


def is_clifford_circuit(circ: QuantumCircuit) -> bool:
    for inst, _, _ in circ.data:
        name = inst.name
        if name in {"measure", "barrier", "delay"}:
            continue
        if (name not in CLIFFORD_ONE_Q) and (name not in CLIFFORD_TWO_Q):
            return False
    return True


def calculate_gate_entropy(circ: QuantumCircuit) -> float:
    """Calculate Shannon entropy of gate distribution."""
    gate_counts = {}
    total_gates = 0
    
    for instruction, _, _ in circ.data:
        name = instruction.name
        if name not in ['measure', 'barrier', 'delay']:
            gate_counts[name] = gate_counts.get(name, 0) + 1
            total_gates += 1
    
    if total_gates == 0:
        return 0.0
    
    entropy = 0.0
    for count in gate_counts.values():
        p = count / total_gates
        if p > 0:
            entropy -= p * math.log2(p)
    
    return entropy


def estimate_entanglement_entropy(circ: QuantumCircuit) -> float:
    """Estimate the entanglement entropy that will be generated."""
    # Count two-qubit gates as entanglement generators
    entangling_gates = 0
    for instruction, qubits, _ in circ.data:
        if instruction.num_qubits == 2 and instruction.name not in ['measure', 'barrier']:
            entangling_gates += 1
    
    if entangling_gates == 0:
        return 0.0
    
    # Rough estimate: each two-qubit gate contributes to entanglement
    # Saturates at log2(2^n) = n for n qubits
    max_entropy = circ.num_qubits
    saturation_factor = 1 - math.exp(-entangling_gates / circ.num_qubits)
    
    return max_entropy * saturation_factor


def estimate_quantum_volume(circ: QuantumCircuit) -> float:
    """Estimate quantum volume based on depth and width."""
    # Quantum volume is 2^m where m = min(depth, width)
    m = min(circ.depth(), circ.num_qubits)
    return 2 ** m


def calculate_parallelization_factor(circ: QuantumCircuit) -> float:
    """Calculate how much parallelization is possible."""
    if circ.depth() == 0:
        return 1.0
    
    total_gates = sum(1 for inst, _, _ in circ.data if inst.name not in ['measure', 'barrier', 'delay'])
    
    if total_gates == 0:
        return 1.0
    
    # Parallelization factor = total_gates / depth
    # Higher values indicate more parallel execution possible
    return total_gates / circ.depth()


def estimate_noise_susceptibility(circ: QuantumCircuit) -> float:
    """Estimate circuit's susceptibility to noise."""
    # Factors: depth (decoherence), two-qubit gates (higher error), total operations
    depth_factor = min(1.0, circ.depth() / 100.0)  # Normalize to reasonable scale
    
    two_qubit_gates = sum(
        1 for inst, _, _ in circ.data 
        if inst.num_qubits == 2 and inst.name not in ['measure', 'barrier']
    )
    
    total_gates = sum(
        1 for inst, _, _ in circ.data 
        if inst.name not in ['measure', 'barrier', 'delay']
    )
    
    if total_gates == 0:
        return 0.0
    
    two_qubit_ratio = two_qubit_gates / total_gates
    
    # Combine factors (0 = low susceptibility, 1 = high susceptibility)
    susceptibility = 0.6 * depth_factor + 0.4 * two_qubit_ratio
    return min(1.0, susceptibility)


def estimate_classical_complexity(circ: QuantumCircuit) -> float:
    """Estimate classical simulation complexity."""
    if is_clifford_circuit(circ):
        # Clifford circuits are polynomial time
        return circ.num_qubits ** 2
    
    # Non-Clifford circuits are exponential
    # Complexity roughly 2^n * depth
    base_complexity = 2 ** circ.num_qubits
    depth_factor = max(1, circ.depth())
    
    return base_complexity * math.log2(depth_factor + 1)


def calculate_connectivity_score(graph: nx.Graph) -> float:
    """Calculate connectivity score of interaction graph."""
    if graph.number_of_nodes() <= 1:
        return 1.0
    
    # Density of the graph
    n = graph.number_of_nodes()
    max_edges = n * (n - 1) // 2
    
    if max_edges == 0:
        return 1.0
    
    density = graph.number_of_edges() / max_edges
    
    # Also consider clustering coefficient
    try:
        clustering = nx.average_clustering(graph)
    except:
        clustering = 0.0
    
    # Combine density and clustering
    return 0.7 * density + 0.3 * clustering


def calculate_gate_diversity(circ: QuantumCircuit) -> float:
    """Calculate diversity of gate types used."""
    gate_types = set()
    total_gates = 0
    
    for instruction, _, _ in circ.data:
        if instruction.name not in ['measure', 'barrier', 'delay']:
            gate_types.add(instruction.name)
            total_gates += 1
    
    if total_gates == 0:
        return 0.0
    
    # Shannon diversity index for gate types
    return len(gate_types) / max(1, math.log2(total_gates + 1))


def calculate_critical_path_length(circ: QuantumCircuit) -> int:
    """Calculate the critical path length through the circuit."""
    # This is effectively the circuit depth for quantum circuits
    # since all operations must respect causality
    return circ.depth()


def calculate_expressivity(circ: QuantumCircuit) -> float:
    """Calculate circuit expressivity measure."""
    # Expressivity relates to how much of Hilbert space the circuit can explore
    # Factors: gate diversity, entangling gates, parameterized gates
    
    gate_types = set()
    entangling_gates = 0
    parameterized_gates = 0
    total_gates = 0
    
    for instruction, qubits, _ in circ.data:
        if instruction.name not in ['measure', 'barrier', 'delay']:
            gate_types.add(instruction.name)
            total_gates += 1
            
            if instruction.num_qubits == 2:
                entangling_gates += 1
            
            # Check for parameterized gates
            if hasattr(instruction, 'params') and instruction.params:
                parameterized_gates += 1
    
    if total_gates == 0:
        return 0.0
    
    # Normalize components
    type_diversity = len(gate_types) / max(1, math.log2(total_gates + 1))
    entangling_ratio = entangling_gates / total_gates
    param_ratio = parameterized_gates / total_gates
    
    # Weighted combination
    expressivity = (
        0.4 * type_diversity +
        0.4 * entangling_ratio +
        0.2 * param_ratio
    )
    
    return min(1.0, expressivity)def interaction_graph(circ: QuantumCircuit) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(circ.num_qubits))
    # Qiskit 2.x no longer exposes .index directly, so pre-compute lookup table
    qubit_index_map = {qubit: idx for idx, qubit in enumerate(circ.qubits)}
    for inst, qargs, _ in circ.data:
        if inst.num_qubits == 2:
            u, v = [qubit_index_map[q] for q in qargs]
            if u != v:
                g.add_edge(u, v)
    return g


def approximate_treewidth(g: nx.Graph) -> int:
    if g.number_of_nodes() == 0:
        return 0
    try:
        # Use a quick heuristic approximation
        from networkx.algorithms.approximation import treewidth_min_fill_in

        width, _ = treewidth_min_fill_in(g)
        return int(width)
    except Exception:
        return max((deg for _, deg in g.degree()), default=0)


def clifford_ratio(circ: QuantumCircuit) -> float:
    total = 0
    cliff = 0
    for inst, _, _ in circ.data:
        name = inst.name
        if name in {"measure", "barrier", "delay"}:
            continue
        total += 1
        if (name in CLIFFORD_ONE_Q) or (name in CLIFFORD_TWO_Q):
            cliff += 1
    return float(cliff) / float(total) if total else 1.0


def light_cone_width_estimate(circ: QuantumCircuit) -> int:
    # Simple proxy: max degree of interaction graph
    g = interaction_graph(circ)
    return max((deg for _, deg in g.degree()), default=0)


def two_qubit_depth(circ: QuantumCircuit) -> int:
    depth = 0
    current_layer_qubits = set()
    for inst, qargs, _ in circ.data:
        if inst.num_qubits == 2:
            # Qiskit 2.x no longer exposes .index directly, so pre-compute lookup table
            qubit_index_map = {qubit: idx for idx, qubit in enumerate(circ.qubits)}
            qubits = {qubit_index_map[q] for q in qargs}
            if current_layer_qubits & qubits:
                depth += 1
                current_layer_qubits = set(qubits)
            else:
                current_layer_qubits |= qubits
    return depth + (1 if current_layer_qubits else 0)


def analyze_circuit(circ: QuantumCircuit) -> dict[str, float | int | bool]:
    """Enhanced circuit analysis with advanced entropy and complexity metrics."""
    g = interaction_graph(circ)
    
    # Basic metrics
    basic_metrics = {
        "num_qubits": circ.num_qubits,
        "depth": int(circ.depth()),
        "two_qubit_depth": two_qubit_depth(circ),
        "edges": g.number_of_edges(),
        "treewidth_estimate": approximate_treewidth(g),
        "light_cone_width": light_cone_width_estimate(circ),
        "clifford_ratio": clifford_ratio(circ),
        "is_clifford": is_clifford_circuit(circ),
    }
    
    # Advanced entropy and complexity metrics
    advanced_metrics = {
        "gate_entropy": calculate_gate_entropy(circ),
        "entanglement_entropy_estimate": estimate_entanglement_entropy(circ),
        "quantum_volume_estimate": estimate_quantum_volume(circ),
        "parallelization_factor": calculate_parallelization_factor(circ),
        "noise_susceptibility": estimate_noise_susceptibility(circ),
        "classical_simulation_complexity": estimate_classical_complexity(circ),
        "connectivity_score": calculate_connectivity_score(g),
        "gate_diversity": calculate_gate_diversity(circ),
        "critical_path_length": calculate_critical_path_length(circ),
        "expressivity_measure": calculate_expressivity(circ)
    }
    
    # Combine all metrics
    return {**basic_metrics, **advanced_metrics}


def should_use_tensor_network(
    circuit: QuantumCircuit, analysis: dict[str, float | int | bool] | None = None
) -> bool:
    """Return ``True`` if the circuit should target a tensor network backend."""

    metrics = analysis or analyze_circuit(circuit)

    if metrics["is_clifford"]:
        return False

    num_qubits = int(metrics["num_qubits"])
    if num_qubits <= 4:
        return False
    if num_qubits > 30:
        return False

    return True

