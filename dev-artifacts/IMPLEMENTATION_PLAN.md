# Ariadne Tensor Network Backend Implementation Plan

## 🎯 Goal: Replace Fake Tensor Network with Real Quantum Simulation

### Current Problem
```python
# Current fake implementation in ariadne/router.py:283
def _simulate_tensor_network(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    # This is just random sampling - NOT quantum simulation!
    total_states = 2 ** min(num_qubits, 10)
    base_count = shots // total_states
    # ... distributes shots evenly across states
```

### Required Solution
Real tensor network simulation using Quimb + Cotengra for efficient simulation of large, sparse quantum circuits.

## 📋 Implementation Steps

### Step 1: Install Dependencies
```bash
pip install quimb cotengra opt_einsum
```

### Step 2: Create Tensor Network Converter
```python
# File: ariadne/converters/tensor_network.py

import numpy as np
import quimb.tensor as qtn
from qiskit import QuantumCircuit
from typing import Dict, Any

def circuit_to_tensor_network(circuit: QuantumCircuit) -> qtn.TensorNetwork:
    """
    Convert a Qiskit circuit to a Quimb tensor network.
    
    Each gate becomes a tensor, and qubits become indices.
    """
    num_qubits = circuit.num_qubits
    tensors = []
    
    # Initialize qubit states as |0⟩
    for i in range(num_qubits):
        state = np.array([1.0, 0.0])  # |0⟩ state
        tensors.append(qtn.Tensor(state, inds=[f'q{i}']))
    
    # Apply gates
    for instruction, qargs, _ in circuit.data:
        gate_name = instruction.name
        
        if gate_name == 'h':
            tensor = _hadamard_tensor()
            tensors.append(qtn.Tensor(tensor, inds=[f'q{qargs[0]}', f'q{qargs[0]}_out']))
            
        elif gate_name == 'x':
            tensor = _pauli_x_tensor()
            tensors.append(qtn.Tensor(tensor, inds=[f'q{qargs[0]}', f'q{qargs[0]}_out']))
            
        elif gate_name == 'y':
            tensor = _pauli_y_tensor()
            tensors.append(qtn.Tensor(tensor, inds=[f'q{qargs[0]}', f'q{qargs[0]}_out']))
            
        elif gate_name == 'z':
            tensor = _pauli_z_tensor()
            tensors.append(qtn.Tensor(tensor, inds=[f'q{qargs[0]}', f'q{qargs[0]}_out']))
            
        elif gate_name == 'cx':
            tensor = _cnot_tensor()
            tensors.append(qtn.Tensor(tensor, inds=[f'q{qargs[0]}', f'q{qargs[1]}', 
                                                   f'q{qargs[0]}_out', f'q{qargs[1]}_out']))
            
        elif gate_name == 'cz':
            tensor = _cz_tensor()
            tensors.append(qtn.Tensor(tensor, inds=[f'q{qargs[0]}', f'q{qargs[1]}', 
                                                   f'q{qargs[0]}_out', f'q{qargs[1]}_out']))
            
        elif gate_name == 't':
            tensor = _t_gate_tensor()
            tensors.append(qtn.Tensor(tensor, inds=[f'q{qargs[0]}', f'q{qargs[0]}_out']))
            
        elif gate_name == 's':
            tensor = _s_gate_tensor()
            tensors.append(qtn.Tensor(tensor, inds=[f'q{qargs[0]}', f'q{qargs[0]}_out']))
            
        elif gate_name == 'ry':
            angle = instruction.params[0]
            tensor = _ry_tensor(angle)
            tensors.append(qtn.Tensor(tensor, inds=[f'q{qargs[0]}', f'q{qargs[0]}_out']))
            
        elif gate_name == 'rz':
            angle = instruction.params[0]
            tensor = _rz_tensor(angle)
            tensors.append(qtn.Tensor(tensor, inds=[f'q{qargs[0]}', f'q{qargs[0]}_out']))
            
        elif gate_name == 'rx':
            angle = instruction.params[0]
            tensor = _rx_tensor(angle)
            tensors.append(qtn.Tensor(tensor, inds=[f'q{qargs[0]}', f'q{qargs[0]}_out']))
    
    return qtn.TensorNetwork(tensors)

def _hadamard_tensor() -> np.ndarray:
    """Hadamard gate tensor."""
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def _pauli_x_tensor() -> np.ndarray:
    """Pauli X gate tensor."""
    return np.array([[0, 1], [1, 0]])

def _pauli_y_tensor() -> np.ndarray:
    """Pauli Y gate tensor."""
    return np.array([[0, -1j], [1j, 0]])

def _pauli_z_tensor() -> np.ndarray:
    """Pauli Z gate tensor."""
    return np.array([[1, 0], [0, -1]])

def _cnot_tensor() -> np.ndarray:
    """CNOT gate tensor."""
    return np.array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
                     [[[0, 0], [0, 1]], [[0, 0], [1, 0]]]])

def _cz_tensor() -> np.ndarray:
    """CZ gate tensor."""
    return np.array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
                     [[[0, 0], [1, 0]], [[0, 0], [0, -1]]]])

def _t_gate_tensor() -> np.ndarray:
    """T gate tensor."""
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

def _s_gate_tensor() -> np.ndarray:
    """S gate tensor."""
    return np.array([[1, 0], [0, 1j]])

def _ry_tensor(angle: float) -> np.ndarray:
    """RY rotation gate tensor."""
    cos = np.cos(angle / 2)
    sin = np.sin(angle / 2)
    return np.array([[cos, -sin], [sin, cos]])

def _rz_tensor(angle: float) -> np.ndarray:
    """RZ rotation gate tensor."""
    return np.array([[np.exp(-1j * angle / 2), 0], 
                     [0, np.exp(1j * angle / 2)]])

def _rx_tensor(angle: float) -> np.ndarray:
    """RX rotation gate tensor."""
    cos = np.cos(angle / 2)
    sin = np.sin(angle / 2)
    return np.array([[cos, -1j * sin], [-1j * sin, cos]])
```

### Step 3: Create Tensor Network Simulator
```python
# File: ariadne/backends/tensor_network_backend.py

import numpy as np
import quimb.tensor as qtn
import cotengra as ctg
from typing import Dict, Any, Optional
from qiskit import QuantumCircuit

class TensorNetworkBackend:
    """
    Real tensor network simulation using Quimb + Cotengra.
    
    Efficient for circuits with low treewidth (sparse entanglement).
    """
    
    def __init__(self, max_bond_dim: int = 32, max_time: float = 60.0):
        self.max_bond_dim = max_bond_dim
        self.max_time = max_time
        
    def simulate(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, int]:
        """
        Simulate circuit using tensor network contraction.
        """
        # Convert circuit to tensor network
        tn = self._circuit_to_tensor_network(circuit)
        
        # Find optimal contraction order
        optimizer = ctg.ReusableHyperOptimizer(
            methods=['greedy', 'kahypar', 'random'],
            max_repeats=16,
            parallel=True,
            max_time=self.max_time
        )
        
        # Contract the tensor network
        try:
            result = tn.contract(optimize=optimizer, max_bond=self.max_bond_dim)
        except Exception as e:
            raise RuntimeError(f"Tensor network contraction failed: {e}")
        
        # Sample measurement outcomes
        return self._sample_from_statevector(result, shots, circuit.num_qubits)
    
    def _circuit_to_tensor_network(self, circuit: QuantumCircuit) -> qtn.TensorNetwork:
        """Convert circuit to tensor network."""
        from .converters.tensor_network import circuit_to_tensor_network
        return circuit_to_tensor_network(circuit)
    
    def _sample_from_statevector(self, statevector: np.ndarray, shots: int, num_qubits: int) -> Dict[str, int]:
        """Sample measurement outcomes from state vector."""
        # Normalize state vector
        statevector = statevector / np.linalg.norm(statevector)
        
        # Calculate probabilities
        probabilities = np.abs(statevector) ** 2
        
        # Sample outcomes
        outcomes = np.random.choice(len(probabilities), size=shots, p=probabilities)
        
        # Count outcomes
        counts = {}
        for outcome in outcomes:
            bit_string = format(outcome, f'0{num_qubits}b')
            counts[bit_string] = counts.get(bit_string, 0) + 1
        
        return counts
```

### Step 4: Update Router
```python
# File: ariadne/router.py (update _simulate_tensor_network method)

def _simulate_tensor_network(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    """
    Real tensor network simulation using Quimb + Cotengra.
    
    For non-Clifford circuits with low treewidth.
    """
    try:
        from .backends.tensor_network_backend import TensorNetworkBackend
        
        backend = TensorNetworkBackend()
        return backend.simulate(circuit, shots)
        
    except ImportError as exc:
        raise RuntimeError("Quimb/Cotengra not installed") from exc
    except Exception as exc:
        # Fallback to Qiskit if tensor network fails
        return self._simulate_qiskit(circuit, shots)
```

### Step 5: Update Routing Logic
```python
# File: ariadne/route/analyze.py (add treewidth estimation)

def estimate_treewidth(circuit: QuantumCircuit) -> int:
    """
    Estimate the treewidth of the circuit's interaction graph.
    
    Lower treewidth = better for tensor network simulation.
    """
    import networkx as nx
    from scipy.sparse.csgraph import minimum_spanning_tree
    
    # Build interaction graph
    G = interaction_graph(circuit)
    
    # Estimate treewidth using various heuristics
    try:
        # Use NetworkX treewidth approximation
        tw = nx.approximation.treewidth_min_degree(G)
        return tw
    except:
        # Fallback: use edge count as proxy
        return min(G.number_of_edges(), 20)

def should_use_tensor_network(circuit: QuantumCircuit) -> bool:
    """
    Determine if circuit should use tensor network simulation.
    """
    analysis = analyze_circuit(circuit)
    
    # Don't use for Clifford circuits (use Stim instead)
    if analysis["is_clifford"]:
        return False
    
    # Don't use for very small circuits (use DDSIM instead)
    if analysis["num_qubits"] <= 4:
        return False
    
    # Don't use for very large circuits (use Qiskit instead)
    if analysis["num_qubits"] > 30:
        return False
    
    # Use if treewidth is low (sparse entanglement)
    treewidth = estimate_treewidth(circuit)
    return treewidth <= 10
```

### Step 6: Add Tests
```python
# File: tests/test_tensor_network_backend.py

import pytest
import numpy as np
from qiskit import QuantumCircuit
from ariadne.backends.tensor_network_backend import TensorNetworkBackend

def test_hadamard_circuit():
    """Test tensor network with simple Hadamard circuit."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.h(1)
    qc.measure_all()
    
    backend = TensorNetworkBackend()
    result = backend.simulate(qc, shots=1000)
    
    # Should get roughly equal distribution
    assert len(result) == 4
    assert all(count > 200 for count in result.values())

def test_entangled_circuit():
    """Test tensor network with entangled circuit."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    backend = TensorNetworkBackend()
    result = backend.simulate(qc, shots=1000)
    
    # Should get |00⟩ and |11⟩ only
    assert "00" in result
    assert "11" in result
    assert "01" not in result
    assert "10" not in result

def test_non_clifford_circuit():
    """Test tensor network with non-Clifford circuit."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.t(2)  # Non-Clifford gate
    qc.measure_all()
    
    backend = TensorNetworkBackend()
    result = backend.simulate(qc, shots=1000)
    
    # Should complete without error
    assert len(result) > 0
    assert sum(result.values()) == 1000
```

## 🧪 Testing Strategy

### Test Circuits
1. **Small Clifford circuits** (should use Stim, not tensor network)
2. **Small non-Clifford circuits** (should use DDSIM, not tensor network)
3. **Medium sparse circuits** (should use tensor network)
4. **Large dense circuits** (should use Qiskit, not tensor network)

### Performance Benchmarks
- Compare against Qiskit state vector simulation
- Measure memory usage and scaling
- Test with various circuit types (QAOA, VQE, etc.)

## 🎯 Success Criteria

1. **Real quantum simulation** - Not random sampling
2. **Correct results** - Match Qiskit for small circuits
3. **Efficient scaling** - Handle larger circuits than Qiskit
4. **Proper routing** - Only use when appropriate
5. **Error handling** - Graceful fallbacks

---

**This will make Ariadne actually useful for non-Clifford circuits!** 🚀
