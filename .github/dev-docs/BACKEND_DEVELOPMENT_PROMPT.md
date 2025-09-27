# Ariadne Backend Development Prompt

## 🚨 CRITICAL ISSUE IDENTIFIED

The current Ariadne implementation has a **fake tensor network backend** that just distributes shots randomly instead of doing real quantum simulation. This makes Ariadne essentially useless for non-Clifford circuits.

## 🎯 MISSION: Build Real Quantum Simulation Backends

### Current State Analysis

**What Works:**
- ✅ Stim backend: Real stabilizer tableau simulation (excellent for Clifford circuits)
- ✅ Qiskit backend: Real state vector simulation (works but slow)
- ✅ DDSIM backend: Real decision diagram simulation (good for small circuits)
- ✅ Hardware detection: Metal/CUDA detection works

**What's Broken:**
- ❌ Tensor Network backend: **FAKE** - just random sampling, not quantum simulation
- ❌ JAX/Metal backend: Incomplete implementation
- ❌ CUDA backend: Incomplete implementation
- ❌ No real tensor network simulation using Quimb/Cotengra

## 🏗️ BACKEND IMPLEMENTATION ROADMAP

### Phase 1: Fix Tensor Network Backend (HIGH PRIORITY)

**Current Fake Implementation:**
```python
def _simulate_tensor_network(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    # This is just random sampling - NOT quantum simulation!
    total_states = 2 ** min(num_qubits, 10)
    base_count = shots // total_states
    # ... distributes shots evenly across states
```

**Required Real Implementation:**
```python
def _simulate_tensor_network(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    """
    Real tensor network simulation using Quimb + Cotengra.
    
    For non-Clifford circuits that are too large for exact simulation
    but have low treewidth (sparse entanglement structure).
    """
    try:
        import quimb.tensor as qtn
        import cotengra as ctg
        from qiskit.quantum_info import Operator
        
        # Convert circuit to tensor network
        tn = self._circuit_to_tensor_network(circuit)
        
        # Use Cotengra to find optimal contraction order
        optimizer = ctg.ReusableHyperOptimizer(
            methods=['greedy', 'kahypar'],
            max_repeats=16,
            parallel=True
        )
        
        # Contract the tensor network
        result = tn.contract(optimize=optimizer)
        
        # Sample from the resulting state
        return self._sample_from_statevector(result, shots)
        
    except ImportError as exc:
        raise RuntimeError("Quimb/Cotengra not installed") from exc
```

**Key Components to Implement:**
1. `_circuit_to_tensor_network()` - Convert Qiskit circuit to Quimb tensor network
2. `_sample_from_statevector()` - Sample measurement outcomes from state vector
3. Treewidth estimation for routing decisions
4. Memory management for large circuits

### Phase 2: Complete JAX/Metal Backend (MEDIUM PRIORITY)

**Current Issues:**
- Metal detection shows "False" but JAX Metal is actually working
- Complex number handling needs improvement
- No proper state vector simulation

**Required Implementation:**
```python
def _simulate_jax_metal(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    """
    Real JAX/Metal simulation with proper complex number handling.
    """
    import jax.numpy as jnp
    from jax import jit, vmap
    
    # Initialize state vector
    num_qubits = circuit.num_qubits
    state = jnp.zeros(2**num_qubits, dtype=jnp.complex64)
    state = state.at[0].set(1.0)
    
    # Apply gates using JAX operations
    for instruction, qargs, _ in circuit.data:
        if instruction.name == 'h':
            state = self._apply_hadamard_jax(state, qargs[0])
        elif instruction.name == 'cx':
            state = self._apply_cnot_jax(state, qargs[0], qargs[1])
        # ... other gates
    
    # Sample measurements
    probabilities = jnp.abs(state) ** 2
    outcomes = jnp.random.choice(2**num_qubits, size=shots, p=probabilities)
    
    return self._count_outcomes(outcomes, num_qubits)
```

### Phase 3: Complete CUDA Backend (MEDIUM PRIORITY)

**Current Issues:**
- Incomplete implementation
- No proper CuPy integration
- Missing gate operations

**Required Implementation:**
```python
def _simulate_cuda(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    """
    Real CUDA simulation using CuPy for GPU acceleration.
    """
    import cupy as cp
    
    # Initialize state vector on GPU
    num_qubits = circuit.num_qubits
    state = cp.zeros(2**num_qubits, dtype=cp.complex128)
    state[0] = 1.0
    
    # Apply gates using CuPy operations
    for instruction, qargs, _ in circuit.data:
        if instruction.name == 'h':
            state = self._apply_hadamard_cuda(state, qargs[0])
        elif instruction.name == 'cx':
            state = self._apply_cnot_cuda(state, qargs[0], qargs[1])
        # ... other gates
    
    # Sample measurements
    probabilities = cp.abs(state) ** 2
    outcomes = cp.random.choice(2**num_qubits, size=shots, p=probabilities)
    
    return self._count_outcomes(cp.asnumpy(outcomes), num_qubits)
```

## 🔧 IMPLEMENTATION CHECKLIST

### Tensor Network Backend
- [ ] Install and configure Quimb + Cotengra
- [ ] Implement `_circuit_to_tensor_network()`
- [ ] Implement `_sample_from_statevector()`
- [ ] Add treewidth estimation for routing
- [ ] Add memory management for large circuits
- [ ] Test with various circuit types (QAOA, VQE, etc.)

### JAX/Metal Backend
- [ ] Fix Metal detection logic
- [ ] Implement proper complex number handling
- [ ] Add all standard quantum gates
- [ ] Optimize with JIT compilation
- [ ] Test on Apple Silicon hardware

### CUDA Backend
- [ ] Complete CuPy integration
- [ ] Implement all quantum gates
- [ ] Add memory management
- [ ] Test on NVIDIA hardware

### Routing Improvements
- [ ] Update routing logic to use real tensor network capabilities
- [ ] Add circuit analysis for treewidth estimation
- [ ] Improve backend selection criteria
- [ ] Add performance benchmarking

## 🧪 TESTING STRATEGY

### Test Circuits
1. **Clifford circuits** (should use Stim)
2. **Small non-Clifford circuits** (should use DDSIM)
3. **Large sparse circuits** (should use tensor network)
4. **Dense circuits** (should use Qiskit/Metal/CUDA)
5. **Mixed circuits** (should route appropriately)

### Performance Benchmarks
- Compare against direct Stim usage
- Compare against Qiskit Aer
- Compare against Cirq simulators
- Measure memory usage and scaling

## 🎯 SUCCESS CRITERIA

**Ariadne should provide:**
1. **Real quantum simulation** for all circuit types
2. **Automatic optimal backend selection** based on circuit properties
3. **Hardware acceleration** when available
4. **Performance improvements** over naive approaches
5. **Unified API** that "just works"

**Performance Targets:**
- Clifford circuits: Match Stim performance
- Non-Clifford circuits: 2-10x speedup over Qiskit
- Large circuits: Handle 50+ qubits efficiently
- Memory usage: Stay within available RAM

## 🚀 IMMEDIATE NEXT STEPS

1. **Fix the tensor network backend** - This is the most critical issue
2. **Test with real quantum algorithms** - QAOA, VQE, etc.
3. **Update documentation** - Remove fake performance claims
4. **Add proper error handling** - Graceful fallbacks
5. **Create comprehensive test suite** - Ensure correctness

## 💡 INSPIRATION

Look at these projects for reference:
- **Quimb**: Excellent tensor network library
- **Cirq**: Good quantum circuit simulation
- **PennyLane**: Good hardware acceleration
- **Qiskit Aer**: Good state vector simulation

The goal is to make Ariadne the **intelligent quantum simulation router** that automatically picks the best backend for each circuit, not just a Stim wrapper with fake backends.

---

**Remember: Real quantum simulation, not random sampling!** 🚀
