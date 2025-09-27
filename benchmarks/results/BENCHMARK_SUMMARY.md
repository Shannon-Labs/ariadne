# Ariadne Benchmark Summary

**Generated**: 2025-09-20 12:45:20

## 🎯 What the Benchmarks Actually Show

Based on real performance data from `results/router_benchmark_results.json`:

| Circuit | Category | Router Backend | Router Time (ms) | Direct Qiskit (ms) | Stim (ms) | Tensor Network (ms) | Notes |
|---------|----------|----------------|------------------|-------------------|-----------|-------------------|-------|
| ghz_chain_10 | Clifford | Stim | 17.9 | 1.47 | 9.43 | 882 | Router overhead + Stim conversion cost more time than running Qiskit directly, but Stim allows scaling beyond 24 qubits |
| random_clifford_12 | Clifford | Stim | 339 | 13.2 | 61.4 | 141 | Router selects Stim correctly, but conversion cost dominates for moderate circuits |
| random_nonclifford_8 | Non-Clifford | Tensor network | 111 | 1.65 | – | 62.3 | Exact tensor contraction is heavy; accuracy gain only matters on larger/structured problems |
| qaoa_maxcut_8_p3 | Algorithmic | Tensor network | 67.6 | 1.34 | – | 80.0 | Router works; no speedup vs. Qiskit because everything falls back to CPU |
| vqe_ansatz_12 | Algorithmic | Tensor network | 68.3 | 5.03 | – | 63.1 | Router roughly matches tensor-network baseline; still slower than Qiskit on CPU |

## 🍎 Metal Backend Results (Apple Silicon)

⚠️ **Metal benchmarks failed.** Every circuit crashed with `UNKNOWN: error: unknown attribute code: 22 (StableHLO_v1.10.9)`, matching the runtime warning we observe when JAX/Metal falls back to the statevector sampler.

See `results/metal_benchmark_results.json` for details (all entries report `success: false` and `execution_time: Infinity`).

## 🚀 CUDA Backend Results (NVIDIA)

⚠️ **CUDA hardware not present on this MacBook.** `cuda_vs_cpu.py` executed, but only Qiskit CPU baselines were recorded. No Ariadne CUDA timings are available.

## 📊 Key Findings

### ✅ What Works
- **Router correctly selects backends** - Stim for Clifford, tensor networks for complex circuits
- **Capability extension** - Can simulate 24+ qubit Clifford circuits that Qiskit Basic can't handle
- **Automatic routing** - No manual backend selection needed
- **Graceful fallbacks** - Router falls back to CPU when GPU backends fail

### ❌ What Doesn't Work
- **Metal backend broken** - StableHLO error 22 prevents GPU acceleration
- **CUDA untested** - No NVIDIA hardware available for testing
- **No speed improvements** - Router overhead makes small circuits slower than direct Qiskit
- **Performance claims** - All previous speedup numbers were from non-functional backends

### 🎯 Honest Assessment
- **NOT "We run faster"** - Router has overhead on small circuits
- **ACTUALLY "We automatically route your circuit to the right simulator and save you from backend limits"**
- **Value proposition** - Capability extension and developer productivity, not raw speed

## 🔧 Usage

```python
from ariadne import simulate
from qiskit import QuantumCircuit

# Automatic backend selection
result = simulate(circuit, shots=1000)
print(f"Backend: {result.backend_used}")
print(f"Time: {result.execution_time:.4f}s")
```

## 📈 Performance Notes

- Metal backend currently unusable on this system because the JAX Metal runtime rejects StableHLO bytecode; router falls back to the CPU statevector path
- CUDA backend cannot be evaluated without NVIDIA hardware
- Qiskit CPU baselines (shots=1000) remain <2 ms for 3–5 qubit cases, ~1.3 ms for the 8-qubit Clifford ladder
- Router overhead is significant for small circuits but enables large circuits that would otherwise crash
