# Ariadne: Intelligent Quantum Router 🔮

## Automatic routing that unlocks larger circuits **and** Apple Silicon acceleration

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Ariadne analyzes your quantum circuit and routes it to the simulator that can actually run it. You get:

- **Automatic Clifford → Stim routing** so 30–50 qubit stabilizer circuits *just work* (no more Qiskit 24-qubit crashes).
- **Hybrid Metal backend** on Apple Silicon delivering 1.4–1.8× speedups over the CPU baseline for general circuits.
- **Single zero-config API** that hides the differences between Qiskit, Stim, tensor networks, and Metal.
- **Graceful fallbacks** when a specialist backend cannot handle a circuit—Ariadne always returns a result.

We are not promising “quantum speedups.” We are making *existing* simulators automatic, accessible, and faster on Apple Silicon.

---

## ✅ What’s working today

| Capability | What it means for you |
|------------|-----------------------|
| **Stim auto-detection** | Clifford circuits are routed to Stim without changing your code. Example: 50-qubit GHZ states simulated in milliseconds. |
| **Metal hybrid backend (Apple Silicon)** | New backend (`ariadne.backends.metal_backend`) bypasses JAX’s Metal bug and yields 1.4–1.8× speedups vs. CPU on M4 Max. |
| **Router intelligence** | Analyzes entropy, treewidth, and gate mix to pick Stim, Metal, generic statevector, tensor network, or DDSIM automatically. |
| **Zero configuration** | `simulate(circuit, shots)` is all you need—no vendor-specific imports. |
| **Open source + extensible** | Apache 2.0, modular backend interface, detailed integration guide. |

### ❗ Honest limitations (September 2025)
- CUDA backend still requires an NVIDIA GPU (not covered in current benchmarks).
- Routing adds overhead for very small circuits; call Qiskit directly if you only need a few qubits.
- Metal backend currently targets Apple Silicon (macOS 14+, Python 3.10+ with the bundled hybrid backend).
- Project is beta—verify results for production workloads.

---

## 🚀 Quickstart (5 minutes)

```bash
pip install ariadne-quantum
```

### 1. Run a 30-qubit Clifford circuit that crashes plain Qiskit
```python
from ariadne import simulate
from qiskit import QuantumCircuit

qc = QuantumCircuit(30, 30)
qc.h(0)
for i in range(29):
    qc.cx(i, i + 1)
qc.measure_all()

result = simulate(qc, shots=1000)
print("Backend:", result.backend_used)  # -> stim
print("Unique outcomes:", len(result.counts))
```

### 2. Verify Metal acceleration on Apple Silicon
```python
from ariadne.backends.metal_backend import MetalBackend
from qiskit import QuantumCircuit

qc = QuantumCircuit(6, 6)
qc.h(range(6))
for i in range(5):
    qc.cx(i, i + 1)
qc.ry(0.42, 2)
qc.measure_all()

backend = MetalBackend()
counts = backend.simulate(qc, shots=1000)
print("Metal mode:", backend.backend_mode)   # 'metal' on Apple Silicon
print("Execution time (s):", backend.last_summary.execution_time)
```

### 3. Ask the router what it’s thinking
```python
from ariadne import QuantumRouter

router = QuantumRouter()
decision = router.select_optimal_backend(qc)
print(decision)
```

---

## 🔌 Backend support matrix

| Backend | Status | Typical use | Notes |
|---------|--------|-------------|-------|
| **Stim** | ✅ | Clifford / stabilizer circuits | Auto-selected when `is_clifford` is true. Enables >24 qubit circuits. |
| **Metal (Apple Silicon)** | ✅ | Dense non-Clifford circuits up to ~12 qubits | Hybrid NumPy + Accelerate path; 1.4–1.8× faster than CPU baseline. |
| **Qiskit Basic** | ✅ | General fallback | Always available; deterministic counts. |
| **Tensor network (Quimb + Cotengra)** | ✅ | Low treewidth, memory-bound circuits | Exact contraction; slower but handles structured circuits. |
| **DDSIM** | ✅ | Decision diagram simulation | Optional extra backend. |
| **CUDA** | ⚠️ (opt-in) | High-parallel circuits on NVIDIA GPUs | Requires `ariadne.backends.cuda_backend` dependencies and hardware. |

---

## 📊 Benchmarks (September 2025)

### Apple Silicon Metal vs. CPU (`python benchmarks/metal_vs_cpu.py --shots 1000`)

| Circuit archetype | Qiskit CPU (ms) | Ariadne Metal (ms) | Speedup |
|-------------------|-----------------|--------------------|---------|
| Small Clifford (H+CX) | 0.64 | 0.45 | **1.43×** |
| Medium Clifford | 1.05 | 0.63 | **1.66×** |
| Small general (H, CX, RY) | 0.76 | 0.42 | **1.82×** |
| Medium general | 1.15 | 0.68 | **1.67×** |
| Large Clifford | 1.90 | 1.34 | **1.41×** |

Numbers come from `results/metal_benchmark_results.json` on an Apple M4 Max MacBook Pro (Python 3.13, Accelerate-enabled NumPy).

### Router comparison (`python benchmarks/router_comparison.py --shots 256 --repetitions 3`)

| Circuit | Router backend | Router mean (ms) | Direct backend mean (ms) | Notes |
|---------|----------------|------------------|--------------------------|-------|
| ghz_chain_10 | Stim | 17.9 | Stim 9.4 / Qiskit 1.5 | Router overhead dominates tiny circuits. |
| random_clifford_12 | Stim | 339 | Stim 61 / Qiskit 13 | Stim conversion is non-trivial; still required for >24 qubits. |
| random_nonclifford_8 | Tensor network | 111 | Qiskit 1.7 | Exact tensor contraction trades speed for fidelity. |
| qaoa_maxcut_8_p3 | Tensor network | 67.6 | Qiskit 1.3 | Router currently prioritizes accuracy over speed. |
| vqe_ansatz_12 | Tensor network | 68.3 | Qiskit 5.0 | Comparable to raw tensor contraction. |

**Takeaway:** Use Ariadne when you need automatic capability selection or Apple Silicon acceleration. For tiny circuits where you already know the right backend, direct calls remain faster.

---

## 🤝 Who benefits from Ariadne?

- **Researchers & students** exploring quantum error correction, stabilizer codes, or anything Clifford-heavy.
- **Developers** wanting a “just run it” API that chooses between Stim, Metal, tensor networks, and vanilla simulators.
- **Apple Silicon users** who want reproducible speedups without patching JAX themselves.
- **Backend authors** looking to plug their simulator into an open routing framework.

---

## ⚙️ Development setup

```bash
git clone https://github.com/Shannon-Labs/ariadne.git
cd ariadne
pip install -e .[dev]

# Run unit tests
make test

# Apple Silicon benchmarks
python benchmarks/metal_vs_cpu.py --shots 1000 --output results/metal_benchmark_results.json

# Routing comparison benchmarks
python benchmarks/router_comparison.py --shots 256 --repetitions 3
```

> Metal backend tip (Apple Silicon): ensure you are using Python ≥3.10 with the system Accelerate BLAS. If JAX detects only CPU devices, reinstall `jax-metal` from PyPI and restart the Python process.

---

## 🧠 Architecture at a glance

1. **Circuit analysis (`ariadne/route/analyze.py`)** — computes entropy, treewidth, Clifford ratio, two-qubit depth.
2. **Capacity scoring (`QuantumRouter.channel_capacity_match`)** — compares circuit metrics against backend profiles (Stim, Metal, Qiskit, tensor network, DDSIM, CUDA).
3. **Routing decision (`RoutingDecision`)** — returns recommended backend, alternatives, and confidence.
4. **Simulation adapters** — real implementations for Stim, Qiskit Basic, tensor networks, Metal hybrid backend, DDSIM, CUDA.
5. **Unified API** — `ariadne.simulate` and `QuantumRouter` coordinate everything.

The modular backend interface makes it straightforward to contribute new simulators—see `docs/INTEGRATION_GUIDE.md`.

---

## 🔭 Roadmap

- Add optional calibration command to derive backend scores from real benchmarks.
- Further tune routing heuristics to leverage Metal on mixed circuits automatically.
- Nightly CI on Apple Silicon runners.
- Optional GPU kernels for the Tensor Network backend.
- CUDA benchmarking once hardware is available.

Have ideas? Open an issue or a PR—we welcome contributions.

---

## 📄 License

Ariadne is released under the [Apache 2.0 License](LICENSE).

Let us know what you build! Tag @ShannonLabs or open a discussion in the repo. Contributions—from docs fixes to new backends—are warmly welcomed.🌟
