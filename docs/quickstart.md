# Quick Start Guide

Get started with Ariadne in just 5 minutes! This guide will help you install Ariadne and run your first quantum circuit with intelligent routing.

## 🚀 Installation

### Option 1: Basic Installation
```bash
pip install ariadne-quantum
```

### Option 2: Apple Silicon (Recommended for Mac users)
```bash
pip install ariadne-quantum[apple]
```

### Option 3: CUDA Support (NVIDIA GPU users)
```bash
pip install ariadne-quantum[cuda]
```

### Option 4: Complete Installation (All features)
```bash
pip install ariadne-quantum[apple,cuda,viz,dev]
```

## 🎯 Your First Circuit

Let's start with a simple example that demonstrates Ariadne's intelligent routing:

```python
from ariadne import simulate
from qiskit import QuantumCircuit

# Create a simple Bell state circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Let Ariadne choose the best backend automatically
result = simulate(qc, shots=1000)

print(f\"Backend used: {result.backend_used}\")
print(f\"Results: {result.counts}\")
```

**Expected output:**
```
Backend used: qiskit_basic
Results: {'00': 496, '11': 504}
```

## 🔬 Clifford Circuit Example

Ariadne automatically detects Clifford circuits and routes them to Stim for massive performance gains:

```python
from ariadne import simulate
from qiskit import QuantumCircuit

# Create a 30-qubit GHZ state (Clifford circuit)
qc = QuantumCircuit(30, 30)
qc.h(0)
for i in range(29):
    qc.cx(i, i + 1)
qc.measure_all()

# This would crash regular Qiskit, but Ariadne routes to Stim
result = simulate(qc, shots=1000)
print(f\"Backend used: {result.backend_used}\")  # -> \"stim\"
print(f\"Simulation completed in: {result.execution_time:.3f}s\")
```

## ⚡ Apple Silicon Acceleration

On Apple Silicon Macs, Ariadne can leverage the Metal backend for hardware acceleration:

```python
from ariadne.backends.metal_backend import MetalBackend
from qiskit import QuantumCircuit

# Create a circuit with both Clifford and non-Clifford gates
qc = QuantumCircuit(8, 8)
qc.h(range(8))
for i in range(7):
    qc.cx(i, i + 1)
qc.ry(0.5, 4)  # Non-Clifford gate
qc.measure_all()

# Use Metal backend directly
backend = MetalBackend()
result = backend.simulate(qc, shots=1000)

print(f\"Metal mode: {backend.backend_mode}\")  # \"metal\" on Apple Silicon
print(f\"Execution time: {backend.last_summary.execution_time:.3f}s\")
```

## 🧠 Understanding Router Decisions

See what the router is thinking when it selects a backend:

```python
from ariadne import QuantumRouter
from qiskit import QuantumCircuit

# Create a mixed circuit
qc = QuantumCircuit(12, 12)
qc.h(range(6))
for i in range(5):
    qc.cx(i, i + 1)
qc.rz(0.25, 3)  # Non-Clifford gate
qc.measure_all()

# Analyze routing decision
router = QuantumRouter()
decision = router.select_optimal_backend(qc)

print(f\"Recommended backend: {decision.backend_name}\")
print(f\"Confidence: {decision.confidence:.2f}\")
print(f\"Reasoning: {decision.reasoning}\")
print(f\"Alternatives: {[alt.backend_name for alt in decision.alternatives]}\")
```

## 📊 Benchmarking

Compare performance across different backends:

```python
from ariadne import simulate
from ariadne.backends import QiskitBasicBackend, MetalBackend
from qiskit import QuantumCircuit
import time

# Create test circuit
qc = QuantumCircuit(10, 10)
qc.h(range(10))
for i in range(9):
    qc.cx(i, i + 1)
qc.measure_all()

# Benchmark automatic routing
start = time.time()
result_auto = simulate(qc, shots=1000)
time_auto = time.time() - start

# Benchmark specific backends
qiskit_backend = QiskitBasicBackend()
start = time.time()
result_qiskit = qiskit_backend.simulate(qc, shots=1000)
time_qiskit = time.time() - start

print(f\"Automatic routing: {time_auto:.3f}s ({result_auto.backend_used})\")
print(f\"Qiskit direct: {time_qiskit:.3f}s\")
print(f\"Speedup: {time_qiskit/time_auto:.2f}x\")
```

## 🎛️ Configuration

Customize Ariadne's behavior with configuration:

```python
from ariadne import QuantumRouter, RouterConfig

# Create custom configuration
config = RouterConfig(
    prefer_accuracy=True,      # Prioritize accuracy over speed
    enable_metal=True,         # Enable Metal backend on Apple Silicon
    enable_cuda=False,         # Disable CUDA (if not available)
    clifford_threshold=0.95,   # Stim threshold for Clifford detection
    max_qubits_statevector=20  # Maximum qubits for statevector simulation
)

# Use custom configuration
router = QuantumRouter(config=config)
result = router.simulate(qc, shots=1000)
```

## 🔧 Troubleshooting

### Common Issues

**Metal backend not available on Apple Silicon:**
```bash
# Reinstall JAX with Metal support
pip uninstall jax jaxlib jax-metal
pip install jax[metal]
```

**CUDA backend issues:**
```bash
# Install CUDA dependencies
pip install ariadne-quantum[cuda]
# Verify CUDA installation
python -c \"import cupy; print(cupy.cuda.runtime.getDeviceCount())\"
```

**Import errors:**
```bash
# Update to latest version
pip install --upgrade ariadne-quantum
```

## 🎯 Next Steps

- 📚 Read the [Core Concepts](concepts/routing.md) to understand how routing works
- 🔧 Explore [Backend Integration](backends/interface.md) for advanced usage
- 📈 Check out [Performance Benchmarks](performance/benchmarks.md) for detailed comparisons
- 🤝 See [Contributing Guide](../CONTRIBUTING.md) to help improve Ariadne

## 💡 Key Takeaways

- **Zero Configuration**: `simulate(circuit, shots)` works out of the box
- **Automatic Optimization**: Ariadne chooses the best backend for your circuit
- **Platform Acceleration**: Native support for Apple Silicon and NVIDIA GPUs
- **Honest Performance**: We provide real benchmarks, not marketing claims
- **No ML, Just Math**: Deterministic routing based on mathematical analysis

---

*Ready to dive deeper? Check out our [Basic Usage Guide](basic-usage.md) for more detailed examples and patterns.*