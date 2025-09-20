# Ariadne API Reference

## High-level simulation

### `simulate(circuit: QuantumCircuit, shots: int = 1024)`
Route the circuit to one of the available backends and execute it.  Returns a
`SimulationResult` containing measurement counts, the backend that was used and
basic timing information.

```python
from qiskit import QuantumCircuit
from ariadne import simulate

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

result = simulate(qc, shots=512)
print(result.backend_used)
print(result.counts)
```

## QuantumRouter

`QuantumRouter` exposes the routing logic without executing the circuit.  Use it
when you want to inspect the recommended simulator before running the job.

```python
from ariadne import QuantumRouter

router = QuantumRouter()
decision = router.select_optimal_backend(qc)
print(decision.recommended_backend)
print(decision.channel_capacity_match)
```

Call `router.simulate(...)` to run the circuit using the router's decision.

## CUDA backend (optional)

The CUDA backend is a compact statevector simulator that can execute on the GPU
via CuPy and automatically falls back to NumPy when no CUDA runtime is detected.

```python
from ariadne.backends.cuda_backend import CUDABackend

backend = CUDABackend()
counts = backend.simulate(qc, shots=256)
```

Set `allow_cpu_fallback=False` to require a working CUDA installation.