# Ariadne: Intelligent Quantum Circuit Routing

Ariadne analyses a quantum circuit and chooses an execution backend that is a
reasonable fit for the circuit structure.  It understands stabiliser-friendly
circuits, dense universal circuits and a handful of hardware-specific options.

The project is primarily a teaching and experimentation tool: it wraps a small
set of existing simulators (Stim, Qiskit, tensor network sketches, MQT DDSIM,
JAX/Metal on Apple Silicon) and provides a consistent API for routing and
simulation.  A lightweight CUDA backend is included for environments where
[CuPy](https://cupy.dev/) is available; it falls back to a NumPy implementation
when no GPU is detected.

## Getting started

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Run the default test suite:

```bash
pytest -q
```

## Quick example

```python
from qiskit import QuantumCircuit
from ariadne import simulate

bell = QuantumCircuit(2, 2)
bell.h(0)
bell.cx(0, 1)
bell.measure_all()

result = simulate(bell, shots=512)
print(result.backend_used)
print(result.counts)
```

To inspect the routing decision directly:

```python
from ariadne import QuantumRouter

router = QuantumRouter()
decision = router.select_optimal_backend(bell)
print(decision.recommended_backend)
print(decision.channel_capacity_match)
```

## Optional CUDA backend

The CUDA backend is intentionally conservative: it provides a correct
statevector simulator that can execute on the GPU when CuPy is installed, and it
uses a well-tested NumPy implementation otherwise.  There are no baked-in
performance guarantees—treat it as an optional accelerator for small and
medium-sized circuits.

```bash
python -m pip install cupy-cuda12x  # choose the wheel that matches your driver
```

```python
from ariadne.backends.cuda_backend import CUDABackend

backend = CUDABackend()
counts = backend.simulate(bell, shots=256)
```

When CUDA support is not available the same code falls back to the CPU.  Set
``allow_cpu_fallback=False`` to require a working GPU environment.

## Development tasks

- `make lint` – run Ruff
- `make format` – format source files with Ruff
- `make typecheck` – MyPy (strict mode)
- `make test` – execute the pytest suite

## Status

Ariadne is a research project.  Expect the API to evolve; measurements produced
by experiments should be validated independently before drawing performance
conclusions.