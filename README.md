# Ariadne: Intelligent Quantum Circuit Routing

Ariadne analyses a quantum circuit and chooses an execution backend that is a
reasonable fit for the circuit structure. It wraps a set of existing simulators
(Stim, Qiskit, tensor-network sketches, MQT DDSIM, and an optional CUDA backend)
and presents a consistent API for routing and simulation.

## Getting started

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Run the default test suite:

```bash
python -m pytest -q
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

The CUDA backend is a correctness-first statevector simulator. It executes on
CuPy when an NVIDIA GPU is available and falls back to NumPy otherwise.

```bash
python -m pip install -e .[dev,cuda]
```

```python
from ariadne.backends.cuda_backend import CUDABackend

backend = CUDABackend()
counts = backend.simulate(bell, shots=256)
```

Disable the CPU fallback by passing ``allow_cpu_fallback=False`` if you want the
backend to raise when CUDA is not present.

## Benchmarks

A lightweight benchmark harness is available in
`benchmarks/cuda_vs_cpu.py`. Run it from the repository root:

```bash
python benchmarks/cuda_vs_cpu.py --cpu --shots 512 --repetitions 2
```

The script measures Ariadne's CUDA and CPU modes alongside Qiskit's
`basic_simulator`, printing a timing table and optional JSON output (see
`python benchmarks/cuda_vs_cpu.py --help`).

## Development tasks

- `make lint`
- `make format`
- `make typecheck`
- `make test`

## Status

Ariadne is a research project. Expect the API to evolve; validate performance
claims independently before relying on them.