# Environment Setup

This guide summarises the steps that are typically required when working on
Ariadne.

## 1. Prerequisites

- Python 3.11 or newer
- Git
- (Optional) NVIDIA GPU with CUDA drivers and a matching CuPy wheel

## 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

_On Unix-like systems use `source .venv/bin/activate` instead._

## 3. Install Ariadne in editable mode

```powershell
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Add the `cuda` extra if you have a supported GPU:

```powershell
python -m pip install -e .[dev,cuda]
```

## 4. Verify the installation

```powershell
pytest -q
python - <<PY
from qiskit import QuantumCircuit
from ariadne import simulate

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

print(simulate(qc, shots=64).backend_used)
PY
```

```powershell
# Optional: run CUDA/CPU comparison benchmarks
python benchmarks\cuda_vs_cpu.py --cpu --shots 512 --repetitions 2
```
To confirm that the CUDA backend is available:

```powershell
python - <<PY
from ariadne.backends.cuda_backend import is_cuda_available
print(f"CUDA available: {is_cuda_available()}")
PY
```

## 5. Common development tasks

- `make lint`
- `make format`
- `make typecheck`
- `make test`

These commands wrap Ruff, MyPy and pytest respectively.
