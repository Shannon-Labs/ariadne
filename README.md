## Ariadne: A Classical Twin for Quantum Programs

Ariadne is a software-only "Classical Twin" for quantum programs. It:

- Parses and verifies OpenQASM 3 programs
- Compiles with noise-aware, auto-mitigation passes (ZNE, CDR, PEC, VD stubs)
- Emulates execution by routing subcircuits to the best simulator
- Estimates fault-tolerant resources for high-level algorithms via Azure RE

Core stack: Python 3.11+, Qiskit 1.0, OpenQASM 3, Mitiq, Stim, PyMatching, PennyLane, quimb/cotengra, NumPy/SciPy/NetworkX/JAX, Rich/Typer, Qualtran, MQT QCEC/DDSIM, Azure Resource Estimator.

Optional CUDA acceleration: cuQuantum/cuStateVec and nvidia-cuda-python. Optional CUDA-Q extras.

### Quickstart

1. Install

```
python -m pip install -U pip
python -m pip install -e .
```

2. Run examples (CPU, < 10 minutes expected):

```
make examples
```

3. Run tests

```
make test
```

### CLI

```
ariadne --help
```

Key commands: `parse-qasm3`, `verify`, `analyze`, `route`, `zne`.

### Apple‑Silicon variant

- A Mac-focused package `ariadne_mac` is included with a CLI `ariadne-mac` tuned for CPU execution on Apple‑Silicon Macs.
- Dispatches to Stim (stabilizer), Qiskit Aer (state‑vector), and quimb+cotengra (TN) with a default 24 GiB memory cap and planning stubs to keep examples fast.
- Optional JAX‑Metal acceleration for real‑valued helpers; automatically falls back to CPU for complex/float64 with a warning. See `docs/apple_silicon_notes.md`.

Quickstart (Mac Studio 36 GB)
- Install: `python -m pip install -U pip && python -m pip install -e .[apple,mitiq,viz,dev]`
- Threads: `export OMP_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 OPENBLAS_NUM_THREADS=8`
- Analyze: `ariadne-mac analyze examples/04_qasm3_qcec.py` (use `--mem-cap-gib` and `--threads`)
- Route & run: `ariadne-mac route --execute --mem-cap-gib 24 --precision fp32 examples/04_qasm3_qcec.py`
- Mitigate: `ariadne-mac mitigate --policy configs/mitigation.yaml examples/04_qasm3_qcec.py`
- Resources: `ariadne-mac resources`

### Design Principles

- Pure-Python control; optional CUDA path guarded
- Emitted equivalence checks around optimizations (skipped for mitigation)
- Deterministic seeds, reproducible decisions, auto-generated reports

### Docs

- See `docs/router_decisions.md` for how routing choices are made.
- See `docs/equivalence_proofs.md` for how equivalence is established.
- Citations in `CITATIONS.bib`.
