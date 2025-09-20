from __future__ import annotations

import pytest


def test_stim_conversion_random_near_clifford() -> None:
    pytest.importorskip("qiskit")
    import numpy as np
    from qiskit import QuantumCircuit
    from ariadne_mac.qiskit_to_stim import qiskit_to_stim

    rng = np.random.default_rng(1234)
    qc = QuantumCircuit(6, 2)
    for _ in range(20):
        i = rng.integers(0, 6)
        gate = rng.choice(["h", "s", "x", "y", "z", "rx", "ry", "cx", "cz", "swap", "measure", "reset"])  # near-Clifford
        if gate in {"rx", "ry"}:
            getattr(qc, gate)(np.pi/2, i)
        elif gate in {"cx", "cz"}:
            j = (i + 1) % 6
            getattr(qc, gate)(i, j)
        elif gate == "swap":
            j = (i + 1) % 6
            qc.swap(i, j)
        elif gate == "measure":
            qc.measure(i, 0)
        elif gate == "reset":
            qc.reset(i)
        else:
            getattr(qc, gate)(i)

    conv = qiskit_to_stim(qc)
    assert conv.circuit is not None

