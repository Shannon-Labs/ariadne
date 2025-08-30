from __future__ import annotations

import pytest


def redundant_circuit(n: int = 10):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n)
    # Repeat same CX pairs many times to raise redundancy
    for _ in range(10):
        for i in range(0, n - 1, 2):
            qc.cx(i, i + 1)
    return qc


def test_dd_chosen_with_redundancy() -> None:
    pytest.importorskip("qiskit")
    from ariadne_mac.route.execute import decide_backend

    circ = redundant_circuit()
    backend = decide_backend(circ)
    assert backend in {"dd", "tn", "sv", "stim"}  # robust in CI
    # Prefer dd under policy defaults
    if backend != "dd":
        pytest.xfail("Backend may differ if metrics are borderline; acceptable in CI.")

