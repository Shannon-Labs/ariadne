from __future__ import annotations

import pytest


def low_treewidth_circuit(n: int = 16):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def test_tn_picked_when_mem_cap_small() -> None:
    pytest.importorskip("qiskit")
    from ariadne_mac.route.execute import decide_backend

    circ = low_treewidth_circuit()
    backend = decide_backend(circ, mem_cap_bytes=1 << 20)  # 1 MiB cap forces TN
    assert backend == "tn"

