from __future__ import annotations

import pytest


def test_defer_measure_keeps_clifford() -> None:
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from ariadne_mac.passes.defer_measure import defer_measure_if_clifford

    qc = QuantumCircuit(2, 1)
    qc.h(0); qc.cx(0,1); qc.measure(1,0)
    qc.x(0).c_if(qc.cregs[0], 1)
    out, ok = defer_measure_if_clifford(qc)
    assert ok is True

