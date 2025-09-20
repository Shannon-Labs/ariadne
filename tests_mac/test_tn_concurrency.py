from __future__ import annotations

import pytest


@pytest.mark.slow
def test_tn_concurrent_smoke() -> None:
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from ariadne_mac.route.execute import _tn_run

    n = 12
    qc = QuantumCircuit(n)
    for i in range(n-1):
        qc.cx(i, i+1)
    out = _tn_run(qc, shots=1, mem_cap_bytes=2**30, run_id="test")
    assert isinstance(out, dict)

