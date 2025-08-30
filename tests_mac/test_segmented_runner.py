from __future__ import annotations

import pytest


def test_segment_manifest_basic() -> None:
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from ariadne_mac.passes.segment import segment_circuit

    qc = QuantumCircuit(4)
    qc.h(0); qc.cx(0,1); qc.t(2); qc.cx(2,3)  # mix Clifford and non-Clifford
    segments, manifest = segment_circuit(qc, {"tn": {"treewidth_threshold": 12}})
    assert segments, "No segments returned"
    assert "segments" in manifest

