from __future__ import annotations

import json
from pathlib import Path
import pytest


def test_qcec_artifact_written(tmp_path: Path) -> None:
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from ariadne_mac.verify.qcec import write_qcec_artifact

    a = QuantumCircuit(1)
    a.h(0); a.h(0)
    b = a.copy()
    out = write_qcec_artifact("peephole", a, b, artifact_dir=tmp_path)
    files = list(tmp_path.glob("*.json"))
    assert files, "artifact file not written"
    record = json.loads(files[0].read_text())
    assert "equivalent" in record

