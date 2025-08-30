from __future__ import annotations

import pytest


def test_qcec_equiv_trivial_if_available() -> None:
    pytest.importorskip("qiskit")
    try:
        pytest.importorskip("mqt.qcec")
    except Exception:
        pytest.skip("QCEC not installed")

    from qiskit import QuantumCircuit
    from ariadne_mac.verify.qcec import assert_equiv

    qc = QuantumCircuit(1)
    qc.h(0); qc.h(0)
    assert_equiv(qc, qc)


def test_zne_improves_simple_model() -> None:
    from ariadne_mac.passes.mitigation import simple_zne

    ideal = 1.0
    noisy = 0.8

    def obs(scale: float) -> float:
        return ideal + (noisy - ideal) * scale

    est = simple_zne(obs, scales=(1.0, 2.0, 3.0), order=2)
    assert abs(est - ideal) < abs(noisy - ideal)

