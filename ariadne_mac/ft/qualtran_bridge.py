from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit


def build_demo_qsp_rotation(theta: float = 0.123) -> QuantumCircuit:
    """Simple single-qubit QSP-style phase rotation demo circuit."""
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.p(theta, 0)
    qc.h(0)
    return qc


def qualtran_to_qiskit(component: Any, *args: Any, **kwargs: Any) -> QuantumCircuit:  # pragma: no cover - stub
    try:
        import qualtran  # noqa: F401
        # TODO: translate a minimal Qualtran component to Qiskit when API is available
        # For now return a placeholder QSP rotation demo
        return build_demo_qsp_rotation(float(kwargs.get("theta", 0.123)))
    except Exception:
        # Graceful fallback when Qualtran not installed
        return build_demo_qsp_rotation(float(kwargs.get("theta", 0.123)))


def build_lcu_block_encoding(n: int = 2) -> QuantumCircuit:
    """Toy LCU-style block-encoding with ancilla prepare and controlled-U."""
    qc = QuantumCircuit(n + 1)
    anc = n
    qc.h(anc)
    for i in range(n):
        qc.cx(anc, i)
    qc.h(anc)
    return qc
