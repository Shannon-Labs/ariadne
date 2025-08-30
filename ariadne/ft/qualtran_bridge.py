from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit


def qualtran_to_qiskit(component: Any, *args: Any, **kwargs: Any) -> QuantumCircuit:  # pragma: no cover - stub
    """Build a Qiskit circuit from a Qualtran component.

    This is a placeholder that expects a Qualtran-like API; adapt as Qualtran stabilizes.
    """
    try:
        import qualtran  # noqa: F401
    except Exception as e:
        raise RuntimeError("Qualtran is required for this feature.") from e

    raise NotImplementedError("Qualtran bridge not yet implemented.")

