from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass
class EquivalenceWitness:
    equivalent: bool
    message: str = ""


def _try_qcec(
    a: QuantumCircuit, b: QuantumCircuit
) -> Optional[EquivalenceWitness]:  # pragma: no cover - optional path
    try:
        import mqt.qcec as qcec  # type: ignore
    except Exception:
        return None

    try:
        # The qcec.verify API returns a result object with an `equivalent` field in recent versions.
        result = qcec.verify(a, b)
        eq = getattr(result, "equivalent", False)
        msg = getattr(result, "message", "") or ("equivalent" if eq else "not equivalent")
        return EquivalenceWitness(eq, msg)
    except Exception as e:
        return EquivalenceWitness(False, f"QCEC error: {e}")


def statevector_equiv(a: QuantumCircuit, b: QuantumCircuit) -> bool:
    """Fallback statevector equivalence check for small circuits.

    Note: This compares statevectors ignoring global phase and does not handle dynamic circuits.
    """
    try:
        sv_a = Statevector.from_instruction(a)
        sv_b = Statevector.from_instruction(b)
        return sv_a.equiv(sv_b)
    except Exception:
        return False


def assert_equiv(a: QuantumCircuit, b: QuantumCircuit) -> None:
    """Assert functional equivalence using MQT QCEC when available, otherwise raise.

    For robust workflows, call `witness_if_not` to get detail instead of raising.
    """
    w = _try_qcec(a, b)
    if w is None:
        raise RuntimeError("mqt.qcec is not available for equivalence checking.")
    if not w.equivalent:
        raise AssertionError(f"Circuits are not equivalent: {w.message}")


def witness_if_not(a: QuantumCircuit, b: QuantumCircuit) -> EquivalenceWitness:
    """Return a witness of equivalence or a reason when not equivalent.

    Falls back to statevector comparison if QCEC is missing.
    """
    w = _try_qcec(a, b)
    if w is not None:
        return w
    ok = statevector_equiv(a, b)
    return EquivalenceWitness(ok, "statevector comparison")

