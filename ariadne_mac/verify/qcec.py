from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass
class EquivalenceWitness:
    equivalent: bool
    message: str = ""


def _try_qcec(a: QuantumCircuit, b: QuantumCircuit) -> Optional[EquivalenceWitness]:  # pragma: no cover
    try:
        import mqt.qcec as qcec  # type: ignore
    except Exception:
        return None

    try:
        result = qcec.verify(a, b)
        eq = getattr(result, "equivalent", False)
        msg = getattr(result, "message", "") or ("equivalent" if eq else "not equivalent")
        return EquivalenceWitness(eq, msg)
    except Exception as e:
        return EquivalenceWitness(False, f"QCEC error: {e}")


def statevector_equiv(a: QuantumCircuit, b: QuantumCircuit) -> bool:
    try:
        sv_a = Statevector.from_instruction(a)
        sv_b = Statevector.from_instruction(b)
        return sv_a.equiv(sv_b)
    except Exception:
        return False


def assert_equiv(a: QuantumCircuit, b: QuantumCircuit) -> None:
    w = _try_qcec(a, b)
    if w is None:
        raise RuntimeError("mqt.qcec not available.")
    if not w.equivalent:
        raise AssertionError(f"Not equivalent: {w.message}")


def counterexample_dump(a: QuantumCircuit, b: QuantumCircuit) -> str:
    """Return a simple counterexample dump or reason when not available."""
    w = _try_qcec(a, b)
    if w is None:
        return "QCEC unavailable; no counterexample."
    if w.equivalent:
        return "Equivalent per QCEC."
    return f"QCEC says not equivalent: {w.message}"


def _circuit_hash_qasm3(c: QuantumCircuit) -> str:
    try:
        from qiskit.qasm3 import dumps as q3_dumps

        text = q3_dumps(c)
    except Exception:
        try:
            text = c.qasm()  # type: ignore[attr-defined]
        except Exception:
            text = str(c)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_qcec_artifact(
    pass_name: str,
    before: QuantumCircuit,
    after: QuantumCircuit,
    artifact_dir: Path = Path("reports") / "qcec",
) -> Dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    w = _try_qcec(before, after)
    record: Dict[str, Any] = {
        "pass": pass_name,
        "before_hash": _circuit_hash_qasm3(before),
        "after_hash": _circuit_hash_qasm3(after),
        "qcec_available": w is not None,
        "equivalent": (w.equivalent if w is not None else None),
        "message": (w.message if w is not None else "QCEC unavailable"),
    }
    path = artifact_dir / f"{pass_name}_{record['before_hash'][:8]}_{record['after_hash'][:8]}.json"
    path.write_text(__import__("json").dumps(record, indent=2))
    return record
