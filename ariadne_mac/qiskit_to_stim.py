from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import math
import stim
from qiskit import QuantumCircuit
from .passes.defer_measure import rewrite_measure_controls


SUPPORTED_ONE_Q = {"i", "x", "y", "z", "h", "s", "sdg"}
SUPPORTED_TWO_Q = {"cx", "cy", "cz", "swap"}


@dataclass
class ConversionResult:
    circuit: stim.Circuit
    measurements: int
    warnings: List[str]


def _is_pi_over_2(theta: float, tol: float = 1e-9) -> Optional[int]:
    # returns +1 for +pi/2, -1 for -pi/2, None otherwise
    twopi = 2 * math.pi
    t = (theta + twopi) % twopi
    if abs(t - (math.pi / 2)) < tol:
        return +1
    if abs(t - (3 * math.pi / 2)) < tol:
        return -1
    return None


def qiskit_to_stim(
    circ: QuantumCircuit,
    strip_measurements: bool = False,
) -> ConversionResult:
    # Attempt to defer measurements to enlarge Clifford regions if possible
    circ, _ = defer_measure_if_clifford(circ)
    s = stim.Circuit()
    warnings: List[str] = []
    qmap = {q: i for i, q in enumerate(circ.qubits)}
    clbit_index: Dict[object, int] = {c: i for i, c in enumerate(circ.clbits)}
    meas_counter = 0

    for inst, qargs, cargs in circ.data:
        name = inst.name.lower()

        # Conditions
        condition = getattr(inst, "condition", None)
        cond_target = None
        if condition is not None:
            cbit, val = condition
            if isinstance(val, int) and val in (0, 1) and len(qargs) == 1 and name in {"x", "z"}:
                # Very limited support: control by last measurement of the same cbit
                # Map to measurement record target; we approximate using rec[-1]
                # More general classical control is not supported.
                cond_target = (clbit_index.get(cbit, -1), val)
            else:
                raise ValueError(f"Unsupported conditional gate: {name} with condition {condition}")

        # Map qubits to indices
        qs = [qmap[q] for q in qargs]

        # Ignore barriers
        if name in {"barrier", "delay"}:
            continue

        # One-qubit Cliffords
        if name in SUPPORTED_ONE_Q:
            if name == "i":
                continue
            if cond_target is not None:
                # Limited: condition on last measurement record
                rec = stim.target_rec(-1)
                if name == "x":
                    s.append_operation("CX", [rec, qs[0]])
                elif name == "z":
                    s.append_operation("CZ", [rec, qs[0]])
                else:
                    raise ValueError(f"Conditional not supported for gate {name}")
                continue
            if name == "x":
                s.append_operation("X", qs)
            elif name == "y":
                s.append_operation("Y", qs)
            elif name == "z":
                s.append_operation("Z", qs)
            elif name == "h":
                s.append_operation("H", qs)
            elif name == "s":
                s.append_operation("S", qs)
            elif name == "sdg":
                s.append_operation("S_DAG", qs)
            continue

        # Two-qubit
        if name in SUPPORTED_TWO_Q:
            if name == "cx":
                s.append_operation("CX", qs)
            elif name == "cy":
                s.append_operation("CY", qs)
            elif name == "cz":
                s.append_operation("CZ", qs)
            elif name == "swap":
                # decompose to 3 CXs
                a, b = qs
                s.append_operation("CX", [a, b])
                s.append_operation("CX", [b, a])
                s.append_operation("CX", [a, b])
            continue

        # RX/RY(pi/2)
        if name in {"rx", "ry"}:
            sign = _is_pi_over_2(float(inst.params[0]))
            if sign is not None:
                op = "SQRT_X" if name == "rx" else "SQRT_Y"
                if sign < 0:
                    op += "_DAG"
                s.append_operation(op, qs)
                continue
            raise ValueError(f"Non-Clifford {name}({inst.params[0]}) not supported; use SV or TN backend.")

        # Reset/Measure
        if name == "reset":
            s.append_operation("R", qs)
            continue
        if name == "measure":
            if strip_measurements:
                continue
            s.append_operation("M", qs)
            meas_counter += len(qs)
            continue

        # Unsupported => structured rejection
        raise ValueError(
            f"Unsupported gate: {name}. This is likely non-Clifford. Try 'sv' or 'tn' backend."
        )

    return ConversionResult(s, meas_counter, warnings)


def stim_observable_expectations(
    circ: QuantumCircuit,
    shots: int,
    observables: Sequence[Tuple[str, Tuple[int, ...]]],
) -> Dict[str, float]:
    # Build a stim circuit without measurements up to end, then measure in chosen basis
    conv = qiskit_to_stim(circ, strip_measurements=True)
    results: Dict[str, float] = {}
    for kind, qubits in observables:
        s = stim.Circuit()
        s += conv.circuit
        if kind == "X":
            for q in qubits:
                s.append_operation("H", [q])
        elif kind == "Z" or kind == "ZZ":
            pass
        else:
            raise ValueError(f"Unsupported observable kind: {kind}")
        # Measure all qubits involved
        s.append_operation("M", list(qubits))
        sampler = s.compile_sampler()
        import numpy as np

        np.random.seed(1234)
        arr = sampler.sample(shots, rand_seed=1234)
        if kind == "Z" or kind == "X":
            b = arr[:, 0]
            exp = float((b == 0).mean() - (b == 1).mean())
        else:  # ZZ
            b0 = arr[:, 0]
            b1 = arr[:, 1]
            exp = float(((b0 ^ b1) == 0).mean() - ((b0 ^ b1) == 1).mean())
        results[f"{kind}{tuple(qubits)}"] = exp
    return results
