from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from qiskit import QuantumCircuit


CLIFFORD_ONE_Q = {"i", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg"}
CLIFFORD_TWO_Q = {"cx", "cz", "swap"}


def t_count(circ: QuantumCircuit) -> int:
    count = 0
    for inst, _, _ in circ.data:
        name = inst.name
        if name in {"t", "tdg"}:
            count += 1
        if name == "p":
            # Heuristic: phase of +/- pi/4
            try:
                theta = float(inst.params[0])
            except Exception:
                continue
            # Normalize to modulo 2pi
            import math

            theta = (theta + math.tau) % math.tau
            if math.isclose(theta, math.pi / 4, rel_tol=1e-9, abs_tol=1e-9) or math.isclose(
                theta, 7 * math.pi / 4, rel_tol=1e-9, abs_tol=1e-9
            ):
                count += 1
    return count


def two_qubit_depth(circ: QuantumCircuit) -> int:
    depth = 0
    current_layer_qubits = set()
    for inst, qargs, _ in circ.data:
        if inst.name in CLIFFORD_TWO_Q or inst.num_qubits == 2:
            # Qiskit 2.x no longer exposes .index directly, so pre-compute lookup table
    qubit_index_map = {qubit: idx for idx, qubit in enumerate(circ.qubits)}
    qubits = {qubit_index_map[q] for q in qargs}
            if current_layer_qubits & qubits:
                depth += 1
                current_layer_qubits = set(qubits)
            else:
                current_layer_qubits |= qubits
    return depth + (1 if current_layer_qubits else 0)


def trivial_cancel(circ: QuantumCircuit) -> QuantumCircuit:
    """Simple local algebraic optimizations: cancel adjacent inverses (H-H, X-X, CX-CX)."""
    out = QuantumCircuit(circ.num_qubits, circ.num_clbits)
    i = 0
    data = list(circ.data)
    while i < len(data):
        inst, qargs, cargs = data[i]
        if i + 1 < len(data):
            inst2, qargs2, cargs2 = data[i + 1]
            if (
                inst.name == inst2.name
                and # Qiskit 2.x no longer exposes .index directly, so pre-compute lookup table
    qubit_index_map = {qubit: idx for idx, qubit in enumerate(circ.qubits)}
    [qubit_index_map[q] for q in qargs] == [qubit_index_map[q] for q in qargs2]
                and # Qiskit 2.x no longer exposes .index directly, so pre-compute lookup table
    clbit_index_map = {clbit: idx for idx, clbit in enumerate(circ.clbits)}
    [clbit_index_map[c] for c in cargs] == [clbit_index_map[c] for c in cargs2]
                and inst.name in {"h", "x", "y", "z", "cx"}
            ):
                i += 2
                continue
        out.append(inst, qargs, cargs)
        i += 1
    return out


@dataclass
class ZXOptReport:
    t_before: int
    t_after: int
    twoq_depth_before: int
    twoq_depth_after: int
    used_engine: str


def zx_optimize(circ: QuantumCircuit) -> Tuple[QuantumCircuit, ZXOptReport]:
    """Optional PyZX/QuiZX optimization; falls back to trivial local cancellations.

    Returns optimized circuit and a report with T-count and two-qubit-depth deltas.
    """
    t_before = t_count(circ)
    d2_before = two_qubit_depth(circ)

    used = "trivial"
    optimized = trivial_cancel(circ)

    # Optional: try PyZX
    try:  # pragma: no cover - optional path
        import pyzx as zx  # type: ignore

        g = zx.Circuit.from_qiskit(circ).to_graph()
        zx.full_reduce(g)
        optimized = zx.extract_circuit(g).to_qiskit()
        used = "pyzx"
    except Exception:
        pass

    # TODO: try QuiZX if available

    t_after = t_count(optimized)
    d2_after = two_qubit_depth(optimized)
    return optimized, ZXOptReport(t_before, t_after, d2_before, d2_after, used)

