from __future__ import annotations

from typing import Tuple

from qiskit import QuantumCircuit

from ..verify.qcec import write_qcec_artifact


def cancel_h_pairs(circ: QuantumCircuit) -> QuantumCircuit:
    out = QuantumCircuit(circ.num_qubits, circ.num_clbits)
    i = 0
    data = list(circ.data)
    while i < len(data):
        inst, qargs, cargs = data[i]
        if i + 1 < len(data):
            inst2, qargs2, cargs2 = data[i + 1]
            if inst.name == "h" and inst2.name == "h" and [q.index for q in qargs] == [
                q.index for q in qargs2
            ]:
                i += 2
                continue
        out.append(inst, qargs, cargs)
        i += 1
    return out


def cancel_adjacent_cx(circ: QuantumCircuit) -> QuantumCircuit:
    out = QuantumCircuit(circ.num_qubits, circ.num_clbits)
    i = 0
    data = list(circ.data)
    while i < len(data):
        inst, qargs, cargs = data[i]
        if i + 1 < len(data):
            inst2, qargs2, cargs2 = data[i + 1]
            if inst.name == inst2.name == "cx" and [q.index for q in qargs] == [
                q.index for q in qargs2
            ]:
                i += 2
                continue
        out.append(inst, qargs, cargs)
        i += 1
    return out


def zx_peephole_optimize(circ: QuantumCircuit, write_artifact: bool = True) -> Tuple[QuantumCircuit, dict]:
    """Apply safe peepholes and write a QCEC artifact record."""
    after = cancel_adjacent_cx(cancel_h_pairs(circ))
    record = {}
    if write_artifact:
        record = write_qcec_artifact("zx_peepholes", circ, after)
    return after, record

