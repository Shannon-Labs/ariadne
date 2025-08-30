from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as qasm3_dumps, loads as qasm3_loads

from ariadne.io.qasm3 import load_qasm3, dump_qasm3
from ariadne.verify.qcec import witness_if_not
from examples._util import write_report


def build_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.t(1)
    qc.cx(1, 2)
    qc.h(2)
    return qc


def main() -> None:
    circ = build_circuit()
    qasm_text = qasm3_dumps(circ)

    # Parse via OpenQASM3 reference parser and roundtrip back
    program = load_qasm3(qasm_text)
    qasm_round = dump_qasm3(program)
    circ_round = qasm3_loads(qasm_round)

    w = witness_if_not(circ, circ_round)
    report = f"""
# QASM3 Roundtrip Report

- Gates: {len(circ.data)}
- Equivalent: {w.equivalent}
- Method: {w.message}
"""
    path = write_report("01_qasm3_roundtrip", report)
    print(f"Wrote report to {path}")


if __name__ == "__main__":
    main()

