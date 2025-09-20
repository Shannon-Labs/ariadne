"""Heuristics for choosing a backend without executing the circuit."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Dict, Literal

from qiskit import QuantumCircuit

from .analyze import analyze_circuit

Backend = Literal["stim", "tn", "sv", "dd"]


def decide_backend(circuit: QuantumCircuit) -> Backend:
    metrics = analyze_circuit(circuit)

    if metrics.get("is_clifford", False):
        return "stim"

    if metrics.get("treewidth_estimate", 0) <= 10 and metrics.get("depth", 0) >= 4:
        return "tn"

    if metrics.get("num_qubits", 0) <= 20 or (
        metrics.get("two_qubit_depth", 0) >= max(1, metrics.get("depth", 1) // 2)
    ):
        return "sv"

    return "dd"


@dataclass
class Trace:
    backend: Backend
    wall_time_s: float
    metrics: Dict[str, float | int | bool]


def execute(circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, object]:  # pragma: no cover - integration helper
    backend = decide_backend(circuit)
    metrics = analyze_circuit(circuit)

    start = perf_counter()
    if backend == "stim":
        payload: Dict[str, object] = {"note": "Stim selected", "counts": None}
    elif backend == "tn":
        payload = {"note": "Tensor-network backend selected", "counts": None}
    elif backend == "sv":
        try:
            from qiskit.quantum_info import Statevector

            statevector = Statevector.from_instruction(circuit)
            payload = {"statevector": statevector.data}
        except Exception as exc:  # pragma: no cover - depends on qiskit features
            payload = {"error": str(exc)}
    else:  # dd
        try:
            import mqt.ddsim as ddsim

            simulator = ddsim.DDSIMProvider().get_backend("qasm_simulator")
            job = simulator.run(circuit, shots=shots)
            payload = {"counts": job.result().get_counts()}
        except Exception as exc:  # pragma: no cover - optional dependency
            payload = {"error": str(exc)}

    elapsed = perf_counter() - start

    trace = Trace(backend=backend, wall_time_s=elapsed, metrics=metrics)
    return {"result": payload, "trace": asdict(trace)}