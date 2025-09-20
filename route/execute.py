from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Dict, Literal

from qiskit import QuantumCircuit

from .analyze import analyze_circuit, is_clifford_circuit


Backend = Literal["stim", "tn", "sv", "dd"]


def decide_backend(circ: QuantumCircuit) -> Backend:
    metrics = analyze_circuit(circ)
    if metrics["is_clifford"]:
        return "stim"
    # Heuristics
    if metrics["treewidth_estimate"] <= 10 and metrics["depth"] >= 4:
        return "tn"
    if metrics["num_qubits"] <= 20 or metrics["two_qubit_depth"] >= metrics["depth"] // 2:
        return "sv"
    return "dd"


@dataclass
class Trace:
    backend: Backend
    wall_time_s: float
    metrics: Dict[str, float | int | bool]


def execute(circ: QuantumCircuit, shots: int = 1024) -> Dict[str, object]:  # pragma: no cover - integration
    backend = decide_backend(circ)
    metrics = analyze_circuit(circ)
    t0 = perf_counter()
    result: Dict[str, object]
    if backend == "stim":
        try:
            import stim  # type: ignore

            # Placeholder: Convert to stim? Nontrivial; for demo just report selection.
            result = {"counts": None, "note": "Stim selected (conversion not implemented)"}
        except Exception:
            result = {"counts": None, "note": "Stim selected, but stim not installed"}
    elif backend == "tn":
        try:
            import quimb  # noqa: F401
            import cotengra  # noqa: F401

            result = {"counts": None, "note": "TN selected (execution stub)"}
        except Exception:
            result = {"counts": None, "note": "TN selected, but quimb/cotengra not installed"}
    elif backend == "sv":
        try:
            from qiskit.quantum_info import Statevector

            sv = Statevector.from_instruction(circ)
            result = {"statevector": sv.data[: min(8, sv.dim)]}
        except Exception:
            result = {"counts": None, "note": "SV selected, but statevector fallback failed"}
    else:  # dd
        try:
            import mqt.ddsim as ddsim  # type: ignore

            # ddsim has multiple simulators; use vector simulator as placeholder
            sim = ddsim.DDSIMProvider().get_backend("qasm_simulator")
            job = sim.run(circ, shots=shots)
            result = {"counts": job.result().get_counts()}
        except Exception:
            result = {"counts": None, "note": "DD selected, but mqt.ddsim not installed"}

    t1 = perf_counter()
    trace = Trace(backend=backend, wall_time_s=t1 - t0, metrics=metrics)
    return {"result": result, "trace": asdict(trace)}

