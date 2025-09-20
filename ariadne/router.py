"""Ariadne Quantum Router.

Implements intelligent backend selection for quantum circuit simulation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np
from qiskit import QuantumCircuit, transpile

try:  # Optional imports
    from qiskit_aer import AerSimulator
except ImportError:  # pragma: no cover - fallback path
    AerSimulator = None  # type: ignore


STIM_GATE_MAP = {
    "h": "H",
    "s": "S",
    "sdg": "S_DAG",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "cx": "CX",
    "cz": "CZ",
    "swap": "SWAP",
}

CLIFFORD_GATES = {
    "h",
    "s",
    "sdg",
    "x",
    "y",
    "z",
    "cx",
    "cz",
    "swap",
    "id",
}


@dataclass
class SimulationResult:
    """Container for simulation results."""

    counts: Dict[str, int]
    backend: str
    time: float
    shots: int
    analysis: Dict[str, Any]


class QuantumRouter:
    """The brain of Ariadne - routes circuits to optimal backends."""

    def analyze_circuit(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Analyze circuit and determine optimal backend."""
        gate_counts = self._count_gates(circuit)
        entropy = self._calculate_entropy(gate_counts)

        non_clifford_gates = gate_counts.get("t", 0) + gate_counts.get("tdg", 0)
        is_clifford = non_clifford_gates == 0 and self._contains_only_clifford_gates(
            gate_counts.keys()
        )

        if is_clifford and circuit.num_qubits <= 500:
            backend = "stim"
            estimated_speedup = 1000
        elif circuit.num_qubits <= 30:
            backend = "qiskit_aer"
            estimated_speedup = 1
        else:
            backend = "tensor_network"
            estimated_speedup = 10  # heuristic

        return {
            "backend": backend,
            "entropy": entropy,
            "is_clifford": is_clifford,
            "gate_counts": gate_counts,
            "estimated_speedup": estimated_speedup,
            "qubits": circuit.num_qubits,
            "depth": int(circuit.depth()),
        }

    def _count_gates(self, circuit: QuantumCircuit) -> Dict[str, int]:
        """Count occurrences of each gate type."""
        counts: Dict[str, int] = {}
        for instruction, _, _ in circuit.data:
            gate_name = instruction.name.lower()
            if gate_name in {"measure", "barrier", "delay"}:
                continue
            counts[gate_name] = counts.get(gate_name, 0) + 1
        return counts

    def _calculate_entropy(self, gate_counts: Dict[str, int]) -> float:
        """Shannon entropy H = -Σ p(g) log2 p(g)."""
        total = sum(gate_counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in gate_counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    def _contains_only_clifford_gates(self, gates: Iterable[str]) -> bool:
        return all(g in CLIFFORD_GATES for g in gates)


def simulate_with_stim(circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, int]:
    """Simulate circuit using Stim for Clifford-only workloads."""
    try:
        import stim
    except ImportError as exc:  # pragma: no cover - requires external dependency
        raise RuntimeError(
            "Stim backend requested but package not installed. Install with `pip install stim`."
        ) from exc

    stim_circuit = stim.Circuit()
    measurement_map: list[tuple[int, int]] = []  # (measurement_index, clbit_index)

    # Qiskit 2.x no longer exposes .index directly, so pre-compute lookup tables.
    qubit_index_map = {qubit: idx for idx, qubit in enumerate(circuit.qubits)}
    clbit_index_map = {clbit: idx for idx, clbit in enumerate(circuit.clbits)}

    measurement_counter = 0
    for instruction, qargs, cargs in circuit.data:
        gate_name = instruction.name.lower()
        qubit_indices = [qubit_index_map[q] for q in qargs]

        if gate_name == "measure":
            if not cargs:
                continue
            for qubit, clbit in zip(qargs, cargs):
                stim_circuit.append("M", [qubit_index_map[qubit]])
                if clbit in clbit_index_map:
                    measurement_map.append((measurement_counter, clbit_index_map[clbit]))
                measurement_counter += 1
            continue
        if gate_name in {"barrier", "delay"}:
            continue
        else:
            stim_gate = STIM_GATE_MAP.get(gate_name)
            if stim_gate is None:
                raise RuntimeError(f"Unsupported gate `{gate_name}` for Stim backend.")
            stim_circuit.append(stim_gate, qubit_indices)

    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(shots)

    num_clbits = circuit.num_clbits if circuit.num_clbits > 0 else circuit.num_qubits
    counts: Dict[str, int] = {}
    for sample in samples:
        bits = ["0"] * num_clbits
        for meas_index, clbit_index in measurement_map:
            if clbit_index < num_clbits:
                bits[clbit_index] = "1" if sample[meas_index] else "0"
        # Qiskit formats classical bitstrings little-endian
        bitstring = "".join(bits[::-1])
        counts[bitstring] = counts.get(bitstring, 0) + 1

    return counts


def simulate_with_qiskit(circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, int]:
    """Simulate circuit using Qiskit Aer (fallback to BasicAer)."""
    backend = None
    if AerSimulator is not None:
        backend = AerSimulator()

    if backend is None:  # pragma: no cover - fallback path
        try:
            from qiskit.providers.basic_provider import BasicProvider

            backend = BasicProvider().get_backend("basic_simulator")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Qiskit backend unavailable. Install qiskit-aer or ensure the BasicSimulator is accessible."
            ) from exc

    compiled = transpile(circuit, backend)
    job = backend.run(compiled, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Guarantee dict[str, int]
    return {str(key): int(val) for key, val in counts.items()}


def simulate_with_tensor_network(circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, int]:
    """Simulate circuit using tensor networks (quimb + cotengra).

    Falls back to Qiskit if tensor network libraries are unavailable.
    """
    try:
        import quimb.tensor as qtn
    except ImportError:  # pragma: no cover - optional dependency
        return simulate_with_qiskit(circuit, shots)

    # Convert circuit to tensor network representation via QASM
    qasm = circuit.qasm()
    qc = qtn.Circuit.from_openqasm(qasm)

    # Sample measurement outcomes using contraction
    samples = qc.sample(shots=shots)
    counts: Dict[str, int] = {}
    for sample in samples:
        bitstring = "".join(map(str, sample))
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


def simulate(circuit: QuantumCircuit, shots: int = 1024) -> SimulationResult:
    """Main entry point - automatically routes to best backend."""
    router = QuantumRouter()
    analysis = router.analyze_circuit(circuit)

    backend_name = analysis["backend"]

    start = time.perf_counter()
    if backend_name == "stim":
        counts = simulate_with_stim(circuit, shots)
    elif backend_name == "qiskit_aer":
        counts = simulate_with_qiskit(circuit, shots)
    else:
        counts = simulate_with_tensor_network(circuit, shots)
    elapsed = time.perf_counter() - start

    return SimulationResult(
        counts=counts,
        backend=backend_name,
        time=elapsed,
        shots=shots,
        analysis=analysis,
    )
