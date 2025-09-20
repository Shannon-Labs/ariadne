from __future__ import annotations

import networkx as nx
from qiskit import QuantumCircuit

CLIFFORD_ONE_Q = {"i", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg"}
CLIFFORD_TWO_Q = {"cx", "cz", "swap"}


def is_clifford_circuit(circ: QuantumCircuit) -> bool:
    for inst, _, _ in circ.data:
        name = inst.name
        if name in {"measure", "barrier", "delay"}:
            continue
        if (name not in CLIFFORD_ONE_Q) and (name not in CLIFFORD_TWO_Q):
            return False
    return True


def interaction_graph(circ: QuantumCircuit) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(circ.num_qubits))
    # Qiskit 2.x no longer exposes .index directly, so pre-compute lookup table
    qubit_index_map = {qubit: idx for idx, qubit in enumerate(circ.qubits)}
    for inst, qargs, _ in circ.data:
        if inst.num_qubits == 2:
            u, v = [qubit_index_map[q] for q in qargs]
            if u != v:
                g.add_edge(u, v)
    return g


def approximate_treewidth(g: nx.Graph) -> int:
    if g.number_of_nodes() == 0:
        return 0
    try:
        # Use a quick heuristic approximation
        from networkx.algorithms.approximation import treewidth_min_fill_in

        width, _ = treewidth_min_fill_in(g)
        return int(width)
    except Exception:
        return max((deg for _, deg in g.degree()), default=0)


def clifford_ratio(circ: QuantumCircuit) -> float:
    total = 0
    cliff = 0
    for inst, _, _ in circ.data:
        name = inst.name
        if name in {"measure", "barrier", "delay"}:
            continue
        total += 1
        if (name in CLIFFORD_ONE_Q) or (name in CLIFFORD_TWO_Q):
            cliff += 1
    return float(cliff) / float(total) if total else 1.0


def light_cone_width_estimate(circ: QuantumCircuit) -> int:
    # Simple proxy: max degree of interaction graph
    g = interaction_graph(circ)
    return max((deg for _, deg in g.degree()), default=0)


def two_qubit_depth(circ: QuantumCircuit) -> int:
    depth = 0
    current_layer_qubits = set()
    for inst, qargs, _ in circ.data:
        if inst.num_qubits == 2:
            # Qiskit 2.x no longer exposes .index directly, so pre-compute lookup table
            qubit_index_map = {qubit: idx for idx, qubit in enumerate(circ.qubits)}
            qubits = {qubit_index_map[q] for q in qargs}
            if current_layer_qubits & qubits:
                depth += 1
                current_layer_qubits = set(qubits)
            else:
                current_layer_qubits |= qubits
    return depth + (1 if current_layer_qubits else 0)


def analyze_circuit(circ: QuantumCircuit) -> dict[str, float | int | bool]:
    g = interaction_graph(circ)
    return {
        "num_qubits": circ.num_qubits,
        "depth": int(circ.depth()),
        "two_qubit_depth": two_qubit_depth(circ),
        "edges": g.number_of_edges(),
        "treewidth_estimate": approximate_treewidth(g),
        "light_cone_width": light_cone_width_estimate(circ),
        "clifford_ratio": clifford_ratio(circ),
        "is_clifford": is_clifford_circuit(circ),
    }

