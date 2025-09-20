from __future__ import annotations

from typing import Dict, Tuple

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
    for inst, qargs, _ in circ.data:
        if inst.num_qubits == 2:
            u, v = [circ.qubits.index(q) for q in qargs]
            if u != v:
                g.add_edge(u, v)
    return g


def approximate_treewidth(g: nx.Graph) -> int:
    if g.number_of_nodes() == 0:
        return 0
    try:
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
    g = interaction_graph(circ)
    return max((deg for _, deg in g.degree()), default=0)


def two_qubit_depth(circ: QuantumCircuit) -> int:
    depth = 0
    current_layer_qubits = set()
    for inst, qargs, _ in circ.data:
        if inst.num_qubits == 2:
            qubits = {circ.qubits.index(q) for q in qargs}
            if current_layer_qubits & qubits:
                depth += 1
                current_layer_qubits = set(qubits)
            else:
                current_layer_qubits |= qubits
    return depth + (1 if current_layer_qubits else 0)


def two_qubit_stats(circ: QuantumCircuit) -> Tuple[int, int, float, float]:
    total_gates = 0
    twoq = 0
    edge_counts: Dict[Tuple[int, int], int] = {}
    for inst, qargs, _ in circ.data:
        name = inst.name
        if name in {"measure", "barrier", "delay"}:
            continue
        total_gates += 1
        if inst.num_qubits == 2:
            u, v = sorted([circ.qubits.index(q) for q in qargs])
            if u != v:
                twoq += 1
                edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1
    density = float(twoq) / float(total_gates) if total_gates else 0.0
    # redundancy: fraction of 2q gates beyond first occurrence on the same edge
    repeated = sum(c - 1 for c in edge_counts.values() if c > 1)
    redundancy = float(repeated) / float(twoq) if twoq else 0.0
    return total_gates, twoq, density, redundancy


def analyze_circuit(circ: QuantumCircuit) -> Dict[str, float | int | bool]:
    g = interaction_graph(circ)
    total_gates, twoq, density, redundancy = two_qubit_stats(circ)
    return {
        "num_qubits": circ.num_qubits,
        "depth": int(circ.depth()),
        "two_qubit_depth": two_qubit_depth(circ),
        "two_qubit_count": twoq,
        "two_qubit_density": density,
        "redundancy_score": redundancy,
        "edges": g.number_of_edges(),
        "treewidth_estimate": approximate_treewidth(g),
        "light_cone_width": light_cone_width_estimate(circ),
        "clifford_ratio": clifford_ratio(circ),
        "is_clifford": is_clifford_circuit(circ),
    }
