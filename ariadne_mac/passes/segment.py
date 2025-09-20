from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit

from .defer_measure import defer_measure_if_clifford
from ..route.analyze import analyze_circuit, is_clifford_circuit


@dataclass
class Segment:
    idx: int
    kind: str  # 'clifford' | 'low_treewidth' | 'dense_sv'
    start: int  # index in instruction list
    end: int  # inclusive
    qubits: Tuple[int, int]  # min,max indices spanned
    circ_hash: str


def _hash_qasm3(c: QuantumCircuit) -> str:
    try:
        from qiskit.qasm3 import dumps as q3_dumps

        text = q3_dumps(c)
    except Exception:
        text = str(c)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def segment_circuit(circ: QuantumCircuit, policy: Dict[str, object]) -> Tuple[List[Segment], Dict[str, object]]:
    """Partition circuit into coarse segments and return manifest metadata.

    Current strategy: perform a single-pass classification and group consecutive
    instructions where the classification remains the same.
    """
    # Try deferred measurement rewrite to enable Clifford classification
    circ_rew, ok = defer_measure_if_clifford(circ)
    data = list(circ_rew.data)
    segments: List[Segment] = []
    start = 0
    n = len(data)
    idx = 0
    while start < n:
        # build a small window and classify
        end = start
        # Create window with same register structure
        window = QuantumCircuit.copy_empty_like(circ)
        while end < n:
            inst, qargs, cargs = data[end]
            window.append(inst, qargs, cargs)
            metrics = analyze_circuit(window)
            if metrics["is_clifford"]:
                kind = "clifford"
            elif int(metrics["treewidth_estimate"]) <= int(
                (policy.get("tn", {}) or {}).get("treewidth_threshold", 12)  # type: ignore
            ):
                kind = "low_treewidth"
            else:
                kind = "dense_sv"
            # Try extending until kind would change
            end_next = end + 1
            if end_next < n:
                tmp = window.copy()
                inst2, qargs2, cargs2 = data[end_next]
                tmp.append(inst2, qargs2, cargs2)
                m2 = analyze_circuit(tmp)
                if m2["is_clifford"] and kind == "clifford":
                    end = end_next
                    window = tmp
                    continue
                if (int(m2["treewidth_estimate"]) <= int((policy.get("tn", {}) or {}).get("treewidth_threshold", 12))) and kind != "dense_sv":
                    end = end_next
                    window = tmp
                    continue
            break
        # finalize segment
        qubits = set()
        for inst, qargs, cargs in window.data:
            for q in qargs:
                qubits.add(circ.qubits.index(q))
        seg = Segment(
            idx=idx,
            kind=kind,
            start=start,
            end=end,
            qubits=(min(qubits) if qubits else 0, max(qubits) if qubits else 0),
            circ_hash=_hash_qasm3(window),
        )
        segments.append(seg)
        idx += 1
        start = end + 1

    manifest = {
        "segments": [asdict(s) for s in segments],
        "boundary_adapters": [],  # future: record basis changes
    }
    return segments, manifest

