#!/usr/bin/env python3
"""SIMPLE DEMO: See Ariadne's intelligent routing in action."""

from __future__ import annotations

import sys
from pathlib import Path

from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ariadne import QuantumRouter, simulate


def main() -> None:
    print("🔮 ARIADNE: Intelligent Quantum Router Demo")
    print("=" * 50)

    print("\n1️⃣ Creating Clifford circuit (30 qubits)...")
    qc = QuantumCircuit(30, 30)
    for idx in range(30):
        qc.h(idx)
        if idx < 29:
            qc.cx(idx, idx + 1)
    qc.measure_all()

    router = QuantumRouter()
    analysis = router.analyze_circuit(qc)

    print("\n📊 Circuit Analysis:")
    print(f"  • Entropy: {analysis['entropy']:.2f} bits")
    print(f"  • Is Clifford? {analysis['is_clifford']}")
    print(f"  • Recommended backend: {analysis['backend']}")
    print(f"  • Expected speedup: {analysis['estimated_speedup']}x")

    print("\n2️⃣ Running simulation...")
    result = simulate(qc, shots=1000)
    print("  ✅ Simulation complete!")
    print(f"  Backend used: {result.backend}")
    print(f"  Time: {result.time:.3f}s")
    print(f"  Shots: {result.shots}")

    print("\n" + "=" * 50)
    print("🎯 Ariadne automatically chose the fastest backend!")
    print("Without Ariadne, you'd have to know this manually.")


if __name__ == "__main__":
    main()
