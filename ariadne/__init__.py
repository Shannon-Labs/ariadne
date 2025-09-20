"""
Ariadne: The Intelligent Quantum Router 🔮
==========================================

Shannon Labs' quantum circuit routing system that delivers 1000× speedups
by automatically selecting optimal simulators based on information content.

Key Features:
    * Intelligent routing based on Bell Labs-style information theory
    * Automatic backend selection (Stim, Qiskit, Tensor Networks, JAX/Metal, DDSIM)
    * 1000× speedup for Clifford circuits
    * Zero configuration - works out of the box
    * Apple Silicon optimized

Example:
    >>> from ariadne import simulate
    >>> from qiskit import QuantumCircuit
    >>>
    >>> circuit = QuantumCircuit(10)
    >>> circuit.h(0)
    >>> for i in range(9):
    >>>     circuit.cx(i, i+1)
    >>> circuit.measure_all()
    >>>
    >>> result = simulate(circuit, shots=1000)
    >>> print(f"Backend used: {result.backend_used}")
    >>> print(f"Execution time: {result.execution_time:.3f}s")
    >>> print(f"Counts: {result.counts}")

© 2025 Shannon Labs Inc. | MIT License
"""

__version__ = "1.0.0"

from .router import (
    QuantumRouter,
    simulate,
    BackendType,
    RoutingDecision,
    SimulationResult,
    BackendCapacity,
)

__all__ = [
    "QuantumRouter",
    "simulate",
    "BackendType",
    "RoutingDecision",
    "SimulationResult",
    "BackendCapacity",
]