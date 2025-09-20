"""Ariadne: intelligent quantum circuit routing."""

__version__ = "1.0.0"

from .router import (
    BackendCapacity,
    BackendType,
    QuantumRouter,
    RoutingDecision,
    SimulationResult,
    simulate,
)
from .backends.cuda_backend import CUDABackend, get_cuda_info, simulate_cuda

__all__ = [
    "QuantumRouter",
    "simulate",
    "BackendType",
    "RoutingDecision",
    "SimulationResult",
    "BackendCapacity",
    "CUDABackend",
    "simulate_cuda",
    "get_cuda_info",
]