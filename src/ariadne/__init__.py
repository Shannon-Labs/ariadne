"""Ariadne: intelligent quantum circuit routing."""

__version__ = "1.0.0"

from .backends.cuda_backend import CUDABackend, get_cuda_info, simulate_cuda
from .backends.metal_backend import MetalBackend, get_metal_info, simulate_metal
from .router import (
    BackendCapacity,
    BackendType,
    RoutingDecision,
    SimulationResult,
    simulate,
)

__all__ = [
    "simulate",
    "BackendType",
    "RoutingDecision",
    "SimulationResult",
    "BackendCapacity",
    "CUDABackend",
    "simulate_cuda",
    "get_cuda_info",
    "MetalBackend",
    "simulate_metal",
    "get_metal_info",
]