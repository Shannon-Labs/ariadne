"""Intelligent routing across the available quantum circuit simulators."""

from __future__ import annotations

import math
import warnings
from time import perf_counter
from typing import Any
import logging

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .types import BackendType, BackendCapacity, RoutingDecision, SimulationResult
from .backends.mps_backend import MPSBackend
from .backends.tensor_network_backend import TensorNetworkBackend
from .route.analyze import analyze_circuit, should_use_tensor_network
from .route.mps_analyzer import should_use_mps
from .route.enhanced_router import EnhancedQuantumRouter, RouterType

try:  # pragma: no cover - import guard for optional CUDA support
    from .backends.cuda_backend import CUDABackend, is_cuda_available
except ImportError:  # pragma: no cover - executed when dependencies missing
    CUDABackend = None  # type: ignore[assignment]

    def is_cuda_available() -> bool:  # type: ignore[override]
        return False

try:  # pragma: no cover - import guard for optional Metal support
    from .backends.metal_backend import MetalBackend, is_metal_available
except ImportError:  # pragma: no cover - executed when dependencies missing
    MetalBackend = None  # type: ignore[assignment]

    def is_metal_available() -> bool:  # type: ignore[override]
        return False


# Global state for Tensor Network Backend instance
_TENSOR_BACKEND: TensorNetworkBackend | None = None

# ------------------------------------------------------------------
# Analysis helpers

def _apple_silicon_boost() -> float:
    import platform

    if platform.system() == "Darwin" and platform.machine() in {"arm", "arm64"}:
        # More realistic boost factor based on actual benchmarks
        return 1.5
    return 1.0

# ------------------------------------------------------------------
# Simulation helpers

def _simulate_stim(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    try:
        from .converters import convert_qiskit_to_stim, simulate_stim_circuit
    except ImportError as exc:
        raise RuntimeError("Stim is not installed") from exc

    stim_circuit, measurement_map = convert_qiskit_to_stim(circuit)
    num_clbits = circuit.num_clbits or circuit.num_qubits
    return simulate_stim_circuit(stim_circuit, measurement_map, shots, num_clbits)

def _simulate_qiskit(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    try:
        from qiskit.providers.basic_provider import BasicProvider
    except ImportError as exc:  # pragma: no cover - depends on qiskit extras
        raise RuntimeError("Qiskit provider not available") from exc

    provider = BasicProvider()
    backend = provider.get_backend("basic_simulator")
    job = backend.run(circuit, shots=shots)
    counts = job.result().get_counts()
    return {str(key): value for key, value in counts.items()}

def _real_tensor_network_simulation(
    circuit: QuantumCircuit, shots: int
) -> dict[str, int]:
    global _TENSOR_BACKEND
    if _TENSOR_BACKEND is None:
        _TENSOR_BACKEND = TensorNetworkBackend()
    return _TENSOR_BACKEND.simulate(circuit, shots)

def _simulate_tensor_network(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    """Simulate ``circuit`` using the tensor network backend."""

    try:
        return _real_tensor_network_simulation(circuit, shots)
    except ImportError as exc:
        raise RuntimeError(
            "Tensor network dependencies are not installed"
        ) from exc
    except Exception as exc:  # pragma: no cover - graceful fallback path
        warnings.warn(
            f"Tensor network simulation failed, falling back to Qiskit: {exc}",
            RuntimeWarning,
        )
        return _simulate_qiskit(circuit, shots)

def _simulate_jax_metal(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    """Simulate using the new hybrid Metal backend for Apple Silicon."""
    logger = logging.getLogger(__name__)
    
    try:
        from .backends.metal_backend import MetalBackend

        # Use our new MetalBackend with hybrid approach
        backend = MetalBackend(allow_cpu_fallback=True)
        result = backend.simulate(circuit, shots)
        
        # Log backend mode for debugging
        logger.debug(f"Metal backend executed in mode: {backend.backend_mode}")
        
        # Check if Metal actually accelerated or fell back to CPU
        if backend.backend_mode == "cpu":
            logger.debug("Metal backend fell back to CPU mode")

        return result

    except ImportError as exc:
        logger.debug(f"MetalBackend not available: {exc}")
        # Fallback to Qiskit if MetalBackend not available
        raise RuntimeError("Metal backend dependencies not available") from exc
    except Exception as exc:
        logger.warning(f"Metal backend execution failed: {exc}")
        # Re-raise for higher-level fallback handling
        raise RuntimeError(f"Metal backend execution failed: {exc}") from exc

def _simulate_ddsim(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    try:
        import mqt.ddsim as ddsim
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("MQT DDSIM not installed") from exc

    simulator = ddsim.DDSIMProvider().get_backend("qasm_simulator")
    job = simulator.run(circuit, shots=shots)
    counts = job.result().get_counts()
    return {str(key): value for key, value in counts.items()}

def _simulate_cuda(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    if not is_cuda_available() or CUDABackend is None:
        raise RuntimeError("CUDA runtime not available")

    backend = CUDABackend()
    return backend.simulate(circuit, shots)

def _simulate_mps(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    """Simulate ``circuit`` using the Matrix Product State backend."""
    try:
        from .backends.mps_backend import MPSBackend
    except ImportError as exc:
        raise RuntimeError("MPS backend dependencies not available") from exc

    backend = MPSBackend()
    return backend.simulate(circuit, shots)

def _simulate_metal(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    if not is_metal_available() or MetalBackend is None:
        raise RuntimeError("JAX with Metal support not available")

    backend = MetalBackend()
    return backend.simulate(circuit, shots)

def _sample_statevector_counts(
    circuit: QuantumCircuit, shots: int, seed: int | None = None
) -> dict[str, int]:
    if shots < 0:
        raise ValueError("shots must be non-negative")
    if shots == 0:
        return {}

    state = Statevector.from_instruction(circuit)
    probabilities = np.abs(state.data) ** 2
    total = probabilities.sum()
    if total == 0.0:
        raise RuntimeError("Statevector sampling produced invalid probabilities")
    if not np.isclose(total, 1.0):
        probabilities = probabilities / total

    rng = np.random.default_rng(seed)
    outcomes = rng.choice(len(probabilities), size=shots, p=probabilities)

    counts: dict[str, int] = {}
    num_qubits = circuit.num_qubits
    for outcome in outcomes:
        bitstring = format(int(outcome), f"0{num_qubits}b")[::-1]
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts

# ------------------------------------------------------------------
# Core Execution Logic

def _execute_simulation(circuit: QuantumCircuit, shots: int, routing_decision: RoutingDecision) -> SimulationResult:
    """Execute simulation based on a routing decision, including fallback logic."""
    
    backend = routing_decision.recommended_backend
    
    # Initialize result tracking
    fallback_reason = None
    warnings_list = []
    
    # Set up logging for backend selection
    logger = logging.getLogger(__name__)
    logger.debug(f"Selected backend: {backend.value} (confidence: {routing_decision.confidence_score:.3f})")

    start = perf_counter()

    try:
        if backend == BackendType.STIM:
            counts = _simulate_stim(circuit, shots)
        elif backend == BackendType.QISKIT:
            counts = _simulate_qiskit(circuit, shots)
        elif backend == BackendType.TENSOR_NETWORK:
            counts = _simulate_tensor_network(circuit, shots)
        elif backend == BackendType.JAX_METAL:
            counts = _simulate_jax_metal(circuit, shots)
        elif backend == BackendType.DDSIM:
            counts = _simulate_ddsim(circuit, shots)
        elif backend == BackendType.MPS:
            counts = _simulate_mps(circuit, shots)
        elif backend == BackendType.CUDA:
            counts = _simulate_cuda(circuit, shots)
        else:
            # Fallback for unknown or unhandled backend types
            counts = _simulate_qiskit(circuit, shots)
            backend = BackendType.QISKIT
            warnings_list.append(f"Unknown backend {backend.value} selected, falling back to Qiskit.")
    except Exception as exc:
        # Log the specific failure for debugging
        logger.warning(f"Backend {backend.value} failed: {exc}. Falling back to Qiskit.")
        fallback_reason = f"Backend {backend.value} failed: {str(exc)}"
        
        # Attempt fallback to Qiskit
        try:
            counts = _simulate_qiskit(circuit, shots)
            backend = BackendType.QISKIT
        except Exception as qiskit_exc:
            # Last resort: log and re-raise the original exception
            logger.error(f"Qiskit fallback also failed: {qiskit_exc}")
            raise RuntimeError(
                f"All backends failed. Original error: {exc}. Qiskit fallback error: {qiskit_exc}"
            ) from exc

    elapsed = perf_counter() - start
    
    # Check for experimental backend warnings
    if backend == BackendType.JAX_METAL and is_metal_available():
        warnings_list.append("JAX-Metal support is experimental and may show warnings")
    elif backend == BackendType.CUDA and not is_cuda_available():
        warnings_list.append("CUDA backend selected but CUDA not available")

    return SimulationResult(
        counts=counts,
        backend_used=backend,
        execution_time=elapsed,
        routing_decision=routing_decision,
        metadata={"shots": shots},
        fallback_reason=fallback_reason,
        warnings=warnings_list if warnings_list else None
    )


def simulate(circuit: QuantumCircuit, shots: int = 1024, backend: str | None = None) -> SimulationResult:
    """Convenience wrapper that routes and executes ``circuit``."""
    
    # Initialize Enhanced Router
    enhanced_router = EnhancedQuantumRouter()
    
    if backend is not None:
        # Force specific backend
        try:
            backend_type = BackendType(backend)
        except ValueError as exc:
            raise ValueError(f"Unknown backend: {backend}") from exc
        
        # Create a forced routing decision
        routing_decision = RoutingDecision(
            circuit_entropy=0.0,
            recommended_backend=backend_type,
            confidence_score=1.0,
            expected_speedup=1.0,
            channel_capacity_match=1.0,
            alternatives=[]
        )
    else:
        # Use Enhanced Router for optimal selection
        routing_decision = enhanced_router.select_optimal_backend(circuit, strategy=RouterType.HYBRID_ROUTER)
    
    return _execute_simulation(circuit, shots, routing_decision)