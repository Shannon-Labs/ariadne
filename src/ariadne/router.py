"""Intelligent routing across the available quantum circuit simulators."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .backends.tensor_network_backend import TensorNetworkBackend
from .route.analyze import analyze_circuit, should_use_tensor_network

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

class BackendType(Enum):
    """Available quantum simulation backends."""

    STIM = "stim"
    QISKIT = "qiskit"
    TENSOR_NETWORK = "tensor_network"
    JAX_METAL = "jax_metal"
    DDSIM = "ddsim"
    CUDA = "cuda"


@dataclass
class BackendCapacity:
    """Simple scoring model for backend suitability."""

    clifford_capacity: float
    general_capacity: float
    memory_efficiency: float
    apple_silicon_boost: float


@dataclass
class RoutingDecision:
    """Information returned by :meth:`QuantumRouter.select_optimal_backend`."""

    circuit_entropy: float
    recommended_backend: BackendType
    confidence_score: float
    expected_speedup: float
    channel_capacity_match: float
    alternatives: list[tuple[BackendType, float]]


@dataclass
class SimulationResult:
    """Container for the output of :func:`simulate`."""

    counts: dict[str, int]
    backend_used: BackendType
    execution_time: float
    routing_decision: RoutingDecision
    metadata: dict[str, Any]


class QuantumRouter:
    """Analyse a circuit and execute it on a suitable backend."""

    def __init__(self) -> None:
        self._cuda_available = is_cuda_available()
        self._metal_available = is_metal_available()
        self._tensor_backend: TensorNetworkBackend | None = None

        self.backend_capacities: dict[BackendType, BackendCapacity] = {
            BackendType.STIM: BackendCapacity(
                clifford_capacity=float("inf"),
                general_capacity=0.0,
                memory_efficiency=1.0,
                apple_silicon_boost=1.0,
            ),
            BackendType.QISKIT: BackendCapacity(
                clifford_capacity=6.0,
                general_capacity=8.0,
                memory_efficiency=0.6,
                apple_silicon_boost=1.0,
            ),
            BackendType.TENSOR_NETWORK: BackendCapacity(
                clifford_capacity=5.0,
                general_capacity=9.0,
                memory_efficiency=0.9,
                apple_silicon_boost=1.0,
            ),
            BackendType.JAX_METAL: BackendCapacity(
                clifford_capacity=8.0 if self._metal_available else 0.0,
                general_capacity=8.0 if self._metal_available else 0.0,
                memory_efficiency=0.8,
                apple_silicon_boost=5.0,
            ),
            BackendType.DDSIM: BackendCapacity(
                clifford_capacity=7.0,
                general_capacity=7.0,
                memory_efficiency=0.8,
                apple_silicon_boost=1.0,
            ),
            BackendType.CUDA: BackendCapacity(
                clifford_capacity=9.0 if self._cuda_available else 0.0,
                general_capacity=10.0 if self._cuda_available else 0.0,
                memory_efficiency=0.9,
                apple_silicon_boost=1.0,
            ),
        }

    # ------------------------------------------------------------------
    # Analysis helpers

    def circuit_entropy(self, circuit: QuantumCircuit) -> float:
        """Return a Shannon-style entropy estimate for the circuit."""

        gate_counts: dict[str, int] = {}
        total_gates = 0

        for instruction, _, _ in circuit.data:
            name = instruction.name
            if name in {"measure", "barrier", "delay"}:
                continue
            gate_counts[name] = gate_counts.get(name, 0) + 1
            total_gates += 1

        if total_gates == 0:
            return 0.0

        entropy = 0.0
        for count in gate_counts.values():
            probability = count / total_gates
            entropy -= probability * math.log2(probability)

        return entropy

    def channel_capacity_match(
        self, circuit: QuantumCircuit, backend: BackendType
    ) -> float:
        """Score how well the circuit maps to the backend's strengths."""

        capacity = self.backend_capacities[backend]
        analysis = analyze_circuit(circuit)

        if backend == BackendType.CUDA and not self._cuda_available:
            return 0.0

        if backend == BackendType.JAX_METAL and not self._metal_available:
            return 0.0

        if backend == BackendType.TENSOR_NETWORK and not should_use_tensor_network(
            circuit, analysis
        ):
            return 0.0

        if analysis["is_clifford"]:
            base_match = capacity.clifford_capacity
        else:
            base_match = capacity.general_capacity

        # Normalise the score to [0, 1].
        base_match = min(1.0, base_match / 10.0)

        if analysis["num_qubits"] > 20:
            base_match *= capacity.memory_efficiency

        if backend == BackendType.JAX_METAL:
            base_match *= self._apple_silicon_boost()

        return float(max(0.0, min(base_match, 1.0)))

    def select_optimal_backend(self, circuit: QuantumCircuit) -> RoutingDecision:
        """Choose the most suitable backend for ``circuit``."""

        entropy = self.circuit_entropy(circuit)

        backend_scores: dict[BackendType, float] = {}
        for backend in BackendType:
            backend_scores[backend] = self.channel_capacity_match(circuit, backend)

        optimal_backend = max(
            backend_scores.keys(), key=lambda key: backend_scores[key]
        )
        optimal_score = backend_scores[optimal_backend]

        # Fall back to Qiskit when every backend is scored at zero.
        if optimal_score == 0.0:
            optimal_backend = BackendType.QISKIT
            optimal_score = backend_scores[BackendType.QISKIT]

        alternatives = [
            (backend, score)
            for backend, score in backend_scores.items()
            if backend != optimal_backend and score >= optimal_score * 0.8
        ]
        alternatives.sort(key=lambda item: item[1], reverse=True)

        baseline = backend_scores.get(BackendType.QISKIT, 0.0)
        expected_speedup = optimal_score / baseline if baseline > 0 else 1.0

        if alternatives:
            confidence = optimal_score / max(alternatives[0][1], 1e-9)
        else:
            confidence = 1.0

        decision = RoutingDecision(
            circuit_entropy=entropy,
            recommended_backend=optimal_backend,
            confidence_score=min(confidence, 1.0),
            expected_speedup=max(expected_speedup, 1.0),
            channel_capacity_match=optimal_score,
            alternatives=alternatives,
        )

        return decision

    # ------------------------------------------------------------------
    # Execution helpers

    def simulate(self, circuit: QuantumCircuit, shots: int = 1024) -> SimulationResult:
        """Simulate ``circuit`` using the selected backend."""

        routing_decision = self.select_optimal_backend(circuit)
        backend = routing_decision.recommended_backend

        start = perf_counter()

        try:
            if backend == BackendType.STIM:
                counts = self._simulate_stim(circuit, shots)
            elif backend == BackendType.QISKIT:
                counts = self._simulate_qiskit(circuit, shots)
            elif backend == BackendType.TENSOR_NETWORK:
                counts = self._simulate_tensor_network(circuit, shots)
            elif backend == BackendType.JAX_METAL:
                counts = self._simulate_jax_metal(circuit, shots)
            elif backend == BackendType.DDSIM:
                counts = self._simulate_ddsim(circuit, shots)
            else:
                counts = self._simulate_cuda(circuit, shots)
        except Exception:
            counts = self._simulate_qiskit(circuit, shots)
            backend = BackendType.QISKIT

        elapsed = perf_counter() - start

        return SimulationResult(
            counts=counts,
            backend_used=backend,
            execution_time=elapsed,
            routing_decision=routing_decision,
            metadata={"shots": shots},
        )

    def _simulate_stim(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        try:
            from .converters import convert_qiskit_to_stim, simulate_stim_circuit
        except ImportError as exc:
            raise RuntimeError("Stim is not installed") from exc

        stim_circuit, measurement_map = convert_qiskit_to_stim(circuit)
        num_clbits = circuit.num_clbits or circuit.num_qubits
        return simulate_stim_circuit(stim_circuit, measurement_map, shots, num_clbits)

    def _simulate_qiskit(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        try:
            from qiskit.providers.basic_provider import BasicProvider
        except ImportError as exc:  # pragma: no cover - depends on qiskit extras
            raise RuntimeError("Qiskit provider not available") from exc

        provider = BasicProvider()
        backend = provider.get_backend("basic_simulator")
        job = backend.run(circuit, shots=shots)
        counts = job.result().get_counts()
        return {str(key): value for key, value in counts.items()}

    def _simulate_tensor_network(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        """Simulate ``circuit`` using the tensor network backend."""

        try:
            return self._real_tensor_network_simulation(circuit, shots)
        except ImportError as exc:
            raise RuntimeError(
                "Tensor network dependencies are not installed"
            ) from exc
        except Exception as exc:  # pragma: no cover - graceful fallback path
            warnings.warn(
                f"Tensor network simulation failed, falling back to Qiskit: {exc}",
                RuntimeWarning,
            )
            return self._simulate_qiskit(circuit, shots)

    def _real_tensor_network_simulation(
        self, circuit: QuantumCircuit, shots: int
    ) -> dict[str, int]:
        if self._tensor_backend is None:
            self._tensor_backend = TensorNetworkBackend()
        return self._tensor_backend.simulate(circuit, shots)

    def _simulate_jax_metal(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        """Simulate using the new hybrid Metal backend for Apple Silicon."""
        try:
            from .backends.metal_backend import MetalBackend

            # Use our new MetalBackend with hybrid approach
            backend = MetalBackend(allow_cpu_fallback=True)
            result = backend.simulate(circuit, shots)

            return result

        except ImportError:
            # Fallback to Qiskit if MetalBackend not available
            return self._simulate_qiskit(circuit, shots)
        except Exception:
            # Fallback to Qiskit for any other errors
            return self._simulate_qiskit(circuit, shots)

    def _simulate_ddsim(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        try:
            import mqt.ddsim as ddsim
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("MQT DDSIM not installed") from exc

        simulator = ddsim.DDSIMProvider().get_backend("qasm_simulator")
        job = simulator.run(circuit, shots=shots)
        counts = job.result().get_counts()
        return {str(key): value for key, value in counts.items()}

    def _simulate_cuda(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        if not self._cuda_available or CUDABackend is None:
            raise RuntimeError("CUDA runtime not available")

        backend = CUDABackend()
        return backend.simulate(circuit, shots)

    def _simulate_metal(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        if not self._metal_available or MetalBackend is None:
            raise RuntimeError("JAX with Metal support not available")

        backend = MetalBackend()
        return backend.simulate(circuit, shots)

    def _sample_statevector_counts(
        self, circuit: QuantumCircuit, shots: int, seed: int | None = None
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

    @staticmethod
    def _apple_silicon_boost() -> float:
        import platform

        if platform.system() == "Darwin" and platform.machine() in {"arm", "arm64"}:
            return 5.0
        return 1.0


def simulate(circuit: QuantumCircuit, shots: int = 1024, backend: str | None = None) -> SimulationResult:
    """Convenience wrapper that routes and executes ``circuit``."""

    router = QuantumRouter()
    
    if backend is not None:
        # Force specific backend
        try:
            backend_type = BackendType(backend)
        except ValueError as exc:
            raise ValueError(f"Unknown backend: {backend}") from exc
        
        # Create a custom router that only uses the specified backend
        class ForcedRouter(QuantumRouter):
            def select_optimal_backend(self, circuit: QuantumCircuit) -> RoutingDecision:
                return RoutingDecision(
                    circuit_entropy=0.0,
                    recommended_backend=backend_type,
                    confidence_score=1.0,
                    expected_speedup=1.0,
                    channel_capacity_match=1.0,
                    alternatives=[]
                )
        
        router = ForcedRouter()
    
    return router.simulate(circuit, shots)