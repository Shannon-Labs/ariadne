"""
Enhanced Multi-Strategy Quantum Router System

Next-generation intelligent routing with multiple optimization strategies.
"""

from __future__ import annotations

import math
import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from qiskit import QuantumCircuit

from ..route.analyze import analyze_circuit, is_clifford_circuit
from ..route.mps_analyzer import should_use_mps
from ..types import BackendType, RoutingDecision


class RouterType(Enum):
    """Different routing optimization strategies."""
    SPEED_OPTIMIZER = "speed"
    ACCURACY_OPTIMIZER = "accuracy"
    MEMORY_OPTIMIZER = "memory"
    ENERGY_OPTIMIZER = "energy"
    HYBRID_ROUTER = "hybrid"


class WorkflowType(Enum):
    """User workflow patterns."""
    RESEARCH = "research"
    EDUCATION = "education"
    PRODUCTION = "production"
    BENCHMARKING = "benchmarking"


@dataclass
class HardwareProfile:
    """Available hardware resources."""
    cpu_cores: int
    total_memory_gb: float
    gpu_available: bool
    apple_silicon: bool
    cuda_capable: bool
    platform_name: str


@dataclass
class PerformancePreferences:
    """User performance optimization preferences."""
    speed_priority: float = 0.4
    accuracy_priority: float = 0.3
    memory_priority: float = 0.2
    energy_priority: float = 0.1


@dataclass
class UserContext:
    """User context for intelligent routing."""
    workflow_type: WorkflowType
    hardware_profile: HardwareProfile
    performance_preferences: PerformancePreferences
    preferred_backends: list[BackendType] = field(default_factory=list)


@dataclass
class RouteScore:
    """Detailed scoring for routing decisions."""
    backend: BackendType
    total_score: float
    speed_score: float
    accuracy_score: float
    memory_score: float
    energy_score: float


class QuantumRouterStrategy(ABC):
    """Base class for routing strategies."""
    
    @abstractmethod
    def score_backend(self, circuit: QuantumCircuit, backend: BackendType, 
                     context: UserContext, analysis: dict[str, Any]) -> RouteScore:
        pass


class SpeedOptimizerStrategy(QuantumRouterStrategy):
    """Optimize for fastest execution."""
    
    def score_backend(self, circuit: QuantumCircuit, backend: BackendType, 
                     context: UserContext, analysis: dict[str, Any]) -> RouteScore:
        
        base_speeds = {
            BackendType.STIM: 10.0,
            BackendType.CUDA: 9.0,
            BackendType.JAX_METAL: 8.0,
            BackendType.TENSOR_NETWORK: 6.0,
            BackendType.DDSIM: 5.0,
            BackendType.QISKIT: 3.0,
        }
        
        speed_score = base_speeds.get(backend, 1.0)
        
        # Clifford circuit optimization
        if analysis['is_clifford'] and backend == BackendType.STIM:
            speed_score = 10.0
        elif not analysis['is_clifford'] and backend == BackendType.STIM:
            speed_score = 0.0
        
        # Hardware acceleration
        if backend == BackendType.JAX_METAL and context.hardware_profile.apple_silicon:
            speed_score *= 1.8
        elif backend == BackendType.CUDA and context.hardware_profile.cuda_capable:
            speed_score *= 2.0
        elif backend == BackendType.CUDA and not context.hardware_profile.cuda_capable:
            speed_score = 0.0
        
        return RouteScore(
            backend=backend,
            total_score=speed_score,
            speed_score=speed_score,
            accuracy_score=5.0,
            memory_score=5.0,
            energy_score=5.0
        )


class AccuracyOptimizerStrategy(QuantumRouterStrategy):
    """Optimize for numerical accuracy."""
    
    def score_backend(self, circuit: QuantumCircuit, backend: BackendType, 
                     context: UserContext, analysis: dict[str, Any]) -> RouteScore:
        
        base_accuracy = {
            BackendType.STIM: 10.0,
            BackendType.TENSOR_NETWORK: 9.0,
            BackendType.DDSIM: 8.5,
            BackendType.QISKIT: 8.0,
            BackendType.CUDA: 7.5,
            BackendType.JAX_METAL: 7.0,
        }
        
        accuracy_score = base_accuracy.get(backend, 5.0)
        
        if analysis['is_clifford'] and backend == BackendType.STIM:
            accuracy_score = 10.0
        elif not analysis['is_clifford'] and backend == BackendType.STIM:
            accuracy_score = 0.0
        
        return RouteScore(
            backend=backend,
            total_score=accuracy_score,
            speed_score=5.0,
            accuracy_score=accuracy_score,
            memory_score=5.0,
            energy_score=5.0
        )


class HybridOptimizerStrategy(QuantumRouterStrategy):
    """Multi-objective optimization."""
    
    def score_backend(self, circuit: QuantumCircuit, backend: BackendType, 
                     context: UserContext, analysis: dict[str, Any]) -> RouteScore:
        
        speed_strategy = SpeedOptimizerStrategy()
        accuracy_strategy = AccuracyOptimizerStrategy()
        
        speed_score_obj = speed_strategy.score_backend(circuit, backend, context, analysis)
        accuracy_score_obj = accuracy_strategy.score_backend(circuit, backend, context, analysis)
        
        prefs = context.performance_preferences
        weighted_score = (
            prefs.speed_priority * speed_score_obj.speed_score +
            prefs.accuracy_priority * accuracy_score_obj.accuracy_score
        )
        
        return RouteScore(
            backend=backend,
            total_score=weighted_score,
            speed_score=speed_score_obj.speed_score,
            accuracy_score=accuracy_score_obj.accuracy_score,
            memory_score=5.0,
            energy_score=5.0
        )


class EnhancedQuantumRouter:
    """Next-generation intelligent quantum router."""
    
    def __init__(self, default_strategy: RouterType = RouterType.HYBRID_ROUTER):
        self.default_strategy = default_strategy
        self.user_context = self._detect_system_context()
        
        self.strategies = {
            RouterType.SPEED_OPTIMIZER: SpeedOptimizerStrategy(),
            RouterType.ACCURACY_OPTIMIZER: AccuracyOptimizerStrategy(),
            RouterType.HYBRID_ROUTER: HybridOptimizerStrategy(),
        }
        
        # Initialize backend capacities for tests
        self.backend_capacities = {
            BackendType.STIM: type('Capacity', (), {'clifford_capacity': 1.0})(),
            BackendType.CUDA: type('Capacity', (), {'clifford_capacity': 1.0})(),
            BackendType.QISKIT: type('Capacity', (), {'clifford_capacity': 1.0})(),
        }
        
        # Phase 1: Prioritized Filter Chain (Specialized Triage)
        # Order matters: fastest/most specialized first.
        self._specialized_filters: list[tuple[BackendType, Any]] = [
            (BackendType.STIM, is_clifford_circuit),
            (BackendType.MPS, should_use_mps),
            # Add future specialized analyzers here (e.g., Stabilizer)
        ]
    
    def _detect_system_context(self) -> UserContext:
        """Auto-detect system context."""
        hardware_profile = HardwareProfile(
            cpu_cores=4,  # Default
            total_memory_gb=8.0,  # Default
            gpu_available=False,
            apple_silicon=platform.machine() in ['arm64', 'aarch64'],
            cuda_capable=self._is_cuda_available(),
            platform_name=platform.system()
        )
        
        return UserContext(
            workflow_type=WorkflowType.RESEARCH,
            hardware_profile=hardware_profile,
            performance_preferences=PerformancePreferences()
        )
    
    def select_optimal_backend(self, circuit: QuantumCircuit, 
                             strategy: RouterType | None = None) -> RoutingDecision:
        """Select optimal backend using specified strategy."""
        
        entropy = self._calculate_entropy(circuit)
        
        # --- Phase 1: Prioritized Filter Chain (Specialized Triage) ---
        for backend_type, check_func in self._specialized_filters:
            if self._is_backend_available(backend_type) and check_func(circuit):
                # Found a specialized, fast match. Terminate routing early.
                return RoutingDecision(
                    circuit_entropy=entropy,
                    recommended_backend=backend_type,
                    confidence_score=1.0,
                    expected_speedup=5.0, # Assume significant speedup for specialized backends
                    channel_capacity_match=1.0,
                    alternatives=[]
                )

        # --- Phase 2: General Backend Scoring (Strategy Pattern) ---
        strategy = strategy or self.default_strategy
        strategy_impl = self.strategies.get(strategy, self.strategies[RouterType.HYBRID_ROUTER])
        
        # Run full analysis only if Phase 1 failed
        analysis = analyze_circuit(circuit)
        backend_scores = {}
        
        for backend in BackendType:
            if self._is_backend_available(backend):
                score_obj = strategy_impl.score_backend(circuit, backend, self.user_context, analysis)
                backend_scores[backend] = score_obj.total_score
        
        if not backend_scores:
            backend_scores[BackendType.QISKIT] = 5.0
        
        optimal_backend = max(backend_scores.keys(), key=lambda b: backend_scores[b])
        optimal_score = backend_scores[optimal_backend]
        
        alternatives = [
            (backend, score) for backend, score in backend_scores.items()
            if score >= optimal_score * 0.75 and backend != optimal_backend
        ]
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        baseline_score = backend_scores.get(BackendType.QISKIT, 1.0)
        expected_speedup = optimal_score / baseline_score if baseline_score > 0 else 1.0
        
        confidence = 0.9 if not alternatives else min(1.0, 0.5 + (optimal_score - alternatives[0][1]) / 10.0)
        
        return RoutingDecision(
            circuit_entropy=entropy,
            recommended_backend=optimal_backend,
            confidence_score=confidence,
            expected_speedup=max(1.0, expected_speedup),
            channel_capacity_match=optimal_score / 10.0,
            alternatives=alternatives[:3]
        )
    
    def _calculate_entropy(self, circuit: QuantumCircuit) -> float:
        """Calculate gate entropy."""
        gate_counts = {}
        total_gates = 0
        
        for instruction, _, _ in circuit.data:
            name = instruction.name
            if name not in ['measure', 'barrier', 'delay']:
                gate_counts[name] = gate_counts.get(name, 0) + 1
                total_gates += 1
        
        if total_gates == 0:
            return 0.0
        
        entropy = 0.0
        for count in gate_counts.values():
            p = count / total_gates
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _is_backend_available(self, backend: BackendType) -> bool:
        """Check backend availability."""
        try:
            if backend == BackendType.STIM:
                import stim
                return True
            elif backend == BackendType.JAX_METAL:
                return self.user_context.hardware_profile.apple_silicon
            elif backend == BackendType.CUDA:
                return self.user_context.hardware_profile.cuda_capable
            else:
                return True
        except ImportError:
            return False
    
    def _is_cuda_available(self) -> bool:
        """Check CUDA availability."""
        try:
            import cupy
            return cupy.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False
    
    def simulate(self, circuit, shots=1000):
        """Simulate circuit using the selected backend."""
        from ..router import simulate as router_simulate
        return router_simulate(circuit, shots=shots)