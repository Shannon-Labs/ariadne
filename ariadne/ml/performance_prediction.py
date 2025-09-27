"""
ML-Based Performance Prediction System

This module uses machine learning to predict quantum circuit execution performance
across different backends, enabling intelligent routing decisions.
"""

from __future__ import annotations

import json
import math
import pickle
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from qiskit import QuantumCircuit

from ..route.analyze import analyze_circuit
from ..router import BackendType


@dataclass
class CircuitFeatures:
    """Extracted features from quantum circuits for ML models."""
    num_qubits: int
    depth: int
    gate_count: int
    two_qubit_gate_count: int
    single_qubit_gate_count: int
    gate_entropy: float
    connectivity_index: float
    clifford_ratio: float
    parallelization_factor: float
    entanglement_complexity: float
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numpy array for ML models."""
        return np.array([
            self.num_qubits,
            self.depth,
            self.gate_count,
            self.two_qubit_gate_count,
            self.single_qubit_gate_count,
            self.gate_entropy,
            self.connectivity_index,
            self.clifford_ratio,
            self.parallelization_factor,
            self.entanglement_complexity
        ], dtype=np.float32)


@dataclass
class PerformanceMetrics:
    """Performance metrics for circuit execution."""
    execution_time: float
    memory_usage_mb: float
    success_probability: float
    accuracy_score: float
    energy_consumption: float


@dataclass
class PredictionResult:
    """Result of performance prediction."""
    backend: BackendType
    predicted_time: float
    predicted_memory_mb: float
    predicted_success_rate: float
    confidence_score: float
    feature_importance: Dict[str, float]


class CircuitFeatureExtractor:
    """Extract ML-relevant features from quantum circuits."""
    
    def extract_features(self, circuit: QuantumCircuit) -> CircuitFeatures:
        """Extract comprehensive features from a quantum circuit."""
        
        # Basic circuit metrics
        num_qubits = circuit.num_qubits
        depth = circuit.depth()
        
        # Gate analysis
        gate_counts = self._count_gates(circuit)
        gate_count = sum(gate_counts.values())
        two_qubit_gate_count = self._count_two_qubit_gates(circuit)
        single_qubit_gate_count = gate_count - two_qubit_gate_count
        
        # Advanced metrics
        gate_entropy = self._calculate_gate_entropy(circuit)
        connectivity_index = self._calculate_connectivity_index(circuit)
        clifford_ratio = self._calculate_clifford_ratio(circuit)
        parallelization_factor = self._calculate_parallelization_factor(circuit)
        entanglement_complexity = self._estimate_entanglement_complexity(circuit)
        
        return CircuitFeatures(
            num_qubits=num_qubits,
            depth=depth,
            gate_count=gate_count,
            two_qubit_gate_count=two_qubit_gate_count,
            single_qubit_gate_count=single_qubit_gate_count,
            gate_entropy=gate_entropy,
            connectivity_index=connectivity_index,
            clifford_ratio=clifford_ratio,
            parallelization_factor=parallelization_factor,
            entanglement_complexity=entanglement_complexity
        )
    
    def _count_gates(self, circuit: QuantumCircuit) -> Dict[str, int]:
        """Count gates by type."""
        gate_counts = {}
        for instruction, _, _ in circuit.data:
            if instruction.name not in ['measure', 'barrier', 'delay']:
                gate_counts[instruction.name] = gate_counts.get(instruction.name, 0) + 1
        return gate_counts
    
    def _count_two_qubit_gates(self, circuit: QuantumCircuit) -> int:
        """Count two-qubit gates."""
        count = 0
        for instruction, _, _ in circuit.data:
            if instruction.num_qubits == 2 and instruction.name not in ['measure', 'barrier']:
                count += 1
        return count
    
    def _calculate_gate_entropy(self, circuit: QuantumCircuit) -> float:
        """Calculate Shannon entropy of gate distribution."""
        gate_counts = self._count_gates(circuit)
        total_gates = sum(gate_counts.values())
        
        if total_gates == 0:
            return 0.0
        
        entropy = 0.0
        for count in gate_counts.values():
            if count > 0:
                p = count / total_gates
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _calculate_connectivity_index(self, circuit: QuantumCircuit) -> float:
        """Calculate connectivity complexity of the circuit."""
        if circuit.num_qubits <= 1:
            return 0.0
        
        # Count unique qubit pairs involved in two-qubit gates
        qubit_pairs = set()
        qubit_map = {qubit: i for i, qubit in enumerate(circuit.qubits)}
        
        for instruction, qubits, _ in circuit.data:
            if instruction.num_qubits == 2:
                q1, q2 = qubits
                i1, i2 = qubit_map[q1], qubit_map[q2]
                qubit_pairs.add(tuple(sorted([i1, i2])))
        
        max_pairs = circuit.num_qubits * (circuit.num_qubits - 1) // 2
        return len(qubit_pairs) / max_pairs if max_pairs > 0 else 0.0
    
    def _calculate_clifford_ratio(self, circuit: QuantumCircuit) -> float:
        """Calculate ratio of Clifford gates."""
        clifford_gates = {'h', 'x', 'y', 'z', 's', 'sdg', 'sx', 'sxdg', 'cx', 'cz', 'swap'}
        
        total_gates = 0
        clifford_count = 0
        
        for instruction, _, _ in circuit.data:
            if instruction.name not in ['measure', 'barrier', 'delay']:
                total_gates += 1
                if instruction.name in clifford_gates:
                    clifford_count += 1
        
        return clifford_count / total_gates if total_gates > 0 else 1.0
    
    def _calculate_parallelization_factor(self, circuit: QuantumCircuit) -> float:
        """Calculate how parallelizable the circuit is."""
        if circuit.depth() == 0:
            return 1.0
        
        total_gates = sum(1 for inst, _, _ in circuit.data 
                         if inst.name not in ['measure', 'barrier', 'delay'])
        
        return total_gates / circuit.depth() if circuit.depth() > 0 else 1.0
    
    def _estimate_entanglement_complexity(self, circuit: QuantumCircuit) -> float:
        """Estimate entanglement generation complexity."""
        two_qubit_gates = self._count_two_qubit_gates(circuit)
        
        if circuit.num_qubits <= 1 or two_qubit_gates == 0:
            return 0.0
        
        # Normalize by maximum possible entanglement
        max_entanglement = circuit.num_qubits * (circuit.num_qubits - 1) // 2
        normalized_gates = two_qubit_gates / max_entanglement
        
        # Apply saturation function
        return 1.0 - math.exp(-normalized_gates)


class SimplePerformanceModel:
    """Simple heuristic-based performance model (fallback when ML unavailable)."""
    
    def __init__(self):
        # Empirically derived performance coefficients
        self.backend_base_times = {
            BackendType.STIM: 0.001,
            BackendType.QISKIT: 0.01,
            BackendType.JAX_METAL: 0.005,
            BackendType.CUDA: 0.003,
            BackendType.TENSOR_NETWORK: 0.02,
            BackendType.DDSIM: 0.008,
        }
        
        self.scaling_factors = {
            BackendType.STIM: {'qubits': 1.0, 'depth': 1.0},  # Linear for Clifford
            BackendType.QISKIT: {'qubits': 2.0, 'depth': 1.2},  # Exponential scaling
            BackendType.JAX_METAL: {'qubits': 1.8, 'depth': 1.1},  # GPU acceleration
            BackendType.CUDA: {'qubits': 1.5, 'depth': 1.1},  # Better GPU acceleration
            BackendType.TENSOR_NETWORK: {'qubits': 1.3, 'depth': 1.5},  # Good for structure
            BackendType.DDSIM: {'qubits': 1.6, 'depth': 1.2},  # Decision diagram efficiency
        }
    
    def predict_execution_time(self, features: CircuitFeatures, backend: BackendType) -> float:
        """Predict execution time using heuristic model."""
        base_time = self.backend_base_times.get(backend, 0.01)
        scaling = self.scaling_factors.get(backend, {'qubits': 2.0, 'depth': 1.2})
        
        # Special case for Clifford circuits with Stim
        if backend == BackendType.STIM and features.clifford_ratio > 0.95:
            # Polynomial scaling for Clifford circuits
            time_estimate = base_time * (features.num_qubits ** 2) * math.log(features.depth + 1)
        else:
            # Exponential scaling for general circuits
            qubit_factor = scaling['qubits'] ** features.num_qubits
            depth_factor = scaling['depth'] ** features.depth
            time_estimate = base_time * qubit_factor * depth_factor
        
        # Apply circuit complexity adjustments
        complexity_factor = 1.0 + features.entanglement_complexity * 0.5
        parallelization_bonus = 1.0 / (1.0 + features.parallelization_factor * 0.1)
        
        return time_estimate * complexity_factor * parallelization_bonus
    
    def predict_memory_usage(self, features: CircuitFeatures, backend: BackendType) -> float:
        """Predict memory usage in MB."""
        # Base memory usage (state vector simulation)
        if backend == BackendType.STIM and features.clifford_ratio > 0.95:
            # Polynomial memory for Clifford circuits
            memory_mb = 0.1 * features.num_qubits ** 2
        else:
            # Exponential memory for general circuits (complex128 state vector)
            state_vector_size = 2 ** features.num_qubits * 16  # bytes
            memory_mb = state_vector_size / (1024 * 1024)
        
        # Backend-specific adjustments
        if backend == BackendType.TENSOR_NETWORK:
            memory_mb *= 0.3  # Tensor contraction is memory efficient
        elif backend in [BackendType.JAX_METAL, BackendType.CUDA]:
            memory_mb *= 1.2  # GPU memory overhead
        
        return max(1.0, memory_mb)  # Minimum 1MB
    
    def predict_success_rate(self, features: CircuitFeatures, backend: BackendType) -> float:
        """Predict execution success probability."""
        base_success = 0.98
        
        # Larger circuits are more likely to fail
        size_penalty = min(0.1, features.num_qubits * 0.005)
        
        # Complex circuits are more likely to fail
        complexity_penalty = features.entanglement_complexity * 0.05
        
        # Backend-specific reliability
        backend_reliability = {
            BackendType.QISKIT: 0.95,
            BackendType.STIM: 0.99,
            BackendType.JAX_METAL: 0.93,
            BackendType.CUDA: 0.92,
            BackendType.TENSOR_NETWORK: 0.88,
            BackendType.DDSIM: 0.94,
        }
        
        backend_factor = backend_reliability.get(backend, 0.90)
        
        success_rate = base_success * backend_factor - size_penalty - complexity_penalty
        return max(0.1, min(1.0, success_rate))


class MLPerformanceModel:
    """Machine learning-based performance prediction model."""
    
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or Path.home() / '.ariadne' / 'ml_models'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Training data storage
        self.training_data: Dict[BackendType, List[Tuple[CircuitFeatures, PerformanceMetrics]]] = {}
        self.models: Dict[BackendType, Any] = {}  # Trained models
        
        # Fallback to simple model
        self.simple_model = SimplePerformanceModel()
        
        # Load existing models
        self._load_models()
    
    def add_training_data(self, backend: BackendType, features: CircuitFeatures, 
                         metrics: PerformanceMetrics) -> None:
        """Add training data point."""
        if backend not in self.training_data:
            self.training_data[backend] = []
        
        self.training_data[backend].append((features, metrics))
        
        # Auto-retrain if we have enough data
        if len(self.training_data[backend]) % 50 == 0:  # Retrain every 50 samples
            self._train_backend_model(backend)
    
    def predict_performance(self, features: CircuitFeatures, backend: BackendType) -> PredictionResult:
        """Predict performance for given circuit features and backend."""
        
        # Try ML model first
        if backend in self.models and self.models[backend] is not None:
            try:
                return self._predict_with_ml_model(features, backend)
            except Exception:
                warnings.warn(f"ML model failed for {backend.value}, using heuristic fallback")
        
        # Fallback to heuristic model
        return self._predict_with_heuristic_model(features, backend)
    
    def _predict_with_ml_model(self, features: CircuitFeatures, backend: BackendType) -> PredictionResult:
        """Make prediction using trained ML model."""
        model = self.models[backend]
        feature_vector = features.to_vector().reshape(1, -1)
        
        # For this implementation, we'll use a simple approach
        # In a real implementation, you'd use sklearn, pytorch, etc.
        predicted_time = self._simple_ml_predict(feature_vector, backend, 'time')
        predicted_memory = self._simple_ml_predict(feature_vector, backend, 'memory')
        predicted_success = self._simple_ml_predict(feature_vector, backend, 'success')
        
        return PredictionResult(
            backend=backend,
            predicted_time=max(0.001, predicted_time),
            predicted_memory_mb=max(1.0, predicted_memory),
            predicted_success_rate=max(0.1, min(1.0, predicted_success)),
            confidence_score=0.8,  # Would be computed from model uncertainty
            feature_importance={
                'num_qubits': 0.3,
                'depth': 0.2,
                'gate_count': 0.15,
                'entanglement_complexity': 0.2,
                'clifford_ratio': 0.15
            }
        )
    
    def _predict_with_heuristic_model(self, features: CircuitFeatures, backend: BackendType) -> PredictionResult:
        """Make prediction using heuristic model."""
        predicted_time = self.simple_model.predict_execution_time(features, backend)
        predicted_memory = self.simple_model.predict_memory_usage(features, backend)
        predicted_success = self.simple_model.predict_success_rate(features, backend)
        
        return PredictionResult(
            backend=backend,
            predicted_time=predicted_time,
            predicted_memory_mb=predicted_memory,
            predicted_success_rate=predicted_success,
            confidence_score=0.6,  # Lower confidence for heuristic
            feature_importance={
                'num_qubits': 0.4,
                'depth': 0.3,
                'clifford_ratio': 0.2,
                'entanglement_complexity': 0.1
            }
        )
    
    def _simple_ml_predict(self, feature_vector: np.ndarray, backend: BackendType, metric: str) -> float:
        """Simple ML prediction (placeholder for real ML implementation)."""
        # This is a simplified placeholder - in reality you'd use proper ML libraries
        features = feature_vector.flatten()
        
        if metric == 'time':
            return self.simple_model.predict_execution_time(
                CircuitFeatures(*features), backend
            )
        elif metric == 'memory':
            return self.simple_model.predict_memory_usage(
                CircuitFeatures(*features), backend
            )
        elif metric == 'success':
            return self.simple_model.predict_success_rate(
                CircuitFeatures(*features), backend
            )
        else:
            return 1.0
    
    def _train_backend_model(self, backend: BackendType) -> None:
        """Train ML model for specific backend."""
        if backend not in self.training_data or len(self.training_data[backend]) < 10:
            return  # Need at least 10 samples
        
        try:
            # Prepare training data
            X = []
            y_time = []
            y_memory = []
            y_success = []
            
            for features, metrics in self.training_data[backend]:
                X.append(features.to_vector())
                y_time.append(metrics.execution_time)
                y_memory.append(metrics.memory_usage_mb)
                y_success.append(metrics.success_probability)
            
            X = np.array(X)
            
            # In a real implementation, you'd train actual ML models here
            # For now, we'll just store the data and use heuristics
            self.models[backend] = {
                'X': X,
                'y_time': np.array(y_time),
                'y_memory': np.array(y_memory),
                'y_success': np.array(y_success),
                'trained': True
            }
            
            # Save model
            self._save_model(backend)
            
        except Exception as e:
            warnings.warn(f"Failed to train model for {backend.value}: {e}")
    
    def _load_models(self) -> None:
        """Load existing models from disk."""
        for backend in BackendType:
            model_file = self.model_dir / f"{backend.value}_model.pkl"
            if model_file.exists():
                try:
                    with open(model_file, 'rb') as f:
                        self.models[backend] = pickle.load(f)
                except Exception:
                    pass  # Ignore loading errors
    
    def _save_model(self, backend: BackendType) -> None:
        """Save model to disk."""
        if backend in self.models:
            model_file = self.model_dir / f"{backend.value}_model.pkl"
            try:
                with open(model_file, 'wb') as f:
                    pickle.dump(self.models[backend], f)
            except Exception:
                pass  # Ignore saving errors


class PerformancePredictor:
    """Main performance prediction interface."""
    
    def __init__(self, use_ml: bool = True):
        self.feature_extractor = CircuitFeatureExtractor()
        self.ml_model = MLPerformanceModel() if use_ml else None
        self.simple_model = SimplePerformanceModel()
    
    def predict_performance(self, circuit: QuantumCircuit, backend: BackendType) -> PredictionResult:
        """Predict performance for circuit on given backend."""
        features = self.feature_extractor.extract_features(circuit)
        
        if self.ml_model:
            return self.ml_model.predict_performance(features, backend)
        else:
            return self.ml_model._predict_with_heuristic_model(features, backend)
    
    def record_actual_performance(self, circuit: QuantumCircuit, backend: BackendType,
                                execution_time: float, memory_usage_mb: float = 0.0,
                                success: bool = True) -> None:
        """Record actual performance for model training."""
        if not self.ml_model:
            return
        
        features = self.feature_extractor.extract_features(circuit)
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=memory_usage_mb,
            success_probability=1.0 if success else 0.0,
            accuracy_score=1.0,  # Placeholder
            energy_consumption=0.0  # Placeholder
        )
        
        self.ml_model.add_training_data(backend, features, metrics)
    
    def get_best_backend_for_circuit(self, circuit: QuantumCircuit, 
                                   available_backends: List[BackendType],
                                   optimize_for: str = 'time') -> Tuple[BackendType, PredictionResult]:
        """Find best backend for circuit based on optimization criterion."""
        predictions = {}
        
        for backend in available_backends:
            predictions[backend] = self.predict_performance(circuit, backend)
        
        if optimize_for == 'time':
            best_backend = min(predictions.keys(), 
                             key=lambda b: predictions[b].predicted_time)
        elif optimize_for == 'memory':
            best_backend = min(predictions.keys(), 
                             key=lambda b: predictions[b].predicted_memory_mb)
        elif optimize_for == 'success':
            best_backend = max(predictions.keys(), 
                             key=lambda b: predictions[b].predicted_success_rate)
        else:
            # Default to time optimization
            best_backend = min(predictions.keys(), 
                             key=lambda b: predictions[b].predicted_time)
        
        return best_backend, predictions[best_backend]


# Convenience functions
def predict_circuit_performance(circuit: QuantumCircuit, backend: BackendType) -> PredictionResult:
    """Convenience function to predict circuit performance."""
    predictor = PerformancePredictor()
    return predictor.predict_performance(circuit, backend)


def find_optimal_backend(circuit: QuantumCircuit, 
                        available_backends: List[BackendType],
                        optimize_for: str = 'time') -> Tuple[BackendType, PredictionResult]:
    """Convenience function to find optimal backend."""
    predictor = PerformancePredictor()
    return predictor.get_best_backend_for_circuit(circuit, available_backends, optimize_for)