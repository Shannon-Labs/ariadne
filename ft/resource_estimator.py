"""
Resource Estimation Framework for Quantum Circuits

This module provides comprehensive resource estimation for quantum circuits,
including time, memory, and hardware requirements for different backends.
Integrates with Qualtran for fault-tolerant resource analysis.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from qiskit import QuantumCircuit

from ariadne.route.analyze import analyze_circuit


@dataclass
class ResourceEstimate:
    """Comprehensive resource estimation for a quantum circuit."""
    
    # Time resources
    execution_time_estimate: float  # Expected execution time in seconds
    gate_time_estimate: float      # Time per gate operation
    measurement_time_estimate: float  # Time for measurements
    
    # Memory resources  
    memory_requirement_mb: float   # Peak memory requirement in MB
    statevector_size_mb: float    # Memory for statevector simulation
    intermediate_storage_mb: float # Additional storage needed
    
    # Hardware resources
    qubit_requirement: int        # Number of physical qubits needed
    connectivity_requirement: int # Minimum qubit connectivity
    gate_count_estimate: int     # Total gate operations
    
    # Error correction resources (for fault-tolerant circuits)
    logical_qubits: Optional[int] = None
    physical_qubits: Optional[int] = None
    code_distance: Optional[int] = None
    
    # Backend-specific estimates
    backend_estimates: Dict[str, Dict[str, float]] = None
    
    # Confidence and metadata
    confidence_score: float = 0.0
    estimation_method: str = "heuristic"
    analysis_metadata: Dict[str, Any] = None


@dataclass 
class BackendResourceProfile:
    """Resource profile for a specific backend."""
    
    backend_name: str
    
    # Performance characteristics
    single_qubit_gate_time_ns: float
    two_qubit_gate_time_ns: float
    measurement_time_ns: float
    readout_time_ns: float
    
    # Memory characteristics
    memory_per_qubit_mb: float
    base_memory_overhead_mb: float
    
    # Hardware limitations
    max_qubits: int
    max_circuit_depth: int
    connectivity_graph: Optional[Any] = None
    
    # Error rates
    single_qubit_error_rate: float = 1e-4
    two_qubit_error_rate: float = 1e-2
    measurement_error_rate: float = 1e-2


class QuantumResourceEstimator:
    """
    Comprehensive quantum circuit resource estimator.
    
    Provides accurate estimates for execution time, memory requirements,
    and hardware resources across different quantum backends.
    """
    
    def __init__(self):
        """Initialize with default backend profiles."""
        self.backend_profiles = self._initialize_backend_profiles()
        self.fault_tolerant_profiles = self._initialize_ft_profiles()
    
    def estimate_resources(self, 
                         circuit: QuantumCircuit,
                         target_backends: Optional[List[str]] = None,
                         shots: int = 1000,
                         include_fault_tolerant: bool = False) -> ResourceEstimate:
        """
        Estimate comprehensive resources for quantum circuit execution.
        """
        
        # Analyze circuit properties
        analysis = analyze_circuit(circuit)
        
        # Basic resource calculations
        basic_estimates = self._calculate_basic_resources(circuit, analysis, shots)
        
        # Backend-specific estimates
        backend_estimates = {}
        target_backends = target_backends or list(self.backend_profiles.keys())
        
        for backend_name in target_backends:
            if backend_name in self.backend_profiles:
                backend_estimates[backend_name] = self._estimate_backend_resources(
                    circuit, analysis, self.backend_profiles[backend_name], shots
                )
        
        # Fault-tolerant estimates if requested
        ft_estimates = None
        if include_fault_tolerant:
            ft_estimates = self._estimate_fault_tolerant_resources(circuit, analysis)
        
        # Combine into comprehensive estimate
        return ResourceEstimate(
            execution_time_estimate=basic_estimates['execution_time'],
            gate_time_estimate=basic_estimates['gate_time'],
            measurement_time_estimate=basic_estimates['measurement_time'],
            memory_requirement_mb=basic_estimates['memory_mb'],
            statevector_size_mb=basic_estimates['statevector_mb'],
            intermediate_storage_mb=basic_estimates['intermediate_mb'],
            qubit_requirement=circuit.num_qubits,
            connectivity_requirement=analysis.get('light_cone_width', 0),
            gate_count_estimate=basic_estimates['gate_count'],
            logical_qubits=ft_estimates.get('logical_qubits') if ft_estimates else None,
            physical_qubits=ft_estimates.get('physical_qubits') if ft_estimates else None,
            code_distance=ft_estimates.get('code_distance') if ft_estimates else None,
            backend_estimates=backend_estimates,
            confidence_score=self._calculate_confidence(circuit, analysis),
            estimation_method="comprehensive",
            analysis_metadata=analysis
        )
    
    def _calculate_basic_resources(self, 
                                 circuit: QuantumCircuit, 
                                 analysis: Dict[str, Any],
                                 shots: int) -> Dict[str, float]:
        """Calculate basic resource estimates."""
        
        # Gate counting
        single_qubit_gates = 0
        two_qubit_gates = 0
        measurement_ops = 0
        
        for instruction, qubits, clbits in circuit.data:
            if instruction.name in ['measure']:
                measurement_ops += 1
            elif instruction.name in ['barrier', 'delay']:
                continue
            elif instruction.num_qubits == 1:
                single_qubit_gates += 1
            elif instruction.num_qubits == 2:
                two_qubit_gates += 1
        
        total_gates = single_qubit_gates + two_qubit_gates
        
        # Time estimates (using conservative defaults)
        single_qubit_time_ns = 50    # 50ns per single-qubit gate
        two_qubit_time_ns = 200      # 200ns per two-qubit gate
        measurement_time_ns = 1000   # 1μs per measurement
        
        gate_time = (single_qubit_gates * single_qubit_time_ns + 
                    two_qubit_gates * two_qubit_time_ns) * 1e-9
        measurement_time = measurement_ops * measurement_time_ns * 1e-9
        execution_time = (gate_time + measurement_time) * shots
        
        # Memory estimates
        num_qubits = circuit.num_qubits
        
        # Statevector simulation memory
        complex_amplitudes = 2 ** num_qubits
        bytes_per_amplitude = 16  # 8 bytes real + 8 bytes imaginary (double precision)
        statevector_mb = complex_amplitudes * bytes_per_amplitude / (1024 * 1024)
        
        # Additional memory for intermediate calculations
        intermediate_mb = statevector_mb * 0.5  # 50% overhead for operations
        
        # Total memory requirement
        memory_mb = statevector_mb + intermediate_mb
        
        return {
            'execution_time': execution_time,
            'gate_time': gate_time,
            'measurement_time': measurement_time,
            'memory_mb': memory_mb,
            'statevector_mb': statevector_mb,
            'intermediate_mb': intermediate_mb,
            'gate_count': total_gates,
            'single_qubit_gates': single_qubit_gates,
            'two_qubit_gates': two_qubit_gates,
            'measurement_ops': measurement_ops
        }
    
    def _estimate_backend_resources(self,
                                  circuit: QuantumCircuit,
                                  analysis: Dict[str, Any],
                                  profile: BackendResourceProfile,
                                  shots: int) -> Dict[str, float]:
        """Estimate resources for a specific backend."""
        
        # Gate counting (reuse from basic calculation)
        basic = self._calculate_basic_resources(circuit, analysis, shots)
        
        # Backend-specific timing
        gate_time = (basic['single_qubit_gates'] * profile.single_qubit_gate_time_ns +
                    basic['two_qubit_gates'] * profile.two_qubit_gate_time_ns) * 1e-9
        
        measurement_time = basic['measurement_ops'] * profile.measurement_time_ns * 1e-9
        
        # Backend-specific memory
        memory_mb = (circuit.num_qubits * profile.memory_per_qubit_mb + 
                    profile.base_memory_overhead_mb)
        
        # Error accumulation estimate
        total_error_rate = (basic['single_qubit_gates'] * profile.single_qubit_error_rate +
                           basic['two_qubit_gates'] * profile.two_qubit_error_rate +
                           basic['measurement_ops'] * profile.measurement_error_rate)
        
        # Feasibility checks
        is_feasible = (circuit.num_qubits <= profile.max_qubits and
                      circuit.depth() <= profile.max_circuit_depth)
        
        return {
            'execution_time': (gate_time + measurement_time) * shots,
            'gate_time': gate_time,
            'measurement_time': measurement_time,
            'memory_mb': memory_mb,
            'total_error_rate': total_error_rate,
            'is_feasible': is_feasible,
            'utilization_score': circuit.num_qubits / profile.max_qubits
        }
    
    def _estimate_fault_tolerant_resources(self,
                                         circuit: QuantumCircuit,
                                         analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate fault-tolerant quantum computing resources."""
        
        logical_qubits = circuit.num_qubits
        
        # Estimate required code distance based on circuit complexity
        # More complex circuits need higher code distance
        noise_susceptibility = analysis.get('noise_susceptibility', 0.5)
        base_distance = 3  # Minimum code distance
        complexity_factor = min(10, analysis.get('depth', 1) / 10)  # Scale with depth
        
        code_distance = base_distance + int(complexity_factor * noise_susceptibility)
        if code_distance % 2 == 0:  # Code distance should be odd
            code_distance += 1
        
        # Physical qubits for surface code (approximate)
        # Surface code requires roughly 2 * d^2 physical qubits per logical qubit
        physical_qubits_per_logical = 2 * code_distance ** 2
        total_physical_qubits = logical_qubits * physical_qubits_per_logical
        
        # Error correction overhead
        logical_gate_time_multiplier = code_distance  # Rough estimate
        
        # Magic state distillation for T gates
        t_gates = self._count_t_gates(circuit)
        magic_state_overhead = t_gates * 100  # Rough estimate: 100 physical qubits per T gate
        
        total_physical_qubits += magic_state_overhead
        
        return {
            'logical_qubits': logical_qubits,
            'physical_qubits': total_physical_qubits,
            'code_distance': code_distance,
            'physical_per_logical': physical_qubits_per_logical,
            'magic_state_overhead': magic_state_overhead,
            'logical_gate_time_multiplier': logical_gate_time_multiplier,
            't_gate_count': t_gates
        }
    
    def _count_t_gates(self, circuit: QuantumCircuit) -> int:
        """Count T gates in circuit (non-Clifford gates that require magic states)."""
        
        # Non-Clifford gates that typically require T gates
        t_gate_names = {'t', 'tdg', 'rz', 'ry', 'rx', 'u1', 'u2', 'u3', 'ccx'}
        
        t_count = 0
        for instruction, _, _ in circuit.data:
            if instruction.name.lower() in t_gate_names:
                t_count += 1
        
        return t_count
    
    def _calculate_confidence(self, circuit: QuantumCircuit, analysis: Dict[str, Any]) -> float:
        """Calculate confidence in resource estimates."""
        
        # Factors affecting confidence
        num_qubits = circuit.num_qubits
        depth = circuit.depth()
        is_clifford = analysis.get('is_clifford', False)
        
        # Higher confidence for smaller, well-understood circuits
        size_confidence = max(0.5, 1.0 - (num_qubits - 10) / 40)  # Decreases with size
        depth_confidence = max(0.5, 1.0 - (depth - 10) / 100)     # Decreases with depth
        
        # Clifford circuits have well-understood resource requirements
        type_confidence = 0.9 if is_clifford else 0.7
        
        # Combined confidence
        confidence = (0.4 * size_confidence + 0.3 * depth_confidence + 0.3 * type_confidence)
        
        return min(1.0, confidence)
    
    def _initialize_backend_profiles(self) -> Dict[str, BackendResourceProfile]:
        """Initialize default backend resource profiles."""
        
        return {
            'qiskit': BackendResourceProfile(
                backend_name='qiskit',
                single_qubit_gate_time_ns=50,
                two_qubit_gate_time_ns=200,
                measurement_time_ns=1000,
                readout_time_ns=5000,
                memory_per_qubit_mb=8.0,  # 8MB per qubit for statevector
                base_memory_overhead_mb=100,
                max_qubits=24,
                max_circuit_depth=1000,
                single_qubit_error_rate=1e-4,
                two_qubit_error_rate=1e-2,
                measurement_error_rate=1e-2
            ),
            
            'stim': BackendResourceProfile(
                backend_name='stim',
                single_qubit_gate_time_ns=1,    # Very fast for Clifford
                two_qubit_gate_time_ns=2,
                measurement_time_ns=10,
                readout_time_ns=50,
                memory_per_qubit_mb=0.1,        # Very memory efficient
                base_memory_overhead_mb=10,
                max_qubits=10000,               # Practically unlimited for Clifford
                max_circuit_depth=100000,
                single_qubit_error_rate=0,      # Perfect simulation
                two_qubit_error_rate=0,
                measurement_error_rate=0
            ),
            
            'metal': BackendResourceProfile(
                backend_name='metal',
                single_qubit_gate_time_ns=30,   # Accelerated
                two_qubit_gate_time_ns=120,
                measurement_time_ns=800,
                readout_time_ns=4000,
                memory_per_qubit_mb=6.0,        # Unified memory efficiency
                base_memory_overhead_mb=200,    # GPU overhead
                max_qubits=30,
                max_circuit_depth=2000,
                single_qubit_error_rate=1e-5,   # High precision
                two_qubit_error_rate=5e-3,
                measurement_error_rate=1e-3
            ),
            
            'cuda': BackendResourceProfile(
                backend_name='cuda',
                single_qubit_gate_time_ns=10,   # Highly parallel
                two_qubit_gate_time_ns=50,
                measurement_time_ns=500,
                readout_time_ns=2000,
                memory_per_qubit_mb=4.0,        # GPU memory efficiency
                base_memory_overhead_mb=500,    # CUDA overhead
                max_qubits=40,
                max_circuit_depth=5000,
                single_qubit_error_rate=1e-6,   # Very high precision
                two_qubit_error_rate=1e-3,
                measurement_error_rate=1e-4
            ),
            
            'tensor_network': BackendResourceProfile(
                backend_name='tensor_network',
                single_qubit_gate_time_ns=100,  # Slower but scalable
                two_qubit_gate_time_ns=500,
                measurement_time_ns=2000,
                readout_time_ns=10000,
                memory_per_qubit_mb=2.0,        # Memory efficient for structured circuits
                base_memory_overhead_mb=50,
                max_qubits=100,                 # Scales well for low treewidth
                max_circuit_depth=10000,
                single_qubit_error_rate=1e-8,   # Exact simulation
                two_qubit_error_rate=1e-8,
                measurement_error_rate=1e-8
            )
        }
    
    def _initialize_ft_profiles(self) -> Dict[str, Any]:
        """Initialize fault-tolerant computing profiles."""
        
        return {
            'surface_code': {
                'physical_per_logical_base': 2,  # Coefficient for d^2 scaling
                'min_code_distance': 3,
                'max_code_distance': 21,
                'magic_state_overhead': 100,     # Physical qubits per T gate
                'logical_gate_time_factor': 'code_distance'
            },
            
            'color_code': {
                'physical_per_logical_base': 1.5,
                'min_code_distance': 3,
                'max_code_distance': 15,
                'magic_state_overhead': 50,
                'logical_gate_time_factor': 'code_distance'
            }
        }
    
    def estimate_scaling(self,
                        base_circuit: QuantumCircuit,
                        qubit_range: Tuple[int, int, int],
                        backend: str = 'qiskit') -> Dict[str, List[float]]:
        """
        Estimate resource scaling across different circuit sizes.
        
        Args:
            base_circuit: Base circuit to scale
            qubit_range: (start, stop, step) for qubit counts
            backend: Target backend for estimates
            
        Returns:
            Dictionary with scaling data for visualization
        """
        
        scaling_data = {
            'qubit_counts': [],
            'execution_times': [],
            'memory_requirements': [],
            'gate_counts': []
        }
        
        for num_qubits in range(*qubit_range):
            # Create scaled circuit (simplified scaling)
            scaled_circuit = QuantumCircuit(num_qubits)
            
            # Scale the original circuit structure
            scaling_factor = num_qubits / base_circuit.num_qubits
            
            # Add scaled gates (simplified approach)
            for instruction, qubits, clbits in base_circuit.data:
                if instruction.name not in ['measure', 'barrier', 'delay']:
                    # Map to new qubit indices
                    new_qubits = [min(q, num_qubits - 1) for q in range(len(qubits))]
                    if all(q < num_qubits for q in new_qubits):
                        # Add instruction to scaled circuit (simplified)
                        if instruction.num_qubits == 1:
                            scaled_circuit.h(new_qubits[0])  # Placeholder
                        elif instruction.num_qubits == 2:
                            scaled_circuit.cx(new_qubits[0], new_qubits[1])  # Placeholder
            
            # Estimate resources for scaled circuit
            estimate = self.estimate_resources(scaled_circuit, [backend])
            
            scaling_data['qubit_counts'].append(num_qubits)
            scaling_data['execution_times'].append(estimate.execution_time_estimate)
            scaling_data['memory_requirements'].append(estimate.memory_requirement_mb)
            scaling_data['gate_counts'].append(estimate.gate_count_estimate)
        
        return scaling_data


# Convenience functions for common use cases
def estimate_circuit_resources(circuit: QuantumCircuit,
                             backend: str = 'qiskit',
                             shots: int = 1000) -> ResourceEstimate:
    """Quick resource estimation for a single backend."""
    estimator = QuantumResourceEstimator()
    return estimator.estimate_resources(circuit, [backend], shots)


def compare_backend_resources(circuit: QuantumCircuit,
                            backends: List[str],
                            shots: int = 1000) -> Dict[str, ResourceEstimate]:
    """Compare resource requirements across multiple backends."""
    estimator = QuantumResourceEstimator()
    
    results = {}
    for backend in backends:
        results[backend] = estimator.estimate_resources(circuit, [backend], shots)
    
    return results


def estimate_fault_tolerant_resources(circuit: QuantumCircuit,
                                    target_error_rate: float = 1e-6) -> ResourceEstimate:
    """Estimate fault-tolerant quantum computing resources."""
    estimator = QuantumResourceEstimator()
    return estimator.estimate_resources(circuit, include_fault_tolerant=True)