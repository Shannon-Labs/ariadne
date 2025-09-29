"""
Comprehensive Backend Unit Tests

This module provides comprehensive unit tests for all Ariadne quantum backends,
including functionality, performance, and integration testing.
"""

import time
from unittest.mock import patch

import pytest
from qiskit import QuantumCircuit

from ariadne.backends.cuda_backend import CUDABackend, is_cuda_available
from ariadne.backends.metal_backend import MetalBackend, is_metal_available
from ariadne.route.analyze import analyze_circuit

# Import Ariadne components
from ariadne.router import BackendType, EnhancedQuantumRouter
from ariadne.simulation import QuantumSimulator, SimulationOptions


class TestEnhancedQuantumRouter:
    """Test cases for the enhanced quantum router."""
    
    def test_router_initialization(self):
        """Test router initialization with default settings."""
        router = EnhancedQuantumRouter()
        assert router is not None
        assert len(router.backend_capacities) > 0
        assert BackendType.STIM in router.backend_capacities
    
    def test_circuit_entropy_calculation(self):
        """Test circuit entropy calculation."""
        router = EnhancedQuantumRouter()
        
        # Simple circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        
        entropy = router.circuit_entropy(qc)
        assert entropy >= 0.0
        assert entropy <= 2.0  # Maximum for 2 gate types
    
    def test_backend_selection_clifford(self):
        """Test backend selection for Clifford circuits."""
        router = EnhancedQuantumRouter()
        
        # Create Clifford circuit
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        
        decision = router.select_optimal_backend(qc)
        
        # Should prefer Stim for Clifford circuits
        assert decision.recommended_backend == BackendType.STIM
        assert decision.confidence_score > 0.5
    
    def test_backend_selection_non_clifford(self):
        """Test backend selection for non-Clifford circuits."""
        router = EnhancedQuantumRouter()
        
        # Create non-Clifford circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.t(0)  # T gate makes it non-Clifford
        qc.cx(0, 1)
        qc.measure_all()
        
        decision = router.select_optimal_backend(qc)
        
        # Should not select Stim
        assert decision.recommended_backend != BackendType.STIM
        assert decision.confidence_score > 0.0
    
    def test_simulation_with_fallback(self):
        """Test simulation with fallback mechanism."""
        router = EnhancedQuantumRouter()
        
        # Simple circuit that should work on any backend
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        result = router.simulate(qc, shots=100)
        
        assert result is not None
        assert sum(result.counts.values()) == 100
        assert result.backend_used in [bt.value for bt in BackendType]
        assert result.execution_time >= 0
    
    def test_capacity_calibration_integration(self):
        """Test integration with calibration system."""
        router = EnhancedQuantumRouter(use_calibration=True)
        
        # Test capacity update
        original_capacity = router.backend_capacities[BackendType.QISKIT].general_capacity
        router.update_capacity(BackendType.QISKIT, general_capacity=15.0)
        
        assert router.backend_capacities[BackendType.QISKIT].general_capacity == 15.0
        assert router.backend_capacities[BackendType.QISKIT].general_capacity != original_capacity


class TestMetalBackend:
    """Test cases for Metal backend."""
    
    def test_metal_availability_detection(self):
        """Test Metal availability detection."""
        available = is_metal_available()
        assert isinstance(available, bool)
    
    def test_metal_backend_initialization(self):
        """Test Metal backend initialization."""
        backend = MetalBackend(allow_cpu_fallback=True)
        assert backend is not None
        assert backend.backend_mode in ["metal", "cpu"]
    
    def test_metal_backend_simulation(self):
        """Test Metal backend simulation."""
        backend = MetalBackend(allow_cpu_fallback=True)
        
        # Simple test circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        result = backend.simulate(qc, shots=100)
        
        assert isinstance(result, dict)
        assert sum(result.values()) == 100
        assert len(result) > 0
    
    def test_metal_memory_management(self):
        """Test Metal backend memory management."""
        backend = MetalBackend(
            allow_cpu_fallback=True,
            memory_pool_size_mb=1024,
            enable_memory_mapping=True
        )
        
        # Test memory statistics
        stats = backend.memory_stats
        assert isinstance(stats, dict)
        assert 'total_allocated_mb' in stats
        assert 'pool_size_mb' in stats
    
    def test_metal_performance_stats(self):
        """Test Metal backend performance statistics."""
        backend = MetalBackend(allow_cpu_fallback=True)
        
        # Run a simulation to generate stats
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        backend.simulate(qc, shots=100)
        
        stats = backend.performance_stats
        assert isinstance(stats, dict)
        assert 'memory_stats' in stats
        assert 'backend_mode' in stats
    
    @pytest.mark.skipif(not is_metal_available(), reason="Metal not available")
    def test_metal_acceleration(self):
        """Test Metal GPU acceleration (if available)."""
        backend = MetalBackend(prefer_gpu=True, allow_cpu_fallback=False)
        
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        
        start_time = time.perf_counter()
        result = backend.simulate(qc, shots=1000)
        execution_time = time.perf_counter() - start_time
        
        assert result is not None
        assert execution_time < 10.0  # Should be reasonably fast
        assert backend.backend_mode == "metal"


class TestCUDABackend:
    """Test cases for CUDA backend."""
    
    def test_cuda_availability_detection(self):
        """Test CUDA availability detection."""
        available = is_cuda_available()
        assert isinstance(available, bool)
    
    def test_cuda_backend_initialization(self):
        """Test CUDA backend initialization."""
        backend = CUDABackend(allow_cpu_fallback=True)
        assert backend is not None
        assert backend.backend_mode in ["cuda", "cpu"]
    
    def test_cuda_backend_simulation(self):
        """Test CUDA backend simulation."""
        backend = CUDABackend(allow_cpu_fallback=True)
        
        # Test circuit
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        
        result = backend.simulate(qc, shots=100)
        
        assert isinstance(result, dict)
        assert sum(result.values()) == 100
        assert len(result) > 0
    
    @pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
    def test_cuda_acceleration(self):
        """Test CUDA GPU acceleration (if available)."""
        backend = CUDABackend(prefer_gpu=True, allow_cpu_fallback=False)
        
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.h(i)
        qc.measure_all()
        
        start_time = time.perf_counter()
        result = backend.simulate(qc, shots=1000)
        execution_time = time.perf_counter() - start_time
        
        assert result is not None
        assert execution_time < 10.0
        assert backend.backend_mode == "cuda"


class TestCircuitAnalysis:
    """Test cases for circuit analysis engine."""
    
    def test_basic_circuit_analysis(self):
        """Test basic circuit analysis functionality."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        
        analysis = analyze_circuit(qc)
        
        assert isinstance(analysis, dict)
        assert 'num_qubits' in analysis
        assert analysis['num_qubits'] == 3
        assert 'depth' in analysis
        assert 'is_clifford' in analysis
        assert analysis['is_clifford']
    
    def test_advanced_circuit_metrics(self):
        """Test advanced circuit analysis metrics."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.t(0)  # Non-Clifford
        qc.cx(0, 1)
        qc.ry(0.5, 2)
        qc.cx(2, 3)
        
        analysis = analyze_circuit(qc)
        
        # Check advanced metrics exist
        assert 'gate_entropy' in analysis
        assert 'entanglement_entropy_estimate' in analysis
        assert 'quantum_volume_estimate' in analysis
        assert 'parallelization_factor' in analysis
        assert 'noise_susceptibility' in analysis
        assert 'expressivity_measure' in analysis
        
        # Verify reasonable values
        assert analysis['gate_entropy'] >= 0
        assert analysis['entanglement_entropy_estimate'] >= 0
        assert analysis['quantum_volume_estimate'] > 0
        assert not analysis['is_clifford']
    
    def test_clifford_detection(self):
        """Test Clifford circuit detection."""
        # Pure Clifford circuit
        clifford_qc = QuantumCircuit(2)
        clifford_qc.h(0)
        clifford_qc.s(0)
        clifford_qc.cx(0, 1)
        
        analysis_clifford = analyze_circuit(clifford_qc)
        assert analysis_clifford['is_clifford']
        
        # Non-Clifford circuit
        non_clifford_qc = QuantumCircuit(2)
        non_clifford_qc.h(0)
        non_clifford_qc.t(0)  # T gate
        non_clifford_qc.cx(0, 1)
        
        analysis_non_clifford = analyze_circuit(non_clifford_qc)
        assert not analysis_non_clifford['is_clifford']


class TestUnifiedSimulationAPI:
    """Test cases for unified simulation API."""
    
    def test_simulator_initialization(self):
        """Test quantum simulator initialization."""
        simulator = QuantumSimulator()
        assert simulator is not None
        assert simulator.router is not None
    
    def test_basic_simulation(self):
        """Test basic simulation functionality."""
        simulator = QuantumSimulator()
        
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        options = SimulationOptions(shots=100)
        result = simulator.simulate(qc, options)
        
        assert result is not None
        assert sum(result.counts.values()) == 100
        assert result.execution_time >= 0
        assert isinstance(result.backend_used, str)
    
    def test_simulation_with_analysis(self):
        """Test simulation with comprehensive analysis."""
        simulator = QuantumSimulator()
        
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        
        options = SimulationOptions(
            shots=100,
            analyze_quantum_advantage=True,
            estimate_resources=True
        )
        
        result = simulator.simulate(qc, options)
        
        assert result.circuit_analysis is not None
        assert result.quantum_advantage is not None
        assert result.resource_estimate is not None
    
    def test_batch_simulation(self):
        """Test batch simulation functionality."""
        simulator = QuantumSimulator()
        
        circuits = []
        for i in range(3):
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            if i > 0:
                qc.cx(0, 1)
            qc.measure_all()
            circuits.append(qc)
        
        options = SimulationOptions(shots=50)
        results = simulator.simulate_batch(circuits, options)
        
        assert len(results) == 3
        for result in results:
            assert sum(result.counts.values()) == 50
    
    def test_backend_comparison(self):
        """Test backend comparison functionality."""
        simulator = QuantumSimulator()
        
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        backends = ['qiskit', 'stim']
        results = simulator.compare_backends(qc, backends, shots=100)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for backend, result in results.items():
            assert backend in backends
            assert sum(result.counts.values()) == 100


class TestQuantumAdvantageDetection:
    """Test cases for quantum advantage detection."""
    
    def test_advantage_detection_clifford(self):
        """Test quantum advantage detection for Clifford circuits."""
        from ariadne.quantum_advantage import detect_quantum_advantage
        
        # Clifford circuit - should have no advantage
        qc = QuantumCircuit(10)
        qc.h(0)
        for i in range(9):
            qc.cx(i, i + 1)
        
        advantage = detect_quantum_advantage(qc)
        
        assert isinstance(advantage, dict)
        assert 'overall_advantage_score' in advantage
        assert 'has_quantum_advantage' in advantage
        assert not advantage['has_quantum_advantage']  # Clifford circuits are classical
    
    def test_advantage_detection_large_circuit(self):
        """Test quantum advantage detection for large non-Clifford circuit."""
        from ariadne.quantum_advantage import detect_quantum_advantage
        
        # Large non-Clifford circuit
        qc = QuantumCircuit(20)
        for i in range(20):
            qc.h(i)
            qc.t(i)  # Non-Clifford
        for i in range(19):
            qc.cx(i, i + 1)
        
        advantage = detect_quantum_advantage(qc)
        
        assert isinstance(advantage, dict)
        assert advantage['overall_advantage_score'] > 0.5  # Should show advantage
        assert 'classical_intractability' in advantage
        assert 'quantum_volume_advantage' in advantage


class TestResourceEstimation:
    """Test cases for resource estimation."""
    
    def test_basic_resource_estimation(self):
        """Test basic resource estimation functionality."""
        from ariadne.ft.resource_estimator import estimate_circuit_resources
        
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)
        
        estimate = estimate_circuit_resources(qc, shots=1000)
        
        assert estimate is not None
        assert estimate.execution_time_estimate > 0
        assert estimate.memory_requirement_mb > 0
        assert estimate.qubit_requirement == 5
        assert estimate.confidence_score > 0
    
    def test_fault_tolerant_estimation(self):
        """Test fault-tolerant resource estimation."""
        from ariadne.ft.resource_estimator import estimate_fault_tolerant_resources
        
        qc = QuantumCircuit(10)
        qc.h(0)
        qc.t(0)  # T gate requires magic states
        for i in range(9):
            qc.cx(i, i + 1)
        
        estimate = estimate_fault_tolerant_resources(qc)
        
        assert estimate is not None
        assert estimate.logical_qubits is not None
        assert estimate.physical_qubits is not None
        assert estimate.code_distance is not None
        assert estimate.physical_qubits > estimate.logical_qubits


class TestConfigurationSystem:
    """Test cases for configuration system."""
    
    def test_config_initialization(self):
        """Test configuration system initialization."""
        from ariadne.config import AriadneConfig, get_config
        
        config = get_config()
        assert isinstance(config, AriadneConfig)
        assert len(config.backends) > 0
    
    def test_backend_preference_configuration(self):
        """Test backend preference configuration."""
        from ariadne.config import get_config_manager
        
        manager = get_config_manager()
        
        # Test setting preference
        manager.get_preferred_backends()
        manager.set_backend_preference('test_backend', 10)
        
        # Verify change (simplified test)
        assert manager.config.backends['test_backend'].priority == 10
    
    def test_platform_specific_configuration(self):
        """Test platform-specific configuration."""
        from ariadne.config import ConfigManager
        
        manager = ConfigManager()
        
        # Test platform detection and configuration
        manager.configure_for_platform("cpu_only")
        
        # Verify GPU backends are disabled
        metal_config = manager.get_backend_config("metal")
        cuda_config = manager.get_backend_config("cuda")
        
        assert not metal_config.enabled
        assert not cuda_config.enabled


class TestErrorHandling:
    """Test cases for error handling and edge cases."""
    
    def test_invalid_circuit_handling(self):
        """Test handling of invalid circuits."""
        router = EnhancedQuantumRouter()
        
        # Empty circuit
        qc = QuantumCircuit(0)
        
        with pytest.raises(ValueError):
            router.simulate(qc, shots=100)
    
    def test_zero_shots_handling(self):
        """Test handling of invalid shot counts."""
        router = EnhancedQuantumRouter()
        
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure_all()
        
        with pytest.raises(ValueError):
            router.simulate(qc, shots=0)
    
    def test_backend_failure_fallback(self):
        """Test fallback behavior when backends fail."""
        router = EnhancedQuantumRouter()
        
        # Create a circuit that should work
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # Simulate backend failure by mocking
        with patch.object(router, '_simulate_stim', side_effect=Exception("Simulated failure")):
            result = router.simulate(qc, shots=100)
            
            # Should still get a result due to fallback
            assert result is not None
            assert sum(result.counts.values()) == 100
            # Should have used fallback backend
            assert result.metadata.get('fallback_used', False)


class TestPerformanceRequirements:
    """Test cases for performance requirements."""
    
    def test_small_circuit_performance(self):
        """Test performance requirements for small circuits."""
        router = EnhancedQuantumRouter()
        
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        
        start_time = time.perf_counter()
        result = router.simulate(qc, shots=1000)
        execution_time = time.perf_counter() - start_time
        
        # Should complete quickly for small circuits
        assert execution_time < 5.0  # 5 seconds max
        assert result is not None
    
    def test_memory_efficiency(self):
        """Test memory efficiency for various circuit sizes."""
        import psutil
        
        router = EnhancedQuantumRouter()
        process = psutil.Process()
        
        # Measure memory before
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Run simulation
        qc = QuantumCircuit(8, 8)
        for i in range(8):
            qc.h(i)
        qc.measure_all()
        
        result = router.simulate(qc, shots=100)
        
        # Measure memory after
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_used = memory_after - memory_before
        
        # Should not use excessive memory
        assert memory_used < 500  # 500 MB max for 8-qubit circuit
        assert result is not None


# Convenience function to run all tests
def run_comprehensive_tests():
    """Run all comprehensive backend tests."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_comprehensive_tests()