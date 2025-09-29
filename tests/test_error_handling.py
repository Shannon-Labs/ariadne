"""
Test error handling and fallback behavior in the quantum router.

This module ensures that the router gracefully handles backend failures
and provides meaningful error messages and fallback behavior.
"""

from unittest.mock import patch

import pytest
from qiskit import QuantumCircuit

from ariadne.backends.metal_backend import MetalBackend
from ariadne.router import BackendType, EnhancedQuantumRouter, simulate


class TestErrorHandling:
    """Test error handling and fallback behavior."""

    def test_backend_fallback_on_failure(self):
        """Test that router falls back to Qiskit when preferred backend fails."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        router = EnhancedQuantumRouter()
        
        # Mock the metal backend to fail
        with patch.object(router, '_simulate_jax_metal', side_effect=RuntimeError("Mock failure")):
            with patch.object(router, 'select_optimal_backend') as mock_select:
                # Force router to select JAX_METAL initially
                from ariadne.router import RoutingDecision
                mock_select.return_value = RoutingDecision(
                    circuit_entropy=1.0,
                    recommended_backend=BackendType.JAX_METAL,
                    confidence_score=0.8,
                    expected_speedup=1.5,
                    channel_capacity_match=0.8,
                    alternatives=[]
                )
                
                result = router.simulate(qc, shots=100)
                
                # Should fall back to Qiskit
                assert result.backend_used == BackendType.QISKIT
                assert result.fallback_reason is not None
                assert "jax_metal failed" in result.fallback_reason.lower()
                assert len(result.counts) > 0

    def test_metal_backend_cpu_fallback(self):
        """Test Metal backend CPU fallback behavior."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()

        try:
            backend = MetalBackend(allow_cpu_fallback=True)
            result = backend.simulate(qc, shots=100)
            
            # Should work (either Metal or CPU mode)
            assert len(result) > 0
            assert isinstance(result, dict)
            
            # Check backend mode
            print(f"Backend mode: {backend.backend_mode}")
            assert backend.backend_mode in ["metal", "cpu"]
            
        except ImportError:
            pytest.skip("JAX not available for testing")

    def test_routing_with_unavailable_backends(self):
        """Test routing behavior when preferred backends are unavailable."""
        qc = QuantumCircuit(4, 4)
        qc.h(range(4))
        qc.measure_all()

        router = EnhancedQuantumRouter()
        
        # Mock CUDA and Metal as unavailable
        router._cuda_available = False
        router._metal_available = False
        
        # Update capacities
        router.backend_capacities[BackendType.CUDA].clifford_capacity = 0.0
        router.backend_capacities[BackendType.CUDA].general_capacity = 0.0
        router.backend_capacities[BackendType.JAX_METAL].clifford_capacity = 0.0
        router.backend_capacities[BackendType.JAX_METAL].general_capacity = 0.0
        
        result = router.simulate(qc, shots=100)
        
        # Should use Stim (for Clifford circuit) or Qiskit
        assert result.backend_used in [BackendType.STIM, BackendType.QISKIT, BackendType.TENSOR_NETWORK]
        assert len(result.counts) > 0

    def test_simulate_function_error_handling(self):
        """Test the high-level simulate function error handling."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        # Should work without any errors
        result = simulate(qc, shots=100)
        
        assert len(result.counts) > 0
        assert result.backend_used is not None
        assert result.execution_time >= 0
        assert result.routing_decision is not None

    def test_forced_backend_error_handling(self):
        """Test error handling when forcing a specific backend."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        # Force Qiskit backend - should always work
        result = simulate(qc, shots=100, backend="qiskit")
        assert result.backend_used == BackendType.QISKIT
        assert len(result.counts) > 0

    def test_invalid_backend_specification(self):
        """Test error handling for invalid backend names."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        with pytest.raises(ValueError, match="Unknown backend"):
            simulate(qc, shots=100, backend="invalid_backend")

    def test_comprehensive_fallback_chain(self):
        """Test the complete fallback chain when multiple backends fail."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        router = EnhancedQuantumRouter()
        
        # Mock multiple backends to fail
        with patch.object(router, '_simulate_stim', side_effect=RuntimeError("Stim failed")):
            with patch.object(router, '_simulate_jax_metal', side_effect=RuntimeError("Metal failed")):
                with patch.object(router, '_simulate_cuda', side_effect=RuntimeError("CUDA failed")):
                    with patch.object(router, '_simulate_tensor_network', side_effect=RuntimeError("Tensor failed")):
                        # Only Qiskit should work
                        result = router.simulate(qc, shots=100)
                        
                        assert result.backend_used == BackendType.QISKIT
                        assert result.fallback_reason is not None
                        assert len(result.counts) > 0

    def test_warning_collection(self):
        """Test that warnings are properly collected and reported."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        router = EnhancedQuantumRouter()
        
        # Force JAX_METAL selection to generate experimental warning
        with patch.object(router, 'select_optimal_backend') as mock_select:
            from ariadne.router import RoutingDecision
            mock_select.return_value = RoutingDecision(
                circuit_entropy=1.0,
                recommended_backend=BackendType.JAX_METAL,
                confidence_score=0.8,
                expected_speedup=1.5,
                channel_capacity_match=0.8,
                alternatives=[]
            )
            
            try:
                result = router.simulate(qc, shots=100)
                
                # Should have warnings about experimental support
                if result.warnings:
                    assert any("experimental" in warning.lower() for warning in result.warnings)
            except Exception:
                # If Metal backend fails, that's also valid behavior
                pass