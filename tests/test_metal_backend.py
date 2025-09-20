"""Tests for the optional Metal backend."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from ariadne.backends.metal_backend import (
    JAX_AVAILABLE,
    MetalBackend,
    get_metal_info,
    is_metal_available,
)


def _bell_state_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    return circuit


def test_get_metal_info_structure() -> None:
    info = get_metal_info()
    assert "available" in info
    assert "device_count" in info

    if info["available"]:
        assert "devices" in info
        assert "is_apple_silicon" in info


def test_backend_without_fallback_requires_metal() -> None:
    if not is_metal_available():
        with pytest.raises(RuntimeError, match="JAX with Metal support not available"):
            MetalBackend(allow_cpu_fallback=False)


def test_backend_prefers_metal_when_available() -> None:
    backend = MetalBackend(allow_cpu_fallback=True)
    # Should not raise an error even if Metal is not available
    assert backend.backend_mode in ["metal", "cpu"]


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_metal_backend_basic_gates() -> None:
    """Test basic quantum gates with Metal backend."""
    backend = MetalBackend(allow_cpu_fallback=True)
    
    # Test H gate
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    
    counts = backend.simulate(circuit, shots=1000)
    assert len(counts) == 2  # Should have 0 and 1
    assert "0" in counts
    assert "1" in counts
    # Should be roughly 50/50
    assert 400 <= counts["0"] <= 600
    assert 400 <= counts["1"] <= 600


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_metal_backend_entanglement() -> None:
    """Test entangled state with Metal backend."""
    backend = MetalBackend(allow_cpu_fallback=True)
    circuit = _bell_state_circuit()
    
    counts = backend.simulate(circuit, shots=1000)
    assert len(counts) == 2  # Should have 00 and 11
    assert "00" in counts
    assert "11" in counts
    # Bell state should be 50/50 between 00 and 11
    assert 400 <= counts["00"] <= 600
    assert 400 <= counts["11"] <= 600


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_metal_backend_gpu_mode() -> None:
    """Test that Metal backend can run in GPU mode when available."""
    backend = MetalBackend(prefer_gpu=True, allow_cpu_fallback=True)
    
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    
    counts = backend.simulate(circuit, shots=100)
    assert len(counts) > 0
    assert backend.last_summary is not None
    assert backend.last_summary.shots == 100


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_metal_backend_statevector() -> None:
    """Test statevector simulation with Metal backend."""
    backend = MetalBackend(allow_cpu_fallback=True)
    
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    
    state, measured = backend.simulate_statevector(circuit)
    assert state.shape == (4,)  # 2 qubits = 4 states
    assert len(measured) == 2
    # Bell state: |00> + |11> (normalized)
    assert abs(state[0]) > 0.7  # |00> component
    assert abs(state[3]) > 0.7  # |11> component
    assert abs(state[1]) < 0.1  # |01> component should be small
    assert abs(state[2]) < 0.1  # |10> component should be small


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_metal_backend_large_circuit() -> None:
    """Test Metal backend with larger circuit."""
    backend = MetalBackend(allow_cpu_fallback=True)
    
    # Create a 3-qubit circuit
    circuit = QuantumCircuit(3, 3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.measure_all()
    
    counts = backend.simulate(circuit, shots=1000)
    assert len(counts) == 2  # Should have 000 and 111
    assert "000" in counts
    assert "111" in counts
    # GHZ state should be 50/50 between 000 and 111
    assert 400 <= counts["000"] <= 600
    assert 400 <= counts["111"] <= 600


def test_metal_backend_invalid_shots() -> None:
    """Test Metal backend with invalid shots parameter."""
    backend = MetalBackend(allow_cpu_fallback=True)
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    
    with pytest.raises(ValueError, match="shots must be a positive integer"):
        backend.simulate(circuit, shots=0)
    
    with pytest.raises(ValueError, match="shots must be a positive integer"):
        backend.simulate(circuit, shots=-1)
