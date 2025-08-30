import pytest
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator

from ariadne_mac.passes.defer_measure import (
    defer_measurements,
    rewrite_measure_controls,
    validate_deferred_circuit,
)


def compute_tvd_from_counts(counts1: dict, counts2: dict) -> float:
    """Compute TVD between two count distributions."""
    all_keys = set(counts1.keys()) | set(counts2.keys())
    total1 = sum(counts1.values()) or 1
    total2 = sum(counts2.values()) or 1
    
    tvd = 0.0
    for key in all_keys:
        p1 = counts1.get(key, 0) / total1
        p2 = counts2.get(key, 0) / total2
        tvd += abs(p1 - p2)
    
    return tvd / 2.0


def create_measure_control_circuit() -> QuantumCircuit:
    """Create a randomized Clifford circuit with measure-controls."""
    np.random.seed(42)
    n = 4
    qc = QuantumCircuit(n, n)
    
    # Random Clifford gates
    clifford_1q = ['h', 's', 'sdg', 'x', 'y', 'z']
    
    # Initial random Cliffords
    for i in range(n):
        gate = np.random.choice(clifford_1q)
        getattr(qc, gate)(i)
    
    # Mid-circuit measurement
    qc.measure(0, 0)
    
    # Classically controlled Clifford gates
    qc.x(1).c_if(qc.clbits[0], 1)
    qc.z(2).c_if(qc.clbits[0], 1)
    
    # More Clifford gates
    qc.cx(1, 2)
    qc.h(3)
    qc.cx(2, 3)
    
    # Another measurement and control
    qc.measure(1, 1)
    qc.h(3).c_if(qc.clbits[1], 1)
    
    # Final measurements
    qc.measure(2, 2)
    qc.measure(3, 3)
    
    return qc


def test_defer_measurements_basic():
    """Test basic deferred measurement rewriting."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.x(1).c_if(qc.clbits[0], 1)
    qc.measure(1, 1)
    
    deferred = defer_measurements(qc)
    
    # Check structure
    assert deferred.num_qubits == 2
    assert deferred.num_clbits == 2
    
    # Measurements should be at the end
    measure_positions = []
    for i, (inst, _, _) in enumerate(deferred.data):
        if inst.name == "measure":
            measure_positions.append(i)
    
    # All measurements should be at the end
    assert all(pos >= len(deferred.data) - 2 for pos in measure_positions)
    
    # Should have a CX instead of classically-controlled X
    has_cx = any(inst.name == "cx" for inst, _, _ in deferred.data)
    assert has_cx


def test_defer_with_non_clifford_fails():
    """Test that deferring fails with non-Clifford classically controlled gates."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.t(1).c_if(qc.clbits[0], 1)  # T gate is non-Clifford
    
    with pytest.raises(ValueError, match="non-Clifford"):
        defer_measurements(qc)


def test_rewrite_measure_controls_safe():
    """Test that rewrite_measure_controls safely handles various circuits."""
    # Circuit with non-Clifford control - should return original
    qc1 = QuantumCircuit(2, 2)
    qc1.measure(0, 0)
    qc1.t(1).c_if(qc1.clbits[0], 1)
    
    result1 = rewrite_measure_controls(qc1)
    assert result1 == qc1  # Should return original
    
    # Circuit with Clifford controls - should defer
    qc2 = QuantumCircuit(2, 2)
    qc2.h(0)
    qc2.measure(0, 0)
    qc2.x(1).c_if(qc2.clbits[0], 1)
    qc2.measure(1, 1)
    
    result2 = rewrite_measure_controls(qc2)
    assert result2 != qc2  # Should be modified
    
    # Circuit with no mid-circuit measurements - should return original
    qc3 = QuantumCircuit(2, 2)
    qc3.h(0)
    qc3.cx(0, 1)
    qc3.measure_all()
    
    result3 = rewrite_measure_controls(qc3)
    # May or may not be same object, but should be equivalent
    assert result3.num_qubits == qc3.num_qubits


@pytest.mark.slow
def test_tvd_with_aer():
    """Test TVD between original and deferred circuits using Aer."""
    # Skip if Aer not available
    try:
        sim = AerSimulator()
    except Exception:
        pytest.skip("Aer not available")
    
    # Create circuit with measure-controls
    original = create_measure_control_circuit()
    
    # Defer measurements
    deferred = rewrite_measure_controls(original)
    
    # Run both with Aer
    shots = 10000
    
    job1 = sim.run(original, shots=shots, seed_simulator=123)
    counts1 = job1.result().get_counts()
    
    job2 = sim.run(deferred, shots=shots, seed_simulator=123)
    counts2 = job2.result().get_counts()
    
    # Compute TVD
    tvd = compute_tvd_from_counts(counts1, counts2)
    
    # Should be very close (< 0.05) for Clifford circuits
    # Note: may not be exactly 0 due to shot noise
    assert tvd <= 0.05, f"TVD {tvd:.3f} > 0.05"


def test_validate_deferred_circuit():
    """Test validation of deferred circuits."""
    # Create a pure Clifford circuit
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.s(1)
    qc.cx(1, 2)
    qc.h(2)
    
    # This should be its own valid deferred version
    assert validate_deferred_circuit(qc, qc)
    
    # Create a modified version
    qc2 = qc.copy()
    qc2.x(0)  # Add an X gate
    
    # These should not validate as equivalent
    try:
        from qiskit.quantum_info import Clifford
        # If Clifford is available, test properly
        assert not validate_deferred_circuit(qc, qc2)
    except ImportError:
        # If not available, validation returns True by default
        assert validate_deferred_circuit(qc, qc2)


def test_multiple_measurements_deferred():
    """Test deferring multiple mid-circuit measurements."""
    qc = QuantumCircuit(4, 4)
    
    # Multiple interleaved measurements and controls
    qc.h(0)
    qc.measure(0, 0)
    qc.x(1).c_if(qc.clbits[0], 1)
    
    qc.h(2)
    qc.measure(2, 2)
    qc.z(3).c_if(qc.clbits[2], 1)
    
    qc.measure(1, 1)
    qc.measure(3, 3)
    
    deferred = defer_measurements(qc)
    
    # Should have all 4 measurements at the end
    measure_count = sum(1 for inst, _, _ in deferred.data if inst.name == "measure")
    assert measure_count == 4
    
    # Check last 4 operations are measurements
    last_4_ops = [inst.name for inst, _, _ in deferred.data[-4:]]
    assert all(op == "measure" for op in last_4_ops)


def test_control_on_zero_value():
    """Test handling of classical control on |0> value."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.x(1).c_if(qc.clbits[0], 0)  # Control on |0> instead of |1>
    qc.measure(1, 1)
    
    # Should handle this case (even if not optimally)
    deferred = defer_measurements(qc)
    assert deferred.num_qubits == 2
    assert deferred.num_clbits == 2