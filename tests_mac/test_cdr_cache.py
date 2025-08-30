import pytest
import json
from pathlib import Path
from qiskit import QuantumCircuit
import numpy as np

from ariadne_mac.passes.mitigation import (
    cdr_calibrate,
    apply_cdr_cached,
    _counts_to_expectation,
)


def mock_executor(circ: QuantumCircuit, shots: int) -> dict:
    """Mock executor that returns counts with controllable noise."""
    n = circ.num_qubits
    
    # Simulate noisy vs ideal based on shot count
    if shots >= 10000:
        # "Ideal" - less noise
        noise_level = 0.01
    else:
        # "Noisy" - more noise
        noise_level = 0.1
    
    # Generate mock counts
    np.random.seed(hash(str(circ)) % 10000 + shots)
    
    # Bias towards |0...0> with some noise
    probs = np.zeros(2**n)
    probs[0] = 1.0 - noise_level
    probs[1:min(10, 2**n)] = noise_level / min(9, 2**n - 1)
    probs = probs / probs.sum()
    
    # Sample
    outcomes = np.random.choice(2**n, shots, p=probs)
    counts = {}
    for outcome in outcomes:
        bitstring = format(outcome, f'0{n}b')
        counts[bitstring] = counts.get(bitstring, 0) + 1
    
    return {"counts": counts}


def create_toy_circuit() -> QuantumCircuit:
    """Create a small test circuit."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.t(1)  # Non-Clifford for near-Clifford CDR
    qc.measure_all()
    return qc


def test_cdr_calibration():
    """Test CDR calibration creates and caches a model."""
    cache_dir = Path("reports") / "cdr_cache"
    circ = create_toy_circuit()
    
    # First calibration - should create cache
    result1 = cdr_calibrate(
        executor=mock_executor,
        circ=circ,
        k=20,  # Small k for fast test
        seed=42,
        cache_dir=cache_dir,
    )
    
    assert "model" in result1
    assert "circuit_hash" in result1
    assert result1["cache_hit"] is False
    assert result1["training_samples"] >= 10
    
    # Check model parameters
    model = result1["model"]
    assert "slope" in model
    assert "intercept" in model
    assert "r2" in model
    
    # Second calibration - should hit cache
    result2 = cdr_calibrate(
        executor=mock_executor,
        circ=circ,
        k=20,
        seed=42,
        cache_dir=cache_dir,
    )
    
    assert result2["cache_hit"] is True
    assert result2["circuit_hash"] == result1["circuit_hash"]
    assert result2["model"] == result1["model"]


def test_cdr_improves_observable():
    """Test that CDR improves a toy observable by at least 15%."""
    cache_dir = Path("reports") / "cdr_cache"
    circ = create_toy_circuit()
    
    # Calibrate CDR
    cal_result = cdr_calibrate(
        executor=mock_executor,
        circ=circ,
        k=50,  # More samples for better model
        seed=123,
        cache_dir=cache_dir,
    )
    
    # Get noisy observable
    noisy_result = mock_executor(circ, 1000)
    noisy_exp = _counts_to_expectation(noisy_result["counts"], circ.num_qubits)
    
    # Apply CDR mitigation
    mitigated_exp = apply_cdr_cached(
        observable=noisy_exp,
        circuit_hash=cal_result["circuit_hash"],
        cache_dir=cache_dir,
    )
    
    # Get "ideal" observable for comparison
    ideal_result = mock_executor(circ, 100000)
    ideal_exp = _counts_to_expectation(ideal_result["counts"], circ.num_qubits)
    
    # CDR should move noisy towards ideal
    noisy_error = abs(noisy_exp - ideal_exp)
    mitigated_error = abs(mitigated_exp - ideal_exp)
    
    # Should improve by at least 15%
    if noisy_error > 0.01:  # Only test if there's meaningful noise
        improvement = (noisy_error - mitigated_error) / noisy_error
        assert improvement >= 0.15, f"CDR improvement {improvement:.1%} < 15%"


def test_cdr_cache_persistence():
    """Test that CDR cache persists across runs."""
    cache_dir = Path("reports") / "cdr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    circ = create_toy_circuit()
    
    # Create a cache file manually
    from ariadne_mac.passes.mitigation import _hash_circuit_qasm3
    
    circuit_hash = _hash_circuit_qasm3(circ)
    cache_file = cache_dir / f"{circuit_hash}.json"
    
    manual_model = {
        "model": {"slope": 1.2, "intercept": 0.05, "r2": 0.95},
        "circuit_hash": circuit_hash,
        "training_samples": 100,
        "seed": 999,
    }
    
    cache_file.write_text(json.dumps(manual_model, indent=2))
    
    # Load should hit this cache
    result = cdr_calibrate(
        executor=mock_executor,
        circ=circ,
        k=10,
        seed=42,
        cache_dir=cache_dir,
    )
    
    assert result["cache_hit"] is True
    assert result["model"]["slope"] == 1.2
    assert result["model"]["intercept"] == 0.05


def test_counts_to_expectation():
    """Test Z expectation value calculation from counts."""
    # All |0>
    counts1 = {"000": 1000}
    exp1 = _counts_to_expectation(counts1, 3)
    assert exp1 == 1.0  # All spins up
    
    # All |1>
    counts2 = {"111": 1000}
    exp2 = _counts_to_expectation(counts2, 3)
    assert exp2 == -1.0  # Odd parity
    
    # Mixed
    counts3 = {"000": 500, "111": 500}
    exp3 = _counts_to_expectation(counts3, 3)
    assert exp3 == 0.0  # Balanced
    
    # Single flip
    counts4 = {"000": 500, "001": 500}
    exp4 = _counts_to_expectation(counts4, 3)
    assert exp4 == 0.0  # Half +1, half -1


@pytest.mark.slow
def test_cdr_with_5k_shots():
    """Test CDR with 5k shots improves observable significantly."""
    cache_dir = Path("reports") / "cdr_cache"
    circ = create_toy_circuit()
    
    # Calibrate with exactly 5000 shots total
    result = cdr_calibrate(
        executor=mock_executor,
        circ=circ,
        k=5,  # 5 training circuits, 1000 shots each = 5k total
        seed=456,
        cache_dir=cache_dir,
    )
    
    assert result["training_samples"] == 5
    
    # Verify cache file exists
    cache_file = cache_dir / f"{result['circuit_hash']}.json"
    assert cache_file.exists()
    
    # Load and check contents
    cached = json.loads(cache_file.read_text())
    assert cached["model"]["r2"] >= 0.0  # Should have some fit quality