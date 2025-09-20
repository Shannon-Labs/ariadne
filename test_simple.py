#!/usr/bin/env python3
"""Simple test to verify the segmented execution works."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qiskit import QuantumCircuit
import numpy as np
import time

# Import what we need
from ariadne_mac.route.execute import execute, execute_segmented, _apply_boundary_adapter
from ariadne_mac.route.analyze import analyze_circuit


def test_basic_execution():
    """Test that basic execution works."""
    print("Testing basic execution...")
    
    # Create a simple circuit
    qc = QuantumCircuit(4, 4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.measure_all()
    
    # Try to execute
    try:
        result = execute(qc, shots=100, mem_cap_bytes=4*(2**30))
        print(f"✓ Basic execution works - backend: {result['trace']['backend']}")
        return True
    except Exception as e:
        print(f"✗ Basic execution failed: {e}")
        return False


def test_segmented():
    """Test segmented execution."""
    print("\nTesting segmented execution...")
    
    # Create a hybrid circuit
    qc = QuantumCircuit(6, 6)
    
    # Clifford section
    qc.h(0)
    for i in range(5):
        qc.cx(i, i+1)
    
    # Non-Clifford section
    qc.t(2)
    qc.rx(np.pi/4, 3)
    
    # Back to Clifford
    for i in range(6):
        qc.s(i)
    
    qc.measure_all()
    
    try:
        result = execute_segmented(qc, mem_cap_bytes=4*(2**30), samples=100)
        print(f"✓ Segmented execution works - {len(result['segments'])} segments")
        for seg in result['segments']:
            print(f"  - Segment {seg['segment_id']}: {seg['segment_backend']}")
        return True
    except Exception as e:
        print(f"✗ Segmented execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_boundary_adapter():
    """Test boundary adapter."""
    print("\nTesting boundary adapter...")
    
    try:
        # Test simple adapter
        adapter_result = _apply_boundary_adapter(
            from_backend="stim",
            to_backend="sv",
            state_data={"counts": {"00": 50, "11": 50}},
            n_qubits=2,
            samples=10,
            seed=42,
        )
        print(f"✓ Boundary adapter works - type: {adapter_result['adapter']}")
        return True
    except Exception as e:
        print(f"✗ Boundary adapter failed: {e}")
        return False


def test_performance():
    """Test performance comparison."""
    print("\nTesting performance comparison...")
    
    # Create a larger circuit
    n = 10
    qc = QuantumCircuit(n, n)
    
    # Large Clifford block
    qc.h(0)
    for i in range(n-1):
        qc.cx(i, i+1)
    for i in range(n):
        qc.s(i)
    
    # Small non-Clifford
    qc.t(n//2)
    
    # More Clifford
    for i in range(n):
        qc.h(i)
    
    qc.measure_all()
    
    try:
        # Time single-engine
        t0 = time.perf_counter()
        single = execute(qc, shots=100, mem_cap_bytes=4*(2**30))
        t1 = time.perf_counter()
        single_time = t1 - t0
        
        # Time segmented
        t0 = time.perf_counter()
        segmented = execute_segmented(qc, mem_cap_bytes=4*(2**30), samples=100)
        t1 = time.perf_counter()
        seg_time = t1 - t0
        
        speedup = single_time / seg_time if seg_time > 0 else 1.0
        
        print(f"✓ Performance test complete")
        print(f"  Single-engine: {single_time:.3f}s ({single['trace']['backend']})")
        print(f"  Segmented: {seg_time:.3f}s ({len(segmented['segments'])} segments)")
        print(f"  Speedup: {speedup:.2f}x")
        
        return True
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Ariadne-mac Segmented Execution Test")
    print("=" * 60)
    
    results = []
    results.append(test_basic_execution())
    results.append(test_segmented())
    results.append(test_boundary_adapter())
    results.append(test_performance())
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
        sys.exit(1)