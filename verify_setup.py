#!/usr/bin/env python3
"""Verification script for Ariadne setup.

This script verifies that all components are working correctly.
"""

import sys
import time
from pathlib import Path

def test_imports():
    """Test that all imports work correctly."""
    print("🔍 Testing imports...")
    
    try:
        from ariadne import simulate, QuantumRouter
        from ariadne.route.analyze import analyze_circuit, is_clifford_circuit
        from ariadne.converters import convert_qiskit_to_stim
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_simulation():
    """Test basic quantum circuit simulation."""
    print("🔍 Testing basic simulation...")
    
    try:
        from qiskit import QuantumCircuit
        from ariadne import simulate
        
        # Create a simple Bell state circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # Simulate with Ariadne
        result = simulate(qc, shots=100)
        
        # Verify results
        assert result.backend_used is not None
        assert len(result.counts) > 0
        assert result.execution_time > 0
        assert result.routing_decision is not None
        
        print(f"✅ Simulation successful: {result.backend_used}")
        print(f"   Execution time: {result.execution_time:.4f}s")
        print(f"   Counts: {len(result.counts)} outcomes")
        return True
        
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        return False

def test_clifford_detection():
    """Test Clifford circuit detection and routing."""
    print("🔍 Testing Clifford detection...")
    
    try:
        from qiskit import QuantumCircuit
        from ariadne import QuantumRouter
        
        # Create Clifford circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.s(1)
        qc.cx(0, 1)
        qc.measure_all()
        
        router = QuantumRouter()
        decision = router.select_optimal_backend(qc)
        
        # Should route to Stim for Clifford circuits
        assert decision.recommended_backend.value == "stim"
        assert decision.confidence_score > 0
        
        print(f"✅ Clifford detection successful")
        print(f"   Recommended backend: {decision.recommended_backend}")
        print(f"   Confidence: {decision.confidence_score:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Clifford detection error: {e}")
        return False

def test_circuit_analysis():
    """Test circuit analysis functionality."""
    print("🔍 Testing circuit analysis...")
    
    try:
        from qiskit import QuantumCircuit
        from ariadne.route.analyze import analyze_circuit, is_clifford_circuit
        
        # Create test circuit
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.t(0)  # Non-Clifford gate
        qc.measure_all()
        
        # Test analysis
        analysis = analyze_circuit(qc)
        is_clifford = is_clifford_circuit(qc)
        
        # Verify analysis results
        assert analysis['num_qubits'] == 3
        assert analysis['depth'] > 0
        assert analysis['is_clifford'] == is_clifford
        assert 0 <= analysis['clifford_ratio'] <= 1
        
        print(f"✅ Circuit analysis successful")
        print(f"   Qubits: {analysis['num_qubits']}")
        print(f"   Depth: {analysis['depth']}")
        print(f"   Is Clifford: {analysis['is_clifford']}")
        print(f"   Clifford ratio: {analysis['clifford_ratio']:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Circuit analysis error: {e}")
        return False

def test_stim_conversion():
    """Test Qiskit to Stim conversion."""
    print("🔍 Testing Stim conversion...")
    
    try:
        from qiskit import QuantumCircuit
        from ariadne.converters import convert_qiskit_to_stim, simulate_stim_circuit
        
        # Create Clifford circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # Convert to Stim
        stim_circuit, measurement_map = convert_qiskit_to_stim(qc)
        
        # Simulate with Stim
        counts = simulate_stim_circuit(stim_circuit, measurement_map, 100, 2)
        
        # Verify results
        assert len(counts) > 0
        assert sum(counts.values()) == 100
        
        print(f"✅ Stim conversion successful")
        print(f"   Stim circuit length: {len(stim_circuit)}")
        print(f"   Measurement outcomes: {len(counts)}")
        return True
        
    except Exception as e:
        print(f"❌ Stim conversion error: {e}")
        return False

def test_examples():
    """Test example scripts."""
    print("🔍 Testing examples...")
    
    try:
        import subprocess
        import sys
        
        # Test clifford circuit example
        result = subprocess.run([
            sys.executable, "examples/clifford_circuit.py"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"❌ Clifford example failed: {result.stderr}")
            return False
        
        # Test bell state example
        result = subprocess.run([
            sys.executable, "examples/bell_state_demo.py"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"❌ Bell state example failed: {result.stderr}")
            return False
        
        print("✅ Examples run successfully")
        return True
        
    except Exception as e:
        print(f"❌ Examples error: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🚀 Ariadne Setup Verification")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_basic_simulation,
        test_clifford_detection,
        test_circuit_analysis,
        test_stim_conversion,
        test_examples,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ariadne is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
