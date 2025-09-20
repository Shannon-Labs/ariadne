#!/usr/bin/env python3
"""
Ariadne Quickstart Example

This example demonstrates the basic usage of Ariadne's intelligent routing.
"""

from qiskit import QuantumCircuit
from ariadne import simulate, QuantumRouter


def main():
    print("=== Ariadne Quickstart ===\n")
    
    # Example 1: Simple Bell State
    print("1. Creating a Bell State")
    bell = QuantumCircuit(2, 2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()
    
    # Simulate with automatic routing
    result = simulate(bell, shots=1000)
    
    print(f"   Backend used: {result.backend_used}")
    print(f"   Execution time: {result.execution_time:.4f}s")
    print(f"   Measurement results: {result.counts}")
    print()
    
    # Example 2: Clifford Circuit (will use Stim)
    print("2. Large Clifford Circuit")
    clifford = QuantumCircuit(20)
    for i in range(20):
        clifford.h(i)
    for i in range(0, 19, 2):
        clifford.cx(i, i + 1)
    clifford.measure_all()
    
    result = simulate(clifford, shots=1000)
    print(f"   Backend used: {result.backend_used}")
    print(f"   Execution time: {result.execution_time:.4f}s")
    print(f"   Sample counts: {list(result.counts.items())[:3]}...")
    print()
    
    # Example 3: General Circuit with T gates
    print("3. General Circuit with T Gates")
    general = QuantumCircuit(3)
    general.h(0)
    general.t(1)  # T gate makes it non-Clifford
    general.cx(0, 1)
    general.cx(1, 2)
    general.measure_all()
    
    result = simulate(general, shots=1000)
    print(f"   Backend used: {result.backend_used}")
    print(f"   Execution time: {result.execution_time:.4f}s")
    print()
    
    # Example 4: Inspect routing decision
    print("4. Inspecting Routing Decisions")
    router = QuantumRouter()
    
    # Analyze the Clifford circuit
    decision = router.select_optimal_backend(clifford)
    print(f"   Clifford circuit analysis:")
    print(f"     - Recommended backend: {decision.recommended_backend}")
    print(f"     - Circuit entropy: {decision.circuit_entropy:.3f}")
    print(f"     - Expected speedup: {decision.expected_speedup:.1f}x")
    print(f"     - Confidence score: {decision.confidence_score:.2f}")
    print()
    
    # Analyze the general circuit
    decision = router.select_optimal_backend(general)
    print(f"   General circuit analysis:")
    print(f"     - Recommended backend: {decision.recommended_backend}")
    print(f"     - Circuit entropy: {decision.circuit_entropy:.3f}")
    print(f"     - Reason: Non-Clifford gates detected")
    print()
    
    # Example 5: Force specific backend
    print("5. Forcing Specific Backend")
    result_qiskit = simulate(bell, shots=100, backend='qiskit')
    print(f"   Forced Qiskit: {result_qiskit.backend_used}")
    
    # Try CUDA if available
    try:
        result_cuda = simulate(bell, shots=100, backend='cuda')
        print(f"   Forced CUDA: {result_cuda.backend_used}")
    except:
        print("   CUDA backend not available")
    
    print("\n=== Quickstart Complete ===")
    print("Ariadne automatically selected the optimal backend for each circuit!")


if __name__ == "__main__":
    main()