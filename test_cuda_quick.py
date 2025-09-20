#!/usr/bin/env python3
"""Quick CUDA backend test for Ariadne."""

import sys
from qiskit import QuantumCircuit

try:
    from ariadne import simulate
    from ariadne.backends.cuda_backend import is_cuda_available, get_cuda_info
    
    print("Ariadne CUDA Backend Test")
    print("=" * 40)
    
    # Check CUDA availability
    cuda_available = is_cuda_available()
    print(f"CUDA Available: {cuda_available}")
    
    # Get CUDA info
    cuda_info = get_cuda_info()
    print(f"CUDA Info: {cuda_info}")
    
    # Create a simple quantum circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    # Simulate with automatic backend selection
    print("\nRunning simulation...")
    result = simulate(qc, shots=1024)
    
    print(f"Backend used: {result.backend_used}")
    print(f"Execution time: {result.execution_time:.4f}s")
    print(f"Counts: {result.counts}")
    
    # Direct CUDA backend test if available
    if cuda_available:
        from ariadne.backends.cuda_backend import CUDABackend
        print("\nDirect CUDA backend test:")
        backend = CUDABackend()
        counts = backend.simulate(qc, shots=1024)
        print(f"CUDA counts: {counts}")
        print(f"Backend mode: {backend.backend_mode}")
    
    print("\n[OK] All tests passed!")
    sys.exit(0)
    
except Exception as e:
    print(f"\n[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)