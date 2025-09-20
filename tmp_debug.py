import cupy as cp
from qiskit.circuit.library import XGate
from ariadne.backends.cuda_backend import CUDABackend

backend = CUDABackend(allow_cpu_fallback=True)
state = backend._xp.zeros(8, dtype=backend._xp.complex128)
state[0] = 1
matrix = backend._instruction_to_matrix(XGate(), 1)
backend._apply_single_qubit_gate(state, matrix, 2)
print(cp.asnumpy(state))