import cupy as cp
state = cp.zeros(8, dtype=cp.complex128)
state[0] = 1
print(cp.asnumpy(state))