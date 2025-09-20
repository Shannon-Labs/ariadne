from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction

try:
    import cupy as cp  # type: ignore[import]

    CUDA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when CuPy is missing
    cp = None  # type: ignore[assignment]
    CUDA_AVAILABLE = False


def is_cuda_available() -> bool:
    """Return ``True`` when CuPy and a CUDA runtime are available."""

    if not CUDA_AVAILABLE:
        return False

    try:  # pragma: no cover - requires CUDA runtime
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def get_cuda_info() -> Dict[str, object]:
    """Return a lightweight description of the detected CUDA devices."""

    if not CUDA_AVAILABLE:
        return {"available": False, "device_count": 0}

    try:  # pragma: no cover - requires CUDA runtime
        device_count = cp.cuda.runtime.getDeviceCount()
        devices: List[Dict[str, object]] = []

        for device_id in range(device_count):
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            devices.append(
                {
                    "device_id": device_id,
                    "name": props.get("name", b"?").decode("utf-8", errors="ignore"),
                    "total_memory": int(props.get("totalGlobalMem", 0)),
                    "multiprocessors": int(props.get("multiProcessorCount", 0)),
                    "compute_capability": f"{props.get('major', 0)}.{props.get('minor', 0)}",
                }
            )

        return {
            "available": device_count > 0,
            "device_count": device_count,
            "devices": devices,
        }
    except Exception as exc:  # pragma: no cover - requires CUDA runtime
        return {"available": False, "error": str(exc)}


@dataclass
class SimulationSummary:
    """Metadata describing the most recent simulation run."""

    shots: int
    measured_qubits: Sequence[int]
    execution_time: float
    backend_mode: str


class CUDABackend:
    """Statevector simulator that optionally executes on the GPU."""

    def __init__(
        self,
        *,
        device_id: int = 0,
        prefer_gpu: bool = True,
        allow_cpu_fallback: bool = True,
    ) -> None:
        self._last_summary: Optional[SimulationSummary] = None
        self._xp: Any = np
        self._mode = "cpu"

        if prefer_gpu and CUDA_AVAILABLE:
            try:  # pragma: no cover - requires CUDA runtime
                cp.cuda.Device(device_id).use()
                self._xp = cp  # type: ignore[assignment]
                self._mode = "cuda"
            except Exception as exc:
                if not allow_cpu_fallback:
                    raise RuntimeError(f"Unable to select CUDA device {device_id}: {exc}")
        elif not allow_cpu_fallback:
            raise RuntimeError(
                "CUDA runtime not available and CPU fallback disabled. "
                "Install CuPy with CUDA support or enable CPU fallback."
            )

    @property
    def backend_mode(self) -> str:
        return self._mode

    @property
    def last_summary(self) -> Optional[SimulationSummary]:
        return self._last_summary

    def simulate(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, int]:
        if shots <= 0:
            raise ValueError("shots must be a positive integer")

        state, measured_qubits, execution_time = self._simulate_statevector(circuit)
        counts = self._sample_measurements(state, measured_qubits, shots)

        self._last_summary = SimulationSummary(
            shots=shots,
            measured_qubits=measured_qubits,
            execution_time=execution_time,
            backend_mode=self._mode,
        )

        return counts

    def simulate_statevector(self, circuit: QuantumCircuit) -> Tuple[np.ndarray, Sequence[int]]:
        state, measured_qubits, _ = self._simulate_statevector(circuit)
        if self._xp is np:
            return state, measured_qubits

        return cp.asnumpy(state), measured_qubits  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Internal helpers

    def _simulate_statevector(
        self, circuit: QuantumCircuit
    ) -> Tuple[Any, Sequence[int], float]:
        xp = self._xp
        num_qubits = circuit.num_qubits
        state = xp.zeros(2**num_qubits, dtype=xp.complex128)
        state[0] = 1.0

        operations, measured_qubits = self._prepare_operations(circuit)

        start = time.perf_counter()
        for instruction, targets in operations:
            gate_matrix = self._instruction_to_matrix(instruction, len(targets))
            self._apply_gate(state, gate_matrix, targets)
        execution_time = time.perf_counter() - start

        return state, measured_qubits, execution_time

    def _prepare_operations(
        self, circuit: QuantumCircuit
    ) -> Tuple[List[Tuple[Instruction, List[int]]], Sequence[int]]:
        operations: List[Tuple[Instruction, List[int]]] = []
        measurement_map: List[Tuple[int, int]] = []

        for item in circuit.data:
            if hasattr(item, "operation"):
                operation = item.operation
                qubits = list(item.qubits)
                clbits = list(item.clbits)
            else:  # Legacy tuple form
                operation, qubits, clbits = item  # type: ignore[misc]

            name = operation.name
            qubit_indices = [circuit.find_bit(qubit).index for qubit in qubits]
            clbit_indices = [circuit.find_bit(clbit).index for clbit in clbits]

            if name in {"barrier", "delay"}:
                continue
            if name == "measure":
                if not qubit_indices:
                    continue
                classical_index = clbit_indices[0] if clbit_indices else len(measurement_map)
                measurement_map.append((classical_index, qubit_indices[0]))
                continue

            operations.append((operation, qubit_indices))

        if not measurement_map:
            measured_qubits: Sequence[int] = list(range(circuit.num_qubits))
        else:
            measurement_map.sort(key=lambda item: item[0])
            measured_qubits = [qubit for _, qubit in measurement_map]

        return operations, measured_qubits

    def _instruction_to_matrix(self, instruction: Instruction, arity: int) -> Any:
        xp = self._xp

        if hasattr(instruction, "to_matrix"):
            matrix = instruction.to_matrix()  # type: ignore[no-untyped-call]
        else:
            matrix = np.eye(2**arity, dtype=np.complex128)

        return xp.asarray(matrix, dtype=xp.complex128)

    def _apply_gate(self, state: Any, matrix: Any, qubits: Sequence[int]) -> None:
        if not qubits:
            return

        if len(qubits) == 1:
            self._apply_single_qubit_gate(state, matrix, qubits[0])
        elif len(qubits) == 2:
            self._apply_two_qubit_gate(state, matrix, qubits[0], qubits[1])
        else:
            self._apply_dense_gate(state, matrix, qubits)

    def _apply_single_qubit_gate(self, state: Any, matrix: Any, qubit: int) -> None:
        stride = 1 << qubit
        period = stride << 1
        for start in range(0, state.shape[0], period):
            for offset in range(stride):
                i0 = start + offset
                i1 = i0 + stride
                amp0 = state[i0]
                amp1 = state[i1]
                state[i0] = matrix[0, 0] * amp0 + matrix[0, 1] * amp1
                state[i1] = matrix[1, 0] * amp0 + matrix[1, 1] * amp1

    def _apply_two_qubit_gate(self, state: Any, matrix: Any, q0: int, q1: int) -> None:
        xp = self._xp
        if q0 == q1:
            raise ValueError("Two-qubit gate requires two distinct qubits")

        if q0 > q1:
            matrix = self._swap_two_qubit_matrix(matrix)
            q0, q1 = q1, q0

        low_bit = 1 << q0
        high_bit = 1 << q1
        period = high_bit << 1
        mid_stride = low_bit << 1

        for base in range(0, state.shape[0], period):
            for offset in range(0, high_bit, mid_stride):
                for inner in range(low_bit):
                    i00 = base + offset + inner
                    i01 = i00 + low_bit
                    i10 = i00 + high_bit
                    i11 = i10 + low_bit
                    vec = xp.array([state[i00], state[i01], state[i10], state[i11]])
                    transformed = matrix @ vec
                    state[i00], state[i01], state[i10], state[i11] = transformed

    def _swap_two_qubit_matrix(self, matrix: Any) -> Any:
        xp = self._xp
        perm = xp.array([0, 2, 1, 3])
        swapped = matrix.take(perm, axis=0).take(perm, axis=1)
        return swapped

    def _apply_dense_gate(self, state: Any, matrix: Any, qubits: Sequence[int]) -> None:
        xp = self._xp
        num_qubits = int(math.log2(state.shape[0]))
        gate_qubits = list(qubits)
        remaining = [index for index in range(num_qubits) if index not in gate_qubits]
        permutation = gate_qubits + remaining
        inverse_perm = _inverse_permutation(permutation)

        tensor = state.reshape([2] * num_qubits)
        tensor = xp.transpose(tensor, permutation)
        tensor = tensor.reshape(2 ** len(gate_qubits), -1)

        matrix = matrix.reshape(2 ** len(gate_qubits), 2 ** len(gate_qubits))
        updated = matrix @ tensor

        updated = updated.reshape([2] * num_qubits)
        updated = xp.transpose(updated, inverse_perm)
        state[:] = updated.reshape(state.shape[0])
    def _sample_measurements(
        self, state: Any, measured_qubits: Sequence[int], shots: int
    ) -> Dict[str, int]:
        xp = self._xp

        if xp is np:
            probabilities = np.abs(state) ** 2
        else:  # pragma: no cover - requires CuPy
            probabilities = xp.abs(state) ** 2
            probabilities = cp.asnumpy(probabilities)  # type: ignore[arg-type]

        probabilities = probabilities / probabilities.sum()
        outcomes = np.random.choice(len(probabilities), size=shots, p=probabilities)

        counts: Dict[str, int] = {}
        measured = list(measured_qubits) or list(range(int(math.log2(len(probabilities)))))

        for outcome in outcomes:
            bit_string = _format_bits(outcome, measured)
            counts[bit_string] = counts.get(bit_string, 0) + 1

        return counts


def _inverse_permutation(values: Sequence[int]) -> List[int]:
    inverse = [0] * len(values)
    for index, value in enumerate(values):
        inverse[value] = index
    return inverse


def _format_bits(state_index: int, qubits: Sequence[int]) -> str:
    bits = ["1" if (state_index >> qubit) & 1 else "0" for qubit in qubits]
    return "".join(reversed(bits))


def simulate_cuda(
    circuit: QuantumCircuit,
    *,
    shots: int = 1024,
    allow_cpu_fallback: bool = True,
) -> Dict[str, int]:
    backend = CUDABackend(allow_cpu_fallback=allow_cpu_fallback)
    return backend.simulate(circuit, shots=shots)