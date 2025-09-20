from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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


def get_cuda_info() -> dict[str, object]:
    """Return a lightweight description of the detected CUDA devices."""

    if not CUDA_AVAILABLE:
        return {"available": False, "device_count": 0}

    try:  # pragma: no cover - requires CUDA runtime
        device_count = cp.cuda.runtime.getDeviceCount()
        devices: list[dict[str, object]] = []

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
        self._last_summary: SimulationSummary | None = None
        self._xp: Any = np
        self._mode = "cpu"

        if prefer_gpu and CUDA_AVAILABLE:
            try:  # pragma: no cover - requires CUDA runtime
                cp.cuda.Device(device_id).use()
                self._xp = cp  # type: ignore[assignment]
                self._mode = "cuda"
            except Exception as exc:
                if not allow_cpu_fallback:
                    raise RuntimeError(f"Unable to select CUDA device {device_id}: {exc}") from exc
        elif not allow_cpu_fallback:
            raise RuntimeError(
                "CUDA runtime not available and CPU fallback disabled. "
                "Install CuPy with CUDA support or enable CPU fallback."
            )

    @property
    def backend_mode(self) -> str:
        return self._mode

    @property
    def last_summary(self) -> SimulationSummary | None:
        return self._last_summary

    def simulate(self, circuit: QuantumCircuit, shots: int = 1024) -> dict[str, int]:
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

    def simulate_statevector(self, circuit: QuantumCircuit) -> tuple[np.ndarray, Sequence[int]]:
        state, measured_qubits, _ = self._simulate_statevector(circuit)
        if self._xp is np:
            return state, measured_qubits

        return cp.asnumpy(state), measured_qubits  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Internal helpers

    def _simulate_statevector(
        self, circuit: QuantumCircuit
    ) -> tuple[Any, Sequence[int], float]:
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
    ) -> tuple[list[tuple[Instruction, list[int]]], Sequence[int]]:
        operations: list[tuple[Instruction, list[int]]] = []
        measurement_map: list[tuple[int, int]] = []

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

        xp = self._xp
        num_qubits = int(math.log2(state.shape[0]))
        k = len(qubits)

        axes = list(qubits)
        tensor = xp.reshape(state, [2] * num_qubits, order="F")
        tensor = xp.moveaxis(tensor, axes, range(k))
        tensor = tensor.reshape(2**k, -1, order="F")

        matrix = matrix.reshape(2**k, 2**k)
        updated = matrix @ tensor

        updated = updated.reshape([2] * k + [-1], order="F")
        updated = xp.moveaxis(updated.reshape([2] * num_qubits, order="F"), range(k), axes)
        state[:] = xp.reshape(updated, state.shape[0], order="F")

    def _sample_measurements(
        self, state: Any, measured_qubits: Sequence[int], shots: int
    ) -> dict[str, int]:
        xp = self._xp

        if xp is np:
            probabilities = np.abs(state) ** 2
        else:  # pragma: no cover - requires CuPy
            probabilities = xp.abs(state) ** 2
            probabilities = cp.asnumpy(probabilities)  # type: ignore[arg-type]

        total = probabilities.sum()
        if not np.isfinite(total) or total == 0:
            raise RuntimeError("Statevector is not normalised")
        probabilities = probabilities / total

        outcomes = np.random.choice(len(probabilities), size=shots, p=probabilities)

        counts: dict[str, int] = {}
        measured = list(measured_qubits) or list(range(int(math.log2(len(probabilities)))))

        for outcome in outcomes:
            bit_string = _format_bits(outcome, measured)
            counts[bit_string] = counts.get(bit_string, 0) + 1

        return counts


def _format_bits(state_index: int, qubits: Sequence[int]) -> str:
    bits = ["1" if (state_index >> qubit) & 1 else "0" for qubit in qubits]
    return "".join(reversed(bits))


def simulate_cuda(
    circuit: QuantumCircuit,
    *,
    shots: int = 1024,
    allow_cpu_fallback: bool = True,
) -> dict[str, int]:
    backend = CUDABackend(allow_cpu_fallback=allow_cpu_fallback)
    return backend.simulate(circuit, shots=shots)