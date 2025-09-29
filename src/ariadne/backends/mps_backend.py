"""A backend for simulating quantum circuits using Matrix Product States."""

from typing import Any

import numpy as np
import quimb.tensor as tn
from qiskit import QuantumCircuit

from .universal_interface import BackendCapability, BackendMetrics, UniversalBackend


class MPSBackend(UniversalBackend):
    """
    A quantum backend that uses a Matrix Product State (MPS) representation
    to simulate quantum circuits. This backend is particularly effective for
    circuits with low entanglement.
    """

    def __init__(self, max_bond_dimension: int = 64):
        """
        Initializes the MPS backend.

        Args:
            max_bond_dimension: The maximum bond dimension to use for the MPS.
        """
        self.max_bond_dimension = max_bond_dimension

    def simulate(self, circuit: QuantumCircuit, shots: int = 1000, **kwargs) -> dict[str, int]:
        """
        Simulates the given quantum circuit using an MPS representation.

        The Matrix Product State (MPS) is a tensor network representation that
        efficiently captures quantum states with limited entanglement. It maps
        the global quantum state of N qubits to a chain of N rank-3 tensors,
        where the central index (bond dimension) controls the amount of
        entanglement that can be represented.

        Args:
            circuit: The quantum circuit to simulate.

        Returns:
            Dictionary of measurement counts.
        """
        n_qubits = circuit.num_qubits
        max_bond = self.max_bond_dimension

        # print(f"Simulating circuit with MPS backend (max_bond_dimension={max_bond})...")

        # 1. Initialize the MPS in the |0...0> state
        # The physical dimension is 2 (qubit), and the bond dimension is 1 (unentangled)
        # Initialize the MPS from the dense |0...0> state vector
        zero_state = np.zeros(2**n_qubits, dtype=complex)
        zero_state[0] = 1.0
        mps = tn.MatrixProductState.from_dense(
            zero_state,
        )
        
        # 2. Apply gates from the Qiskit circuit
        for instruction, qargs, _cargs in circuit.data:
            gate_name = instruction.name
            # Qiskit's find_bit is used to map Qubit objects to their integer index
            qubits = [circuit.find_bit(q).index for q in qargs]
            
            # Get the gate matrix (unitary) from Qiskit
            try:
                gate_matrix = instruction.to_matrix()
            except Exception:
                # Handle gates that don't have a matrix representation (e.g., measurements, barriers)
                if gate_name in ['measure', 'barrier', 'reset']:
                    continue
                raise NotImplementedError(f"Gate '{gate_name}' not supported by MPS backend.")

            # Reshape the matrix for quimb (2^k x 2^k -> 2, 2, ..., 2, 2)
            k = len(qubits)
            gate_tensor = gate_matrix.reshape([2] * 2 * k)
            
            # Apply the gate to the MPS, truncating bond dimension if necessary
            mps.gate(
                gate_tensor,
                qubits,
                max_bond=max_bond,
                tags=gate_name
            )

        # 3. Sample counts directly from the MPS
        # This avoids the exponential O(2^N) contraction step, restoring polynomial scaling.
        samples = mps.sample(n_samples=shots)
        
        # Convert samples (list of integers) to Qiskit-style counts dictionary (bitstrings)
        counts = {}
        # n_qubits is defined on line 42
        
        for state_int in samples:
            # Convert integer state to bitstring (MSB first, e.g., 0101)
            bitstring = f'{state_int:0{n_qubits}b}'
            counts[bitstring] = counts.get(bitstring, 0) + 1
            
        return counts


    # Implement required abstract methods from UniversalBackend
    def get_backend_info(self) -> dict[str, Any]:
        return {"name": "mps", "max_bond_dimension": self.max_bond_dimension}

    def get_capabilities(self) -> list[BackendCapability]:
        return [BackendCapability.STATE_VECTOR_SIMULATION]

    def get_metrics(self) -> BackendMetrics:
        return BackendMetrics(
            max_qubits=50,
            typical_qubits=30,
            memory_efficiency=0.9,
            speed_rating=0.9,
            accuracy_rating=0.9,
            stability_rating=0.9,
            capabilities=self.get_capabilities(),
            hardware_requirements=['CPU'],
            estimated_cost_factor=0.1
        )

    def can_simulate(self, circuit: QuantumCircuit, **kwargs) -> tuple[bool, str]:
        if circuit.num_qubits > 50:
            return False, "Too many qubits for MPS backend"
        return True, "Can simulate"
