import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Assuming MPSBackend is importable from the source path
from ariadne.backends.mps_backend import MPSBackend

class TestMPSBackendRigor:
    """
    Rigorously tests the core simulation capabilities of the MPSBackend.
    Focuses on correctness, entanglement handling, and robustness against truncation.
    """

    @pytest.fixture(scope="class")
    def backend(self):
        """Fixture to provide a fresh MPSBackend instance."""
        return MPSBackend()

    def test_mps_simulates_bell_state_phi_plus(self, backend):
        """
        Test 1: Verifies correct simulation of the maximally entangled Bell state |Φ+⟩.
        Circuit: H(0), CNOT(0, 1). Expected state: (|00⟩ + |11⟩) / √2.
        This confirms basic gate application, entanglement generation, and state representation fidelity.
        """
        num_qubits = 2
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.cx(0, 1)

        # Calculate the reference state vector using Qiskit's simulator
        reference_state = Statevector(qc).data
        
        # Simulate using the MPS backend. Assumes simulate returns the state vector.
        simulated_state = backend.simulate(qc)
        
        # Assertions for correctness
        assert simulated_state.shape == reference_state.shape
        
        # Use allclose for complex vector comparison with high precision
        np.testing.assert_allclose(
            simulated_state, 
            reference_state, 
            atol=1e-8, 
            rtol=1e-5,
            err_msg="MPS Bell state simulation failed to match reference state."
        )


    def test_mps_simulates_low_entanglement_product_state(self, backend):
        """
        Test 2: Verifies correct simulation of a low-entanglement product state (separable state).
        Circuit: RZ(π/2) on 0, X on 1, H on 2 (3 qubits).
        This ensures the MPS representation correctly handles separable states where bond dimension D=1 is sufficient.
        """
        num_qubits = 3
        qc = QuantumCircuit(num_qubits)
        qc.rz(np.pi/2, 0)
        qc.x(1)
        qc.h(2)

        reference_state = Statevector(qc).data
        
        # Simulate using the MPS backend
        simulated_state = backend.simulate(qc)
        
        # Assertions for correctness
        assert simulated_state.shape == reference_state.shape
        
        np.testing.assert_allclose(
            simulated_state, 
            reference_state, 
            atol=1e-8, 
            rtol=1e-5,
            err_msg="MPS product state simulation failed to match reference state."
        )

    def test_mps_handles_high_entanglement_with_truncation(self, backend):
        """
        Test 3: Checks robustness when simulating a highly entangled circuit (e.g., deep GHZ-like)
        with a severely restricted maximum bond dimension (D=2).
        The test ensures the simulation runs without error and produces a state vector
        of the correct size, demonstrating the backend's ability to handle truncation
        gracefully when entanglement exceeds the bond dimension limit.
        """
        num_qubits = 6
        qc = QuantumCircuit(num_qubits)
        
        # Create a highly entangled circuit
        for i in range(num_qubits):
            qc.h(i)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        for i in range(num_qubits):
            qc.rx(0.5, i)
        
        max_bond_dimension = 2
        
        # Simulate using the MPS backend with truncation
        simulated_state = backend.simulate(qc, max_bond_dimension=max_bond_dimension)
            
        # The resulting state vector size should be 2^N
        expected_size = 2**num_qubits
        
        # Assertions for robustness and size correctness
        assert simulated_state is not None, "Simulation failed to return a state vector."
        assert simulated_state.size == expected_size, f"Expected state size {expected_size}, got {simulated_state.size}."