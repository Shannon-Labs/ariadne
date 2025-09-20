"""Test suite for Ariadne's intelligent routing system."""

import pytest
from qiskit import QuantumCircuit

from ariadne import QuantumRouter, simulate
from ariadne.route.analyze import analyze_circuit


class TestQuantumRouter:
    """Test the QuantumRouter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.router = QuantumRouter()
    
    def test_router_initialization(self):
        """Test router initializes correctly."""
        assert self.router is not None
        assert hasattr(self.router, 'select_optimal_backend')
    
    def test_clifford_circuit_routing(self):
        """Test that pure Clifford circuits route to Stim."""
        # Create a Clifford-only circuit
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.s(2)
        qc.measure_all()
        
        decision = self.router.select_optimal_backend(qc)
        assert decision.recommended_backend == 'stim'
        assert decision.circuit_entropy < 0.1  # Low entropy for Clifford
    
    def test_general_circuit_routing(self):
        """Test that circuits with T gates don't route to Stim."""
        # Create circuit with T gates
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.t(0)  # Non-Clifford gate
        qc.cx(0, 1)
        qc.measure_all()
        
        decision = self.router.select_optimal_backend(qc)
        assert decision.recommended_backend != 'stim'
        assert decision.circuit_entropy > 0.1
    
    def test_large_circuit_routing(self):
        """Test routing for large circuits."""
        # Create a large sparse circuit
        qc = QuantumCircuit(100)
        for i in range(0, 100, 10):
            qc.h(i)
            if i + 1 < 100:
                qc.cx(i, i + 1)
        qc.measure_all()
        
        decision = self.router.select_optimal_backend(qc)
        # Large sparse circuits might use tensor networks
        assert decision.recommended_backend in ['stim', 'tensor_network']
    
    def test_parameterized_circuit_routing(self):
        """Test routing for parameterized circuits."""
        qc = QuantumCircuit(3)
        qc.rx(0.5, 0)
        qc.ry(1.2, 1)
        qc.rz(2.1, 2)
        qc.measure_all()
        
        decision = self.router.select_optimal_backend(qc)
        # Parameterized circuits should not use Stim
        assert decision.recommended_backend != 'stim'
    
    @pytest.mark.parametrize("n_qubits,expected_backend", [
        (2, 'stim'),      # Small Clifford
        (10, 'stim'),     # Medium Clifford
        (50, 'stim'),     # Large Clifford
        (100, 'stim'),    # Very large Clifford
    ])
    def test_scalability(self, n_qubits, expected_backend):
        """Test routing scales with circuit size."""
        qc = QuantumCircuit(n_qubits)
        # Create GHZ state (Clifford circuit)
        qc.h(0)
        for i in range(1, n_qubits):
            qc.cx(0, i)
        qc.measure_all()
        
        decision = self.router.select_optimal_backend(qc)
        assert decision.recommended_backend == expected_backend


class TestSimulateFunction:
    """Test the main simulate() function."""
    
    def test_simulate_basic(self):
        """Test basic simulation functionality."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        result = simulate(qc, shots=1000)
        
        assert hasattr(result, 'counts')
        assert hasattr(result, 'backend_used')
        assert hasattr(result, 'execution_time')
        
        # Check measurement results
        assert sum(result.counts.values()) == 1000
        assert set(result.counts.keys()).issubset({'00', '11'})
    
    def test_simulate_forced_backend(self):
        """Test forcing a specific backend."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.measure_all()
        
        # Force Qiskit backend
        result = simulate(qc, shots=100, backend='qiskit')
        assert result.backend_used == 'qiskit'
    
    def test_simulate_shots_parameter(self):
        """Test different shot counts."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()
        
        for shots in [1, 10, 100, 1000]:
            result = simulate(qc, shots=shots)
            assert sum(result.counts.values()) == shots
    
    def test_simulate_deterministic_circuit(self):
        """Test deterministic circuit gives consistent results."""
        qc = QuantumCircuit(1)
        qc.x(0)  # |0> -> |1>
        qc.measure_all()
        
        result = simulate(qc, shots=100)
        assert result.counts == {'1': 100}
    
    def test_simulate_empty_circuit(self):
        """Test empty circuit simulation."""
        qc = QuantumCircuit(3)
        qc.measure_all()
        
        result = simulate(qc, shots=100)
        assert result.counts == {'000': 100}


class TestCircuitAnalysis:
    """Test circuit analysis functionality."""
    
    def test_analyze_circuit_basic(self):
        """Test basic circuit analysis."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        
        analysis = analyze_circuit(qc)
        
        assert analysis['num_qubits'] == 3
        assert analysis['depth'] == 3
        assert 'gate_counts' in analysis
        assert 'clifford_ratio' in analysis
        assert 'is_clifford' in analysis
    
    def test_clifford_detection(self):
        """Test Clifford circuit detection."""
        # Pure Clifford circuit
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)
        qc1.s(1)
        
        analysis1 = analyze_circuit(qc1)
        assert analysis1['is_clifford'] == True
        assert analysis1['clifford_ratio'] == 1.0
        
        # Non-Clifford circuit
        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.t(0)  # T gate is not Clifford
        
        analysis2 = analyze_circuit(qc2)
        assert analysis2['is_clifford'] == False
        assert analysis2['clifford_ratio'] < 1.0
    
    def test_gate_counting(self):
        """Test gate counting functionality."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.t(2)
        
        analysis = analyze_circuit(qc)
        counts = analysis['gate_counts']
        
        assert counts.get('h', 0) == 2
        assert counts.get('cx', 0) == 2
        assert counts.get('t', 0) == 1
    
    def test_two_qubit_depth(self):
        """Test two-qubit depth calculation."""
        qc = QuantumCircuit(3)
        qc.h(0)  # Layer 1 (single-qubit)
        qc.h(1)  # Layer 1 (single-qubit)
        qc.cx(0, 1)  # Layer 2 (two-qubit)
        qc.h(2)  # Can be parallel with cx
        qc.cx(1, 2)  # Layer 3 (two-qubit)
        
        analysis = analyze_circuit(qc)
        assert analysis['two_qubit_depth'] == 2


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for routing decisions."""
    
    def test_routing_speed(self, benchmark):
        """Test that routing decisions are fast."""
        qc = QuantumCircuit(20)
        for i in range(20):
            qc.h(i)
        for i in range(0, 19, 2):
            qc.cx(i, i + 1)
        
        router = QuantumRouter()
        
        # Routing should be very fast (< 10ms)
        result = benchmark(router.select_optimal_backend, qc)
        assert result is not None
    
    def test_analysis_speed(self, benchmark):
        """Test that circuit analysis is fast."""
        qc = QuantumCircuit(50)
        for i in range(50):
            qc.h(i)
            if i > 0:
                qc.cx(i - 1, i)
        
        # Analysis should be fast even for larger circuits
        result = benchmark(analyze_circuit, qc)
        assert result['num_qubits'] == 50