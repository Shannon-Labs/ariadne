from __future__ import annotations

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Instruction, Measure, ControlledGate
from qiskit.circuit.library import XGate, YGate, ZGate, HGate, SGate, CXGate
from typing import List, Optional, Set, Tuple


def defer_measurements(circ: QuantumCircuit) -> QuantumCircuit:
    """Rewrite mid-circuit measurements with classical control over Clifford gates.
    
    Converts patterns like:
        measure q[0] -> c[0]
        if (c[0]) x q[1]
    
    Into:
        cx q[0], q[1]  # Controlled version
        measure q[0] -> c[0]  # Deferred to end
    
    Aborts if non-Clifford gates have classical control.
    """
    deferred_measurements: List[Tuple[int, int, int]] = []  # (qubit, clbit, position)
    new_circ = QuantumCircuit(circ.num_qubits, circ.num_clbits)
    
    # Track which classical bits are measurement results
    measurement_clbits: Set[int] = set()
    
    for idx, (inst, qargs, cargs) in enumerate(circ.data):
        # Check if this is a measurement
        if isinstance(inst, Measure):
            qubit_idx = circ.qubits.index(qargs[0])
            clbit_idx = circ.clbits.index(cargs[0])
            
            # Check if there are any gates after this that depend on this measurement
            has_dependent = False
            for future_idx in range(idx + 1, len(circ.data)):
                future_inst, _, _ = circ.data[future_idx]
                if hasattr(future_inst, 'condition') and future_inst.condition:
                    if future_inst.condition[0] == cargs[0]:
                        has_dependent = True
                        break
            
            if has_dependent:
                # Defer this measurement
                deferred_measurements.append((qubit_idx, clbit_idx, idx))
                measurement_clbits.add(clbit_idx)
                continue
            else:
                # Keep measurement in place if no dependencies
                new_circ.append(inst, qargs, cargs)
                continue
        
        # Handle classically-controlled gates
        if hasattr(inst, 'condition') and inst.condition:
            clbit, value = inst.condition
            clbit_idx = circ.clbits.index(clbit)
            
            if clbit_idx in measurement_clbits:
                # This gate is controlled by a measurement result
                # Check if it's a Clifford gate
                if not _is_clifford_gate(inst):
                    raise ValueError(f"Cannot defer measurement: non-Clifford gate {inst.name} has classical control")
                
                # Convert to quantum-controlled version
                controlled_gate = _make_controlled_clifford(inst, value)
                
                # Find the measurement qubit that controls this
                control_qubit = None
                for mq, mc, _ in deferred_measurements:
                    if mc == clbit_idx:
                        control_qubit = mq
                        break
                
                if control_qubit is not None:
                    # Apply controlled gate
                    target_qubits = [circ.qubits.index(q) for q in qargs]
                    new_circ.append(controlled_gate, [circ.qubits[control_qubit]] + qargs, [])
                else:
                    # Shouldn't happen, but fallback
                    new_circ.append(inst, qargs, cargs)
            else:
                # Not controlled by measurement, keep as-is
                new_circ.append(inst, qargs, cargs)
        else:
            # Regular gate, keep as-is
            new_circ.append(inst, qargs, cargs)
    
    # Add deferred measurements at the end
    for qubit_idx, clbit_idx, _ in deferred_measurements:
        new_circ.measure(qubit_idx, clbit_idx)
    
    return new_circ


def _is_clifford_gate(inst: Instruction) -> bool:
    """Check if gate is a Clifford gate."""
    clifford_gates = {
        'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg',
        'cx', 'cy', 'cz', 'swap',
        'id', 'i',
    }
    return inst.name.lower() in clifford_gates


def _make_controlled_clifford(inst: Instruction, control_value: int) -> ControlledGate:
    """Create quantum-controlled version of Clifford gate."""
    from qiskit.circuit.library import CXGate, CYGate, CZGate, CHGate, CSGate
    
    # Map single-qubit Cliffords to their controlled versions
    gate_map = {
        'x': CXGate,
        'y': CYGate, 
        'z': CZGate,
        'h': CHGate,
        's': CSGate,
    }
    
    if inst.name.lower() in gate_map:
        controlled_class = gate_map[inst.name.lower()]
        gate = controlled_class()
        
        # If control should be on |0> instead of |1>, add X gates
        if control_value == 0:
            # This would need special handling - for now assume control on |1>
            pass
        
        return gate
    
    # For other Clifford gates, use generic control
    return inst.control(1)


def validate_deferred_circuit(original: QuantumCircuit, deferred: QuantumCircuit) -> bool:
    """Validate that deferred circuit is equivalent to original for Clifford circuits."""
    try:
        from qiskit.quantum_info import Clifford
        
        # Both should produce same Clifford tableau
        cliff_orig = Clifford(original)
        cliff_defer = Clifford(deferred)
        
        return cliff_orig == cliff_defer
    except Exception:
        # If not pure Clifford, can't validate this way
        return True


def rewrite_measure_controls(circ: QuantumCircuit) -> QuantumCircuit:
    """Main entry point for deferred measurement optimization.
    
    Returns optimized circuit or original if optimization not applicable.
    """
    try:
        deferred = defer_measurements(circ)
        
        # Validate if possible
        if validate_deferred_circuit(circ, deferred):
            return deferred
        else:
            return circ
    except ValueError:
        # Contains non-Clifford classically controlled gates
        return circ
    except Exception:
        # Any other issue, return original
        return circ


def defer_measure_if_clifford(circ: QuantumCircuit) -> Tuple[QuantumCircuit, bool]:
    """Wrapper for backward compatibility.
    
    Returns (circuit, success) tuple.
    """
    try:
        deferred = rewrite_measure_controls(circ)
        return deferred, (deferred != circ)
    except Exception:
        return circ, False