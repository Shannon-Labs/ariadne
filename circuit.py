"""
Ariadne Quantum Circuit
=======================

Quantum circuit representation and manipulation.
"""

from typing import List, Optional
from rich.console import Console

console = Console()

class QuantumCircuit:
    """Simple quantum circuit representation."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.gates: List[str] = []
        console.print(f"[green]✅ QuantumCircuit created with {num_qubits} qubits[/green]")

    def h(self, qubit: int) -> 'QuantumCircuit':
        """Add Hadamard gate."""
        self.gates.append(f"H({qubit})")
        return self

    def cx(self, control: int, target: int) -> 'QuantumCircuit':
        """Add CNOT gate."""
        self.gates.append(f"CNOT({control},{target})")
        return self

    def depth(self) -> int:
        """Get circuit depth."""
        return len(self.gates)

    def size(self) -> int:
        """Get number of gates."""
        return len(self.gates)