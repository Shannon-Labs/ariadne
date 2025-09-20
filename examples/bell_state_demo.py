#!/usr/bin/env python3
"""
Ariadne Bell State Demo
=======================

This demo showcases Ariadne's quantum circuit simulation capabilities
by creating and measuring a Bell state (quantum entanglement).

Run with: python bell_state_demo.py
"""

import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

# Import Ariadne components
from ariadne.simulator import QuantumSimulator
from ariadne.circuit import QuantumCircuit
from ariadne.backends import QiskitBackend

console = Console()

def create_bell_state_circuit():
    """Create a quantum circuit that generates a Bell state."""
    console.print("[cyan]Creating Bell state quantum circuit...[/cyan]")

    # Create a 2-qubit circuit
    circuit = QuantumCircuit(2)

    # Create Bell state |00⟩ + |11⟩
    # 1. Put first qubit in superposition
    circuit.h(0)  # Hadamard gate

    # 2. Entangle with second qubit
    circuit.cx(0, 1)  # CNOT gate

    console.print("[green]✅ Bell state circuit created[/green]")
    console.print(f"Circuit depth: {circuit.depth}")
    console.print(f"Gate count: {circuit.size()}")

    return circuit

def simulate_bell_state(circuit: QuantumCircuit, shots: int = 1000):
    """Simulate the Bell state circuit."""
    console.print(f"[cyan]Simulating Bell state with {shots} shots...[/cyan]")

    # Use Qiskit backend for reliable simulation
    backend = QiskitBackend()
    simulator = QuantumSimulator(backend)

    with Progress() as progress:
        task = progress.add_task("Running simulation...", total=shots)

        start_time = time.time()
        result = simulator.simulate(circuit, shots=shots)
        end_time = time.time()

        progress.update(task, completed=shots)

    simulation_time = end_time - start_time
    console.print(f"[green]✅ Simulation completed in {simulation_time:.3f}s[/green]")

    return result

def analyze_results(result):
    """Analyze and display the measurement results."""
    console.print("[cyan]Analyzing measurement results...[/cyan]")

    # Display results in a nice table
    results_table = Table(title="🔔 Bell State Measurement Results")
    results_table.add_column("State", style="cyan", no_wrap=True)
    results_table.add_column("Count", style="magenta")
    results_table.add_column("Probability", style="green")
    results_table.add_column("Expected", style="yellow")

    total_shots = sum(result.counts.values())

    # Bell state should give ~50% |00⟩ and ~50% |11⟩
    for state, count in result.counts.items():
        probability = count / total_shots
        expected = "50%" if state in ["00", "11"] else "0%"

        results_table.add_row(
            f"|{state}⟩",
            str(count),
            f"{probability:.1%}",
            expected
        )

    console.print(results_table)

    # Verify entanglement
    state_00 = result.counts.get("00", 0)
    state_11 = result.counts.get("11", 0)

    entanglement_ratio = min(state_00, state_11) / max(state_00, state_11) if max(state_00, state_11) > 0 else 0

    console.print(f"\n[bold cyan]Entanglement Analysis:[/bold cyan]")
    console.print(f"State |00⟩: {state_00} measurements ({state_00/total_shots:.1%})")
    console.print(f"State |11⟩: {state_11} measurements ({state_11/total_shots:.1%})")
    console.print(f"Entanglement ratio: {entanglement_ratio:.3f} (closer to 1.0 = better entanglement)")

    if entanglement_ratio > 0.8:
        console.print("[green]✅ Strong quantum entanglement detected![/green]")
    else:
        console.print("[yellow]⚠️  Weak entanglement - may be measurement error[/yellow]")

def main():
    """Main Bell state demonstration."""
    console.print(Panel.fit(
        "[bold cyan]🔔 Ariadne Bell State Demo[/bold cyan]\n\n"
        "[yellow]Creating and measuring quantum entanglement[/yellow]\n"
        "[dim]Simulating quantum circuits on your laptop[/dim]",
        title="🔬 Shannon Labs",
        border_style="cyan"
    ))

    # Create Bell state circuit
    circuit = create_bell_state_circuit()

    # Simulate measurement
    result = simulate_bell_state(circuit, shots=1000)

    # Analyze results
    analyze_results(result)

    # Final summary
    console.print(Panel(
        "[bold green]🎉 Demo Complete![/bold green]\n\n"
        "Successfully simulated quantum entanglement on a classical computer!\n"
        "This Bell state demonstrates the power of quantum computing.\n\n"
        "[yellow]Ready to explore more quantum algorithms?[/yellow]",
        title="🔬 Quantum Computing on Your Laptop",
        border_style="green"
    ))

    console.print("\n[dim]💡 Tip: Try running with more shots (e.g., 10000) for better statistics[/dim]")
    console.print("[dim]🔗 Learn more quantum computing at entruptor.com[/dim]")

if __name__ == "__main__":
    main()