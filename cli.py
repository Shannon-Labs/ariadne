"""
Ariadne Command Line Interface
==============================

CLI for quantum circuit simulation and management.
"""

import typer
from rich.console import Console

console = Console()
app = typer.Typer()

@app.command()
def simulate():
    """Simulate quantum circuits."""
    console.print("[green]Ariadne quantum simulator CLI[/green]")
    console.print("Run: python -m ariadne.examples.bell_state_demo")

@app.command()
def info():
    """Show Ariadne information."""
    console.print("[bold cyan]Ariadne: Quantum Computing on Your Laptop[/bold cyan]")
    console.print("Version: 1.0.0")
    console.print("Built by Shannon Labs")
    console.print("Production quantum security: [link]https://entruptor.com[/link]")

if __name__ == "__main__":
    app()
