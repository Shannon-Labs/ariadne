from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print

from .io.qasm3 import load_qasm3, dump_qasm3, load_qasm3_file, dump_qasm3_file
from .route.analyze import analyze_circuit
from .route.execute import decide_backend
from .passes.mitigation import simple_zne
from .verify.qcec import assert_equiv, statevector_equiv

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("parse-qasm3")
def parse_qasm3(
    path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    out: Optional[Path] = typer.Option(None, help="Optional output path to write back QASM3"),
) -> None:
    """Parse an OpenQASM 3 file and optionally write a normalized dump."""
    program = load_qasm3_file(path)
    text = dump_qasm3(program)
    if out:
        dump_qasm3_file(program, out)
        print(f"[green]Wrote[/green] normalized QASM3 to {out}")
    else:
        print(text)


@app.command("verify")
def verify_equiv(
    qpy_a: Path = typer.Argument(..., help="QPY or QASM3 of circuit A"),
    qpy_b: Path = typer.Argument(..., help="QPY or QASM3 of circuit B"),
    fallback_statevector: bool = typer.Option(True, help="Fallback to SV check if QCEC unavailable"),
) -> None:
    from qiskit import QuantumCircuit
    from qiskit import qpy

    def load(path: Path) -> QuantumCircuit:
        if path.suffix == ".qpy":
            with open(path, "rb") as f:
                return qpy.load(f)[0]
        else:
            from qiskit.qasm3 import loads as qiskit_qasm3_loads

            return qiskit_qasm3_loads(path.read_text())

    circ_a = load(qpy_a)
    circ_b = load(qpy_b)
    try:
        assert_equiv(circ_a, circ_b)
        print("[green]Equivalent[/green] per MQT QCEC")
    except Exception as e:
        if fallback_statevector:
            ok = statevector_equiv(circ_a, circ_b)
            if ok:
                print("[yellow]Equivalent[/yellow] by statevector fallback (QCEC failed)")
                raise typer.Exit(code=0)
        raise typer.Exit(code=1) from e


@app.command("analyze")
def analyze(
    qasm3: Path = typer.Argument(..., exists=True),
) -> None:
    from qiskit.qasm3 import loads as qiskit_qasm3_loads

    circ = qiskit_qasm3_loads(qasm3.read_text())
    report = analyze_circuit(circ)
    print(json.dumps(report, indent=2))


@app.command("route")
def route(
    qasm3: Path = typer.Argument(..., exists=True),
) -> None:
    from qiskit.qasm3 import loads as qiskit_qasm3_loads

    circ = qiskit_qasm3_loads(qasm3.read_text())
    backend = decide_backend(circ)
    print(f"Chosen backend: [bold]{backend}[/bold]")


@app.command("zne")
def zne_demo(
    noisy_value: float = typer.Option(0.85, help="Observed noisy expectation value"),
) -> None:
    def fake_expectation(scale: float) -> float:
        # Simple linear bias that increases with scale
        ideal = 1.0
        bias = (noisy_value - ideal) * scale
        return ideal + bias

    est = simple_zne(fake_expectation, scales=(1.0, 2.0, 3.0))
    print(f"ZNE estimate: {est:.6f} (noisy: {noisy_value:.6f})")

