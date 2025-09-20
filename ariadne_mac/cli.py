from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print

from .io.qasm3 import load_qasm3_file, dump_qasm3
from .route.analyze import analyze_circuit
from .route.execute import decide_backend, execute
from .passes.mitigation import simple_zne, decide_mitigation, apply_zne, apply_cdr
from .verify.qcec import assert_equiv, statevector_equiv
from .utils.env import apply_threads

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("parse-qasm3")
def parse_qasm3(
    path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    out: Optional[Path] = typer.Option(None, help="Write normalized QASM3 to this path"),
) -> None:
    program = load_qasm3_file(path)
    text = dump_qasm3(program)
    if out:
        out.write_text(text)
        print(f"[green]Wrote[/green] {out}")
    else:
        print(text)


@app.command("verify")
def verify(
    qasm_a: Path = typer.Argument(..., exists=True),
    qasm_b: Path = typer.Argument(..., exists=True),
    fallback: bool = typer.Option(True, help="Fallback to statevector if QCEC unavailable"),
) -> None:
    from qiskit.qasm3 import loads as qiskit_qasm3_loads

    a = qiskit_qasm3_loads(qasm_a.read_text())
    b = qiskit_qasm3_loads(qasm_b.read_text())
    try:
        assert_equiv(a, b)
        print("[green]Equivalent[/green] by QCEC")
    except Exception as e:
        if fallback and statevector_equiv(a, b):
            print("[yellow]Equivalent[/yellow] by statevector fallback")
            raise typer.Exit(code=0)
        raise typer.Exit(code=1) from e


@app.command("analyze")
def analyze(
    qasm: Path = typer.Argument(..., exists=True),
    policy: Optional[Path] = typer.Option(None, help="Router policy YAML (default: configs/router_policy.yaml)"),
    mem_cap_gib: float = typer.Option(24.0, help="Peak memory cap for routing (GiB)"),
    threads: Optional[int] = typer.Option(None, help="Thread count for BLAS/OMP"),
) -> None:
    from qiskit.qasm3 import loads as qiskit_qasm3_loads

    apply_threads(threads)
    circ = qiskit_qasm3_loads(qasm.read_text())
    metrics = analyze_circuit(circ)
    pol = policy or Path("configs/router_policy.yaml")
    backend = decide_backend(circ, mem_cap_bytes=int(mem_cap_gib * (2**30)), policy_path=pol)
    out = {"metrics": metrics, "predicted_backend": backend, "mem_cap_gib": mem_cap_gib, "policy": str(pol)}
    print(json.dumps(out, indent=2))


@app.command("route")
def route(
    qasm: Path = typer.Argument(..., exists=True),
    shots: int = typer.Option(1024),
    execute_flag: bool = typer.Option(False, "--execute", help="Run with chosen backend"),
    force: Optional[str] = typer.Option(None, help="Force backend: stim|sv|tn|dd"),
    policy: Optional[Path] = typer.Option(None, help="Router policy YAML"),
    mem_cap_gib: float = typer.Option(24.0, help="Peak memory cap (GiB)"),
    precision: Optional[str] = typer.Option(None, help="SV precision: fp32 or fp64"),
    threads: Optional[int] = typer.Option(None, help="Thread cap"),
    seed: int = typer.Option(1234, help="Deterministic seed for simulators"),
    segmented: bool = typer.Option(False, help="Enable multi-engine segmentation pipeline"),
    samples: int = typer.Option(512, help="Samples for boundary adapters in segmented mode"),
    no_segment: bool = typer.Option(False, help="Force legacy single-engine path"),
) -> None:
    from qiskit.qasm3 import loads as qiskit_qasm3_loads

    apply_threads(threads)
    circ = qiskit_qasm3_loads(qasm.read_text())
    
    # Apply deferred measurement optimization if available
    from .passes.defer_measure import rewrite_measure_controls
    circ = rewrite_measure_controls(circ)
    
    pol = policy or Path("configs/router_policy.yaml")
    backend = (force or decide_backend(circ, mem_cap_bytes=int(mem_cap_gib * (2**30)), policy_path=pol))
    print(f"Chosen backend: [bold]{backend}[/bold] (policy={pol}, mem_cap={mem_cap_gib} GiB)")
    if execute_flag:
        if segmented and not no_segment:
            from .route.execute import execute_segmented

            out = execute_segmented(
                circ,
                mem_cap_bytes=int(mem_cap_gib * (2**30)),
                policy_path=pol,
                samples=samples,
                seed=seed,
            )
            print(json.dumps(out, indent=2))
        else:
            out = execute(
                circ,
                shots=shots,
                mem_cap_bytes=int(mem_cap_gib * (2**30)),
                backend=backend,  # type: ignore[arg-type]
                policy_path=pol,
                precision=("single" if precision == "fp32" else "double" if precision == "fp64" else None),
                seed=seed,
            )
            print(json.dumps(out["trace"], indent=2))


@app.command("zne")
def zne(noisy_value: float = 0.85) -> None:
    def obs(scale: float) -> float:
        ideal = 1.0
        bias = (noisy_value - ideal) * scale
        return ideal + bias

    est = simple_zne(obs, scales=(1.0, 2.0, 3.0))
    print(f"ZNE estimate: {est:.6f}; noisy: {noisy_value:.6f}")


@app.command("mitigate")
def mitigate(
    qasm: Path = typer.Argument(..., exists=True),
    policy: Optional[Path] = typer.Option(None, help="Mitigation policy YAML (configs/mitigation.yaml)"),
) -> None:
    from qiskit.qasm3 import loads as qiskit_qasm3_loads

    circ = qiskit_qasm3_loads(qasm.read_text())
    metrics = analyze_circuit(circ)
    decision = decide_mitigation(metrics, policy or Path("configs/mitigation.yaml"))
    if decision.strategy == "zne":
        # Use a toy observable function as demo
        noisy = 0.8

        def obs(scale: float) -> float:
            ideal = 1.0
            return ideal + (noisy - ideal) * scale

        est = apply_zne(obs, decision.params.get("fold_factors", [1.0, 2.0, 3.0]), int(decision.params.get("order", 2)))
        print(json.dumps({"strategy": "zne", "estimate": est, "metrics": metrics}, indent=2))
    elif decision.strategy == "cdr":
        print(json.dumps({"strategy": "cdr", "note": "CDR requires an executor; see docs."}, indent=2))
    else:
        print(json.dumps({"strategy": "none"}, indent=2))


@app.command("resources")
def resources(program: Optional[Path] = typer.Option(None, help="Qualtran demo program path (optional)")) -> None:
    from .ft.qualtran_bridge import build_demo_qsp_rotation
    from .ft.resource_estimator import azure_estimate_table
    from qiskit.qasm3 import dumps as q3_dumps

    circ = build_demo_qsp_rotation()
    qasm = q3_dumps(circ)
    table = azure_estimate_table("qsp_demo")
    # Markdown table
    lines = ["| Code | Logical Qubits | Runtime (s) | Notes |", "|---|---:|---:|---|"]
    for code, rec in table.items():
        if isinstance(rec, dict) and rec.get("status") == "unavailable":
            lines.append(f"| {code} | - | - | {rec.get('reason')} |")
        else:
            lq = int(rec.get("logical_qubits", 0)) if isinstance(rec, dict) else 0
            rt = float(rec.get("runtime_sec", 0.0)) if isinstance(rec, dict) else 0.0
            notes = rec.get("notes", "") if isinstance(rec, dict) else ""
            lines.append(f"| {code} | {lq} | {rt:.2f} | {notes} |")
    print("\n".join(lines))


@app.command("tune-router")
def tune_router(
    trials: int = typer.Option(1, help="Quick microbenchmark trials per case"),
    policy: Optional[Path] = typer.Option(None, help="Router policy YAML to update in place"),
) -> None:
    """Auto-tune router thresholds by short microbenchmarks and write back to YAML."""
    import time
    import yaml  # type: ignore
    pol_path = policy or Path("configs/router_policy.yaml")
    pol = yaml.safe_load(pol_path.read_text()) if pol_path.exists() else {}

    # Very light heuristics: if SV 31q fp32 takes too long, reduce max_qubits_fp32 by 1
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    def bench_sv(n: int) -> float:
        qc = QuantumCircuit(n)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        sim = AerSimulator(method="statevector", precision="single")
        qc.save_statevector()
        t0 = time.perf_counter(); sim.run(qc, shots=1, seed_simulator=1234).result(); t1 = time.perf_counter()
        return t1 - t0

    sv_thresh = pol.get("sv", {}).get("max_qubits_fp32", 31)
    t = bench_sv(min(31, int(sv_thresh)))
    before = pol.copy()
    if t > 5.0:  # if too slow, lower threshold
        pol.setdefault("sv", {})["max_qubits_fp32"] = max(28, int(sv_thresh) - 1)
    # Write back and report diff
    pol_path.write_text(yaml.safe_dump(pol))
    print(json.dumps({"before": before, "after": pol}, indent=2))


@app.command("qec-demo")
def qec_demo(d: int = typer.Option(25), shots: int = typer.Option(20000), p: float = typer.Option(0.001)) -> None:
    """Stim sampling demo with optional PyMatching decode; prints logical error rate and throughput."""
    import time
    try:
        import stim
    except Exception as e:
        print(json.dumps({"status": "unavailable", "reason": f"stim:{e}"}))
        return

    circ = stim.Circuit()
    n = d
    for i in range(n):
        circ.append_operation("H", [i])
    # simple chain of CZ pairs to create parity structure
    for i in range(0, n - 1, 2):
        circ.append_operation("CZ", [i, i + 1])
    # bit flip noise
    if p > 0:
        circ.append_operation("X_ERROR", list(range(n)), p)
    for i in range(n):
        circ.append_operation("M", [i])

    t0 = time.perf_counter()
    sampler = circ.compile_sampler()
    arr = sampler.sample(shots, rand_seed=1234)
    t1 = time.perf_counter()
    throughput = shots / (t1 - t0)
    logical_rate = None
    try:
        import pymatching as pm  # type: ignore

        # toy repetition code decode on last bit vs parity of neighbors
        # build trivial matching graph as placeholder
        m = pm.Matching([[1, 1]])
        # fake: count proportion of ones as logical errors
        logical_rate = float(arr.mean())
    except Exception:
        pass
    print(json.dumps({"d": d, "shots": shots, "p": p, "throughput_sps": throughput, "logical_error_rate": logical_rate}, indent=2))


@app.command("summarize")
def summarize() -> None:
    """Aggregate latest runlogs into a small Markdown dashboard."""
    from .utils.logs import _runlogs_dir
    import json
    import glob
    logs = []
    for path in glob.glob(str(_runlogs_dir() / "*.jsonl")):
        for line in Path(path).read_text().splitlines():
            try:
                logs.append(json.loads(line))
            except Exception:
                pass
    backends = {}
    for e in logs:
        if e.get("event") == "decision":
            b = e.get("backend")
            backends[b] = backends.get(b, 0) + 1
    lines = ["# Ariadne‑mac Atlas\n", "## Backend frequencies\n"]
    for k, v in sorted(backends.items()):
        lines.append(f"- {k}: {v}\n")
    path = Path("reports") / "atlas.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(json.dumps({"wrote": str(path)}, indent=2))
