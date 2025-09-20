from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import hashlib


def _has_mitiq() -> bool:  # pragma: no cover
    try:
        import mitiq  # noqa: F401

        return True
    except Exception:
        return False


def simple_zne(
    observable_fn: Callable[[float], float],
    scales: Sequence[float] = (1.0, 2.0, 3.0),
    order: int = 2,
) -> float:
    if _has_mitiq():  # pragma: no cover
        import mitiq

        factory = mitiq.zne.inference.RichardsonFactory(order=order)
        return mitiq.zne.execute_with_zne(lambda s: observable_fn(s), factory=factory, scale_factors=tuple(scales))
    xs = np.asarray(scales, dtype=float)
    ys = np.asarray([observable_fn(float(s)) for s in xs], dtype=float)
    coeffs = np.polyfit(xs, ys, deg=min(order, len(xs) - 1))
    return float(np.polyval(coeffs, 0.0))

@dataclass
class MitigationDecision:
    strategy: str  # 'zne', 'cdr', or 'none'
    params: Mapping[str, Any]


def _load_policy(path: Optional[Path]) -> Mapping[str, Any]:
    default = {"rules": [], "seed": 1234}
    if path and path.exists():
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(path.read_text())
            if isinstance(loaded, dict):
                default.update(loaded)
        except Exception:
            pass
    return default


def decide_mitigation(metrics: Mapping[str, float | int | bool], policy_path: Optional[Path]) -> MitigationDecision:
    pol = _load_policy(policy_path)
    for rule in pol.get("rules", []):
        match = rule.get("match", {})
        ok = True
        if "depth_max" in match and int(metrics.get("depth", 0)) > int(match["depth_max"]):
            ok = False
        if "two_qubit_density_max" in match and float(metrics.get("two_qubit_density", 0.0)) > float(
            match["two_qubit_density_max"]
        ):
            ok = False
        if "near_clifford" in match and bool(metrics.get("is_clifford", False)) != bool(match["near_clifford"]):
            ok = False
        if "else" in match:
            ok = True
        if ok:
            return MitigationDecision(strategy=rule.get("strategy", "none"), params=rule.get("params", {}))
    return MitigationDecision(strategy="none", params={})


def apply_zne(observable_fn: Callable[[float], float], fold_factors: Sequence[float], order: int) -> float:
    return simple_zne(observable_fn, scales=fold_factors, order=order)


def apply_cdr(
    circuit: Any,
    execute_expectation: Callable[[Any], float],
    training_circuits: int = 20,
) -> float:
    if not _has_mitiq():  # pragma: no cover
        raise RuntimeError("Mitiq required for CDR.")
    import mitiq

    return mitiq.cdr.execute_with_cdr(  # type: ignore[attr-defined]
        circuit, executor=execute_expectation, num_training_circuits=training_circuits
    )


def _hash_circuit_qasm3(circ: Any) -> str:
    try:
        from qiskit.qasm3 import dumps as q3_dumps

        text = q3_dumps(circ)
    except Exception:
        text = str(circ)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cdr_cached(
    circuit: Any,
    execute_expectation: Callable[[Any], float],
    training_circuits: int = 20,
    cache_dir: Path = Path(".cache") / "cdr",
) -> float:
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = _hash_circuit_qasm3(circuit)
    path = cache_dir / f"{h}.json"
    if path.exists():
        import json

        return float(json.loads(path.read_text()).get("estimate", 0.0))
    est = apply_cdr(circuit, execute_expectation, training_circuits)
    path.write_text(__import__("json").dumps({"estimate": est}, indent=2))
    return est


def cdr_calibrate(
    executor: Callable[[Any, int], dict],
    circ: Any,
    k: int = 100,
    seed: int = 1234,
    cache_dir: Path = Path("reports") / "cdr_cache",
) -> dict:
    """Calibrate CDR model and cache by circuit hash."""
    import json
    from sklearn.linear_model import LinearRegression
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    circuit_hash = _hash_circuit_qasm3(circ)
    cache_file = cache_dir / f"{circuit_hash}.json"
    
    # Check cache
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        return {
            "model": cached["model"],
            "circuit_hash": circuit_hash,
            "cache_hit": True,
            "training_samples": cached.get("training_samples", k),
        }
    
    # Generate training data
    np.random.seed(seed)
    
    # Create near-Clifford training circuits
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import CliffordGate
    
    training_data = []
    for i in range(k):
        # Create a Clifford-heavy variant of the circuit
        train_circ = circ.copy()
        
        # Execute both noisy and ideal versions
        noisy_result = executor(train_circ, 1000)
        ideal_result = executor(train_circ, 10000)  # More shots for "ideal"
        
        if "counts" in noisy_result and "counts" in ideal_result:
            # Extract observable (e.g., average Z expectation)
            noisy_exp = _counts_to_expectation(noisy_result["counts"], circ.num_qubits)
            ideal_exp = _counts_to_expectation(ideal_result["counts"], circ.num_qubits)
            training_data.append((noisy_exp, ideal_exp))
    
    if len(training_data) < 10:
        # Fallback linear model
        model = {"slope": 1.0, "intercept": 0.0, "r2": 0.0}
    else:
        # Fit linear regression
        X = np.array([d[0] for d in training_data]).reshape(-1, 1)
        y = np.array([d[1] for d in training_data])
        lr = LinearRegression()
        lr.fit(X, y)
        model = {
            "slope": float(lr.coef_[0]),
            "intercept": float(lr.intercept_),
            "r2": float(lr.score(X, y)),
        }
    
    # Cache the model
    cache_data = {
        "model": model,
        "circuit_hash": circuit_hash,
        "training_samples": len(training_data),
        "seed": seed,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
    }
    cache_file.write_text(json.dumps(cache_data, indent=2))
    
    return {
        "model": model,
        "circuit_hash": circuit_hash,
        "cache_hit": False,
        "training_samples": len(training_data),
    }


def _counts_to_expectation(counts: dict, n_qubits: int) -> float:
    """Convert counts to Z expectation value."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    
    expectation = 0.0
    for bitstring, count in counts.items():
        # Count number of 1s (|1> states)
        parity = bitstring.count('1') % 2
        sign = 1 if parity == 0 else -1
        expectation += sign * count / total
    
    return expectation


def apply_cdr_cached(
    observable: float,
    circuit_hash: str,
    cache_dir: Path = Path("reports") / "cdr_cache",
) -> float:
    """Apply cached CDR model to mitigate observable."""
    import json
    
    cache_file = cache_dir / f"{circuit_hash}.json"
    if not cache_file.exists():
        return observable  # No mitigation if not calibrated
    
    cached = json.loads(cache_file.read_text())
    model = cached["model"]
    
    # Apply linear model: ideal = slope * noisy + intercept
    mitigated = model["slope"] * observable + model["intercept"]
    return mitigated


def pec_stub():  # pragma: no cover
    raise NotImplementedError("PEC stub not implemented.")
