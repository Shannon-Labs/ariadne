from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any
import os
from pathlib import Path
import json
from ..utils.logs import new_run_id


@dataclass
class ResourceEstimate:
    logical_qubits: int
    runtime_sec: float
    code: str
    notes: str = ""


def azure_estimate_table(qir_or_prog: str, codes: tuple[str, ...] = ("surface", "floquet")) -> Dict[str, Any]:  # pragma: no cover - stub
    """Return a per-code resource estimate or an 'unavailable' record.

    If Azure workspace env variables are not set, returns structured unavailable entries with setup hints.
    """
    env_vars = [
        "AZURE_QUANTUM_SUBSCRIPTION_ID",
        "AZURE_QUANTUM_RESOURCE_GROUP",
        "AZURE_QUANTUM_WORKSPACE_NAME",
        "AZURE_QUANTUM_LOCATION",
    ]
    if any(os.getenv(v) is None for v in env_vars):
        msg = {
            "status": "unavailable",
            "reason": "no_workspace",
            "hint": "Set AZURE_QUANTUM_* env vars to enable Resource Estimator.",
        }
        return {code: msg | {"code": code} for code in codes}

    try:  # pragma: no cover - integration
        from azure.quantum import Workspace  # type: ignore
        from azure.quantum.target.microsoft import MicrosoftEstimator  # type: ignore

        ws = Workspace(
            subscription_id=os.environ.get("AZURE_QUANTUM_SUBSCRIPTION_ID"),
            resource_group=os.environ.get("AZURE_QUANTUM_RESOURCE_GROUP"),
            name=os.environ.get("AZURE_QUANTUM_WORKSPACE_NAME"),
            location=os.environ.get("AZURE_QUANTUM_LOCATION"),
        )
        target = MicrosoftEstimator(workspace=ws)
        # Placeholder input data: in practice we would submit QIR from Qualtran bridge
        # Here we submit a tiny built-in estimator job if available
        run_id = new_run_id("azure")
        reports_dir = Path("reports") / "resources"
        reports_dir.mkdir(parents=True, exist_ok=True)
        table: Dict[str, Any] = {}
        for code in codes:
            try:
                params = {"errorBudget": 1e-3, "code": code}
                job = target.submit(input_data={}, params=params)
                result = job.get_results()
                rec = {
                    "logical_qubits": int(result.get("logicalQubits", 0)),
                    "runtime_sec": float(result.get("runtime", 0.0)),
                    "phys_qubits_est": int(result.get("physicalQubits", 0)),
                }
                table[code] = rec
            except Exception as e:  # pragma: no cover
                table[code] = {"status": "unavailable", "reason": f"sdk_error:{e}"}
        (reports_dir / f"{run_id}.json").write_text(json.dumps(table, indent=2))
        return table
    except Exception as e:
        return {code: {"status": "unavailable", "reason": f"sdk_error:{e}", "code": code} for code in codes}
