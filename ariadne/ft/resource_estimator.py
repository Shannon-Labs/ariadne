from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ResourceEstimate:
    logical_qubits: int
    runtime_sec: float
    code: str
    notes: str = ""


def estimate_with_azure(
    qir_or_broombridge: str,
    target: str = "microsoft.estimator",
    code: str = "surface",
    error_budget: Optional[float] = None,
) -> ResourceEstimate:  # pragma: no cover - stub
    """Wrapper around Azure Quantum Resource Estimator.

    Parameters
    ----------
    qir_or_broombridge: str
        Path to QIR or chemistry description supported by Azure RE.
    target: str
        Azure target name for the estimator.
    code: str
        Error-correcting code (e.g., surface, floquet) to compare.
    error_budget: Optional[float]
        Optional target failure probability.
    """
    try:
        from azure.quantum import Workspace  # type: ignore
    except Exception as e:
        raise RuntimeError("Azure Quantum SDK is required (azure-quantum).") from e

    # Placeholder stub; real flow requires authenticated workspace and job submission
    return ResourceEstimate(
        logical_qubits=0, runtime_sec=0.0, code=code, notes="Stub: connect to Azure workspace to run"
    )

