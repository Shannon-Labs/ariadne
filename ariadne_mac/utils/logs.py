from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from enum import Enum


class ReasonCode(str, Enum):
    """Standardized reason codes for routing decisions."""
    CLIFFORD_CIRCUIT = "clifford_circuit"
    SV_FITS_MEMORY = "sv_fits_memory"
    HIGH_REDUNDANCY = "high_redundancy"
    LOW_TREEWIDTH = "low_treewidth"
    EXCEEDS_MEMORY = "exceeds_memory"
    NOT_CLIFFORD = "not_clifford"
    FALLBACK = "fallback"
    SEGMENTED_BOUNDARY = "segmented_boundary"
    CDR_APPLIED = "cdr_applied"


def _runlogs_dir() -> Path:
    d = Path("reports") / "runlogs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_run_id(prefix: str = "run") -> str:
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    pid = os.getpid()
    return f"{prefix}_{now}_{pid}"


def runlog_path(run_id: str) -> Path:
    return _runlogs_dir() / f"{run_id}.jsonl"


_OPEN_FILES: dict[str, Any] = {}
_LOGGED_EVENTS: set[tuple[str, str]] = set()


def log_event(run_id: str, event: Dict[str, Any], dedupe_key: Optional[str] = None) -> None:
    """Log event with schema v2 and deduplication support.
    
    Args:
        run_id: Unique run identifier
        event: Event data to log
        dedupe_key: Optional key for deduplication within run
    """
    event = dict(event)
    event.setdefault("ts", datetime.utcnow().isoformat() + "Z")
    event.setdefault("schema_version", 2)
    
    # Add reason code if applicable
    if "backend" in event and "reason" not in event:
        backend = event["backend"]
        if backend == "stim":
            event["reason_code"] = ReasonCode.CLIFFORD_CIRCUIT
        elif backend == "sv":
            event["reason_code"] = ReasonCode.SV_FITS_MEMORY
        elif backend == "dd":
            event["reason_code"] = ReasonCode.HIGH_REDUNDANCY
        elif backend == "tn":
            event["reason_code"] = ReasonCode.LOW_TREEWIDTH
    
    # Deduplication check
    if dedupe_key:
        event_key = (run_id, dedupe_key)
        if event_key in _LOGGED_EVENTS:
            return  # Skip duplicate
        _LOGGED_EVENTS.add(event_key)
    
    p = runlog_path(run_id)
    
    # Use cached file handle if available (within same process)
    if run_id not in _OPEN_FILES:
        _OPEN_FILES[run_id] = p.open("a", encoding="utf-8", buffering=1)
    
    f = _OPEN_FILES[run_id]
    f.write(json.dumps(event, default=str) + "\n")
    f.flush()


def close_run_logs(run_id: str) -> None:
    """Close log file for a run."""
    if run_id in _OPEN_FILES:
        _OPEN_FILES[run_id].close()
        del _OPEN_FILES[run_id]
        # Clear dedup cache for this run
        _LOGGED_EVENTS = {k for k in _LOGGED_EVENTS if k[0] != run_id}


def summarize_run(run_id: str) -> Dict[str, Any]:
    """Generate summary statistics for a run."""
    p = runlog_path(run_id)
    if not p.exists():
        return {}
    
    events = []
    with p.open("r") as f:
        for line in f:
            events.append(json.loads(line))
    
    summary = {
        "run_id": run_id,
        "total_events": len(events),
        "schema_version": 2,
        "backends_used": list(set(e.get("backend") for e in events if "backend" in e)),
        "segments": len([e for e in events if e.get("event") == "segment"]),
        "total_time_s": sum(e.get("wall_time_s", 0) for e in events if "wall_time_s" in e),
    }
    
    return summary
