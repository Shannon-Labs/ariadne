from __future__ import annotations

import tracemalloc
from typing import Optional


def try_peak_memory_gib() -> Optional[float]:  # pragma: no cover - optional
    try:
        import psutil  # type: ignore

        p = psutil.Process()
        info = p.memory_info()
        return float(info.rss) / (1024 ** 3)
    except Exception:
        return None


class TrackPeak:
    def __enter__(self) -> "TrackPeak":
        tracemalloc.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        tracemalloc.stop()

    @staticmethod
    def current_gib() -> float:
        try:
            current, peak = tracemalloc.get_traced_memory()
            return float(peak) / (1024 ** 3)
        except Exception:
            return 0.0

