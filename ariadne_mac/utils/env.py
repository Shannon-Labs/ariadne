from __future__ import annotations

import os
from typing import Optional


def apply_threads(threads: Optional[int]) -> None:
    if threads is None or threads <= 0:
        return
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(threads)

