from __future__ import annotations

from typing import Callable, Iterable, Sequence, Tuple

import numpy as np


def _try_mitiq() -> bool:  # pragma: no cover - optional path
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
    """Zero-Noise Extrapolation via simple polynomial fit.

    - If Mitiq is available, delegates to Mitiq's Richardson ZNE.
    - Otherwise fits a degree-`order` polynomial in the scale and extrapolates to 0 noise.

    Parameters
    ----------
    observable_fn: Callable[[float], float]
        Function returning the noisy observable measured at a given noise scale factor.
    scales: Sequence[float]
        Noise scaling factors to sample.
    order: int
        Polynomial order for the fallback fit.
    """
    if _try_mitiq():  # pragma: no cover - optional path
        import mitiq

        factory = mitiq.zne.inference.RichardsonFactory(order=order)
        return mitiq.zne.execute_with_zne(lambda s: observable_fn(s), factory=factory, scale_factors=tuple(scales))

    # Fallback polynomial fit at integer scale factors
    xs = np.asarray(scales, dtype=float)
    ys = np.asarray([observable_fn(float(s)) for s in xs], dtype=float)
    deg = min(order, len(xs) - 1)
    coeffs = np.polyfit(xs, ys, deg=deg)
    # Extrapolate to zero noise (scale=0) => value is constant term of polynomial in descending order
    return float(np.polyval(coeffs, 0.0))


def build_cdr_training_set():  # pragma: no cover - stub
    """Stub: Build a training set of near-Clifford circuits for CDR.

    Returns a list of (circuit, ideal_value) pairs when implemented.
    """
    raise NotImplementedError("CDR training set builder not implemented yet.")


def pec_stub():  # pragma: no cover - stub
    raise NotImplementedError("Probabilistic error cancellation stub.")


def vd_stub():  # pragma: no cover - stub
    raise NotImplementedError("Virtual distillation stub.")

