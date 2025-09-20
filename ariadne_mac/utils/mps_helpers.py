from __future__ import annotations

import warnings
from typing import Callable, Iterable, Optional

import numpy as np

_METAL_WARNED = False


def _can_use_jax_metal() -> bool:  # pragma: no cover
    try:
        import jax  # noqa: F401
        import jax.experimental.metal  # noqa: F401
        return True
    except Exception:
        return False


def _warn_once(msg: str) -> None:
    global _METAL_WARNED
    if not _METAL_WARNED:
        warnings.warn(msg)
        _METAL_WARNED = True


def plan_real_contraction(scores: np.ndarray, scorer: Optional[Callable[[np.ndarray], float]] = None) -> float:
    """Optionally accelerate a real-valued planning score with JAX‑Metal.

    If `scores` is not float32 real, falls back to CPU and warns once.
    """
    arr = np.asarray(scores)
    if arr.dtype != np.float32:
        _warn_once("JAX‑Metal only supports float32 real; falling back to CPU.")
        return float(np.mean(arr)) if scorer is None else float(scorer(arr))

    if not _can_use_jax_metal():  # pragma: no cover
        return float(np.mean(arr)) if scorer is None else float(scorer(arr))

    import jax
    import jax.numpy as jnp

    fn = (lambda x: jnp.mean(x)) if scorer is None else jax.jit(scorer)
    return float(fn(jnp.asarray(arr)))

