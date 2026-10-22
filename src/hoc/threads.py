"""Pin BLAS/OpenMP threads so results do not depend on the machine's core count.

Environment variables only take effect if they are set before numpy loads its
BLAS, so ``import hoc`` sets them as defaults (inherited by every subprocess and
joblib worker), and ``threadpoolctl`` limits OpenBLAS/MKL pools that are already
loaded. Apple Accelerate cannot be changed after loading; launch through
``run_all.py`` or ``make`` (which set the variables first) to pin it too.

Set ``HOC_BLAS_THREADS`` to use more threads.
"""

from __future__ import annotations

import os

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_limiter = None


def blas_threads() -> int:
    return int(os.environ.get("HOC_BLAS_THREADS", "1"))


def thread_env(n: int | None = None) -> dict[str, str]:
    """The variables to put in a child process's environment."""
    n = blas_threads() if n is None else n
    return {name: str(n) for name in THREAD_ENV_VARS}


def pin_threads(n: int | None = None) -> None:
    """Default the thread variables and limit already-loaded BLAS pools."""
    global _limiter
    n = blas_threads() if n is None else n
    for name, value in thread_env(n).items():
        os.environ.setdefault(name, value)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl is a dependency
        return
    _limiter = threadpool_limits(limits=n)


def threadpool_record() -> list[dict]:
    """The BLAS/OpenMP pools actually in use (for the run manifest)."""
    try:
        from threadpoolctl import threadpool_info
    except ImportError:  # pragma: no cover
        return []
    keys = ("user_api", "internal_api", "version", "num_threads")
    return [{k: pool.get(k) for k in keys} for pool in threadpool_info()]
