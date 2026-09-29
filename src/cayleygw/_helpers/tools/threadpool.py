"""Private helpers of :mod:`cayleygw.tools.parallel`: the cached controller and the core count."""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

from threadpoolctl import ThreadpoolController


_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

_controller: Any = None
_controller_key: int | None = None


def _threadpool_controller() -> Any:
    """Return the process's cached ``ThreadpoolController``.

    Building one walks every loaded shared object under the dynamic loader
    lock, which can deadlock against a LAPACK call in another thread that is
    loading a library, and costs about sixty times the limit it installs. So it
    is rebuilt only after a new module import, the one event that can load a
    library the last walk did not see.
    """

    global _controller, _controller_key
    key = len(sys.modules)
    if _controller is None or key != _controller_key:
        _controller = ThreadpoolController()
        _controller_key = key
    return _controller


def _install_limit(count: int) -> Callable[[], None]:
    """Install ``count`` and return the call that releases it."""

    limiter = _threadpool_controller().limit(limits=count)
    limiter.__enter__()
    return lambda: limiter.__exit__(None, None, None)


def _available_cores() -> int:
    """Return the thread count set in the environment, else the cores this process may use."""

    for name in _THREAD_VARIABLES:
        value = os.environ.get(name, "").strip()
        if value.isdigit() and int(value) > 0:
            return int(value)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1
