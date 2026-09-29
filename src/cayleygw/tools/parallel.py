"""Python workers and native-thread limits for the loops over nodes and candidates.

Each contour node or closure candidate is BLAS and LAPACK work, which releases
the GIL, so Python threads run them at the same time. The thread-count
environment variables are read only when a native library loads, so this module
sets the limit through ``threadpoolctl``.
"""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from itertools import islice
from typing import Any, Callable, Iterator, Literal, Sequence

from .._helpers.tools.threadpool import (
    _available_cores,
    _install_limit,
    _threadpool_controller,
)
from .._helpers.validate import _check

__all__ = [
    "ambient_native_threads",
    "evaluate_by_index",
    "limited_native_threads",
    "resolve_native_threads",
]


def resolve_native_threads(native_threads: int | Literal["auto"], n_workers: int = 1) -> int:
    """Return the native threads each worker may use.

    ``"auto"`` shares the cores among the workers: the count set in
    ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS`` or ``MKL_NUM_THREADS`` if one
    is set, otherwise the cores this process may run on, which respects
    ``taskset``, SLURM and container limits.

    Args:
        native_threads: A positive count, or ``"auto"``.
        n_workers: Python workers that share the cores.

    Returns:
        The count, at least one.

    Raises:
        ValidationError: If either argument is not a positive integer or ``"auto"``.
    """

    workers = _check.positive_integer(n_workers, "n_workers")
    if isinstance(native_threads, str) and native_threads == "auto":
        return max(1, _available_cores() // workers)
    return _check.positive_integer(native_threads, "native_threads")


def ambient_native_threads() -> int | None:
    """Return the native-thread count in force now, if it can be read.

    The package pins its own dense work to ``native_threads``, but PySCF runs
    the mean field before that at whatever count the caller left, and another
    count gives other orbitals and so other moments. Neon at ``n_conserved=5``
    realizes at dimension 339 with one ambient thread and 340 with four. For a
    reproducible number, run the mean field and the moments inside one
    :func:`limited_native_threads` block and record this count beside the
    result: conservation ratios compare only at the same count.

    Returns:
        The count, or ``None`` if no BLAS library is loaded.
    """

    blas = _threadpool_controller().select(user_api="blas").lib_controllers
    return int(blas[0].num_threads) if blas else None


# The shared limit: regions inside it on any thread, their count, and the call that ends it.
_LIMIT_LOCK = threading.Lock()
_limit_depth = 0
_limit_count: int | None = None
_limit_release: Callable[[], None] | None = None


def _leave_shared_limit() -> None:
    """Leave the shared region; the last one out restores the ambient count."""

    global _limit_depth, _limit_count, _limit_release
    with _LIMIT_LOCK:
        _limit_depth -= 1
        if _limit_depth > 0:
            return
        release = _limit_release
        _limit_release = None
        _limit_count = None
    if release is not None:
        release()


@contextmanager
def limited_native_threads(count: int) -> Iterator[None]:
    """Limit native BLAS and LAPACK threads to ``count`` inside a block.

    The previous count comes back on exit, also when the block raises. A region
    entered while another holds the same count, such as the candidate scan
    inside a sector, joins it, and the last region out restores the ambient
    count. So the count does not depend on which thread finishes first, and
    nested entries skip the library walk of :func:`_threadpool_controller`. A
    region with another count installs its own and restores the enclosing one.

    Args:
        count: Native threads for the block.

    Raises:
        ValidationError: If ``count`` is not a positive integer.
    """

    global _limit_depth, _limit_count, _limit_release

    count = _check.positive_integer(count, "count")
    with _LIMIT_LOCK:
        if _limit_depth > 0 and _limit_count == count:
            _limit_depth += 1
            mode = "shared"
            release = None
        elif _limit_depth > 0:
            mode = "nested"
            release = _install_limit(count)
        else:
            mode = "shared"
            _limit_release = _install_limit(count)
            _limit_depth = 1
            _limit_count = count
            release = None
    try:
        yield
    finally:
        if mode == "shared":
            _leave_shared_limit()
        elif release is not None:
            release()


def evaluate_by_index(
    function: Callable[[int], Any],
    indices: Sequence[int],
    *,
    n_workers: int = 1,
    native_threads: int = 1,
    sink: Callable[[int, Any], None] | None = None,
    on_complete: Callable[[int], None] | None = None,
) -> tuple[dict[int, Any], list[tuple[int, Exception]]]:
    """Evaluate ``function`` at each index, serially or on ``n_workers`` threads.

    Native threads are pinned on both paths, so ``n_workers`` changes the cost
    but not the answer: a last-bit difference from another BLAS count can change
    an integer rank in the block-Toeplitz test downstream. Results are keyed by
    index and failures sorted by index, so the lowest failing index is reported
    whatever the thread timing. Serial evaluation stops at the first failure;
    parallel evaluation finishes the work in flight and submits no more.

    Args:
        function: Applied to one index at a time; must not change state shared
            with other indices.
        indices: Indices to evaluate.
        n_workers: Python threads; one evaluates in the calling thread.
        native_threads: BLAS and LAPACK threads per worker while the region runs.
        sink: Receives each ``(index, value)`` as it completes, and the value is
            not kept, so one ``(naux, naux)`` matrix per contour node is not held
            twice.
        on_complete: Called with each index after its result is read, on the
            calling thread and after ``sink``; for progress counts.

    Returns:
        Results keyed by index (none if ``sink`` is given), and the
        ``(index, exception)`` pairs sorted by index.

    Raises:
        ValidationError: If ``native_threads`` is not a positive integer.
    """

    results: dict[int, Any] = {}
    failures: list[tuple[int, Exception]] = []
    with limited_native_threads(native_threads):
        if n_workers == 1:
            for index in indices:
                try:
                    value = function(index)
                    if sink is None:
                        results[index] = value
                    else:
                        sink(index, value)
                    if on_complete is not None:
                        on_complete(index)
                except Exception as error:  # noqa: BLE001 - reported, not consumed
                    failures.append((index, error))
                    break
            return results, failures

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Bounded window: an unread future holds its result, so queuing every index holds
            # them all.
            remaining = iter(indices)
            in_flight: deque[tuple[Future, int]] = deque()
            window = 2 * n_workers
            for index in islice(remaining, window):
                in_flight.append((executor.submit(function, index), index))
            while in_flight:
                future, index = in_flight.popleft()
                try:
                    value = future.result()
                    if sink is None:
                        results[index] = value
                    else:
                        sink(index, value)
                    if on_complete is not None:
                        on_complete(index)
                except Exception as error:  # noqa: BLE001 - reported below
                    failures.append((index, error))
                finally:
                    # Release the value before the next result() blocks.
                    future = value = None
                if failures:
                    # Drain the window but submit nothing more.
                    continue
                for index in islice(remaining, 1):
                    in_flight.append((executor.submit(function, index), index))
    failures.sort(key=lambda item: item[0])
    return results, failures
