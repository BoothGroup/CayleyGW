"""Logging for the workflow: stage lines, progress counts and a timing summary.

The package logs through the :mod:`logging` logger ``cayleygw`` and its module
children. ``verbose=0`` logs warnings only; ``1``, the default, logs the start
and end of each stage (``INFO``) and a summary of stage times per main call;
``2`` adds progress counts over contour nodes and closure candidates
(``DEBUG``). The console view of :mod:`cayleygw.tools.console` is on from
import. :func:`enable_logging` changes its level or style and can add the
plain-text log file of :mod:`cayleygw.tools.logfile`. Both read the values each
record carries as its ``cayleygw`` attribute. The package never writes to
standard output.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Literal, TextIO

from .._helpers.errors import ValidationError
from .._helpers.tools.logging import _ACTIVE, _ACTIVE_LOCK, _duration, _Function
from .console import ConsoleHandler
from .logfile import LogFileHandler

LOGGER = logging.getLogger("cayleygw")

#: Logging level implied by each ``verbose`` value; larger values map to DEBUG.
LEVELS = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

LogStyle = Literal["standard", "bare"]


def enable_logging(
    verbose: int = 1,
    *,
    stream: TextIO | None = None,
    style: LogStyle = "standard",
    log_file: str | os.PathLike[str] | None = None,
) -> logging.Logger:
    """Set the level and style of the log, and optionally add a log file.

    The package calls ``enable_logging()`` on import, so the standard console
    view is on without it. A later call replaces the handlers an earlier one
    installed and closes its log file.

    Args:
        verbose: ``0`` warnings only, ``1`` stages and summaries, ``2`` adds
            progress counts.
        stream: Text stream of the console view; ``sys.stderr`` by default.
        style: ``"standard"`` draws options, sizes, a spinner naming the running
            stage, each automatic ``N_q`` doubling and a summary panel with the
            stage times. ``"bare"`` keeps the key results: the selected ``N_q``,
            no contour geometry, and only the total time of each call.
        log_file: Plain-text log to append to, beside the console view: the
            versions, the command and the native threads, then a timestamped
            line per stage, every option and each result in full. ``None``
            writes no file.

    Returns:
        The ``cayleygw`` logger.

    Raises:
        ValidationError: If ``style`` is not ``"standard"`` or ``"bare"``.
    """

    if style not in ("standard", "bare"):
        raise ValidationError(f"style must be 'standard' or 'bare'; got {style!r}")
    level = LEVELS[max(0, min(int(verbose), 2))]
    for handler in list(LOGGER.handlers):
        if getattr(handler, "cayleygw_handler", False):
            LOGGER.removeHandler(handler)
            handler.close()
    handlers: list[logging.Handler] = [ConsoleHandler(stream, bare=style == "bare")]
    if log_file is not None:
        handlers.append(LogFileHandler(log_file))
    for handler in handlers:
        handler.cayleygw_handler = True  # type: ignore[attr-defined]
        LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
    return LOGGER


class Stages:
    """Stage lines and a timing summary for one public call.

    :meth:`stage` logs a line on entry with the sizes and one on exit with the
    wall time. The outermost recorder of a call tree collects the stages of
    every nested call, worker threads included, so its :meth:`summary` lists
    them all.

    Args:
        verbose: ``0`` silent, ``1`` stages and summary, ``2`` adds progress counts.
        logger: Module logger under ``cayleygw`` that the lines go to.
    """

    __slots__ = ("verbose", "logger", "_rows", "_lock", "_started")

    def __init__(self, verbose: int, logger: logging.Logger) -> None:
        self.verbose = int(verbose)
        self.logger = logger
        self._rows: list[tuple[str, float]] = []
        self._lock = threading.Lock()
        self._started = time.perf_counter()

    def __enter__(self) -> "Stages":
        """Make this the collecting recorder for the stages of nested calls."""

        with _ACTIVE_LOCK:
            _ACTIVE.append(self)
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exception: object) -> None:
        """Stop collecting the stages of nested calls."""

        with _ACTIVE_LOCK:
            if self in _ACTIVE:
                _ACTIVE.remove(self)

    @property
    def is_root(self) -> bool:
        """Whether this recorder collects, or no recorder is collecting."""

        with _ACTIVE_LOCK:
            return not _ACTIVE or _ACTIVE[0] is self

    def _record(self, name: str, seconds: float) -> None:
        """Add a stage time to the collecting recorder, or to this one."""

        with _ACTIVE_LOCK:
            target = _ACTIVE[0] if _ACTIVE else self
        with target._lock:
            target._rows.append((name, seconds))

    @property
    def rows(self) -> tuple[tuple[str, float], ...]:
        """Stage names and wall times recorded so far, in completion order."""

        with self._lock:
            return tuple(self._rows)

    @contextmanager
    def stage(self, name: str, **sizes: Any) -> Iterator[None]:
        """Log ``name`` with ``sizes`` on entry and its wall time on exit."""

        if self.verbose >= 1:
            event = {"event": "stage-start", "name": name, "sizes": sizes}
            if sizes:
                described = ", ".join(f"{key}={value}" for key, value in sizes.items())
                self.logger.info("%s: %s", name, described, extra={"cayleygw": event})
            else:
                self.logger.info("%s", name, extra={"cayleygw": event})
        started = time.perf_counter()
        ending = "%s: %s"
        try:
            yield
        except BaseException:
            ending = "%s: failed after %s"
            raise
        finally:
            elapsed = time.perf_counter() - started
            self._record(name, elapsed)
            if self.verbose >= 1:
                self.logger.info(
                    ending,
                    name,
                    _duration(elapsed),
                    extra={"cayleygw": {"event": "stage-end", "name": name}},
                )

    def progress(self, name: str, done: int, total: int | None = None) -> None:
        """Log a running count at ``verbose >= 2``, at every tenth and at the end."""

        if self.verbose < 2:
            return
        event = {"event": "progress", "name": name, "done": done, "total": total}
        if total is None:
            self.logger.debug("%s: %d done", name, done, extra={"cayleygw": event})
            return
        step = max(1, total // 10)
        if done >= total or done % step == 0:
            self.logger.debug("%s: %d of %d", name, done, total, extra={"cayleygw": event})

    def options(
        self,
        title: str,
        options: dict[str, Any],
        sizes: dict[str, Any],
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log the settings and sizes a call runs with.

        The console view draws ``options`` and ``sizes``; ``details`` are the
        remaining settings, which only the log file writes.
        """

        if self.verbose >= 1:
            self.logger.info(
                "%s: %s",
                title,
                ", ".join(f"{key}={value}" for key, value in options.items()),
                extra={
                    "cayleygw": {
                        "event": "options",
                        "title": title,
                        "options": options,
                        "sizes": sizes,
                        "details": details or {},
                    }
                },
            )

    def result(self, title: str, value: Any, **context: Any) -> None:
        """Log a result for the console view and the log file to describe."""

        if self.verbose >= 1:
            self.logger.info(
                "%s: returned %s",
                title,
                type(value).__name__,
                extra={
                    "cayleygw": {
                        "event": "result",
                        "title": title,
                        "value": value,
                        **context,
                    }
                },
            )

    def summary(self, title: str, result: Any = None) -> None:
        """Log the stage times, from the outermost recorder only.

        ``result`` is the call's return value, ``None`` if it failed; the
        console view draws its summary panel from it.
        """

        if self.verbose < 1 or not self.is_root:
            return
        event = {
            "event": "summary",
            "title": title,
            "rows": self.rows,
            "total": time.perf_counter() - self._started,
            "result": result,
        }
        self.logger.info("%s: stage times", title, extra={"cayleygw": event})


def summarized(title: str, logger: logging.Logger) -> Callable[[_Function], _Function]:
    """Collect the stages of a public call and log their summary when it ends.

    The decorated function takes ``verbose`` as a keyword argument; at
    ``verbose >= 1``, given or by default, the stages of every nested call are
    collected and the summary is logged when the call returns or fails.
    """

    def decorate(function: _Function) -> _Function:
        default = inspect.signature(function).parameters["verbose"].default

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            verbose = kwargs.get("verbose", default)
            level = (
                verbose
                if isinstance(verbose, int) and not isinstance(verbose, bool) and verbose >= 0
                else 0
            )
            result = None
            with Stages(level, logger) as root:
                try:
                    result = function(*args, **kwargs)
                    return result
                finally:
                    if root.rows:
                        root.summary(title, result)

        return wrapper  # type: ignore[return-value]

    return decorate


__all__ = ["LEVELS", "LOGGER", "Stages", "enable_logging", "summarized"]
