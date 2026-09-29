"""The default log: a console view of the workflow, drawn with rich.

:class:`ConsoleHandler` draws the values the records of
:mod:`cayleygw.tools.logger` carry, not their text: a banner, the options and
sizes of each main call, one line per automatic ``N_q`` doubling, warnings, a
summary panel per main call and the quasiparticle table. A spinner names the
running stage. Stage lines are left to the log file. On a stream that is not a
terminal there is no spinner and no colour.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, TextIO

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from .._helpers.tools.logging import (
    _RESULTS,
    _SUMMARIES,
    PACKAGES,
    _duration,
    _git_hash,
    _table,
    _timings,
    _value,
    _version,
)
from .._helpers.version import __version__

BANNER = r"""                 _             ______        __
  ___ __ _ _   _| | ___ _   _ / ___\ \      / /
 / __/ _` | | | | |/ _ \ | | | |  _ \ \ /\ / /
| (_| (_| | |_| | |  __/ |_| | |_| | \ V  V /
 \___\__,_|\__, |_|\___|\__, |\____|  \_/\_/
           |___/        |___/"""

THEME = Theme(
    {
        "good": "green",
        "ok": "yellow",
        "bad": "red",
        "option": "bold cyan",
        "output": "bold blue",
        "comment": "dim",
    }
)


class ConsoleHandler(logging.Handler):
    """Draw the package's log records as a console view.

    Args:
        stream: Text stream to write to; ``None`` follows ``sys.stderr``.
        bare: Draw only the key results: no ``N_q`` doublings, no contour
            geometry and no stage times.
    """

    def __init__(self, stream: TextIO | None = None, *, bare: bool = False) -> None:
        super().__init__()
        self.bare = bare
        self.console = Console(
            file=stream,
            stderr=stream is None,
            highlight=False,
            theme=THEME,
        )
        self._running: list[tuple[int | None, str]] = []
        self._counts: dict[str, str] = {}
        self._status: Status | None = None
        self._banner_shown = False
        self._after_blank = True

    # -- records ---------------------------------------------------------

    def emit(self, record: logging.LogRecord) -> None:
        """Draw a record's values, or print a warning or error on one line."""

        try:
            event = getattr(record, "cayleygw", None)
            draw = None if event is None else self._EVENTS.get(event["event"])
            if draw is not None:
                draw(self, event, record)
            elif record.levelno >= logging.WARNING:
                # On one line, unwrapped, so that it can be searched for.
                label = "Warning: " if record.levelno == logging.WARNING else "Error: "
                self._banner_once()
                self._after_blank = False
                self.console.print(
                    Text.assemble((label, "bad"), record.getMessage()),
                    soft_wrap=True,
                )
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """Stop the spinner and close the handler."""

        self._stop_status()
        super().close()

    def _stage_start(self, event: dict, record: logging.LogRecord) -> None:
        """Add a running stage to the spinner."""

        self._running.append((record.thread, event["name"]))
        self._update_status()

    def _stage_end(self, event: dict, record: logging.LogRecord) -> None:
        """Remove a finished stage, and its count, from the spinner."""

        entry = (record.thread, event["name"])
        for index in range(len(self._running) - 1, -1, -1):
            if self._running[index] == entry:
                del self._running[index]
                break
        if not any(name == event["name"] for _, name in self._running):
            self._counts.pop(event["name"], None)
        self._update_status()

    def _progress(self, event: dict, record: logging.LogRecord) -> None:
        """Show a progress count on the spinner."""

        done, total = event["done"], event["total"]
        self._counts[event["name"]] = f"{done}" if total is None else f"{done}/{total}"
        self._update_status()

    def _options(self, event: dict, record: logging.LogRecord) -> None:
        """Draw the options and sizes tables of a main call."""

        options = _table("Options")
        options.add_column("Option", justify="right")
        options.add_column("Value", justify="right", style="option")
        for key, value in event["options"].items():
            options.add_row(key, _value(value))
        layout = Table.grid(padding=(0, 4))
        if event["sizes"]:
            sizes = _table("Sizes")
            sizes.add_column("Space", justify="right")
            sizes.add_column("Size", justify="right")
            for key, value in event["sizes"].items():
                sizes.add_row(key, _value(value))
            layout.add_row(options, sizes)
        else:
            layout.add_row(options)
        self._write("")
        self._write(f"[bold underline]{event['title']}[/]", comment="options")
        self._write(layout)
        self._write("")

    def _n_q(self, event: dict, record: logging.LogRecord) -> None:
        """Draw one automatic ``N_q`` doubling and its verdict."""

        if self.bare:
            return
        error = event["error"]
        if event["converged"]:
            style, verdict = "good", "converged"
        elif error < 1.0:
            style, verdict = "ok", "moments converged, feasibility not yet"
        else:
            style, verdict = "bad", ""
        self._write(
            f"N_q {event['coarse']:>5} -> {event['fine']:<5}  "
            f"error/tolerance = [{style}]{error:.3e}[/]",
            comment=verdict,
        )

    def _summary(self, event: dict, record: logging.LogRecord) -> None:
        """Draw the summary panel of a main call."""

        title, result, total = event["title"], event["result"], event["total"]
        if result is None:
            parts: list[RenderableType] = [f"{title} [bad]failed[/] after {_duration(total)}."]
        else:
            describe = _SUMMARIES.get(title)
            parts = (
                [f"{title} finished in {_duration(total)}."]
                if describe is None
                else describe(result, total, self.bare)
            )
        if not self.bare:
            parts.append(_timings(event["rows"], total))
        spaced: list[RenderableType] = []
        for part in parts:
            spaced.extend((part, ""))
        self._write("")
        self._write(Panel(Group(*spaced[:-1]), title=title, padding=(1, 2), expand=False))
        self._write("")

    def _result(self, event: dict, record: logging.LogRecord) -> None:
        """Draw a result that has a renderer."""

        draw = _RESULTS.get(event["title"])
        if draw is not None:
            self._write(draw(event))

    _EVENTS: dict[str, Callable[["ConsoleHandler", dict, logging.LogRecord], None]] = {
        "stage-start": _stage_start,
        "stage-end": _stage_end,
        "progress": _progress,
        "options": _options,
        "n_q": _n_q,
        "summary": _summary,
        "result": _result,
    }

    # -- output ----------------------------------------------------------

    def _write(self, message: RenderableType, *, comment: str | None = None) -> None:
        """Print one block, with ``comment`` dimmed at the right."""

        self._banner_once()
        # One blank line separates blocks, however many of them ask for one.
        blank = message == "" and not comment
        if blank and self._after_blank:
            return
        self._after_blank = blank
        if comment:
            grid = Table.grid(expand=True)
            grid.add_column(justify="left")
            grid.add_column(justify="right", style="comment")
            grid.add_row(message, comment)
            message = grid
        self.console.print(message)

    def _update_status(self) -> None:
        """Start, update or stop the spinner for the running stages."""

        if not self._running:
            self._stop_status()
            return
        if not self._on_a_terminal():
            return
        names = [name for _, name in self._running]
        text = " > ".join(names)
        count = self._counts.get(names[-1])
        if count is not None:
            text += f" ({count})"
        if self._status is None:
            self._status = Status(text, console=self.console)
            self._status.start()
        else:
            self._status.update(text)

    def _on_a_terminal(self) -> bool:
        """Whether the stream is a real terminal, which can erase the spinner.

        ``FORCE_COLOR``, which IDE consoles set, makes rich treat any stream as
        a terminal.  That is right for the colours, but a console that ignores
        cursor movement keeps every spinner frame, one line per stage.
        """

        isatty = getattr(self.console.file, "isatty", None)
        try:
            return self.console.is_interactive and isatty is not None and isatty()
        except ValueError:  # the stream was closed
            return False

    def _stop_status(self) -> None:
        """Stop the spinner."""

        if self._status is not None:
            self._status.stop()
            self._status = None

    def _banner_once(self) -> None:
        """Draw the banner before whatever the log prints first."""

        if not self._banner_shown:
            self._banner_shown = True
            self._banner()

    def _banner(self) -> None:
        """Draw the logo, the package versions and the OpenMP thread count."""

        lines = BANNER.splitlines()
        width = max(len(line) for line in lines)
        lines[-1] += f"{__version__:>{width - len(lines[-1])}}"
        self._write("[bold]" + "\n".join(lines) + "[/]")
        packages = _table(show_header=False, box=None)
        packages.add_column(style="bold")
        packages.add_column(justify="right")
        packages.add_column(style="comment")
        commit = _git_hash()
        packages.add_row("cayleygw", __version__, "" if commit is None else f"git {commit}")
        for name in PACKAGES:
            packages.add_row(name, _version(name) or "N/A", "")
        packages.add_row("OpenMP threads", os.environ.get("OMP_NUM_THREADS", "unset"), "")
        self._write(packages)


__all__ = ["ConsoleHandler"]
