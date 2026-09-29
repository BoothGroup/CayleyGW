"""The logger: stage lines, progress, summaries, verbosity, console views, log file."""

from __future__ import annotations

import io
import logging
import subprocess
import sys

import numpy as np
import pytest

from cayleygw import (
    Sector,
    ValidationError,
    enable_logging,
)
from cayleygw.realization.sector import SectorSelfEnergyRealization
from cayleygw.tools.logger import LEVELS, LOGGER, Stages, summarized


def test_enable_logging_installs_one_handler_at_the_implied_level():
    stream = io.StringIO()
    logger = enable_logging(2, stream=stream)
    assert logger is LOGGER
    assert logger.level == LEVELS[2] == logging.DEBUG
    handlers = [h for h in logger.handlers if getattr(h, "cayleygw_handler", False)]
    assert len(handlers) == 1
    # Calling it again replaces the handler rather than stacking a second one.
    enable_logging(0, stream=stream)
    handlers = [h for h in logger.handlers if getattr(h, "cayleygw_handler", False)]
    assert len(handlers) == 1
    assert logger.level == logging.WARNING
    enable_logging()


def test_the_console_view_is_on_from_import():
    probe = (
        "import cayleygw\n"
        "from cayleygw.tools.console import ConsoleHandler\n"
        "from cayleygw.tools.logger import LOGGER\n"
        "print(sum(isinstance(h, ConsoleHandler) for h in LOGGER.handlers), LOGGER.level)\n"
    )
    shown = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.split()
    assert shown == ["1", str(logging.INFO)]


def test_stage_lines_carry_sizes_and_wall_time(caplog):
    logger = logging.getLogger("cayleygw.test")
    with caplog.at_level(logging.DEBUG, logger="cayleygw"):
        stages = Stages(1, logger)
        with stages.stage("contour node solves", N_q=64, n_aux=30):
            stages.progress("contour node solves", 32, 64)
    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == "contour node solves: N_q=64, n_aux=30"
    assert messages[1].startswith("contour node solves: ") and messages[1].endswith(" s")
    assert len(messages) == 2, "progress is silent at verbose=1"
    assert stages.rows[0][0] == "contour node solves"


def test_verbose_zero_is_silent_and_two_adds_progress(caplog):
    logger = logging.getLogger("cayleygw.test")
    with caplog.at_level(logging.DEBUG, logger="cayleygw"):
        silent = Stages(0, logger)
        with silent.stage("quiet", n=1):
            silent.progress("quiet", 1, 10)
        silent.summary("quiet call")
    assert not caplog.records
    with caplog.at_level(logging.DEBUG, logger="cayleygw"):
        chatty = Stages(2, logger)
        with chatty.stage("scan"):
            for done in range(1, 21):
                chatty.progress("scan", done, 20)
    messages = [record.getMessage() for record in caplog.records]
    assert "scan: 20 of 20" in messages
    assert "scan: 3 of 20" not in messages, "progress is reported at every tenth"


def test_a_failed_stage_still_logs_its_time(caplog):
    logger = logging.getLogger("cayleygw.test")
    stages = Stages(1, logger)
    with caplog.at_level(logging.INFO, logger="cayleygw"):
        with pytest.raises(RuntimeError):
            with stages.stage("doomed"):
                raise RuntimeError("no")
    messages = [record.getMessage() for record in caplog.records]
    assert messages[-1].startswith("doomed: failed after")
    assert stages.rows[0][0] == "doomed"


def test_the_summarized_call_collects_nested_stages(caplog):
    logger = logging.getLogger("cayleygw.test")

    def nested(verbose):
        inner = Stages(verbose, logger)
        with inner.stage("hole Gram eigensolve", rank=4):
            pass
        inner.summary("nested")  # not the root, so silent

    @summarized("outer call", logger)
    def outer(*, verbose=0):
        stages = Stages(verbose, logger)
        with stages.stage("outer stage"):
            nested(verbose)
        return "done"

    with caplog.at_level(logging.INFO, logger="cayleygw"):
        assert outer(verbose=1) == "done"
    messages = [record.getMessage() for record in caplog.records]
    summaries = [
        record.cayleygw
        for record in caplog.records
        if getattr(record, "cayleygw", {}).get("event") == "summary"
    ]
    assert len(summaries) == 1 and summaries[0]["title"] == "outer call"
    stages = [name for name, _ in summaries[0]["rows"]]
    assert "hole Gram eigensolve" in stages
    assert "outer stage" in stages
    assert summaries[0]["total"] >= 0.0
    assert not any(m.startswith("nested: stage times") for m in messages)
    # Outside any summarized call a recorder is its own root again.
    assert Stages(1, logger).is_root


def test_public_functions_validate_verbose():
    with pytest.raises(ValidationError, match="verbose"):
        SectorSelfEnergyRealization.scan_closures(
            np.asarray([[[2.0 + 0.0j]]]),
            Sector.PARTICLE,
            verbose=-1,
        )


@pytest.fixture
def stream():
    """A text stream to log to; the default console view is restored afterwards."""

    stream = io.StringIO()
    yield stream
    enable_logging()


def _call(logger, *, fail=False, verbose=1):
    """A summarized call that logs one of everything the console view draws."""

    @summarized("demo call", logger)
    def call(*, verbose=0):
        stages = Stages(verbose, logger)
        stages.options("demo call", {"n_conserved": 3, "label": "[bold]"}, {"orbitals": 24})
        with stages.stage("contour node solves", N_q=64):
            stages.progress("contour node solves", 64, 64)
        logger.info(
            "Automatic N_q refinement 64 -> 128",
            extra={
                "cayleygw": {
                    "event": "n_q",
                    "coarse": 64,
                    "fine": 128,
                    "error": 6.0e-3,
                    "converged": True,
                }
            },
        )
        logger.warning("a refusal that does not raise, at [0, 1]")
        stages.result("demo call", "unused")
        if fail:
            raise RuntimeError("no")
        return "done"

    return call(verbose=verbose)


def test_the_console_view_is_the_default(stream):
    enable_logging(1, stream=stream)
    logger = logging.getLogger("cayleygw.test")
    assert _call(logger) == "done"
    text = stream.getvalue()
    assert text.count("|___/") == 2, "the banner is drawn once, before the first call"
    assert "Options" in text and "n_conserved" in text and "[bold]" in text
    assert "orbitals" in text and "24" in text
    assert "6.000e-03" in text and "converged" in text
    assert "Warning: a refusal that does not raise, at [0, 1]" in text
    assert "demo call finished in" in text
    assert "contour node solves" in text and "total wall time" in text
    assert "contour node solves: N_q=64" not in text, "no stage lines"


def test_the_console_view_reports_a_failed_call(stream):
    enable_logging(1, stream=stream)
    with pytest.raises(RuntimeError):
        _call(logging.getLogger("cayleygw.test"), fail=True)
    assert "demo call failed after" in stream.getvalue()


def test_the_log_file_holds_every_stage_line_and_option(stream, tmp_path):
    path = tmp_path / "run.log"
    enable_logging(2, stream=stream, log_file=path)
    assert _call(logging.getLogger("cayleygw.test"), verbose=2) == "done"
    enable_logging(stream=stream)  # closes the file
    text = path.read_text()
    assert text.startswith("=" * 78) and "log opened" in text
    assert "numpy" in text and "OMP_NUM_THREADS=" in text
    body = [line[13:] for line in text.splitlines() if line[:2].isdigit()]
    assert "INFO    contour node solves: N_q=64" in body
    assert "DEBUG   contour node solves: 64 of 64" in body
    assert "INFO    Automatic N_q refinement 64 -> 128" in body
    assert "WARNING a refusal that does not raise, at [0, 1]" in body
    assert "INFO    demo call: options" in body and "n_conserved  3" in text
    assert "demo call: stage times" in text
    assert "contour node solves: N_q=64" not in stream.getvalue(), "not on the console"
    # A second file opened on the same path appends rather than overwrites.
    enable_logging(stream=stream, log_file=path)
    enable_logging(stream=stream)
    assert path.read_text().count("log opened") == 2


def test_the_bare_view_keeps_the_results_and_drops_the_rest(stream):
    enable_logging(1, stream=stream, style="bare")
    assert _call(logging.getLogger("cayleygw.test")) == "done"
    text = stream.getvalue()
    assert "n_conserved" in text and "orbitals" in text
    assert "Warning: a refusal that does not raise" in text
    assert "demo call finished in" in text
    assert "error/tolerance" not in text, "the N_q doublings are left out"
    assert "Timings" not in text and "total wall time" not in text


@pytest.mark.pyscf
def test_verbose_zero_writes_nothing_to_the_console_or_the_file(stream, tmp_path, h2_rhf_df):
    from cayleygw import CayleyGW

    path = tmp_path / "quiet.log"
    enable_logging(1, stream=stream, log_file=path)
    CayleyGW(h2_rhf_df, verbose=0).kernel(1)
    enable_logging(stream=stream)  # closes the file
    assert stream.getvalue() == "", "not even the banner"
    records = [line[13:] for line in path.read_text().splitlines() if line[:2].isdigit()]
    assert not any(record.startswith(("INFO", "DEBUG")) for record in records)


def test_an_unknown_style_is_refused(stream):
    with pytest.raises(ValueError, match="style"):
        enable_logging(1, stream=stream, style="pretty")
