r"""Report how far the machine-sensitive quantities sit from a recorded reference.

The suite asserts what holds on any machine: an identity, an ordering, or an
inequality with margin. A weight, a residual, a retained rank or an attempt
count moves with the BLAS build, the core count and the LAPACK version, so it
goes through :func:`record` instead. That compares it with a single-threaded
reference and warns past a tolerance, but never fails. It is the rule
``tests/realization/test_sector.py`` applies: a hard-coded weight tests the
eigensolver, an identity tests the repair.

``machine_reference.json`` is one machine's answer, kept so another machine's
can be described against it. It is not a target. Regenerate it single-threaded,
or it records the core count as well as the BLAS; ``tests/conftest.py`` writes
it at session end::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        CAYLEYGW_DRIFT_CAPTURE=1 pytest tests/
"""

from __future__ import annotations

import json
import math
import os
import platform
import warnings
from pathlib import Path
from typing import Any

REFERENCE_PATH = Path(__file__).resolve().parent / "machine_reference.json"
#: Relative gap before a warning; the measured spread makes anything tighter warn every run.
DEFAULT_RELATIVE = 0.5


class MachineDriftWarning(UserWarning):
    """A machine-sensitive quantity departed from its recorded reference."""


def _load() -> dict[str, Any]:
    if not REFERENCE_PATH.exists():
        return {"metadata": {}, "values": {}}
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


_REFERENCE = _load()
_OBSERVED: dict[str, Any] = {}


def machine_description() -> dict[str, str]:
    """Return the identifying facts that make two runs differ."""

    try:
        import numpy as np

        numpy_version = np.__version__
    except Exception:  # pragma: no cover - numpy is a hard dependency
        numpy_version = "?"
    threads = {
        name: os.environ.get(name, "<unset>")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "?",
        "cpu_count": str(os.cpu_count()),
        "numpy": numpy_version,
        **threads,
    }


def record(key: str, value: Any, *, relative: float = DEFAULT_RELATIVE) -> Any:
    """Compare ``value`` with its reference, warn on a material gap, and return it."""

    _OBSERVED[key] = value
    reference = _REFERENCE.get("values", {}).get(key)
    if reference is None:
        warnings.warn(
            f"{key}: no recorded reference (observed {value!r}); "
            "regenerate tests/machine_reference.json to adopt it",
            MachineDriftWarning,
            stacklevel=2,
        )
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if value != reference:
            warnings.warn(
                f"{key}: {value!r} against reference {reference!r}",
                MachineDriftWarning,
                stacklevel=2,
            )
        return value
    if reference == 0.0:
        gap = math.inf if value != 0.0 else 0.0
    else:
        gap = abs(value - reference) / abs(reference)
    if gap > relative:
        warnings.warn(
            f"{key}: {value:.6g} against reference {reference:.6g} "
            f"({gap:.1%} away, tolerated {relative:.0%}) -- this quantity is "
            "machine-sensitive and is reported, not gated",
            MachineDriftWarning,
            stacklevel=2,
        )
    return value


def observed() -> dict[str, Any]:
    """Return everything :func:`record` saw this session."""

    return dict(_OBSERVED)


def write_reference(path: Path | None = None) -> Path:
    """Write everything observed this session as the new reference."""

    target = REFERENCE_PATH if path is None else Path(path)
    payload = {
        "metadata": {
            "note": "One machine's answer, kept so another machine's can be "
            "described relative to it. Not a target.",
            **machine_description(),
        },
        "values": _OBSERVED,
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(json.dumps(machine_description(), indent=2))
