"""The automatic ``N_q`` ladder of :mod:`cayleygw.moments`."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ...tools.logger import Stages
from ...tools.records.contour import ContourSectorMoments
from ...tools.records.moments import (
    AutomaticNQDiagnostics,
    AutomaticNQSectorDiagnostics,
)
from ..errors import ValidationError
from ..types import Sector
from ..validate import FloatArray
from .feasibility import feasibility_violations

LOGGER = logging.getLogger("cayleygw.moments")

#: First and largest complete-contour node counts of the automatic ladder.
_N_Q_INITIAL = 128
_N_Q_MAXIMUM = 4096


def _moment_errors(
    coarse: NDArray[np.complex128],
    fine: NDArray[np.complex128],
    tolerance: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return one sector's absolute errors, relative errors and thresholds per order."""

    difference = np.linalg.norm(fine - coarse, axis=(1, 2))
    fine_norms = np.linalg.norm(fine, axis=(1, 2))
    relative = np.divide(
        difference,
        fine_norms,
        out=np.zeros_like(difference),
        where=fine_norms > 0.0,
    )
    return difference, relative, tolerance * fine_norms


def _automatic_n_q_moments(
    evaluate: Callable[[int, str], dict[Sector, ContourSectorMoments]],
    n_max: int,
    *,
    tolerance: float,
    verbose: int,
) -> tuple[dict[Sector, ContourSectorMoments], AutomaticNQDiagnostics]:
    """Double ``N_q`` from 128 until the moments converge and pass the feasibility checks.

    ``evaluate(n_points, rule)`` builds both sectors' moments. The ladder uses
    the trapezoidal rule, whose nodes nest: the ``2N`` grid is the ``N`` grid and
    its midpoint nodes, so each doubling solves only those and averages,
    ``T_2N = (T_N + M_N) / 2``. The feasibility checks run on the fine moments
    only once both sectors meet the error tolerance, and only for ``n_max >= 1``.
    """

    stages = Stages(verbose, LOGGER)

    def build(n_q: int, n_points: int, rule: str) -> dict[Sector, ContourSectorMoments]:
        with stages.stage("automatic N_q", N_q=n_q, n_max=n_max):
            return evaluate(n_points, rule)

    def refined(
        coarse: dict[Sector, ContourSectorMoments],
        added: dict[Sector, ContourSectorMoments],
    ) -> dict[Sector, ContourSectorMoments]:
        result = {}
        for sector in Sector:
            moments = 0.5 * (coarse[sector].moments + added[sector].moments)
            moments.setflags(write=False)
            result[sector] = ContourSectorMoments(moments=moments)
        return result

    def violations(
        fine: dict[Sector, ContourSectorMoments],
        sector: Sector,
    ) -> tuple[str, ...]:
        with stages.stage(f"automatic N_q feasibility ({sector.name.lower()})"):
            return feasibility_violations(fine[sector].moments, sector)

    coarse_n_q = _N_Q_INITIAL
    coarse = build(coarse_n_q, coarse_n_q, "trapezoid")
    refinements: list[
        tuple[AutomaticNQSectorDiagnostics, AutomaticNQSectorDiagnostics]
    ] = []
    while 2 * coarse_n_q <= _N_Q_MAXIMUM:
        fine_n_q = 2 * coarse_n_q
        fine = refined(coarse, build(fine_n_q, coarse_n_q, "midpoint"))
        errors = {
            sector: _moment_errors(
                coarse[sector].moments, fine[sector].moments, tolerance
            )
            for sector in Sector
        }
        check_feasibility = n_max >= 1 and all(
            bool(np.all(absolute <= thresholds))
            for absolute, _, thresholds in errors.values()
        )
        hole, particle = (
            AutomaticNQSectorDiagnostics(
                sector,
                coarse_n_q,
                fine_n_q,
                *errors[sector],
                feasibility_violations=(
                    violations(fine, sector) if check_feasibility else ()
                ),
            )
            for sector in Sector
        )
        refinements.append((hole, particle))
        error = max(hole.maximum_normalized_error, particle.maximum_normalized_error)
        converged = hole.converged and particle.converged
        if stages.verbose >= 1:
            LOGGER.info(
                "Automatic N_q refinement %d -> %d: maximum normalized moment "
                "error %.3e; converged=%s",
                coarse_n_q,
                fine_n_q,
                error,
                converged,
                extra={
                    "cayleygw": {
                        "event": "n_q",
                        "coarse": coarse_n_q,
                        "fine": fine_n_q,
                        "error": error,
                        "converged": converged,
                    }
                },
            )
        if converged:
            return fine, AutomaticNQDiagnostics(
                initial_n_q=_N_Q_INITIAL,
                maximum_n_q=_N_Q_MAXIMUM,
                tolerance=tolerance,
                refinements=tuple(refinements),
            )
        coarse_n_q = fine_n_q
        coarse = fine

    candidates = refinements[-1]
    worst = max(candidates, key=lambda item: item.maximum_normalized_error)
    reason = (
        f"worst moment={worst.sector.name.lower()} C_{worst.worst_order}, "
        f"normalized error={worst.maximum_normalized_error:.3e}"
    )
    feasibility = tuple(
        f"{item.sector.name.lower()}: {', '.join(item.feasibility_violations)}"
        for item in candidates
        if item.feasibility_violations
    )
    if feasibility:
        reason += "; feasibility violations=" + " | ".join(feasibility)
    raise ValidationError(
        "automatic n_q selection did not converge: "
        f"initial_n_q={_N_Q_INITIAL}, maximum_n_q={_N_Q_MAXIMUM}, "
        f"tolerance={tolerance:.3e}; {reason}. Loosen n_q_tolerance, or "
        "converge omega_p first."
    )
