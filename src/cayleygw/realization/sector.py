r"""One sector's self-energy: the terminal closure and the pole sum.

A backend realizes the sector's moments as a unitary, a scalar terminal phase
closes it, and its nodes :math:`u_\ell` go back through the inverse Cayley map
to real poles :math:`d_\ell`. With the couplings :math:`w_\ell` the sector
self-energy is

.. math::

   \Sigma^{\lessgtr}(z) = \sum_\ell \frac{w_\ell w_\ell^\dagger}{z - d_\ell} .

The terminal block :math:`\beta = e^{i\phi} I_t` keeps every fitted moment
but sets the higher ones and so the poles. The scan samples :math:`\phi` and
keeps the best closure that passes the arc and weight gates. Atoms off the
sector's semicircle or near :math:`u = 1` are discarded, the fitted moments
are checked again, and a congruence of the couplings can restore
:math:`C^{(0)}` without moving a pole.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np

from .._helpers import tolerances as limits
from .._helpers.cayley import CayleyMap
from .._helpers.errors import RefusalError, ValidationError
from .._helpers.tolerances import DEFAULT_TOLERANCES, Tolerances
from .._helpers.types import Sector
from .._helpers.validate import ComplexArray, FloatArray, _check
from ..tools.logger import Stages
from ..tools.records.realization import (
    MatrixCayleyMoments,
    SectorClosureDiagnostics,
    SectorClosureScan,
)
from ._helpers.base import _AUTOMATIC_BACKEND_ORDER
from ._helpers.sector import (
    _candidate_counter,
    _closure_candidate,
    _finalize_first_viable_closure,
    _floor_text,
    _phase_evaluator,
    _restricted_phase_search,
    _retry_advice,
    _selected_candidate,
)
from .base import PhaseRefinement, RealizationAlgorithm
from .block_cmv import BlockCMVRealization
from .toeplitz import ToeplitzRealization

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SectorSelfEnergyRealization:
    r"""One sector's self-energy as a sum of positive poles.

    Retained nodes lie on the sector's open semicircle and map to finite poles
    strictly on the sector's side of the Cayley centre. Residues are stored as
    coupling columns, so they are positive semidefinite by construction. The
    two ``*_is_marginal`` flags must be kept in any record of the result.

    Attributes:
        sector: Hole or particle sector.
        mapping: Cayley map between nodes and poles.
        closure_scan: Every closure candidate the scan built, and its ranking.
        nodes: Retained unit-circle nodes :math:`u_\ell`.
        poles: Real poles :math:`d_\ell` in Hartree.
        couplings: One coupling column :math:`w_\ell` per pole.
        moment_residuals: Frobenius residual of each supplied order after the discard.
        discarded_wrong_arc_weight: Trace weight of the atoms off the sector's open semicircle.
        discarded_near_singular_weight: Trace weight of the atoms near :math:`u = 1`.
        discarded_total_weight: Trace weight of every discarded atom.
        closure_rank: Rank of the closure used; nonzero when better closures failed conservation.
        conservation_is_marginal: Whether a fitted residual passed only inside the margin band.
        maximum_conservation_ratio: Largest fitted residual over its unmargined threshold.
        zeroth_weight_redistributed: Whether the couplings carry the congruence restoring ``C[0]``.
        redistributed_zeroth_weight: Size of the congruence's change to ``C[0]``, applied or not.
    """

    sector: Sector
    mapping: CayleyMap
    closure_scan: SectorClosureScan
    nodes: ComplexArray = field(repr=False)
    poles: FloatArray = field(repr=False)
    couplings: ComplexArray = field(repr=False)
    moment_residuals: FloatArray = field(repr=False)
    discarded_wrong_arc_weight: float
    discarded_near_singular_weight: float
    discarded_total_weight: float
    closure_rank: int = 0
    conservation_is_marginal: bool = False
    maximum_conservation_ratio: float = 0.0
    zeroth_weight_redistributed: bool = False
    redistributed_zeroth_weight: float = 0.0

    @property
    def support_is_marginal(self) -> bool:
        """Whether the closure used passed its arc-weight gate only inside the margin band.

        Such a closure is admitted only because the post-discard conservation
        check passed.
        """

        return bool(self.selected_closure.support_marginal)

    @property
    def conserved_order(self) -> int:
        """Highest moment order used to build the selected closure."""

        return self.closure_scan.n_conserved

    @property
    def selected_closure(self) -> SectorClosureDiagnostics:
        """Diagnostics of the closure these poles came from, at :attr:`closure_rank`."""

        ranked = self.closure_scan.ranked_indices
        if not ranked:
            return self.closure_scan.require_selected()
        return self.closure_scan.candidates[ranked[self.closure_rank]]

    @property
    def terminal_dimension(self) -> int:
        """Size ``t`` of the free terminal block, or zero if the moments fix the closure.

        The scalar arc weight has ``t`` local minima, so a grid of fewer than
        ``t`` phases can miss one.
        """

        return self.closure_scan.terminal_dimension

    @property
    def selected_phase(self) -> float | None:
        """Terminal phase of the closure used, or ``None`` if the moments fix it."""

        return self.selected_closure.phase

    @property
    def maximum_conserved_moment_residual(self) -> float:
        """Largest moment residual over the conserved orders ``0..n_conserved``."""

        return float(np.max(self.moment_residuals[: self.conserved_order + 1]))

    @classmethod
    def realize(
        cls,
        moments: MatrixCayleyMoments | Any,
        sector: Sector,
        mapping: CayleyMap,
        *,
        n_conserved: int | None = None,
        phase_count: int = 32,
        n_workers: int = 1,
        native_threads: int = 1,
        phase_refinement: PhaseRefinement = "fixed",
        realization_algorithm: RealizationAlgorithm = "auto",
        tolerances: Tolerances = DEFAULT_TOLERANCES,
        verbose: int = 0,
    ) -> SectorSelfEnergyRealization:
        r"""Realize one sector's moments as a checked sum of poles.

        Scans the terminal phases, takes the best-ranked closure that still
        conserves the fitted moments once its unsupported atoms are discarded,
        and maps the retained nodes to poles. ``"auto"`` tries block-CMV first
        and Toeplitz if block-CMV refuses; the two can pick different closures.

        Args:
            moments: Cayley moments ``C[0..n_max]`` from any theory.
            sector: Hole or particle sector.
            mapping: Cayley map between nodes and real poles.
            n_conserved: Highest order the realization conserves; higher orders only rank
                closures. Every supplied order by default.
            phase_count: Equally spaced terminal phases on the grid.
            n_workers: Python workers over closure candidates.
            native_threads: BLAS and LAPACK threads per worker.
            phase_refinement: ``"fixed"`` or ``"restricted"``, as in :meth:`scan_closures`.
            realization_algorithm: ``"auto"``, ``"block-cmv"`` or ``"toeplitz"``.
            tolerances: Numerical thresholds; ``rank_floor`` sets the rank cuts.
            verbose: ``0`` silent, ``1`` the stages, ``2`` adds progress counts.

        Returns:
            The poles and couplings with every closure, discard and moment diagnostic.

        Raises:
            ValidationError: If an argument has the wrong type or value.
            RefusalError: If no closure passes the gates and conserves the fitted
                moments; the message names the failed check and the rank floors to retry.
        """

        if not isinstance(sector, Sector):
            raise ValidationError("sector must be a Sector value")
        if not isinstance(mapping, CayleyMap):
            raise ValidationError("mapping must be a CayleyMap object")
        if realization_algorithm not in ("auto", "toeplitz", "block-cmv"):
            raise ValidationError(
                "realization_algorithm must be 'auto', 'toeplitz', or 'block-cmv'"
            )
        _check.tolerances(tolerances)
        stages = Stages(verbose, LOGGER)

        # Fall back after finalization: a closure can pass every gate and fail the later checks.
        backends: tuple[RealizationAlgorithm, ...] = cast(
            "tuple[RealizationAlgorithm, ...]",
            _AUTOMATIC_BACKEND_ORDER
            if realization_algorithm == "auto"
            else (realization_algorithm,),
        )
        # If every backend fails, report a finalization error, which carries the scan.
        scan_error: RefusalError | None = None
        construction_error: RefusalError | None = None
        result: SectorSelfEnergyRealization | None = None
        with stages.stage(
            f"{sector.name.lower()} realization",
            rank_floor=f"{tolerances.rank_floor:.1e}",
            n_conserved=n_conserved,
        ):
            for backend in backends:
                try:
                    scan = cls.scan_closures(
                        moments,
                        sector,
                        n_conserved=n_conserved,
                        phase_count=phase_count,
                        n_workers=n_workers,
                        native_threads=native_threads,
                        phase_refinement=phase_refinement,
                        realization_algorithm=backend,
                        tolerances=tolerances,
                        verbose=verbose,
                    )
                except RefusalError as error:
                    if error.kind != "sector":
                        raise
                    construction_error = error
                    continue
                if realization_algorithm == "auto":
                    scan = replace(scan, requested_realization_algorithm=realization_algorithm)
                try:
                    with stages.stage(f"{sector.name.lower()} poles and couplings"):
                        result = cls(
                            **_finalize_first_viable_closure(scan, sector, mapping, tolerances)
                        )
                except RefusalError as error:
                    if error.kind != "sector":
                        raise
                    scan_error = scan_error or error
                    continue
                break
        if result is None:
            error = scan_error or construction_error
            assert error is not None
            raise RefusalError(
                f"{sector.name.lower()} sector refused at rank_floor="
                f"{_floor_text(tolerances.rank_floor)}: "
                f"{str(error).rstrip('.')}.{_retry_advice(tolerances)}",
                kind="sector",
                diagnostics=error.diagnostics,
            ) from error
        return result

    @classmethod
    def scan_closures(
        cls,
        moments: MatrixCayleyMoments | Any,
        sector: Sector,
        *,
        n_conserved: int | None = None,
        phase_count: int = 32,
        n_workers: int = 1,
        native_threads: int = 1,
        phase_refinement: PhaseRefinement = "fixed",
        realization_algorithm: RealizationAlgorithm = "block-cmv",
        tolerances: Tolerances = DEFAULT_TOLERANCES,
        verbose: int = 0,
    ) -> SectorClosureScan:
        r"""Build and rank the closure candidates of one sector.

        The identity closure comes first and is the only candidate when the
        moments fix the terminal block. Otherwise the candidates are equally
        spaced phases in :math:`[0, 2\pi)`: all ``phase_count`` of them under
        ``"fixed"``, or under ``"restricted"`` only those the withheld residual
        cannot rule out.

        A candidate is acceptable if its near-singular weight is at most the
        weight threshold and its wrong-arc weight, which counts atoms within
        roundoff of the real axis, at most ``arc_margin`` times
        it. Clean candidates rank before marginal ones, and marginal ones by
        that weight. Then, with withheld moments, the largest scaled withheld
        residual ranks them and the nearest node's distance from :math:`u = 1`
        breaks ties; without them the largest distance wins.

        Args:
            moments: Cayley moments ``C[0..n_max]`` from any theory.
            sector: Hole or particle sector.
            n_conserved: Highest order the realization conserves; higher orders only rank
                closures. Every supplied order by default.
            phase_count: Equally spaced terminal phases on the grid.
            n_workers: Python workers over closure candidates.
            native_threads: BLAS and LAPACK threads per worker.
            phase_refinement: ``"fixed"`` realizes every grid phase. ``"restricted"``
                fits each squared withheld residual, a trigonometric polynomial in
                the phase, on a sublattice of the grid, realizes only the phases
                whose predicted floor beats the best feasible value, and bisects
                the feasible arc's edge toward the minimum. It needs a moment above
                ``n_conserved`` and is never worse than ``"fixed"``.
            realization_algorithm: ``"block-cmv"`` or ``"toeplitz"``.
            tolerances: Numerical thresholds; ``rank_floor`` sets the rank cuts.
            verbose: ``0`` silent, ``1`` the stages, ``2`` adds progress counts.

        Returns:
            Every candidate, identity first and then in phase order, with the
            index of the best acceptable one or ``None``.

        Raises:
            ValidationError: If an argument has the wrong type or value.
            RefusalError: If the backend cannot realize the fitted moments.
        """

        workers = _check.positive_integer(n_workers, "n_workers")
        native = _check.positive_integer(native_threads, "native_threads")
        count = _check.positive_integer(phase_count, "phase_count")
        _check.tolerances(tolerances)
        stages = Stages(_check.nonnegative_integer(verbose, "verbose"), LOGGER)
        if not isinstance(sector, Sector):
            raise ValidationError("sector must be a Sector value")
        if realization_algorithm not in ("toeplitz", "block-cmv"):
            raise ValidationError("realization_algorithm must be 'toeplitz' or 'block-cmv'")
        if phase_refinement not in ("fixed", "restricted"):
            raise ValidationError("phase_refinement must be 'fixed' or 'restricted'")
        source = (
            moments if isinstance(moments, MatrixCayleyMoments) else MatrixCayleyMoments(moments)
        )
        if n_conserved is None:
            fitted_order = source.n_max
        else:
            fitted_order = _check.nonnegative_integer(n_conserved, "n_conserved")
            if fitted_order > source.n_max:
                raise ValidationError("n_conserved cannot exceed the highest supplied moment order")
        # The exclusion radius around the inverse-map singularity ``u = 1``.
        minimum_distance = limits.ABSOLUTE_TOLERANCE
        fitted = MatrixCayleyMoments(source.values[: fitted_order + 1])
        try:
            cmv_context = stages.stage(
                f"{sector.name.lower()} {realization_algorithm} realization",
                n_conserved=fitted_order,
                rank=int(source.values.shape[1]),
            )
            with cmv_context:
                if realization_algorithm == "toeplitz":
                    canonical_realization = ToeplitzRealization.realize(
                        fitted,
                        tolerances=tolerances,
                        gram_native_threads=native,
                    )
                else:
                    # Only the terminal block depends on the closure; candidates reuse the rest.
                    canonical_realization = BlockCMVRealization.realize(
                        fitted,
                        tolerances=tolerances,
                    )
        except RefusalError as error:
            if error.kind != "block-cmv":
                raise
            raise RefusalError(
                f"{realization_algorithm} construction failed fitting C_0 through "
                f"C_{fitted_order}: {error}",
                kind="sector",
                diagnostics=error,
            ) from error
        canonical_phase = None if canonical_realization.terminal_from_moments else 0.0
        # One stage for the whole scan: per-candidate timings of concurrent work would overcount.
        scan_progress_label = f"{sector.name.lower()} terminal candidates"
        note_candidate = _candidate_counter(stages, scan_progress_label)

        with stages.stage(
            scan_progress_label,
            phase_count=phase_count,
            refinement=phase_refinement,
        ):
            canonical_candidate = _closure_candidate(
                canonical_realization,
                canonical_phase,
                sector,
                source,
                fitted_order,
                minimum_distance,
                tolerances,
            )
            note_candidate()
            if canonical_realization.terminal_from_moments:
                # The moments fix the terminal block, so there is no phase to search.
                scan_terminal_dimension = 0
                candidates = [canonical_candidate]
            else:
                terminal_dimension = canonical_realization.terminal_unitary.shape[0]
                scan_terminal_dimension = int(terminal_dimension)
                evaluate_phases = _phase_evaluator(
                    canonical_realization,
                    terminal_dimension,
                    sector,
                    source,
                    fitted_order,
                    minimum_distance,
                    tolerances,
                    workers=workers,
                    native=native,
                    note_candidate=note_candidate,
                    realization_algorithm=realization_algorithm,
                )

                if phase_refinement == "fixed":
                    # The identity closure is grid phase 0, so only phases 1 onward are evaluated.
                    candidates = [canonical_candidate]
                    candidates.extend(
                        evaluate_phases([2.0 * np.pi * index / count for index in range(1, count)])
                    )
                else:
                    candidates = _restricted_phase_search(
                        evaluate_phases,
                        canonical_candidate,
                        source,
                        fitted_order,
                        count,
                    )
        selected_index = _selected_candidate(candidates)
        return SectorClosureScan(
            sector=sector,
            requested_realization_algorithm=realization_algorithm,
            realization_algorithm=realization_algorithm,
            moments=source,
            n_conserved=fitted_order,
            candidates=tuple(candidates),
            selected_index=selected_index,
            terminal_dimension=scan_terminal_dimension,
        )


__all__ = [
    "SectorSelfEnergyRealization",
    "SectorClosureDiagnostics",
    "SectorClosureScan",
]
