"""Private helpers of :mod:`cayleygw.realization.sector`."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..._helpers import tolerances as limits
from ..._helpers.cayley import CayleyMap
from ..._helpers.errors import RefusalError, ValidationError
from ..._helpers.tolerances import Tolerances
from ..._helpers.types import Sector
from ..._helpers.validate import BoolArray, ComplexArray, FloatArray, _check
from ...tools.parallel import evaluate_by_index
from ...tools.records.realization import SectorClosureDiagnostics
from .base import _atomic_matrix_moments, _ranking_key, _selection_key

if TYPE_CHECKING:
    from ...tools.logger import Stages
    from ...tools.records.realization import (
        BlockCMVSpectrum,
        MatrixCayleyMoments,
        NormalizedMatrixCayleyMoments,
        SectorClosureScan,
    )
    from ..base import UnitaryMomentRealization


def _floor_text(value: float) -> str:
    """Return a rank floor as a caller would type it, ``1e-9`` not ``1e-09``."""

    return f"{value:g}".replace("e-0", "e-")


def _retry_advice(tolerances: Tolerances) -> str:
    """Return the sentence naming the higher rank floors to retry, in order."""

    # The floors that rescued production refusals, from the lowest.
    ladder = (3.0e-10, 1.0e-9, 2.0e-9, 5.0e-9, 1.0e-8)
    higher = [_floor_text(floor) for floor in ladder if floor > tolerances.rank_floor]
    if not higher:
        return ""
    then = f", then {', '.join(higher[1:])}" if higher[1:] else ""
    return f" Retry with rank_floor={higher[0]}{then}."


def _finalize_first_viable_closure(
    scan: SectorClosureScan,
    sector: Sector,
    mapping: CayleyMap,
    tolerances: Tolerances,
) -> dict[str, Any]:
    """Return the fields of the best-ranked closure of ``scan`` that survives finalization."""

    scan.require_selected()
    # A lower-ranked closure can conserve the moments where the first misses by a few percent.
    first_error: RefusalError | None = None
    result: dict[str, Any] | None = None
    ranked = scan.ranked_indices
    for rank, index in enumerate(ranked):
        try:
            result = _finalize_sector_candidate(scan, index, rank, sector, mapping, tolerances)
        except RefusalError as error:
            if error.kind != "sector":
                raise
            first_error = first_error or error
            continue
        break
    if result is None:
        assert first_error is not None
        raise first_error
    return result


def _conservation_ratio(
    moment_residuals: FloatArray,
    fitted_thresholds: FloatArray,
    n_conserved: int,
) -> float:
    """Return the worst fitted moment residual in units of its own threshold."""

    return float(np.max(moment_residuals[: n_conserved + 1] / fitted_thresholds))


def _zeroth_moment_congruence(
    couplings: ComplexArray,
    normalization: NormalizedMatrixCayleyMoments,
    tolerances: Tolerances,
    diagnostics: SectorClosureScan,
) -> ComplexArray:
    r"""Return the congruence ``T`` that restores the part of ``C[0]`` a discard removed.

    Discarded residues add in phase at order 0, so ``C[0]`` loses their full
    weight; at higher orders they partly cancel. With ``C[0] = R R.H`` and the
    retained couplings ``W``, ``T = R G**-0.5 R+`` for ``G = R+ W W.H R+.H``
    gives ``(T W)(T W).H = R R.H`` exactly. Every residue stays rank one and
    positive, no node moves, and every moment transforms as ``C[n] -> T C[n] T.H``.

    Raises:
        RefusalError: If ``G`` has an eigenvalue further than ``SECTOR_REDISTRIBUTION`` from one.
    """

    factor = normalization.support_factor
    pseudoinverse = normalization.support_pseudoinverse
    compressed = pseudoinverse @ couplings
    gram = compressed @ compressed.conj().T
    gram = 0.5 * (gram + gram.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    deficit = float(np.max(np.abs(eigenvalues - 1.0)))
    if deficit > limits.SECTOR_REDISTRIBUTION:
        raise RefusalError(
            "discarded atoms carry too much of the zeroth moment to "
            f"redistribute (deficit={deficit:.3e}, "
            f"threshold={limits.SECTOR_REDISTRIBUTION:.3g}); increase N_q or omega_p",
            kind="sector",
            diagnostics=diagnostics,
        )
    inverse_root = (eigenvectors / np.sqrt(eigenvalues)[None, :]) @ eigenvectors.conj().T
    return factor @ inverse_root @ pseudoinverse


def _finalize_sector_candidate(
    scan: SectorClosureScan,
    index: int,
    rank: int,
    sector: Sector,
    mapping: CayleyMap,
    tolerances: Tolerances,
) -> dict[str, Any]:
    """Discard one candidate's unsupported atoms and return the fields of what remains.

    Raises a ``"sector"`` :class:`RefusalError` if what remains fails a check,
    so the caller can try the next-ranked candidate.
    """

    selected = scan.candidates[index]
    spectrum = selected.spectrum
    retained_mask = selected.correct_arc_mask & ~selected.near_singular_mask
    discarded_mask = ~retained_mask
    nodes = spectrum.nodes[retained_mask]
    couplings = spectrum.couplings[:, retained_mask]
    if nodes.size:
        poles = np.asarray(mapping.inverse(nodes), dtype=np.float64)
    else:
        poles = np.zeros(0, dtype=np.float64)
    pole_offsets = sector.arc_sign * (poles - mapping.center)
    if np.any(pole_offsets <= 0.0):
        raise RefusalError(
            "inverse-Cayley poles do not lie strictly in the requested sector",
            kind="sector",
            diagnostics=scan,
        )

    reconstructed = _atomic_matrix_moments(
        nodes,
        couplings,
        scan.moments.n_max,
    )
    target = scan.moments.values
    moment_residuals = np.linalg.norm(reconstructed - target, axis=(1, 2))
    thresholds = selected.realization.physical_moment_acceptance_thresholds(
        base_tolerance=tolerances.moment_conservation,
    )
    fitted_thresholds = thresholds[: scan.n_conserved + 1]

    # The congruence fixes order 0 but moves the others; keep it only if the ratio is no worse.
    redistributed_zeroth_weight = 0.0
    zeroth_weight_redistributed = False
    if np.any(discarded_mask) and nodes.size:
        transform = _zeroth_moment_congruence(
            couplings,
            selected.realization.normalization,
            tolerances,
            scan,
        )
        repaired = np.einsum(
            "ap,kpq,bq->kab",
            transform,
            reconstructed,
            transform.conj(),
            optimize=True,
        )
        repaired_residuals = np.linalg.norm(repaired - target, axis=(1, 2))
        redistributed_zeroth_weight = float(
            np.linalg.norm(repaired[0] - reconstructed[0], ord="fro")
        )
        if _conservation_ratio(
            repaired_residuals, fitted_thresholds, scan.n_conserved
        ) <= _conservation_ratio(moment_residuals, fitted_thresholds, scan.n_conserved):
            zeroth_weight_redistributed = True
            couplings = transform @ couplings
            reconstructed = repaired
            moment_residuals = repaired_residuals

    fitted_residuals = moment_residuals[: scan.n_conserved + 1]
    # Residuals in (threshold, margin * threshold] pass but are flagged; a margin of 1 is sharp.
    margin = tolerances.conservation_margin
    failed = np.flatnonzero(fitted_residuals > margin * fitted_thresholds)
    if failed.size:
        failed_order = int(failed[0])
        # Refusal records parse ``order=``, the lowest breached order, so it stays first.
        raise RefusalError(
            "removing unsupported or inverse-unsafe atoms does not conserve "
            f"a guaranteed moment: order={failed_order}, "
            f"residual={moment_residuals[failed_order]:.3e}, "
            f"threshold={thresholds[failed_order]:.3e}, "
            f"margin={margin:.3g}, "
            f"breached={failed.size}, highest={int(failed[-1])} "
            f"(closure rank {rank} of {len(scan.ranked_indices)} acceptable); "
            "increase N_q or lower n_conserved",
            kind="sector",
            diagnostics=scan,
        )
    maximum_conservation_ratio = _conservation_ratio(
        moment_residuals, fitted_thresholds, scan.n_conserved
    )
    weights = spectrum.trace_weights
    discarded_total_weight = float(np.sum(weights[discarded_mask]))
    result = dict(
        sector=sector,
        mapping=mapping,
        closure_scan=scan,
        nodes=_check.readonly_complex(nodes, "sector nodes"),
        poles=_check.readonly_real(poles, "sector poles"),
        couplings=_check.readonly_complex(couplings, "sector couplings"),
        moment_residuals=_check.readonly_real(moment_residuals, "moment_residuals"),
        discarded_wrong_arc_weight=float(np.sum(weights[selected.wrong_arc_mask])),
        discarded_near_singular_weight=float(np.sum(weights[selected.near_singular_mask])),
        discarded_total_weight=discarded_total_weight,
        closure_rank=rank,
        conservation_is_marginal=bool(np.any(fitted_residuals > fitted_thresholds)),
        maximum_conservation_ratio=maximum_conservation_ratio,
        zeroth_weight_redistributed=zeroth_weight_redistributed,
        redistributed_zeroth_weight=redistributed_zeroth_weight,
    )
    return result


def _candidate_diagnostics(
    spectrum: BlockCMVSpectrum,
    phase: float | None,
    sector: Sector,
    source: MatrixCayleyMoments,
    n_conserved: int,
    minimum_node_distance: float,
    tolerances: Tolerances,
) -> SectorClosureDiagnostics:
    """Classify every atom of one candidate's spectrum against the sector gates.

    The dense squares are released before returning, since a scan keeps one
    candidate per phase.
    """

    nodes = spectrum.nodes
    # An atom within roundoff of the real axis counts as wrong-arc.
    correct = sector.arc_sign * nodes.imag > limits.ROUNDOFF_TOLERANCE
    wrong = ~correct
    distances = np.abs(nodes - 1.0)
    near_singular = distances <= minimum_node_distance
    weights = spectrum.trace_weights
    total_weight = spectrum.total_weight
    weight_threshold = limits.ABSOLUTE_TOLERANCE + limits.RELATIVE_TOLERANCE * total_weight

    def selected_weight(mask: BoolArray | NDArray[np.bool_]) -> float:
        return float(np.sum(weights[mask]))

    wrong_weight = selected_weight(wrong)
    near_singular_weight = selected_weight(near_singular)
    significant_correct = correct & (weights > weight_threshold)
    if np.any(significant_correct):
        minimum_distance = float(np.min(distances[significant_correct]))
    elif total_weight <= weight_threshold:
        minimum_distance = math.inf
    else:
        minimum_distance = 0.0
    if nodes.size:
        distance_floor = math.sqrt(np.finfo(np.float64).tiny)
        denominator = np.maximum(distances, distance_floor) ** 2
        inverse_penalty = float(np.sum(weights / denominator))
    else:
        inverse_penalty = 0.0

    if n_conserved < source.n_max:
        predicted = spectrum.realization.matrix_moments(source.n_max)[n_conserved + 1 :]
        target = source.values[n_conserved + 1 :]
        residuals = np.linalg.norm(predicted - target, axis=(1, 2))
        relative = residuals / (1.0 + np.linalg.norm(target, axis=(1, 2)))
    else:
        residuals = np.zeros(0, dtype=np.float64)
        relative = np.zeros(0, dtype=np.float64)
    # The withheld residuals above are the last reader of the dense matrices.
    spectrum = spectrum.without_matrices()
    realization = spectrum.realization
    return SectorClosureDiagnostics(
        phase=phase,
        realization=realization,
        spectrum=spectrum,
        correct_arc_mask=_check.readonly_bool(correct, "correct_arc_mask"),
        wrong_arc_mask=_check.readonly_bool(wrong, "wrong_arc_mask"),
        near_singular_mask=_check.readonly_bool(near_singular, "near_singular_mask"),
        total_weight=total_weight,
        weight_threshold=weight_threshold,
        wrong_arc_weight=wrong_weight,
        near_singular_weight=near_singular_weight,
        minimum_correct_node_distance_from_one=minimum_distance,
        inverse_conditioning_penalty=inverse_penalty,
        withheld_moment_residuals=_check.readonly_real(residuals, "withheld_moment_residuals"),
        withheld_relative_residuals=_check.readonly_real(relative, "withheld_relative_residuals"),
        support_acceptable=(wrong_weight <= weight_threshold),
        # Arc weight moves by tens of percent with the BLAS thread count, hence the band.
        support_marginal=(
            weight_threshold < wrong_weight <= tolerances.arc_margin * weight_threshold
        ),
        inverse_safe=(near_singular_weight <= weight_threshold),
    )


def _candidate_counter(
    stages: Stages,
    scan_progress_label: str,
) -> Callable[[], None]:
    """Return the live candidate count of one scan: each call advances and reports it."""

    scan_completed = 0

    def note_candidate() -> None:
        """Advance the live candidate count by one and report it."""

        nonlocal scan_completed
        scan_completed += 1
        stages.progress(scan_progress_label, scan_completed)

    return note_candidate


def _closure_candidate(
    realization: UnitaryMomentRealization,
    phase: float | None,
    sector: Sector,
    source: MatrixCayleyMoments,
    n_conserved: int,
    minimum_node_distance: float,
    tolerances: Tolerances,
) -> SectorClosureDiagnostics:
    """Extract the spectrum of one closed realization and classify it."""

    return _candidate_diagnostics(
        realization.spectrum(),
        phase,
        sector,
        source,
        n_conserved,
        minimum_node_distance,
        tolerances,
    )


def _phase_evaluator(
    canonical_realization: UnitaryMomentRealization,
    terminal_dimension: int,
    sector: Sector,
    source: MatrixCayleyMoments,
    fitted_order: int,
    minimum_distance: float,
    tolerances: Tolerances,
    *,
    workers: int,
    native: int,
    note_candidate: Callable[[], None],
    realization_algorithm: str,
) -> Callable[[Sequence[float]], list[SectorClosureDiagnostics]]:
    """Return a function that realizes and classifies a batch of closure phases."""

    identity = np.eye(terminal_dimension, dtype=np.complex128)

    def evaluate_phase(phase: float) -> SectorClosureDiagnostics:
        """Build and classify one closure; ``reclose`` never writes the shared cache."""

        terminal = np.exp(1.0j * phase) * identity
        realization = canonical_realization.reclose(
            terminal,
            tolerances=tolerances,
        )
        return _closure_candidate(
            realization,
            phase,
            sector,
            source,
            fitted_order,
            minimum_distance,
            tolerances,
        )

    def evaluate_phases(
        phases: Sequence[float],
    ) -> list[SectorClosureDiagnostics]:
        """Evaluate a batch of phases, returned in input order whatever the worker count."""

        if not phases:
            return []
        evaluated, failures = evaluate_by_index(
            lambda index: evaluate_phase(phases[index]),
            range(len(phases)),
            n_workers=workers,
            native_threads=native,
            on_complete=lambda _index: note_candidate(),
        )
        if failures:
            index, error = failures[0]
            if isinstance(error, RefusalError) and error.kind == "block-cmv":
                # Re-raised as a "sector" refusal so the "auto" fallback catches it.
                raise RefusalError(
                    f"the {realization_algorithm} closure at phase "
                    f"{float(phases[index]):.6f} failed its realization "
                    f"contracts: {str(error).rstrip('.')}",
                    kind="sector",
                    diagnostics=error,
                ) from error
            raise error
        return [evaluated[index] for index in range(len(phases))]

    return evaluate_phases


def _restricted_phase_search(
    evaluate_phases: Callable[[Sequence[float]], list[SectorClosureDiagnostics]],
    canonical_candidate: SectorClosureDiagnostics,
    source: MatrixCayleyMoments,
    fitted_order: int,
    count: int,
) -> list[SectorClosureDiagnostics]:
    """Realize only the phases the reconstructed withheld objective cannot rule out.

    Returns every candidate realized, the identity closure first and the rest
    in phase order.
    """

    if fitted_order >= source.n_max:
        raise ValidationError(
            "phase_refinement='restricted' needs a withheld moment above "
            "n_conserved; lower n_conserved or use phase_refinement='fixed'"
        )
    denominators = 1.0 + np.linalg.norm(
        np.asarray(source.values)[fitted_order + 1 : source.n_max + 1],
        axis=(1, 2),
    )
    grid = [2.0 * np.pi * index / count for index in range(count)]
    seen: dict[int, SectorClosureDiagnostics] = {0: canonical_candidate}

    def take(wanted: Any) -> list[SectorClosureDiagnostics]:
        """Realize the requested grid indices, reusing what exists."""

        indices = list(wanted)
        fresh = [i for i in sorted(set(indices)) if i not in seen]
        if fresh:
            for index, candidate in zip(fresh, evaluate_phases([grid[i] for i in fresh])):
                seen[index] = candidate
        return [seen[i] for i in indices]

    def objective_of(candidate: SectorClosureDiagnostics) -> float:
        return float(_selection_key(candidate)[0])

    def fit(step_size: int):
        """Return the objective predicted from the sublattice of stride ``step_size``."""

        ordered = [seen[i] for i in range(0, count, step_size)]
        squares = np.asarray(
            [np.asarray(item.withheld_moment_residuals) ** 2 for item in ordered],
            dtype=np.float64,
        )
        spectra = np.fft.fft(squares, axis=0) / len(ordered)
        frequencies = np.fft.fftfreq(len(ordered), d=1.0 / len(ordered))

        def predict(thetas: Any) -> FloatArray:
            basis = np.exp(1.0j * np.outer(np.asarray(thetas, dtype=np.float64), frequencies))
            return np.max(
                np.sqrt(np.maximum(np.real(basis @ spectra), 0.0)) / denominators,
                axis=1,
            )

        return predict

    # Coarsest sublattice that fits the degree, refined until it predicts the phases it skipped.
    raw_degree = max(1, source.n_max - fitted_order)
    stride = count
    while stride > 1 and count // stride < 2 * raw_degree + 1:
        stride //= 2
    take(range(0, count, stride))
    predict = fit(stride)
    while stride > 1:
        checks = list(range(stride // 2, count, stride))[:3]
        actual = take(checks)
        forecast = predict([grid[i] for i in checks])
        scale = max([objective_of(item) for item in seen.values()] + [np.finfo(np.float64).tiny])
        if max(abs(float(a) - objective_of(b)) for a, b in zip(forecast, actual)) <= 1.0e-6 * scale:
            break
        # Aliased: complete the finer level and reconstruct on it.
        stride //= 2
        take(range(0, count, stride))
        predict = fit(stride)

    # Walk the grid by predicted objective and stop at the best feasible value; the fit is exact.
    best_value = math.inf
    for item in seen.values():
        if item.acceptable:
            best_value = min(best_value, objective_of(item))
    # Bound each interval by its floor, not its grid value: the optimum can lie between points.
    dense = np.linspace(0.0, 2.0 * np.pi, max(4096, 64 * count), endpoint=False)
    dense_values = predict(dense)
    owner = np.round(dense * count / (2.0 * np.pi)).astype(int) % count
    cell_floor = np.full(count, np.inf, dtype=np.float64)
    np.minimum.at(cell_floor, owner, dense_values)
    pending = sorted(
        (index for index in range(count) if index not in seen),
        key=lambda i: cell_floor[i],
    )
    for index in pending:
        if cell_floor[index] >= best_value:
            break
        item = seen[index] = evaluate_phases([grid[index]])[0]
        if item.acceptable:
            best_value = min(best_value, objective_of(item))

    extra: list[SectorClosureDiagnostics] = []

    def arc_edge(outside: float, inside: float) -> None:
        """Bisect one feasible/infeasible transition."""

        for _ in range(limits.PHASE_REFINE_STEPS):
            middle = 0.5 * (outside + inside)
            candidate = evaluate_phases([float(middle)])[0]
            extra.append(candidate)
            if candidate.acceptable:
                inside = middle
            else:
                outside = middle

    feasible = [i for i, item in seen.items() if item.acceptable]
    if feasible:
        # Walk only toward the exact minimizer; the arc's far edge cannot hold the optimum.
        span = 2.0 * np.pi / count
        anchor = min(feasible, key=lambda i: objective_of(seen[i]))
        centre = grid[anchor]
        target = float(dense[int(np.argmin(dense_values))])
        offset = (target - centre + np.pi) % (2.0 * np.pi) - np.pi
        direction = 1 if offset >= 0.0 else -1
        inside, steps, blocked = centre, 0, False
        limit = int(abs(offset) / span)
        while steps < limit:
            index = (anchor + direction * (steps + 1)) % count
            item = seen.get(index)
            if item is None:
                item = seen[index] = evaluate_phases([grid[index]])[0]
            if not item.acceptable:
                blocked = True
                break
            steps += 1
            inside = centre + direction * steps * span
        if blocked:
            arc_edge(inside + direction * span, inside)
        else:
            probe = evaluate_phases([target])[0]
            extra.append(probe)
            if not probe.acceptable:
                # The common case: the optimum is the arc's edge toward the minimizer.
                arc_edge(target, inside)

    candidates = [seen[0]] + sorted(
        [item for index, item in seen.items() if index] + extra,
        key=lambda item: 0.0 if item.phase is None else float(item.phase),
    )
    return candidates


def _selected_candidate(
    candidates: list[SectorClosureDiagnostics],
) -> int | None:
    """Return the index of the best-ranked acceptable candidate, or ``None``."""

    feasible = [index for index, candidate in enumerate(candidates) if candidate.acceptable]
    if feasible:
        selected_index = min(
            feasible,
            key=lambda index: _ranking_key(candidates[index]),
        )
    else:
        selected_index = None
    return selected_index
