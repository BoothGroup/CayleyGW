"""Settings, log and sector realizations of :func:`cayleygw.upfold.build_upfolded_hamiltonian`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, NamedTuple

from ...tools.parallel import evaluate_by_index, resolve_native_threads
from ..errors import ValidationError
from ..tolerances import Tolerances
from ..types import Sector
from ..validate import _check

if TYPE_CHECKING:
    from ...moments import G0W0CayleyMoments
    from ...realization.sector import SectorSelfEnergyRealization
    from ...tools.logger import Stages


class _UpfoldSettings(NamedTuple):
    """The checked settings of one realization of both sectors."""

    n_conserved: int
    n_max: int
    phase_count: int
    phase_refinement: str
    tolerances: Tolerances
    threads: int


def _checked_settings(
    moments: G0W0CayleyMoments,
    *,
    n_conserved: Any,
    terminal_selection: Any,
    terminal_phase_count: Any,
    n_workers: Any,
    native_threads: Any,
    realization: Any,
    tolerances: Any,
) -> _UpfoldSettings:
    """Check the realization arguments against the moments and fill in the defaults."""

    if terminal_selection not in ("scan", "restricted"):
        raise ValidationError("terminal_selection must be 'scan' or 'restricted'")
    if realization not in ("auto", "block-cmv", "toeplitz"):
        raise ValidationError("realization must be 'auto', 'block-cmv', or 'toeplitz'")
    if n_conserved is None:
        selected_n_conserved = moments.n_conserved
    else:
        selected_n_conserved = _check.nonnegative_integer(n_conserved, "n_conserved")
        if selected_n_conserved > moments.n_conserved:
            raise ValidationError(
                f"n_conserved={selected_n_conserved} exceeds the "
                f"{moments.n_conserved} conserved orders the moments were "
                "built with"
            )
    # A lower n_conserved keeps one spare: order n_conserved + 1.
    realized_n_max = selected_n_conserved + 1
    if terminal_phase_count is None:
        selected_phase_count = 64
    else:
        selected_phase_count = _check.positive_integer(
            terminal_phase_count,
            "terminal_phase_count",
        )
    phase_refinement = "restricted" if terminal_selection == "restricted" else "fixed"
    selected_tolerances = _check.tolerances(tolerances)
    # Resolved once so the sectors and the later Dyson solve use one count.
    threads = resolve_native_threads(native_threads, n_workers)
    return _UpfoldSettings(
        selected_n_conserved,
        realized_n_max,
        selected_phase_count,
        phase_refinement,
        selected_tolerances,
        threads,
    )


def _log_settings(
    stages: Stages,
    moments: G0W0CayleyMoments,
    settings: _UpfoldSettings,
    *,
    realization: str,
    terminal_selection: str,
    n_workers: int,
) -> None:
    """Log the settings and sizes the realization runs with."""

    (
        selected_n_conserved,
        realized_n_max,
        selected_phase_count,
        phase_refinement,
        selected_tolerances,
        threads,
    ) = settings
    stages.options(
        "build_upfolded_hamiltonian",
        {
            "n_conserved": selected_n_conserved,
            "realization": realization,
            "rank_floor": selected_tolerances.rank_floor,
            "terminal_selection": terminal_selection,
            "terminal_phase_count": selected_phase_count,
            "n_workers": n_workers,
            "native_threads": threads,
        },
        {
            "orbitals": moments.adapter.reference.nmo,
            "moment orders": f"C_0..C_{realized_n_max}",
        },
        {
            "phase_refinement": phase_refinement,
            "moments built": f"C_0..C_{moments.n_max} at N_q={moments.n_q}",
            "tolerances": selected_tolerances,
        },
    )


def _realize_sectors(
    realize: Callable[..., SectorSelfEnergyRealization],
    moments: G0W0CayleyMoments,
    settings: _UpfoldSettings,
    *,
    realization: str,
    n_workers: int,
    verbose: int,
    stages: Stages,
) -> dict[Sector, SectorSelfEnergyRealization]:
    """Realize the hole and particle sectors, concurrently when there is more than one worker."""

    (
        selected_n_conserved,
        realized_n_max,
        selected_phase_count,
        phase_refinement,
        selected_tolerances,
        threads,
    ) = settings
    common = {
        "n_conserved": selected_n_conserved,
        "phase_count": selected_phase_count,
        "phase_refinement": phase_refinement,
        "realization_algorithm": realization,
        "tolerances": selected_tolerances,
        "n_workers": n_workers,
        "native_threads": threads,
    }
    # The sectors are independent; for molecules they are often the only parallel work.
    sector_inputs = (
        (
            Sector.HOLE,
            moments.hole.moments[: realized_n_max + 1],
            moments.mapping,
        ),
        (
            Sector.PARTICLE,
            moments.particle.moments[: realized_n_max + 1],
            moments.mapping,
        ),
    )
    # Each sector's candidate scan gets half the workers, so the total stays at n_workers.
    sector_workers = 2 if n_workers > 1 else 1
    inner_workers = max(1, n_workers // sector_workers)
    sector_common = dict(common)
    sector_common["n_workers"] = inner_workers

    def realize_one(index: int) -> SectorSelfEnergyRealization:
        sector, sector_moments, mapping = sector_inputs[index]
        return realize(
            sector_moments,
            sector,
            mapping,
            verbose=verbose,
            **sector_common,
        )

    with stages.stage(
        "sector realizations",
        n_conserved=selected_n_conserved,
        realization=realization,
    ):
        completed, sector_failures = evaluate_by_index(
            realize_one,
            range(len(sector_inputs)),
            n_workers=sector_workers,
            native_threads=threads,
        )
        if sector_failures:
            # Hole is index 0, so this raises what a serial run would, unchanged.
            raise sector_failures[0][1]
    realized: dict[Sector, SectorSelfEnergyRealization] = {
        sector_inputs[index][0]: completed[index] for index in range(len(sector_inputs))
    }
    return realized
