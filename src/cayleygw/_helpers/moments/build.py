"""Argument checks and option logging of :meth:`G0W0CayleyMoments.build`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from ..errors import ValidationError
from ..types import Screening
from ..validate import _check

if TYPE_CHECKING:
    from ...tools.logger import Stages
    from ..pyscf import RestrictedPySCFAdapter


class _BuildOptions(NamedTuple):
    """The checked arguments of one moment build."""

    conserved: int
    order: int
    automatic_n_q: bool
    count: int | None
    n_q_rtol: float
    verbose: int
    omega_p: float
    orbital_block_size: int
    screening: Screening


def _checked_options(
    *,
    n_conserved: Any,
    n_q: Any,
    n_q_tolerance: Any,
    verbose: Any,
    omega_p: Any,
    spectral_bound: Any,
    contour_orbital_block_size: Any,
    screening: Any,
) -> _BuildOptions:
    """Check the arguments of a moment build and raise on the first failure."""

    conserved = _check.nonnegative_integer(n_conserved, "n_conserved")
    order = conserved + 1
    automatic_n_q = n_q == "auto"
    if isinstance(n_q, str) and not automatic_n_q:
        raise ValidationError(f"n_q must be a positive integer or 'auto'; got {n_q!r}")
    if automatic_n_q:
        n_q_rtol = _check.positive_real(n_q_tolerance, "n_q_tolerance")
        count = None
    else:
        count = _check.positive_integer(n_q, "n_q", minimum=2)
        if count % 2:
            raise ValidationError(f"n_q must be even; got {count}")
        n_q_rtol = 0.0
    level = _check.nonnegative_integer(verbose, "verbose")
    selected_omega_p = _check.positive_real(omega_p, "omega_p")
    if spectral_bound not in ("frobenius", "spectral"):
        raise ValidationError("spectral_bound must be 'frobenius' or 'spectral'")
    selected_orbital_block_size = _check.positive_integer(
        contour_orbital_block_size,
        "contour_orbital_block_size",
    )
    try:
        selected_screening = Screening(screening)
    except ValueError as error:
        raise ValidationError(
            "screening must be Screening.RPA, Screening.TDA, 'rpa', or 'tda'"
        ) from error
    return _BuildOptions(
        conserved,
        order,
        automatic_n_q,
        count,
        n_q_rtol,
        level,
        selected_omega_p,
        selected_orbital_block_size,
        selected_screening,
    )


def _log_options(
    stages: Stages,
    mean_field: Any,
    adapter: RestrictedPySCFAdapter,
    options: _BuildOptions,
    *,
    spectral_bound: str,
    n_workers: int,
    native_threads: int,
) -> None:
    """Log the settings and sizes a moment build runs with."""

    (
        conserved,
        _,
        automatic_n_q,
        count,
        n_q_rtol,
        _,
        selected_omega_p,
        selected_orbital_block_size,
        selected_screening,
    ) = options
    reference = adapter.reference
    stages.options(
        "build_cayley_moments",
        {
            "n_conserved": conserved,
            "n_q": "auto" if automatic_n_q else count,
            **({"n_q_tolerance": n_q_rtol} if automatic_n_q else {}),
            "omega_p": selected_omega_p,
            "screening": selected_screening.label,
            "n_workers": n_workers,
            "native_threads": native_threads,
        },
        {
            "orbitals": reference.nmo,
            "occupied": reference.nocc,
            "virtual": reference.nvir,
            **({"frozen": len(adapter.frozen_indices)} if adapter.frozen_indices else {}),
            "transitions": reference.nocc * reference.nvir,
        },
        {
            "reference": type(mean_field).__name__
            + (f" ({mean_field.xc})" if hasattr(mean_field, "xc") else ""),
            "frozen orbitals": adapter.frozen_indices,
            "spectral_bound": spectral_bound,
            "contour_orbital_block_size": selected_orbital_block_size,
        },
    )
