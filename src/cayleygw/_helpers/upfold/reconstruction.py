"""Private helpers of the upfolding functions of :mod:`cayleygw.upfold`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...tools.records.upfold import SectorMomentReconstruction
from ..errors import ValidationError
from ..types import Sector
from ..validate import _check

if TYPE_CHECKING:
    from ...upfold import UpfoldedDysonHamiltonian


def _sector_reconstruction(
    hamiltonian: UpfoldedDysonHamiltonian,
    sector: Sector,
    n_max: int | None,
) -> SectorMomentReconstruction:
    """Rebuild one sector's moments from its poles and couplings in the Hamiltonian."""

    source = hamiltonian.hole if sector is Sector.HOLE else hamiltonian.particle
    supplied = source.closure_scan.moments.values
    if n_max is None:
        order = source.closure_scan.moments.n_max
    else:
        order = _check.nonnegative_integer(n_max, "n_max")
        if order > source.closure_scan.moments.n_max:
            raise ValidationError(
                "n_max exceeds the moments retained by the realization"
            )
    input_moments = supplied[: order + 1]
    nodes = np.asarray(source.mapping.forward(source.poles), dtype=np.complex128)
    powers = np.ones((order + 1, nodes.size), dtype=np.complex128)
    for moment_order in range(order):
        powers[moment_order + 1] = powers[moment_order] * nodes
    reconstructed = np.einsum(
        "pl,kl,ql->kpq",
        source.couplings,
        powers,
        source.couplings.conj(),
        optimize=True,
    )
    absolute = np.linalg.norm(reconstructed - input_moments, axis=(1, 2))
    scales = np.linalg.norm(input_moments, axis=(1, 2))
    relative = absolute / np.maximum(scales, np.finfo(np.float64).eps)
    generic_thresholds = hamiltonian.tolerances.moment_conservation * (1.0 + scales)
    fitted_thresholds = (
        source.selected_closure.realization.physical_moment_acceptance_thresholds(
            base_tolerance=hamiltonian.tolerances.moment_conservation,
        )[: order + 1]
    )
    if fitted_thresholds.size == order + 1:
        thresholds = fitted_thresholds
    else:
        # Orders above a lower n_conserved have no rank-deflation budget, so keep the generic rule.
        thresholds = np.concatenate(
            (fitted_thresholds, generic_thresholds[fitted_thresholds.size:])
        )
    conserved_order = min(source.conserved_order, order)
    return SectorMomentReconstruction(
        sector=sector,
        input_moments=input_moments,
        reconstructed_moments=reconstructed,
        absolute_errors=absolute,
        relative_errors=relative,
        acceptance_thresholds=thresholds,
        conserved_order=conserved_order,
    )
