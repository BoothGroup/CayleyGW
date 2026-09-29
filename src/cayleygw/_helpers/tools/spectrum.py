"""Private helpers of :mod:`cayleygw.tools.spectrum`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import tolerances as limits
from ..errors import ValidationError
from ..validate import _check

if TYPE_CHECKING:
    from ...upfold import UpfoldedDysonHamiltonian


def _chemical_potential_from_problem(
    problem: UpfoldedDysonHamiltonian,
) -> float:
    """Return the Cayley-map center both realized sectors share, as the chemical potential."""

    centers: list[float] = []
    for sector in (problem.hole, problem.particle):
        mapping = getattr(sector, "mapping", None)
        center = getattr(mapping, "center", None)
        if center is not None:
            centers.append(_check.finite_real(center, "Cayley-map center"))
    if len(centers) != 2:
        raise ValidationError(
            "chemical_potential is required when the sectors carry no Cayley map"
        )
    scale = max(abs(centers[0]), abs(centers[1]))
    threshold = limits.ABSOLUTE_TOLERANCE + limits.RELATIVE_TOLERANCE * scale
    if abs(centers[0] - centers[1]) > threshold:
        raise ValidationError(
            "hole and particle realizations have different Cayley centers"
        )
    return 0.5 * (centers[0] + centers[1])
