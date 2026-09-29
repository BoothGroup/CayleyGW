"""Numerical thresholds: the safeguards a caller may set and the internal constants.

The constants after :class:`Tolerances` are internal; readers look them up on
this module, so one patch here reaches all.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from .errors import ValidationError


@dataclass(frozen=True, slots=True)
class Tolerances:
    """The safeguards of the realization: thresholds that decide whether a result is trusted.

    Loosening one can admit a wrong answer and tightening one can refuse a good
    one, so change them to study a refusal, not to silence it. A margin passes
    a value just above its threshold but flags it, because a decision within a
    few percent of a bare threshold flips with the BLAS reduction order; a
    margin of ``1.0`` disables the band.

    Attributes:
        rank_floor: Rank cut of the block-Toeplitz Gram and the block-CMV defect matrices.
        moment_conservation: Absolute-plus-relative tolerance on conserved moments after discards.
        conservation_margin: Multiple of ``moment_conservation`` that passes as marginal.
        arc_margin: Multiple of the negligible weight that wrong-arc atoms may carry, flagged marginal.
        positivity: Allowed breach of moment positivity, in the Toeplitz Gram and the Schur contractions.

    Raises:
        ValidationError: If a value is not a positive finite number, or a margin is below one.
    """

    rank_floor: float = 1.0e-10
    moment_conservation: float = 1.0e-9
    conservation_margin: float = 2.0
    arc_margin: float = 100.0
    positivity: float = 1.0e-10

    def __post_init__(self) -> None:
        names = ("rank_floor", "moment_conservation", "conservation_margin", "arc_margin", "positivity")
        for name in names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValidationError(f"{name} must be a finite real number")
            if value <= 0.0:
                raise ValidationError(f"{name} must be strictly positive")
        for name in ("conservation_margin", "arc_margin"):
            if getattr(self, name) < 1.0:
                raise ValidationError(f"{name} must be at least 1")


DEFAULT_TOLERANCES = Tolerances()

"""The tolerance set every function uses unless it is given another."""

# The general tolerances of checks that never decide a result.
#: Absolute part of every check with a shared absolute-plus-relative threshold.
ABSOLUTE_TOLERANCE = 1.0e-12
#: Relative part of the same checks, times the scale of the quantity checked.
RELATIVE_TOLERANCE = 1.0e-9
#: Roundoff allowed where zero or an identity is exact: unitarity, the unit circle, symmetry.
ROUNDOFF_TOLERANCE = 1.0e-10
#: Occupation error allowed, and smallest gap and excitation energy in Hartree, of the reference.
REFERENCE_TOLERANCE = 1.0e-8

# The realization backends and the sector closure.
#: Absolute-plus-relative tolerance on moments rebuilt from a block-CMV realization.
BLOCK_CMV_MOMENT_TOLERANCE = 1.0e-9
#: Largest zeroth-moment deficit after a discard that the congruence may restore.
SECTOR_REDISTRIBUTION = 0.5
#: Bisection steps of the ``"restricted"`` closure search on the feasible-arc boundary.
PHASE_REFINE_STEPS = 8
