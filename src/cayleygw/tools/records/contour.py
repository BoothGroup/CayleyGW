"""Records of :mod:`cayleygw.contour`: bounds, equioscillation and sector moments."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..._helpers.errors import ValidationError
from ..._helpers.validate import ComplexArray, _check


@dataclass(frozen=True, slots=True)
class SpectralBounds:
    """Certified interval of the positive excitation energies.

    Attributes:
        lower: Lower bound in Hartree.
        upper: Upper bound in Hartree.
        source: How the bounds were obtained.
    """

    lower: float
    upper: float
    source: str

    def __post_init__(self) -> None:
        lower = _check.positive_real(self.lower, "lower")
        upper = _check.positive_real(self.upper, "upper")
        if upper < lower:
            raise ValidationError("upper must not be smaller than lower")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValidationError("source must be a nonempty string")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "source", self.source.strip())


@dataclass(frozen=True, slots=True)
class EquioscillationDiagnostics:
    """How the confocal equioscillation chose a contour.

    Attributes:
        sigma_min: Confocal parameter of the nearest excluded singularity; the
            error rate is half of it.
        sigma_c: Confocal parameter used, half of ``sigma_min``.
        limiting_singularity: ``"kernel"`` or ``"mirror"``, the family of that singularity.
    """

    sigma_min: float
    sigma_c: float
    limiting_singularity: str


@dataclass(frozen=True, slots=True)
class ContourSectorMoments:
    """Cayley moments of one sector from the contour quadrature.

    Attributes:
        moments: Moments summed over the internal orbitals, shape
            ``(n_max + 1, nmo, nmo)``.
    """

    moments: ComplexArray = field(repr=False)
