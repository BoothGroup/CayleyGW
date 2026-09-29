"""Records of :mod:`cayleygw.upfold`: the moment reconstruction check and the Dyson spectrum."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from ..._helpers.errors import ValidationError
from ..._helpers.types import Sector
from ..._helpers.validate import ComplexArray, FloatArray, _check

if TYPE_CHECKING:
    from ...upfold import UpfoldedDysonHamiltonian


@dataclass(frozen=True, slots=True)
class SectorMomentReconstruction:
    """Order-by-order moment reconstruction of one sector.

    Attributes:
        sector: Hole or particle.
        input_moments: Moments the sector was realized from, one per order.
        reconstructed_moments: Moments rebuilt from the poles and couplings.
        absolute_errors: Frobenius norm of the difference at each order.
        relative_errors: Absolute error over the input norm, floored at machine epsilon.
        acceptance_thresholds: Largest absolute error the realization accepts at each order.
        conserved_order: Highest conserved order; higher orders are reported only.
    """

    sector: Sector
    input_moments: ComplexArray = field(repr=False)
    reconstructed_moments: ComplexArray = field(repr=False)
    absolute_errors: FloatArray
    relative_errors: FloatArray
    acceptance_thresholds: FloatArray = field(repr=False)
    conserved_order: int

    @property
    def passed(self) -> bool:
        """Whether every conserved order is within its threshold."""

        stop = self.conserved_order + 1
        return bool(np.all(self.absolute_errors[:stop] <= self.acceptance_thresholds[:stop]))


@dataclass(frozen=True, slots=True)
class MomentReconstructionResult:
    """Moment reconstruction of both sectors from the poles and couplings in the Hamiltonian.

    It checks the matrix the Dyson solve uses, not diagnostics kept by the
    realization.

    Attributes:
        hole: Reconstruction of the hole sector.
        particle: Reconstruction of the particle sector.
    """

    hole: SectorMomentReconstruction
    particle: SectorMomentReconstruction

    @property
    def maximum_conserved_absolute_error(self) -> float:
        """Largest absolute error over the conserved orders of both sectors."""

        return max(
            float(np.max(sector.absolute_errors[: sector.conserved_order + 1]))
            for sector in (self.hole, self.particle)
        )

    @property
    def maximum_conserved_relative_error(self) -> float:
        """Largest relative error over the conserved orders of both sectors."""

        return max(
            float(np.max(sector.relative_errors[: sector.conserved_order + 1]))
            for sector in (self.hole, self.particle)
        )

    @property
    def passed(self) -> bool:
        """Whether both sectors are within threshold at every conserved order."""

        return self.hole.passed and self.particle.passed


class DysonProblem:
    """Empty base of the upfolded Hamiltonian, so a spectrum can check its type."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class DysonSpectrum:
    """Eigenvalues, eigenvectors and physical weights of an upfolded Hamiltonian.

    The physical rows of an eigenvector are its Dyson orbital, and their squared
    norm is its physical spectral weight. Nothing is renormalized or clipped.

    Attributes:
        energies: Ascending eigenvalues in Hartree.
        eigenvectors: Orthonormal eigenvectors as columns.
        problem: The Hamiltonian they diagonalize.

    Raises:
        ValidationError: If ``problem`` is not an upfolded Hamiltonian or the
            shapes do not match its dimension.
    """

    energies: FloatArray = field(repr=False)
    eigenvectors: ComplexArray = field(repr=False)
    problem: UpfoldedDysonHamiltonian = field(repr=False, compare=False)
    _orbital_weights: FloatArray = field(init=False, repr=False, compare=False)
    _physical_weights: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.problem, DysonProblem):
            raise ValidationError("problem must be an UpfoldedDysonHamiltonian instance")
        dimension = self.problem.dimension
        if np.shape(self.energies) != (dimension,) or np.shape(self.eigenvectors) != (
            dimension,
            dimension,
        ):
            raise ValidationError(
                "energies and eigenvectors must have one entry and one column "
                "per upfolded dimension"
            )
        orbital_weights = np.abs(self.eigenvectors[self.problem.physical_slice]) ** 2
        physical_weights = np.sum(orbital_weights, axis=0)
        for name, value in (
            ("_orbital_weights", orbital_weights),
            ("_physical_weights", physical_weights),
        ):
            value = np.ascontiguousarray(value, dtype=np.float64)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def nphysical(self) -> int:
        """Number of physical orbitals, the length of each Dyson orbital."""

        return self.problem.nphysical

    @property
    def nstates(self) -> int:
        """Number of charged states, the dimension of the Hamiltonian."""

        return int(self.energies.size)

    @property
    def orbital_weights(self) -> FloatArray:
        """Physical spectral weight per orbital and state, shape (nphysical, nstates)."""

        return self._orbital_weights

    @property
    def physical_weights(self) -> FloatArray:
        """Physical spectral weight of every charged state, its squared Dyson-orbital norm."""

        return self._physical_weights

    def physical_weight_below(self, boundary: Any) -> float:
        r"""Total physical weight of the states strictly below an energy.

        For a restricted reference this is the correlated electron count per
        spin. Nothing checks it: one-shot :math:`G_0W_0` does not conserve it
        even with every pole kept, and with moment order it converges to the
        all-pole value.

        Args:
            boundary: Energy in Hartree between the removal and addition poles.

        Returns:
            :math:`\sum_n w_n` over the states with :math:`E_n <` ``boundary``.

        Raises:
            ValidationError: If ``boundary`` is not a finite real number.
        """

        edge = _check.finite_real(boundary, "boundary")
        selected = np.asarray(self.energies) < edge
        return float(np.sum(np.asarray(self.physical_weights)[selected]))

    def spectral_function(
        self,
        frequency: ArrayLike,
        *,
        broadening: float,
        orbital_position: int | None = None,
    ) -> FloatArray:
        """Lorentzian-broadened physical spectral function.

        Args:
            frequency: Real energies in Hartree, a scalar or an array.
            broadening: Lorentzian half-width in Hartree.
            orbital_position: Physical orbital whose diagonal spectrum to return;
                the trace over orbitals by default.

        Returns:
            The spectrum, with the shape of ``frequency``.

        Raises:
            ValidationError: If an argument is not finite, ``broadening`` is not
                positive, or ``orbital_position`` is out of range.
        """

        eta = _check.positive_real(broadening, "broadening")
        values = _check.readonly_real(frequency, "frequency")
        if orbital_position is None:
            weights = self.physical_weights
        else:
            orbital = _check.nonnegative_integer(orbital_position, "orbital_position")
            if orbital >= self.nphysical:
                raise ValidationError("orbital_position is out of range")
            weights = self.orbital_weights[orbital]
        denominator = (values[..., None] - self.energies) ** 2 + eta**2
        result = np.sum(weights * eta / (np.pi * denominator), axis=-1)
        return _check.readonly_real(result, "spectral function")
