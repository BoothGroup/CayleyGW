"""Records of :mod:`cayleygw.realization`: moments, Schur steps, spectra and closure scans."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from ..._helpers.errors import RefusalError, ValidationError
from ..._helpers.types import Sector
from ..._helpers.validate import BoolArray, ComplexArray, FloatArray, _check
from ...realization._helpers.base import _ranking_key

if TYPE_CHECKING:
    from ...realization.base import (
        RealizationAlgorithm,
        UnitaryMomentRealization,
    )


@dataclass(frozen=True, slots=True)
class MatrixCayleyMoments:
    r"""One-sided moments of a positive matrix measure on the unit circle.

    ``C[n]`` is the integral of ``u**n dSigma(u)``, and ``C[-n] = C[n].H``.
    Only shape and finiteness are checked here; positivity and support are
    checked when the moments are normalized. The record holds no
    theory-specific data, so any moment producer can build it.

    Attributes:
        values: ``C[0], ..., C[n_max]``, of shape ``(n_max + 1, n, n)``.
    """

    values: ComplexArray = field(repr=False)

    def __post_init__(self) -> None:
        values = _check.readonly_complex(self.values, "values")
        if (
            values.ndim != 3
            or values.shape[0] == 0
            or values.shape[1] == 0
            or values.shape[1] != values.shape[2]
        ):
            raise ValidationError(
                "values must have shape (order + 1, nphysical, nphysical) "
                "with positive dimensions"
            )
        object.__setattr__(self, "values", values)

    @property
    def n_max(self) -> int:
        """Highest moment order supplied."""

        return int(self.values.shape[0] - 1)

    @property
    def nphysical(self) -> int:
        """Row and column dimension of every moment."""

        return int(self.values.shape[1])


@dataclass(frozen=True, slots=True)
class NormalizedMatrixCayleyMoments:
    r"""Moments compressed to the support of ``C[0]`` and normalized to ``C[0] = I``.

    Attributes:
        source: The unnormalized moments.
        values: Support-coordinate moments ``R+ C[n] R+.H``; the zeroth is the identity.
        support_eigenvalues: Positive eigenvalues of ``C[0]`` kept on the support.
        support_factor: Rectangular factor ``R`` with ``C[0] = R R.H`` on the support.
        support_pseudoinverse: Moore-Penrose inverse ``R+`` used to normalize.
    """

    source: MatrixCayleyMoments
    values: ComplexArray = field(repr=False)
    support_eigenvalues: FloatArray = field(repr=False)
    support_factor: ComplexArray = field(repr=False)
    support_pseudoinverse: ComplexArray = field(repr=False)

    @property
    def rank(self) -> int:
        """Rank of the support of ``C[0]`` that was kept."""

        return int(self.support_eigenvalues.size)

    @property
    def n_max(self) -> int:
        """Highest moment order supplied."""

        return self.source.n_max


@dataclass(frozen=True, slots=True)
class BlockCMVSpectrum:
    r"""Atomic spectrum of one finite unitary realization.

    Attributes:
        realization: The realization the spectrum came from.
        nodes: Schur-form diagonal, projected radially onto the circle to remove roundoff.
        couplings: Physical coupling columns ``W = B @ Z``, one per node.
        trace_weights: Traces ``||w_l||**2`` of the rank-one residues.
        moment_residuals: Residual of the atomic moments against the realized ones, per order.
    """

    realization: UnitaryMomentRealization
    nodes: ComplexArray = field(repr=False)
    couplings: ComplexArray = field(repr=False)
    trace_weights: FloatArray = field(repr=False)
    moment_residuals: FloatArray = field(repr=False)

    def without_matrices(self) -> "BlockCMVSpectrum":
        """Return this spectrum with its realization's dense squares released.

        Classifying and finalizing a candidate reads only nodes, couplings and masks.
        """

        return replace(self, realization=self.realization.without_matrices())

    @property
    def total_weight(self) -> float:
        """Total trace weight of all atoms."""

        return float(np.sum(self.trace_weights))


@dataclass(frozen=True, slots=True)
class BlockSchurStep:
    r"""One rank-revealing operator Schur step.

    The left and right defect bases differ for a nonnormal ``Gamma``, but
    their spectra agree.

    Attributes:
        choice_parameter: Reflected choice coefficient ``Gamma``; the first is ``C[1]``.
        left_defect_basis: Orthonormal basis kept for the left defect space.
        right_defect_basis: Orthonormal basis kept for the right defect space.
        defect_eigenvalues: Positive eigenvalues kept in each squared defect.
        rotation: Reduced Julia rotation, unitary even after rank deflation.
        contraction_ratio: Top singular value's excess over one before projection, in units of the positivity tolerance.
    """

    choice_parameter: ComplexArray = field(repr=False)
    left_defect_basis: ComplexArray = field(repr=False)
    right_defect_basis: ComplexArray = field(repr=False)
    defect_eigenvalues: FloatArray = field(repr=False)
    rotation: ComplexArray = field(repr=False)
    contraction_ratio: float = 0.0

    @property
    def verblunsky_coefficient(self) -> ComplexArray:
        """Conventional matrix Verblunsky coefficient ``alpha = Gamma.H``."""

        return self.choice_parameter.conj().T

    @property
    def input_dimension(self) -> int:
        """Dimension the choice coefficient acts on."""

        return int(self.choice_parameter.shape[0])

    @property
    def defect_rank(self) -> int:
        """Numerical rank kept in both defect spaces."""

        return int(self.defect_eigenvalues.size)

    @property
    def terminated(self) -> bool:
        """Whether the coefficient is unitary after rank detection."""

        return self.defect_rank == 0


@dataclass(frozen=True, slots=True)
class BlockSchurParameters:
    """Operator Schur steps fixed by the moments, possibly rank-deflated.

    Attributes:
        steps: The Schur steps, with a natural terminal step but never a chosen closure.
        terminated: Whether the moments fix a unitary terminal step.
        initial_dimension: Normalized support dimension before the recursion.
    """

    steps: tuple[BlockSchurStep, ...]
    terminated: bool
    initial_dimension: int


@dataclass(frozen=True, slots=True)
class GramSpectrum:
    """Eigendecomposition of a block-Toeplitz Gram matrix.

    Attributes:
        gram: The symmetrized Gram matrix.
        hermitian_residual: Asymmetry of the Gram matrix before symmetrization.
        eigenvalues: Ascending eigenvalues of ``gram``.
        eigenvectors: Matching eigenvectors of ``gram``.
        native_threads: Eigensolve threads, pinned because the smallest eigenvalue is roundoff.
    """

    gram: ComplexArray
    hermitian_residual: float
    eigenvalues: FloatArray
    eigenvectors: ComplexArray
    native_threads: int


@dataclass(frozen=True, slots=True)
class ToeplitzDiagnostics:
    """Conditioning and roundoff diagnostics of a Toeplitz realization.

    Attributes:
        minimum_toeplitz_eigenvalue: Smallest eigenvalue of the normalized block Toeplitz matrix.
        eigenvalue_floor: Scale-aware floor that separates null directions from small positive ones.
        numerical_rank: Rank kept in the square-root Gram factor.
        gram_residual: Frobenius residual of the Gram square root.
        determined_shift_residual: Frobenius residual ``||U D - R||`` on adjacent Gram block columns.
        terminal_dimension: ``D - min(D, F * r)``, left for the terminal unitary at full shift rank.
    """

    minimum_toeplitz_eigenvalue: float
    eigenvalue_floor: float
    numerical_rank: int
    gram_residual: float
    determined_shift_residual: float
    terminal_dimension: int


@dataclass(frozen=True, slots=True)
class SectorClosureDiagnostics:
    r"""Diagnostics of one scalar terminal closure ``beta = exp(1j * phase) I``.

    Attributes:
        phase: Terminal phase, or ``None`` when the moments fix the terminal block.
        realization: The closed realization, with its dense matrices released.
        spectrum: Its atomic spectrum.
        correct_arc_mask: Atoms on the sector's semicircle, beyond roundoff of the axis.
        wrong_arc_mask: The other atoms: the other semicircle and within roundoff of the axis.
        near_singular_mask: Atoms near :math:`u = 1`, where the inverse Cayley map diverges.
        total_weight: Trace weight of all atoms.
        weight_threshold: ``ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * total_weight``.
        wrong_arc_weight: Trace weight of the wrong-arc atoms.
        near_singular_weight: Trace weight near :math:`u = 1`.
        minimum_correct_node_distance_from_one: Closest approach to ``u = 1`` of a significant correct-arc atom.
        inverse_conditioning_penalty: Sum over atoms of weight over squared distance from ``u = 1``.
        withheld_moment_residuals: Residual of each moment above ``n_conserved``.
        withheld_relative_residuals: The same residuals over ``1 + ||C[n]||``.
        support_acceptable: Whether the wrong-arc weight is at most the threshold.
        inverse_safe: Whether the near-singular weight is at most the threshold.
        support_marginal: Whether the arc weight passed only inside the margin band.
    """

    phase: float | None
    realization: UnitaryMomentRealization
    spectrum: BlockCMVSpectrum
    correct_arc_mask: BoolArray = field(repr=False)
    wrong_arc_mask: BoolArray = field(repr=False)
    near_singular_mask: BoolArray = field(repr=False)
    total_weight: float
    weight_threshold: float
    wrong_arc_weight: float
    near_singular_weight: float
    minimum_correct_node_distance_from_one: float
    inverse_conditioning_penalty: float
    withheld_moment_residuals: FloatArray = field(repr=False)
    withheld_relative_residuals: FloatArray = field(repr=False)
    support_acceptable: bool
    inverse_safe: bool
    support_marginal: bool = False

    @property
    def acceptable(self) -> bool:
        """Whether support passes, cleanly or in the margin band, and the closure is inverse-safe.

        Inverse safety has no band: it guards the singularity of the inverse Cayley map.
        """

        return (self.support_acceptable or self.support_marginal) and self.inverse_safe

    @property
    def maximum_withheld_relative_residual(self) -> float:
        """Largest relative residual over the withheld moments, or ``0.0`` if none."""

        if self.withheld_relative_residuals.size == 0:
            return 0.0
        return float(np.max(self.withheld_relative_residuals))


@dataclass(frozen=True, slots=True)
class SectorClosureScan:
    """Every closure candidate of one sector's phase scan, and the one it selected.

    Attributes:
        sector: Hole or particle sector.
        requested_realization_algorithm: Backend policy the caller asked for, ``"auto"`` included.
        realization_algorithm: Backend that built the candidates.
        moments: Every supplied moment, withheld ones included.
        n_conserved: Highest moment order every candidate conserves.
        candidates: The identity closure first, then the others in phase order.
        selected_index: Index of the best acceptable candidate, or ``None``.
        terminal_dimension: Size ``t`` of the free terminal block; zero when the moments fix it.
    """

    sector: Sector
    requested_realization_algorithm: RealizationAlgorithm
    realization_algorithm: RealizationAlgorithm
    moments: MatrixCayleyMoments
    n_conserved: int
    candidates: tuple[SectorClosureDiagnostics, ...]
    selected_index: int | None
    terminal_dimension: int = 0

    @property
    def selected(self) -> SectorClosureDiagnostics | None:
        """The selected acceptable candidate, or ``None`` if there is none."""

        if self.selected_index is None:
            return None
        return self.candidates[self.selected_index]

    @property
    def ranked_indices(self) -> tuple[int, ...]:
        """Indices of the acceptable candidates, best first; the first is ``selected_index``.

        Finalization walks this order, because a candidate that passes every
        gate can still fail conservation once its unsupported atoms are discarded.
        """

        if self.selected_index is None:
            return ()
        feasible = [
            index
            for index, candidate in enumerate(self.candidates)
            if candidate.acceptable
        ]
        feasible.sort(key=lambda index: _ranking_key(self.candidates[index]))
        return tuple(feasible)

    def require_selected(self) -> SectorClosureDiagnostics:
        """Return the selected closure, or raise a refusal that carries the whole scan."""

        selected = self.selected
        if selected is None:
            best = min(self.candidates, key=lambda item: item.wrong_arc_weight)
            lever = (
                "lower n_conserved, since the moments fix the terminal block"
                if best.realization.terminal_from_moments
                else "increase terminal_phase_count"
            )
            count = len(self.candidates)
            raise RefusalError(
                "no sector-supported, inverse-safe realization among "
                f"{count} closure{'' if count == 1 else 's'}: the best has "
                f"invalid arc weight {best.wrong_arc_weight:.3e} and "
                f"near-singular weight {best.near_singular_weight:.3e} against "
                f"a threshold of {best.weight_threshold:.3e}; {lever}",
                kind="sector",
                diagnostics=self,
            )
        return selected
