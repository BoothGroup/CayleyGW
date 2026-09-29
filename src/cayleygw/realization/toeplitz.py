r"""Block Toeplitz realization by orthogonal completion.

The Schur recursion of :mod:`.block_cmv` divides by defect factors and loses
accuracy as a coefficient nears the boundary of the contraction ball. This
backend avoids those divisions. The block Toeplitz Gram matrix of the
normalized moments has entries :math:`(U^i E)^\dagger (U^j E)`, so its
positive square root exposes the orbit :math:`U^j E` of the selector, and
the shift between adjacent block columns is a unitary Procrustes problem.
Only the complement the moments leave unconstrained is closed with the
terminal unitary. The Gram rank cut and the block-CMV defect cut share
``rank_floor``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._helpers.tolerances import DEFAULT_TOLERANCES, Tolerances
from .._helpers.validate import ComplexArray, FloatArray, _check
from ..tools.records.realization import MatrixCayleyMoments, ToeplitzDiagnostics
from ._helpers.base import _canonical_terminal_unitary
from ._helpers.toeplitz import (
    _certified_minimum,
    _gram_square_root,
    _moment_acceptance_thresholds,
    _projection_scale,
    _seal,
    _shift_completion,
    _UnitarityCertificate,
    _validated_realization,
    build_gram_spectrum,
)
from .base import UnitaryMomentRealization


@dataclass(frozen=True, slots=True, kw_only=True)
class ToeplitzRealization(UnitaryMomentRealization):
    r"""Unitary realization of matrix Cayley moments from a Toeplitz Gram square root.

    It has the fields of
    :class:`~cayleygw.realization.base.UnitaryMomentRealization` and these.
    The completion factors are kept, so :meth:`reclose` changes the closure
    without repeating the Gram eigensolve or the SVD. No Schur coefficient is
    formed, so the contraction ratio is ``0.0``.

    Attributes:
        unitarity_residual: The Frobenius residual ``||U.H U - I||`` of this closure.
        diagnostics: The Gram rank, floor and residuals of the construction.
    """

    unitarity_residual: float = 0.0
    diagnostics: ToeplitzDiagnostics
    _fixed_matrix: ComplexArray | None = field(default=None, repr=False)
    _terminal_left_basis: ComplexArray | None = field(default=None, repr=False)
    _terminal_right_adjoint: ComplexArray | None = field(default=None, repr=False)
    # Closed-form unitarity residual for reclose; ``None`` if the moments fix the closure.
    _unitarity_certificate: _UnitarityCertificate | None = field(
        default=None, repr=False, compare=False
    )

    def physical_moment_acceptance_thresholds(
        self,
        *,
        base_tolerance: float | None = None,
        absolute_tolerance: float | None = None,
    ) -> FloatArray:
        """Return the strict thresholds plus a rank-deflation backward-error budget.

        A rank-deficient Toeplitz problem lies on the boundary of the positive
        moment cone. If the Gram cut removed directions, each order's budget
        adds the feasibility tolerance and the measured Gram-projection and
        shift residuals; a full-rank problem keeps the strict thresholds.
        """

        return _moment_acceptance_thresholds(
            self.normalization.source.values,
            self.normalization,
            self.diagnostics,
            base_tolerance=base_tolerance,
            absolute_tolerance=absolute_tolerance,
            projection_scale=_projection_scale(self.normalization),
        )

    def reclose(
        self,
        terminal_unitary: Any,
        *,
        tolerances: Tolerances = DEFAULT_TOLERANCES,
    ) -> "ToeplitzRealization":
        """Return this realization closed by another terminal, reusing its factors."""

        _check.tolerances(tolerances)
        if self.terminal_from_moments:
            return self
        terminal = _canonical_terminal_unitary(
            terminal_unitary,
            self.diagnostics.terminal_dimension,
        )
        return ToeplitzRealization(
            **_validated_realization(
                normalization=self.normalization,
                terminal=terminal,
                fixed_matrix=self._fixed_matrix,
                terminal_left_basis=self._terminal_left_basis,
                terminal_right_adjoint=self._terminal_right_adjoint,
                selector=self.selector,
                base_diagnostics=self.diagnostics,
                certificate=self._unitarity_certificate,
            )
        )

    @classmethod
    def realize(
        cls,
        moments: MatrixCayleyMoments | Any,
        *,
        tolerances: Tolerances = DEFAULT_TOLERANCES,
        gram_native_threads: int = 1,
    ) -> ToeplitzRealization:
        r"""Realize physical matrix Cayley moments without inverse defect factors.

        Gram eigenvalues at or below the backward-error floor are dropped as
        null directions, so quadrature noise does not become a full-rank
        auxiliary space. The result is accepted only if the physical and
        normalized moments it rebuilds fit the reconstruction-plus-feasibility
        budget.

        Args:
            moments: The moments ``C[0], ..., C[n_max]``, as a
                :class:`MatrixCayleyMoments` or a compatible array.
            tolerances: Thresholds; ``rank_floor`` sets the Gram rank cut.
            gram_native_threads: BLAS threads for the Gram eigensolve; see
                :func:`build_gram_spectrum`.

        Returns:
            The realization closed by the identity terminal block;
            :meth:`reclose` changes the closure.

        Raises:
            ValidationError: If the shape, the values or ``tolerances`` are invalid.
            RefusalError: If positivity, support, the Gram rank, unitarity or
                moment conservation fails.
        """

        _check.tolerances(tolerances)
        normalization = cls.normalize(moments)
        rank = normalization.rank
        if rank == 0:
            return cls.empty(
                normalization,
                diagnostics=ToeplitzDiagnostics(
                    minimum_toeplitz_eigenvalue=0.0,
                    eigenvalue_floor=tolerances.rank_floor,
                    numerical_rank=0,
                    gram_residual=0.0,
                    determined_shift_residual=0.0,
                    terminal_dimension=0,
                ),
            )

        gram_spectrum = build_gram_spectrum(
            normalization.values,
            native_threads=gram_native_threads,
        )
        hermitian_residual = gram_spectrum.hermitian_residual
        minimum = _certified_minimum(gram_spectrum.eigenvalues, tolerances.positivity)
        root, floor, gram_residual = _gram_square_root(gram_spectrum, tolerances.rank_floor)
        dimension = root.shape[0]
        (
            selector,
            fixed_matrix,
            terminal_left,
            terminal_right_adjoint,
            domain_rank,
            shift_residual,
        ) = _shift_completion(root, rank, normalization.n_max)
        terminal_dimension = dimension - domain_rank
        terminal = _canonical_terminal_unitary(None, terminal_dimension)
        base_diagnostics = ToeplitzDiagnostics(
            minimum_toeplitz_eigenvalue=minimum,
            eigenvalue_floor=floor,
            numerical_rank=dimension,
            gram_residual=max(gram_residual, hermitian_residual),
            determined_shift_residual=shift_residual,
            terminal_dimension=terminal_dimension,
        )
        return cls(
            **_validated_realization(
                normalization=normalization,
                terminal=terminal,
                fixed_matrix=_seal(fixed_matrix, "fixed_matrix"),
                terminal_left_basis=_check.readonly_complex(terminal_left, "terminal_left_basis"),
                terminal_right_adjoint=_check.readonly_complex(
                    terminal_right_adjoint, "terminal_right_adjoint"
                ),
                selector=selector,
                base_diagnostics=base_diagnostics,
            )
        )


__all__ = [
    "ToeplitzDiagnostics",
    "ToeplitzRealization",
]
