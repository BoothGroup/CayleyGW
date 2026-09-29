r"""The contract both realization backends share: normalization, realization and poles.

A positive matrix measure on the unit circle has moments
:math:`C^{(n)} = \int u^n\, d\Sigma(u)` with :math:`C^{(0)} \succeq 0`. Every
backend factors :math:`C^{(0)} = \mathcal{B}\mathcal{B}^\dagger` at the
support-rank cut and works from the normalized moments
:math:`\widehat{C}^{(n)} = \mathcal{B}^{+} C^{(n)} \mathcal{B}^{+\dagger}`,
whose zeroth moment is the identity. A realization
:math:`C^{(n)} = B U^n B^\dagger`, with coupling
:math:`B = \mathcal{B} E^\dagger`, diagonalizes as
:math:`U = Z\,\mathrm{diag}(u_\ell)\,Z^\dagger`, which gives the atoms
:math:`C^{(n)} = \sum_\ell u_\ell^n\, w_\ell w_\ell^\dagger` with
:math:`W = B Z`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeAlias

import numpy as np

from .._helpers import tolerances as limits
from .._helpers.errors import RefusalError
from .._helpers.tolerances import DEFAULT_TOLERANCES, Tolerances
from .._helpers.validate import ComplexArray, FloatArray, _check
from ..tools.records.realization import (
    BlockCMVSpectrum,
    MatrixCayleyMoments,
    NormalizedMatrixCayleyMoments,
)
from ._helpers.base import (
    _selector_moment_chain,
    _support_compression,
    _unit_circle_atoms,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class UnitaryMomentRealization:
    r"""A finite unitary realization ``C[n] = B U**n B.H`` of physical moments.

    Both backends return one, and the sector, inverse-Cayley and Dyson code
    read it.

    Attributes:
        normalization: The support compression of ``C[0]`` and the normalized moments.
        terminal_unitary: The closure on the space the moments leave unconstrained.
        terminal_from_moments: Whether the moments fixed the terminal block.
        matrix: The read-only unitary ``U``, or ``None`` after :meth:`without_matrices`.
        selector: The isometry ``E`` with ``E.H @ U**n @ E`` the normalized ``C[n]``.
        coupling: The physical coupling ``B = support_factor @ E.H``.
        reconstructed_moments: The physical moments rebuilt through every supplied order.
        moment_residuals: The Frobenius residual of each supplied physical moment.
        maximum_contraction_ratio: Largest contraction-ball excess of a Schur
            step over the positivity tolerance; ``0.0`` without Schur steps.
    """

    normalization: NormalizedMatrixCayleyMoments
    terminal_unitary: ComplexArray = field(repr=False)
    terminal_from_moments: bool
    matrix: ComplexArray | None = field(repr=False)
    selector: ComplexArray = field(repr=False)
    coupling: ComplexArray = field(repr=False)
    reconstructed_moments: ComplexArray = field(repr=False)
    moment_residuals: FloatArray = field(repr=False)
    maximum_contraction_ratio: float = 0.0
    # The contract check's normalized moments and chain state; later orders continue it.
    _normalized_reconstructed: ComplexArray | None = field(
        default=None, repr=False, compare=False
    )
    _selector_state: ComplexArray | None = field(
        default=None, repr=False, compare=False
    )

    @staticmethod
    def normalize(moments: MatrixCayleyMoments | Any) -> NormalizedMatrixCayleyMoments:
        r"""Compress ``C[0]`` to its positive support and normalize it to the identity.

        Args:
            moments: A :class:`MatrixCayleyMoments` or a compatible array; it
                is never altered.

        Returns:
            The support factorization and the normalized moments.

        Raises:
            ValidationError: If the shape or the values are invalid.
            RefusalError: If ``C[0]`` is non-Hermitian or indefinite, a higher
                moment leaves its support, or the normalized ``C[0]`` is not
                the identity.
        """

        source = (
            moments
            if isinstance(moments, MatrixCayleyMoments)
            else MatrixCayleyMoments(moments)
        )
        return NormalizedMatrixCayleyMoments(
            source=source, **_support_compression(source)
        )

    @classmethod
    def empty(
        cls, normalization: NormalizedMatrixCayleyMoments, **fields: Any
    ) -> Any:
        """Return the zero-dimensional realization of a measure with no support."""

        empty = _check.readonly_complex(np.zeros((0, 0)), "empty matrix")
        return cls(
            normalization=normalization,
            terminal_unitary=empty,
            terminal_from_moments=True,
            matrix=empty,
            selector=empty,
            coupling=_check.readonly_complex(
                np.zeros((normalization.source.nphysical, 0)), "coupling"
            ),
            reconstructed_moments=_check.readonly_complex(
                np.zeros_like(normalization.source.values),
                "reconstructed_moments",
            ),
            moment_residuals=_check.readonly_real(
                np.linalg.norm(normalization.source.values, axis=(1, 2)),
                "moment_residuals",
            ),
            _normalized_reconstructed=_check.readonly_complex(
                np.zeros((normalization.n_max + 1, 0, 0)),
                "normalized reconstructed moments",
            ),
            _selector_state=empty,
            **fields,
        )

    @property
    def dimension(self) -> int:
        """The dimension of ``U``, read from the selector so it survives release."""

        return int(self.selector.shape[0])

    @property
    def conserved_order(self) -> int:
        """The highest supplied and verified moment order."""

        return self.normalization.n_max

    def without_matrices(self) -> Any:
        """Return this realization with its dense square matrices released.

        Everything a sector scan reads afterwards is already computed and is
        smaller than one square.
        """

        return replace(self, matrix=None, _selector_state=None)

    def require_matrix(self) -> ComplexArray:
        """Return the dense matrix, or refuse if it was released."""

        if self.matrix is None:
            raise RefusalError(
                "this realization released its dense matrices, which the "
                "moment methods need",
                kind="block-cmv",
            )
        return self.matrix

    def physical_moment_acceptance_thresholds(
        self,
        *,
        base_tolerance: float | None = None,
        absolute_tolerance: float | None = None,
    ) -> FloatArray:
        """Return the reconstruction thresholds ``absolute + base * ||C[n]||``.

        Args:
            base_tolerance: The ``base``; the tolerance of the realization's
                own contract check by default.
            absolute_tolerance: The ``absolute``; ``base`` by default.
        """

        base = (
            limits.BLOCK_CMV_MOMENT_TOLERANCE
            if base_tolerance is None
            else float(base_tolerance)
        )
        scales = np.linalg.norm(self.normalization.source.values, axis=(1, 2))
        if absolute_tolerance is None:
            thresholds = base * (1.0 + scales)
        else:
            thresholds = absolute_tolerance + base * scales
        return _check.readonly_real(thresholds, "physical moment thresholds")

    def reclose(
        self,
        terminal_unitary: Any,
        *,
        tolerances: Tolerances = DEFAULT_TOLERANCES,
    ) -> Any:
        """Return the realization closed by another terminal unitary.

        Only the terminal block changes and everything the moments determine
        is reused. A realization whose terminal block the moments fixed is
        returned unchanged.
        """

        raise NotImplementedError

    def matrix_moments(self, n_max: int) -> ComplexArray:
        r"""Return the physical moments ``B U**n B.H`` through order ``n_max``."""

        factor = self.normalization.support_factor
        return _check.readonly_complex(
            factor @ self.normalized_matrix_moments(n_max) @ factor.conj().T,
            "reconstructed moments",
        )

    def normalized_matrix_moments(self, n_max: int) -> ComplexArray:
        r"""Return the normalized moments ``E.H U**n E`` through order ``n_max``.

        Orders the contract check computed come from its cache, and later ones
        continue its chain, so the result is bit-identical to a full chain.
        """

        order = _check.nonnegative_integer(n_max, "n_max")
        # Refuse on a released realization even when the order is cached.
        matrix = self.require_matrix()
        cached = self._normalized_reconstructed
        computed = cached.shape[0] - 1
        if order <= computed:
            return cached[: order + 1]
        tail, _ = _selector_moment_chain(
            matrix,
            self.selector,
            order,
            state=matrix @ self._selector_state,
            first_order=computed + 1,
        )
        return _check.readonly_complex(
            np.concatenate((cached, tail), axis=0),
            "normalized reconstructed moments",
        )

    def spectrum(self) -> BlockCMVSpectrum:
        r"""Return the unit-circle nodes and positive semidefinite residues.

        A valid realization is normal and unitary, so its complex Schur form
        is diagonal. Nodes are projected onto the circle only after the Schur
        off-diagonal and radial residuals pass their thresholds.

        Returns:
            The nodes in angle order with their coupling columns, trace weights
            and moment residuals.

        Raises:
            RefusalError: If the matrix is materially non-unitary or the atoms
                do not conserve the moments.
        """

        return BlockCMVSpectrum(realization=self, **_unit_circle_atoms(self))


RealizationAlgorithm: TypeAlias = Literal["auto", "toeplitz", "block-cmv"]


PhaseRefinement: TypeAlias = Literal["fixed", "restricted"]


__all__ = [
    "MatrixCayleyMoments",
    "NormalizedMatrixCayleyMoments",
    "UnitaryMomentRealization",
    "BlockCMVSpectrum",
    "PhaseRefinement",
    "RealizationAlgorithm",
]
