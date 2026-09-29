r"""Block-CMV realization by the operator Schur recursion.

The normalized moments define a matrix Schur function. Peeling it gives the
matrix Verblunsky coefficients :math:`\alpha_j`, starting from
:math:`\alpha_0^\dagger = \widehat{C}^{(1)}`, and their Julia rotations
assemble into the CMV unitary :math:`U`. The coupling
:math:`B = \mathcal{B} E_0^\dagger` completes the realization. Defects are
rank-deflated at the rank tolerance, a coefficient outside the contraction
ball by more than the block-Schur band is refused, and the caller supplies
the terminal block.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .._helpers.tolerances import DEFAULT_TOLERANCES, Tolerances
from .._helpers.validate import _check
from ..tools.records.realization import BlockSchurParameters, MatrixCayleyMoments
from ._helpers.block_cmv import (
    _AssemblyPrefix,
    _realize_block_cmv_from_prefix,
    _schur_parameters,
)
from .base import UnitaryMomentRealization


@dataclass(frozen=True, slots=True, kw_only=True)
class BlockCMVRealization(UnitaryMomentRealization):
    r"""Unitary realization of matrix Cayley moments by the operator Schur recursion.

    The fields are those of
    :class:`~cayleygw.realization.base.UnitaryMomentRealization`. The terminal
    unitary is the last Verblunsky unitary; a scalar phase closure is
    ``exp(1j * phase) * I``. The Schur steps and the closure-independent part
    of the assembly are kept, so :meth:`reclose` assembles only the closure's
    block. That part is ``None`` if the moments fixed the closure or the
    matrices were released.
    """

    # Kept for :meth:`reclose`.
    _parameters: BlockSchurParameters | None = field(
        default=None, repr=False, compare=False
    )
    _prefix: _AssemblyPrefix | None = field(
        default=None, repr=False, compare=False
    )

    def without_matrices(self) -> "BlockCMVRealization":
        """Release the dense matrices as the base class does, and the cached assembly."""

        return replace(self, matrix=None, _selector_state=None, _prefix=None)

    def reclose(
        self,
        terminal_unitary: Any,
        *,
        tolerances: Tolerances = DEFAULT_TOLERANCES,
    ) -> "BlockCMVRealization":
        """Return this realization closed by another terminal unitary.

        The moments alone fix the support compression, the Schur recursion
        and every block of the factors but the closure's, so only that block
        is assembled again. ``tolerances`` must be the ones this realization
        was built with.
        """

        if self.terminal_from_moments:
            # Rank-terminated data admits one closure and it is already built.
            return self
        if self._prefix is None:
            self.require_matrix()
        return BlockCMVRealization(
            **_realize_block_cmv_from_prefix(
                self.normalization,
                self._parameters,
                terminal_unitary,
                tolerances,
                self._prefix,
            )
        )

    @classmethod
    def realize(
        cls,
        moments: MatrixCayleyMoments | Any,
        *,
        tolerances: Tolerances = DEFAULT_TOLERANCES,
    ) -> BlockCMVRealization:
        r"""Realize physical matrix Cayley moments as ``B U**n B.H``.

        Args:
            moments: The moments ``C[0], ..., C[n_max]``, as a
                :class:`MatrixCayleyMoments` or a compatible array.
            tolerances: Thresholds; ``rank_floor`` sets the defect cut.

        Returns:
            The realization closed by the identity terminal block;
            :meth:`reclose` changes the closure.

        Raises:
            ValidationError: If the shape, the values or ``tolerances`` are invalid.
            RefusalError: If positivity, support, Schur contractivity,
                unitarity or moment conservation fails.
        """

        _check.tolerances(tolerances)
        normalization = cls.normalize(moments)
        if normalization.rank == 0:
            # A zero measure, from a vanishing self-energy, has no auxiliary space.
            return cls.empty(normalization)

        parameters = _schur_parameters(normalization.values, tolerances)
        return cls(
            **_realize_block_cmv_from_prefix(
                normalization,
                parameters,
                None,
                tolerances,
                None,
            )
        )


__all__ = ["BlockCMVRealization"]
