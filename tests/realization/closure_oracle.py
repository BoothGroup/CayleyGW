"""Test oracle: the sector closure gates as a continuous function of the terminal unitary.

:class:`ClosureObjective` runs the reclosure and classification of
:meth:`~cayleygw.realization.sector.SectorSelfEnergyRealization.scan_closures`
at any terminal unitary: the scalar family the scan samples, one phase per
terminal direction, or the full unitary group. The tests use it to check that
the scan's candidates are points of one continuous objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
from scipy.linalg import expm

from cayleygw._helpers.errors import (
    RefusalError,
    ValidationError,
)
from cayleygw._helpers.tolerances import (
    ABSOLUTE_TOLERANCE,
    DEFAULT_TOLERANCES,
    ROUNDOFF_TOLERANCE,
    Tolerances,
)
from cayleygw._helpers.types import Sector
from cayleygw._helpers.validate import ComplexArray, _check
from cayleygw.realization._helpers.base import _ranking_key
from cayleygw.realization._helpers.sector import _candidate_diagnostics
from cayleygw.realization.base import (
    MatrixCayleyMoments,
    UnitaryMomentRealization,
)
from cayleygw.realization.block_cmv import BlockCMVRealization
from cayleygw.realization.toeplitz import ToeplitzRealization
from cayleygw.tools.records.realization import SectorClosureDiagnostics

ObjectiveBackend: TypeAlias = Literal["block-cmv", "toeplitz"]


@dataclass(frozen=True, slots=True)
class ClosureObjective:
    """The sector closure gates evaluated at any terminal unitary.

    Build with :func:`build_closure_objective`. The closure-independent part of
    the realization is built once; each :meth:`evaluate` costs one ``reclose``
    and one spectral extraction.

    Attributes:
        sector: Sector whose semicircle decides arc validity.
        source: Supplied moments, including orders withheld above ``fitted_order``.
        fitted_order: Highest moment order the realization is fitted to.
        minimum_distance: Node distance from ``u = 1`` below which an atom is inverse-unsafe.
        backend: Backend probed; the two fail on disjoint populations.
        canonical: The closure-independent realization, with the canonical terminal.
        tolerances: Thresholds for arc classification and both gates.
    """

    sector: Sector
    source: MatrixCayleyMoments = field(repr=False)
    fitted_order: int
    minimum_distance: float
    backend: ObjectiveBackend
    canonical: UnitaryMomentRealization = field(repr=False)
    tolerances: Tolerances = field(repr=False)

    @property
    def terminal_dimension(self) -> int:
        """Return ``t``, the free terminal dimension, or zero when the moments fix the block."""

        if self.canonical.terminal_from_moments:
            return 0
        return int(self.canonical.terminal_unitary.shape[0])

    @property
    def terminal_is_fixed(self) -> bool:
        """Return whether the moments already determine the terminal block."""

        return bool(self.canonical.terminal_from_moments)

    @property
    def scalar_parameter_count(self) -> int:
        """Return the real dimension of the scalar family, ``1`` when free."""

        return 0 if self.terminal_is_fixed else 1

    @property
    def diagonal_parameter_count(self) -> int:
        """Return the real dimension of the diagonal family, ``t``."""

        return self.terminal_dimension

    @property
    def unitary_parameter_count(self) -> int:
        """Return the real dimension of the full unitary family, ``t**2``."""

        dimension = self.terminal_dimension
        return dimension * dimension

    def _require_free_terminal(self) -> int:
        """Return ``t``, or raise when the moments fix the terminal block."""

        if self.terminal_is_fixed:
            raise ValidationError(
                "the moments determine the terminal block, so no closure "
                "family exists; terminal_dimension is zero"
            )
        return self.terminal_dimension

    def scalar(self, phase: float) -> ComplexArray:
        """Return ``exp(i phase) I``, the family the scan samples, with ``phase`` in radians."""

        dimension = self._require_free_terminal()
        value = float(phase)
        if not np.isfinite(value):
            raise ValidationError("phase must be finite")
        identity = np.eye(dimension, dtype=np.complex128)
        return np.exp(1.0j * value) * identity

    def diagonal(self, phases: Any) -> ComplexArray:
        """Return ``diag(exp(i phases))``, one phase in radians per terminal direction."""

        dimension = self._require_free_terminal()
        values = np.asarray(phases, dtype=np.float64).reshape(-1)
        if values.size != dimension:
            raise ValidationError(f"phases must have length {dimension}, got {values.size}")
        if not np.all(np.isfinite(values)):
            raise ValidationError("phases must be finite")
        return np.diag(np.exp(1.0j * values)).astype(np.complex128)

    def generated(self, generator: Any) -> ComplexArray:
        """Return ``expm(generator)``, a full ``U(t)`` point from a skew-Hermitian generator."""

        dimension = self._require_free_terminal()
        matrix = np.asarray(generator, dtype=np.complex128)
        if matrix.shape != (dimension, dimension):
            raise ValidationError(
                f"generator must have shape ({dimension}, {dimension}), got {matrix.shape}"
            )
        if not np.all(np.isfinite(matrix.real)) or not np.all(np.isfinite(matrix.imag)):
            raise ValidationError("generator must be finite")
        skew_residual = float(np.linalg.norm(matrix + matrix.conj().T))
        threshold = ROUNDOFF_TOLERANCE * max(1.0, float(np.linalg.norm(matrix)))
        if skew_residual > threshold:
            raise ValidationError(
                "generator must be skew-Hermitian within block_cmv_unitarity: "
                f"residual={skew_residual:.3e}, threshold={threshold:.3e}"
            )
        return np.asarray(expm(matrix), dtype=np.complex128)

    def random_terminal(self, generator: np.random.Generator) -> ComplexArray:
        """Return a Haar-distributed unitary on the terminal defect space.

        The terminal block places only ``t`` of the realization's nodes, so no
        closure moves weight on moment-fixed nodes. The best invalid arc weight
        over many samples bounds what any closure search could reach.
        """

        dimension = self._require_free_terminal()
        if not isinstance(generator, np.random.Generator):
            raise ValidationError("generator must be a numpy.random.Generator")
        sample = generator.normal(size=(dimension, dimension)) + 1.0j * generator.normal(
            size=(dimension, dimension)
        )
        factor, upper = np.linalg.qr(sample)
        # Divide out the phases LAPACK puts on the diagonal of ``upper``, or the draw is not Haar.
        diagonal = np.diagonal(upper).copy()
        diagonal /= np.abs(diagonal)
        return np.asarray(factor * diagonal, dtype=np.complex128)

    def evaluate(
        self,
        terminal: Any = None,
        *,
        phase: float | None = None,
    ) -> SectorClosureDiagnostics:
        """Return the scan's diagnostics at ``terminal``, or at the canonical closure for ``None``.

        ``phase`` only labels the result. The ranking uses it as the last
        tiebreaker, so passing it makes a scalar candidate rank exactly as in
        the scan; a diagonal or general closure leaves it ``None``. A failed
        reclosure raises a sector :class:`RefusalError`.
        """

        if terminal is None or self.terminal_is_fixed:
            realization = self.canonical
        else:
            candidate = np.asarray(terminal, dtype=np.complex128)
            try:
                realization = self.canonical.reclose(
                    candidate,
                    tolerances=self.tolerances,
                )
            except RefusalError as error:
                if error.kind != "block-cmv":
                    raise
                raise RefusalError(
                    f"{self.sector.name.lower()} closure reclosure failed at "
                    f"the supplied terminal unitary: {error}",
                    kind="sector",
                    diagnostics=error,
                ) from error
        if self.terminal_is_fixed:
            label = None
        elif phase is None:
            label = None
        else:
            label = float(phase)
            if not np.isfinite(label):
                raise ValidationError("phase must be finite")
        return _candidate_diagnostics(
            realization.spectrum(),
            label,
            self.sector,
            self.source,
            self.fitted_order,
            self.minimum_distance,
            self.tolerances,
        )

    def invalid_arc_weight(self, terminal: Any = None) -> float:
        """Return the wrong-arc trace weight that a support search minimises."""

        return self.evaluate(terminal).wrong_arc_weight

    def scalar_arc_weight(self, phase: float) -> float:
        """Return the wrong-arc weight at one scalar phase."""

        return self.evaluate(self.scalar(phase), phase=phase).wrong_arc_weight

    def diagonal_arc_weight(self, phases: Any) -> float:
        """Return the wrong-arc weight at one diagonal closure."""

        return self.evaluate(self.diagonal(phases)).wrong_arc_weight

    @property
    def weight_threshold(self) -> float:
        """Return the sharp acceptance threshold on wrong-arc weight."""

        return self.evaluate(None).weight_threshold

    def ranking_key(self, candidate: SectorClosureDiagnostics) -> tuple[float, ...]:
        """Return the scan's own ranking key for one evaluated candidate."""

        if not isinstance(candidate, SectorClosureDiagnostics):
            raise ValidationError("candidate must be a SectorClosureDiagnostics instance")
        return _ranking_key(candidate)


def build_closure_objective(
    moments: MatrixCayleyMoments | Any,
    sector: Sector,
    *,
    n_conserved: int | None = None,
    backend: ObjectiveBackend = "toeplitz",
    minimum_node_distance: float | None = None,
    tolerances: Tolerances = DEFAULT_TOLERANCES,
) -> ClosureObjective:
    """Build the closure-independent part of a sector realization once.

    The setup mirrors :meth:`SectorSelfEnergyRealization.scan_closures`, so an
    objective and a scan over the same inputs see the same candidates. It takes
    no ``mapping``, since the gates read only the sector's arc sign, and no
    ``"auto"`` backend, since the two backends fail on disjoint populations.
    ``n_conserved`` defaults to every supplied order, which leaves no withheld
    moments.
    """

    if not isinstance(sector, Sector):
        raise ValidationError("sector must be a Sector value")
    if backend not in ("block-cmv", "toeplitz"):
        raise ValidationError("backend must be 'block-cmv' or 'toeplitz'")
    _check.tolerances(tolerances)
    source = moments if isinstance(moments, MatrixCayleyMoments) else MatrixCayleyMoments(moments)
    if n_conserved is None:
        fitted_order = source.n_max
    else:
        if isinstance(n_conserved, bool) or not isinstance(n_conserved, int):
            raise ValidationError("n_conserved must be a nonnegative integer")
        if n_conserved < 0:
            raise ValidationError("n_conserved must be a nonnegative integer")
        if n_conserved > source.n_max:
            raise ValidationError("n_conserved cannot exceed the highest supplied moment order")
        fitted_order = int(n_conserved)
    if minimum_node_distance is None:
        minimum_distance = ABSOLUTE_TOLERANCE
    else:
        value = float(minimum_node_distance)
        if not np.isfinite(value) or value <= 0.0:
            raise ValidationError("minimum_node_distance must be a positive finite number")
        minimum_distance = value

    fitted = MatrixCayleyMoments(source.values[: fitted_order + 1])
    try:
        if backend == "toeplitz":
            canonical: UnitaryMomentRealization = ToeplitzRealization.realize(
                fitted,
                tolerances=tolerances,
            )
        else:
            canonical = BlockCMVRealization.realize(fitted, tolerances=tolerances)
    except RefusalError as error:
        if error.kind != "block-cmv":
            raise
        raise RefusalError(
            f"{sector.name.lower()} sector closure-independent construction "
            f"failed while fitting C_0 through C_{fitted_order}: {error}",
            kind="sector",
            diagnostics=error,
        ) from error

    return ClosureObjective(
        sector=sector,
        source=source,
        fitted_order=fitted_order,
        minimum_distance=minimum_distance,
        backend=backend,
        canonical=canonical,
        tolerances=tolerances,
    )
