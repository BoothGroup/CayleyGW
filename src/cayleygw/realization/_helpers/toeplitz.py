"""Private helpers of :mod:`cayleygw.realization.toeplitz`."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..._helpers import tolerances as limits
from ..._helpers.errors import RefusalError
from ..._helpers.validate import ComplexArray, FloatArray, _check
from ...tools.parallel import limited_native_threads
from ...tools.records.realization import GramSpectrum
from .base import _conserved_fields, _hermitian_square

if TYPE_CHECKING:
    from ...tools.records.realization import (
        NormalizedMatrixCayleyMoments,
        ToeplitzDiagnostics,
    )


def _seal(values: ComplexArray, name: str) -> ComplexArray:
    """Check a freshly built array is finite and mark it read-only, without a copy."""

    if not np.all(np.isfinite(values)):
        raise RefusalError(f"{name} must contain finite values", kind="block-cmv")
    values.setflags(write=False)
    return values


def _block_toeplitz(values: ComplexArray) -> ComplexArray:
    """Return the Hermitian block Toeplitz Gram matrix of one-sided moments."""

    order = values.shape[0] - 1
    block = values.shape[1]
    result = np.empty(
        ((order + 1) * block, (order + 1) * block),
        dtype=np.complex128,
    )
    for row in range(order + 1):
        row_slice = slice(row * block, (row + 1) * block)
        for column in range(order + 1):
            column_slice = slice(column * block, (column + 1) * block)
            difference = column - row
            result[row_slice, column_slice] = (
                values[difference]
                if difference >= 0
                else values[-difference].conj().T
            )
    return result


def _polar_unitary(matrix: ComplexArray) -> ComplexArray:
    """Return the closest square unitary polar factor of ``matrix``."""

    left, _, right_adjoint = np.linalg.svd(matrix, full_matrices=False)
    return left @ right_adjoint


def _rank_was_deflated(
    normalization: NormalizedMatrixCayleyMoments,
    diagnostics: ToeplitzDiagnostics,
) -> bool:
    """Return whether the Toeplitz square root discarded null directions."""

    full_dimension = (normalization.n_max + 1) * normalization.rank
    return diagnostics.numerical_rank < full_dimension


def _moment_acceptance_thresholds(
    values: ComplexArray,
    normalization: NormalizedMatrixCayleyMoments,
    diagnostics: ToeplitzDiagnostics,
    *,
    base_tolerance: float | None = None,
    absolute_tolerance: float | None = None,
    projection_scale: float = 1.0,
) -> FloatArray:
    """Return strict thresholds plus a rank-deflation backward-error budget.

    The budget applies only if the Gram cut removed directions. It adds the
    feasibility allowance once per side of the two-sided congruence, and the
    Gram-projection residual plus the telescoped shift bound
    ``n * shift_residual * sqrt(1 + gram_residual)`` at order ``n``, scaled by
    ``projection_scale``, which is ``||C0||_2`` in physical space.
    """

    scales = np.linalg.norm(values, axis=(1, 2))
    base = _check.nonnegative_real(
        limits.BLOCK_CMV_MOMENT_TOLERANCE if base_tolerance is None else base_tolerance,
        "base_tolerance",
    )
    _check.nonnegative_real(projection_scale, "projection_scale")
    if absolute_tolerance is None:
        thresholds = base * (1.0 + scales)
    else:
        thresholds = absolute_tolerance + base * scales
    if _rank_was_deflated(normalization, diagnostics):
        thresholds += 2.0 * (
            limits.ABSOLUTE_TOLERANCE + limits.RELATIVE_TOLERANCE * (1.0 + scales)
        )
        orders = np.arange(values.shape[0], dtype=np.float64)
        shift_bounds = (
            orders
            * diagnostics.determined_shift_residual
            * math.sqrt(1.0 + diagnostics.gram_residual)
        )
        thresholds += projection_scale * (
            diagnostics.gram_residual + shift_bounds
        )
    return _check.readonly_real(thresholds, "moment acceptance thresholds")


@dataclass(frozen=True, slots=True)
class _UnitarityCertificate:
    r"""Closed-form ``||M(T).H M(T) - I||_F`` for any terminal ``T``, in ``O(t**3)``.

    Every closure of one base realization is ``M(T) = F + L T.H R`` with the
    same ``F``, ``L`` and ``R``, so

    .. math::

        M^H M - I = S_0 + R^H T A^H + A T^H R + R^H (T N T^H + T T^H - I) R,

    with ``S0 = F.H F + R.H R - I``, ``A = F.H L`` and ``N = L.H L - I``. The
    ``T`` part is the low-rank ``Y = Yl @ Yr`` with ``Yl = [R.H, A, R.H]``,
    ``Yr = [T A.H; T.H R; C R]`` and ``C = T N T.H + T T.H - I``, and
    ``||S0 + Y||**2 = ||S0||**2 + 2 Re tr(S0 Y) + ||Y||**2`` reduces to the
    ``t x t`` blocks stored here. ``S0``, ``A`` and ``N`` are the
    construction's own roundoff-small defects, so nothing cancels and the
    result is accurate relative to the residual itself. It replaces a dense
    ``D**3`` square per candidate; ``scale`` is the largest defect norm, and a
    caller uses the dense square if it is not small against the threshold.
    """

    s0_square: float
    ag: ComplexArray = field(repr=False)
    rh: ComplexArray = field(repr=False)
    rg: ComplexArray = field(repr=False)
    rr: ComplexArray = field(repr=False)
    ra: ComplexArray = field(repr=False)
    aa: ComplexArray = field(repr=False)
    nn: ComplexArray = field(repr=False)
    scale: float

    @classmethod
    def build(
        cls,
        fixed_matrix: ComplexArray,
        terminal_left_basis: ComplexArray,
        terminal_right_adjoint: ComplexArray,
    ) -> "_UnitarityCertificate":
        """Form the blocks once: one Hermitian square and a few thin products."""

        left = terminal_left_basis
        right = terminal_right_adjoint
        dimension = fixed_matrix.shape[0]
        terminal_dimension = left.shape[1]
        s0 = _hermitian_square(fixed_matrix)
        s0 += right.conj().T @ right
        s0[np.diag_indices(dimension)] -= 1.0
        s0 = 0.5 * (s0 + s0.conj().T)
        a = fixed_matrix.conj().T @ left
        g = s0 @ right.conj().T
        h = s0 @ a
        s0_square = float(np.vdot(s0, s0).real)
        del s0
        rr = right @ right.conj().T
        nn = left.conj().T @ left
        nn[np.diag_indices(terminal_dimension)] -= 1.0
        scale = max(
            math.sqrt(s0_square),
            float(np.linalg.norm(a, ord="fro")),
            float(np.linalg.norm(nn, ord="fro")),
            float(np.linalg.norm(
                rr - np.eye(terminal_dimension, dtype=np.complex128), ord="fro"
            )),
        )
        return cls(
            s0_square=s0_square,
            ag=a.conj().T @ g,
            rh=right @ h,
            rg=right @ g,
            rr=rr,
            ra=right @ a,
            aa=a.conj().T @ a,
            nn=nn,
            scale=scale,
        )

    def residual(self, terminal: ComplexArray) -> float:
        """Return ``||M(T).H M(T) - I||_F`` for the given terminal unitary."""

        t = np.asarray(terminal, dtype=np.complex128)
        th = t.conj().T
        identity = np.eye(t.shape[0], dtype=np.complex128)
        c = t @ self.nn @ th + t @ th - identity
        ch = c.conj().T
        cross = (
            np.trace(t @ self.ag) + np.trace(th @ self.rh) + np.trace(c @ self.rg)
        )
        rah = self.ra.conj().T
        # ||Y||**2 = sum_ij tr(gl[i][j] gr[j][i]), gl = Yl.H Yl fixed, gr = Yr Yr.H.
        gl = (
            (self.rr, self.ra, self.rr),
            (rah, self.aa, rah),
            (self.rr, self.ra, self.rr),
        )
        gr = (
            (t @ self.aa @ th, t @ rah @ t, t @ rah @ ch),
            (th @ self.ra @ th, th @ self.rr @ t, th @ self.rr @ ch),
            (c @ self.ra @ th, c @ self.rr @ t, c @ self.rr @ ch),
        )
        square = 0.0j
        for i in range(3):
            for j in range(3):
                square += np.trace(gl[i][j] @ gr[j][i])
        value = self.s0_square + 2.0 * cross.real + square.real
        return math.sqrt(max(value, 0.0))


def _projection_scale(normalization: NormalizedMatrixCayleyMoments) -> float:
    """Return ``||C0||_2``, the congruence factor back to physical space."""

    if normalization.support_eigenvalues.size == 0:
        return 0.0
    return float(np.max(normalization.support_eigenvalues))


def _validated_realization(
    *,
    normalization: NormalizedMatrixCayleyMoments,
    terminal: ComplexArray,
    fixed_matrix: ComplexArray,
    terminal_left_basis: ComplexArray,
    terminal_right_adjoint: ComplexArray,
    selector: ComplexArray,
    base_diagnostics: ToeplitzDiagnostics,
    certificate: _UnitarityCertificate | None = None,
) -> dict[str, Any]:
    """Assemble one closure from the cached factors, check it, and return its fields.

    A trusted ``certificate`` replaces the dense ``D**3`` unitarity square;
    without one the square is formed and the certificate built for later
    closures.
    """

    matrix = fixed_matrix.copy()
    if terminal.shape[0]:
        # As in block-CMV, ``U`` holds the adjoint of the Verblunsky terminal.
        matrix += (
            terminal_left_basis
            @ terminal.conj().T
            @ terminal_right_adjoint
        )
    matrix = _seal(matrix, "matrix")
    unitarity_threshold = limits.ROUNDOFF_TOLERANCE * max(
        1.0, math.sqrt(matrix.shape[0])
    )
    # Trust the closed form while its roundoff, eps * scale**2, is far below threshold**2.
    if (
        certificate is not None
        and certificate.scale <= 1.0e3 * unitarity_threshold
    ):
        unitarity_residual = certificate.residual(terminal)
    else:
        # Subtract the identity in place: the same arithmetic, two fewer D x D arrays.
        unitarity_square = _hermitian_square(matrix)
        unitarity_square[np.diag_indices_from(unitarity_square)] -= 1.0
        unitarity_residual = float(np.linalg.norm(unitarity_square, ord="fro"))
        del unitarity_square
        if certificate is None and terminal.shape[0]:
            certificate = _UnitarityCertificate.build(
                fixed_matrix, terminal_left_basis, terminal_right_adjoint
            )
    if unitarity_residual > unitarity_threshold:
        raise RefusalError(
            "Toeplitz orthogonal completion is not unitary within "
            f"tolerance: residual={unitarity_residual:.3e}, "
            f"threshold={unitarity_threshold:.3e}",
            kind="block-cmv",
        )
    fields = _conserved_fields(
        normalization,
        matrix,
        selector,
        _moment_acceptance_thresholds(
            normalization.source.values,
            normalization,
            base_diagnostics,
            projection_scale=_projection_scale(normalization),
        ),
        _moment_acceptance_thresholds(
            normalization.values,
            normalization,
            base_diagnostics,
            projection_scale=1.0,
        ),
        "Toeplitz realization",
        # Always zero; kept so the refusal text keeps its fields.
        suffix="; structured_ridge=0.000e+00",
    )
    return dict(
        normalization=normalization,
        terminal_unitary=terminal,
        terminal_from_moments=(terminal.shape[0] == 0),
        matrix=matrix,
        selector=selector,
        unitarity_residual=unitarity_residual,
        diagnostics=base_diagnostics,
        _fixed_matrix=fixed_matrix,
        _terminal_left_basis=terminal_left_basis,
        _terminal_right_adjoint=terminal_right_adjoint,
        _unitarity_certificate=certificate,
        **fields,
    )


def build_gram_spectrum(
    normalized_values: ComplexArray,
    *,
    native_threads: int = 1,
) -> GramSpectrum:
    """Eigendecompose the block Toeplitz Gram matrix of normalized moments.

    Args:
        normalized_values: Normalized moments from
            :meth:`~cayleygw.realization.base.UnitaryMomentRealization.normalize`.
        native_threads: BLAS threads for the eigensolve. They are pinned
            because the smallest eigenvalue is roundoff that moves with the
            thread count, and with it the retained rank and the pole count.

    Returns:
        The decomposition, tagged with the thread count it used.
    """

    gram = _block_toeplitz(normalized_values)
    # Symmetrize in place: two D x D temporaries instead of five, bit for bit.
    adjoint = gram.conj().T
    difference = gram - adjoint
    hermitian_residual = float(np.linalg.norm(difference, ord="fro"))
    del difference
    gram += adjoint
    gram *= 0.5
    del adjoint
    with limited_native_threads(native_threads):
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
    return GramSpectrum(
        gram=gram,
        hermitian_residual=hermitian_residual,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        native_threads=native_threads,
    )


def gram_rank_cut(
    eigenvalues: NDArray[np.float64] | Any,
    rank_floor: float,
) -> tuple[int, float]:
    """Return the retained count and the floor of the Gram cut.

    The floor is ``max(rank_floor, eps * scale)`` with ``scale = max(1,
    |lambda_min|, |lambda_max|)`` over the ascending ``eigenvalues``, and
    eigenvalues above it are kept. This is the one place the floor becomes a
    rank. ``rank_floor = 0`` gives the floor from roundoff alone.
    """

    values = np.asarray(eigenvalues, dtype=np.float64)
    epsilon_floor = 0.0
    if values.size:
        scale = max(1.0, abs(float(values[0])), abs(float(values[-1])))
        epsilon_floor = float(np.finfo(np.float64).eps) * scale
    floor = max(float(rank_floor), epsilon_floor)
    # ``cut`` is the index of the first retained eigenvalue.
    cut = int(np.searchsorted(values, floor, side="right"))
    return int(values.size) - cut, floor


def _certified_minimum(eigenvalues: FloatArray, positivity: float) -> float:
    """Return the smallest Gram eigenvalue after checking the positivity breach.

    The breach is the most negative eigenvalue over the largest, zero if none
    is negative. A breach over the positivity tolerance is refused.
    """

    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    scale = max(1.0, abs(maximum), abs(minimum))
    positivity_threshold = positivity * scale
    if minimum < -positivity_threshold:
        raise RefusalError(
            "normalized block Toeplitz moment matrix is materially indefinite: "
            f"minimum_eigenvalue={minimum:.3e}, "
            f"threshold={positivity_threshold:.3e}, "
            f"breach={-minimum / scale:.3e}, "
            f"positivity={positivity:.3e}. Increase N_q.",
            kind="block-cmv",
        )
    return minimum


def _gram_square_root(
    gram_spectrum: GramSpectrum,
    rank_floor: float,
) -> tuple[ComplexArray, float, float]:
    """Return the Gram square root above the rank floor, the floor and its residual.

    The residual is ``||root.H root - gram||_F``. A Gram matrix with nothing
    above the floor is refused.
    """

    gram = gram_spectrum.gram
    eigenvalues = gram_spectrum.eigenvalues
    eigenvectors = gram_spectrum.eigenvectors
    retained_count, floor = gram_rank_cut(eigenvalues, rank_floor)
    small_count = int(eigenvalues.size) - retained_count
    if small_count == eigenvalues.size:
        raise RefusalError(
            "normalized block Toeplitz moment matrix has zero numerical rank",
            kind="block-cmv",
        )

    # A prefix cut of the ascending spectrum, decided by ``gram_rank_cut``.
    retained = np.zeros(eigenvalues.shape, dtype=bool)
    retained[small_count:] = True
    retained_values = np.maximum(eigenvalues[retained], 0.0)
    retained_vectors = eigenvectors[:, retained]
    root = np.sqrt(retained_values)[:, None] * retained_vectors.conj().T
    del retained_vectors
    # In place: the square minus the Gram is one D x D temporary, not two.
    gram_square = _hermitian_square(root)
    gram_square -= gram
    gram_residual = float(np.linalg.norm(gram_square, ord="fro"))
    del gram_square
    return root, floor, gram_residual


def _shift_completion(
    root: ComplexArray,
    rank: int,
    order: int,
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray, int, float]:
    """Complete the shift between adjacent orbit blocks of the Gram square root.

    Returns:
        The selector, the part of the shift the moments determine, the two
        oriented bases of the terminal completion, the rank of the determined
        part and the residual of the provisional shift.
    """

    blocks = tuple(
        root[:, index * rank : (index + 1) * rank]
        for index in range(order + 1)
    )
    selector = _check.readonly_complex(blocks[0], "selector")
    dimension = root.shape[0]
    if order == 0:
        fixed_matrix = np.zeros((dimension, dimension), dtype=np.complex128)
        terminal_left = np.eye(dimension, dtype=np.complex128)
        terminal_right_adjoint = np.eye(dimension, dtype=np.complex128)
        domain_rank = 0
        shift_residual = 0.0
    else:
        domain = np.concatenate(blocks[:-1], axis=1)
        target = np.concatenate(blocks[1:], axis=1)
        cross = target @ domain.conj().T
        left, singular_values, right_adjoint = np.linalg.svd(
            cross, full_matrices=True
        )
        domain_rank = min(dimension, domain.shape[1])
        if domain_rank and singular_values[domain_rank - 1] <= 0.0:
            raise RefusalError(
                "Toeplitz determined shift lost numerical rank: "
                f"expected_rank={domain_rank}, "
                f"smallest_retained_singular_value="
                f"{singular_values[domain_rank - 1]:.3e}",
                kind="block-cmv",
            )
        del cross
        fixed_matrix = left[:, :domain_rank] @ right_adjoint[:domain_rank]
        # Copies, not views, so the two D x D singular-vector factors are freed here.
        terminal_left = left[:, domain_rank:].copy()
        terminal_right_adjoint = right_adjoint[domain_rank:].copy()
        del left, right_adjoint
        # Orient the null bases by their endpoints, so a phase means what it does in CMV.
        terminal_dimension = terminal_left.shape[1]
        if terminal_dimension:
            right_terminal_basis = terminal_right_adjoint.conj().T
            left_endpoint = terminal_left.conj().T @ blocks[0]
            right_endpoint = right_terminal_basis.conj().T @ blocks[-1]
            # Both endpoints are ``(t, r)``, so square exactly when ``t == r``.
            if terminal_dimension == rank:
                left_orientation = _polar_unitary(left_endpoint)
                right_orientation = _polar_unitary(right_endpoint)
            else:
                # LAPACK's gauge stays: orienting here tested worse on real calculations.
                left_orientation = np.eye(terminal_dimension, dtype=complex)
                right_orientation = np.eye(terminal_dimension, dtype=complex)
            terminal_left = terminal_left @ left_orientation
            terminal_right_adjoint = (
                right_orientation.conj().T @ terminal_right_adjoint
            )
        provisional = fixed_matrix.copy()
        if terminal_left.shape[1]:
            provisional += terminal_left @ terminal_right_adjoint
        shift_residual = float(
            np.linalg.norm(provisional @ domain - target, ord="fro")
        )
        del provisional, domain, target
    return (
        selector,
        fixed_matrix,
        terminal_left,
        terminal_right_adjoint,
        domain_rank,
        shift_residual,
    )
