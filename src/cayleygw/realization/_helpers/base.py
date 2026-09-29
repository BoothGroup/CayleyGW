"""Private helpers of :mod:`cayleygw.realization.base`: normalization, poles, ranking.

The records module imports this one, so records appear here as types only and
the caller builds them.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import get_blas_funcs, schur

from ..._helpers import tolerances as limits
from ..._helpers.errors import RefusalError, ValidationError
from ..._helpers.validate import ComplexArray, FloatArray, _check

if TYPE_CHECKING:
    from ...tools.records.realization import (
        MatrixCayleyMoments,
        NormalizedMatrixCayleyMoments,
        SectorClosureDiagnostics,
    )
    from ..base import UnitaryMomentRealization


def _hermitian_square(values: ComplexArray) -> ComplexArray:
    """Return ``values.conj().T @ values`` through BLAS ``zherk``.

    ``zherk`` computes one triangle and reads the operand in place, so it is
    about three times faster than ``@``. It agrees with ``@`` to roundoff, not
    bitwise: identical on NumPy 1.26 / SciPy 1.11, ``5.7e-13`` apart on
    entries of ``3e+02`` on NumPy 2.4 / SciPy 1.17. Both callers compare a
    residual with a threshold, so no rank or pole count depends on that last
    bit. Reflecting the triangle makes the result exactly Hermitian.
    """

    if values.shape[0] == 0:
        # ``herk`` returns an uninitialized buffer for an empty inner dimension.
        return np.zeros((values.shape[1], values.shape[1]), dtype=np.complex128)
    (herk,) = get_blas_funcs(("herk",), (values,))
    # ``trans=2`` is ``'C'``; ``herk`` rejects ``'T'`` only with a message on stderr.
    upper = herk(1.0, values, trans=2, lower=0)
    return upper + np.triu(upper, 1).conj().T


def _unitarity_residual(matrix: ComplexArray) -> float:
    """Return the Frobenius residual ``||matrix.H @ matrix - I||``."""

    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    return float(np.linalg.norm(matrix.conj().T @ matrix - identity, ord="fro"))


def _canonical_terminal_unitary(value: Any, dimension: int) -> ComplexArray:
    """Validate and polar-project a terminal unitary; ``None`` is the identity."""

    if value is None:
        matrix = np.eye(dimension, dtype=np.complex128)
    else:
        matrix = _check.readonly_complex(value, "terminal_unitary")
        if matrix.shape != (dimension, dimension):
            raise ValidationError(f"terminal_unitary must have shape ({dimension}, {dimension})")
    if dimension == 0:
        return _check.readonly_complex(matrix, "terminal_unitary")
    left_residual = _unitarity_residual(matrix)
    right_residual = _unitarity_residual(matrix.conj().T)
    threshold = limits.ROUNDOFF_TOLERANCE * max(1.0, math.sqrt(dimension))
    if max(left_residual, right_residual) > threshold:
        raise RefusalError(
            "terminal_unitary is not unitary within tolerance: "
            f"left_residual={left_residual:.3e}, "
            f"right_residual={right_residual:.3e}, "
            f"threshold={threshold:.3e}",
            kind="block-cmv",
        )
    left, _, right_adjoint = np.linalg.svd(matrix)
    return _check.readonly_complex(left @ right_adjoint, "terminal_unitary")


def _selector_moment_chain(
    matrix: ComplexArray,
    selector: ComplexArray,
    n_max: int,
    *,
    state: ComplexArray | None = None,
    first_order: int = 0,
) -> tuple[ComplexArray, ComplexArray]:
    """Return ``E.H @ U**n @ E`` for ``first_order .. n_max`` and the final state.

    The chain iterates the thin ``U**n @ E``, never the dense ``U**n``.
    ``state`` is ``U**first_order @ E``, and ``None`` starts from ``E``. The
    returned state is ``U**n_max @ E``, so a later call from ``matrix @ state``
    continues the same chain bit for bit.
    """

    rank = selector.shape[1]
    result = np.empty((n_max + 1 - first_order, rank, rank), dtype=np.complex128)
    running = np.array(selector, dtype=np.complex128) if state is None else state
    adjoint = selector.conj().T
    for order in range(first_order, n_max + 1):
        result[order - first_order] = adjoint @ running
        if order < n_max:
            running = matrix @ running
    return _check.readonly_complex(result, "normalized reconstructed moments"), running


def _check_conservation(
    residuals: FloatArray,
    thresholds: FloatArray,
    failure: str,
    suffix: str = "",
) -> None:
    """Refuse at the lowest order whose residual exceeds its threshold."""

    failed = np.flatnonzero(residuals > thresholds)
    if failed.size:
        order = int(failed[0])
        raise RefusalError(
            f"{failure}: order={order}, residual={residuals[order]:.3e}, "
            f"threshold={thresholds[order]:.3e}{suffix}",
            kind="block-cmv",
        )


def _conserved_fields(
    normalization: NormalizedMatrixCayleyMoments,
    matrix: ComplexArray,
    selector: ComplexArray,
    thresholds: FloatArray,
    normalized_thresholds: FloatArray,
    realization: str,
    suffix: str = "",
) -> dict[str, Any]:
    """Rebuild the supplied moments from ``U`` and ``E``, check them, and return the fields.

    The physical ``B U**n B.H`` is the normalized ``E.H U**n E`` under the
    support congruence, so no power of ``U`` is formed in the physical basis.
    """

    factor = normalization.support_factor
    coupling = _check.readonly_complex(factor @ selector.conj().T, "coupling")
    normalized_reconstructed, selector_state = _selector_moment_chain(
        matrix, selector, normalization.n_max
    )
    reconstructed = _check.readonly_complex(
        factor @ normalized_reconstructed @ factor.conj().T,
        "reconstructed moments",
    )
    normalized_residuals = np.linalg.norm(
        normalized_reconstructed - normalization.values,
        axis=(1, 2),
    )
    residuals = np.linalg.norm(
        reconstructed - normalization.source.values,
        axis=(1, 2),
    )
    _check_conservation(
        residuals,
        thresholds,
        f"{realization} does not conserve a supplied matrix moment",
        suffix,
    )
    _check_conservation(
        normalized_residuals,
        normalized_thresholds,
        f"{realization} does not conserve a supplied normalized matrix moment",
    )
    return {
        "coupling": coupling,
        "reconstructed_moments": reconstructed,
        "moment_residuals": _check.readonly_real(residuals, "moment_residuals"),
        "_normalized_reconstructed": normalized_reconstructed,
        "_selector_state": _check.readonly_complex(selector_state, "selector state"),
    }


#: Backends the ``"auto"`` policy tries, in order: block-CMV, then Toeplitz.
_AUTOMATIC_BACKEND_ORDER: tuple[str, ...] = ("block-cmv", "toeplitz")


def _atomic_matrix_moments(
    nodes: ComplexArray,
    couplings: ComplexArray,
    n_max: int,
) -> ComplexArray:
    """Return the atomic moments ``sum_l u_l**n w_l w_l.H`` through ``n_max``."""

    nphysical = couplings.shape[0]
    result = np.empty(
        (n_max + 1, nphysical, nphysical),
        dtype=np.complex128,
    )
    powers = np.ones(nodes.shape, dtype=np.complex128)
    for order in range(n_max + 1):
        result[order] = np.einsum(
            "pl,l,ql->pq",
            couplings,
            powers,
            couplings.conj(),
            optimize=True,
        )
        powers = powers * nodes
    return _check.readonly_complex(result, "atomic matrix moments")


def _unitary_eigendecomposition(
    matrix: ComplexArray,
    residual_threshold: float,
) -> tuple[ComplexArray, ComplexArray, ComplexArray, float]:
    r"""Diagonalize a unitary matrix, through a Hermitian eigensolver where it can.

    A unitary matrix is normal, so the eigenvectors of its Hermitian part
    ``(U + U.H) / 2``, with eigenvalues ``cos(theta)``, are eigenvectors of
    ``U`` wherever ``cos(theta)`` is simple. That costs a fraction of a complex
    Schur decomposition, and this runs once per closure candidate. Atoms
    mirrored on the two arcs share a ``cos(theta)``, and each such cluster is
    fixed by a small Schur decomposition of its block. If the off-diagonal
    residual still exceeds ``residual_threshold``, the full Schur
    decomposition is returned instead, for the caller to check against the
    same threshold.

    Args:
        matrix: Square matrix expected to be unitary.
        residual_threshold: Largest accepted Frobenius norm of the
            off-diagonal part; it decides only whether the cheap route is kept.

    Returns:
        ``(nodes, eigenvectors, matrix @ eigenvectors, offdiagonal_residual)``;
        the caller needs the product for its own residual.
    """

    dimension = matrix.shape[0]

    def residual_of(
        vectors: ComplexArray,
        product: ComplexArray | None = None,
    ) -> tuple[ComplexArray, ComplexArray, float]:
        # Off-diagonal norm of ``Q.H U Q``; a difference of squared norms would cancel.
        if product is None:
            product = matrix @ vectors
        diagonal = np.einsum("ji,ji->i", vectors.conj(), product)
        residual = float(np.linalg.norm(product - vectors * diagonal[None, :], ord="fro"))
        return diagonal, product, residual

    # ``(M + M.H) / 2`` with one temporary instead of three, element for element.
    hermitian = matrix.conj().T
    hermitian += matrix
    hermitian *= 0.5
    _, vectors = np.linalg.eigh(hermitian)
    del hermitian
    diagonal, product, residual = residual_of(vectors)
    if residual <= residual_threshold:
        return diagonal, vectors, product, residual

    # Mirrored atoms share a ``cos(theta)`` and mix; repair each cluster on its block.
    values = np.real(diagonal)
    order = np.argsort(values, kind="stable")
    vectors = np.ascontiguousarray(vectors[:, order])
    product = np.ascontiguousarray(product[:, order])
    values = values[order]
    # A gap ``d`` blurs eigenvectors by ``eps / d``; merge gaps below ``eps / threshold``.
    epsilon = float(np.finfo(np.float64).eps)
    gap_threshold = min(0.5, max(8.0 * epsilon, epsilon / max(residual_threshold, epsilon)))
    boundaries = np.flatnonzero(np.diff(values) > gap_threshold) + 1
    clusters = np.split(np.arange(dimension), boundaries)
    largest = max((c.size for c in clusters), default=0)
    # A cluster spanning everything would cost as much as the fallback.
    if 1 < largest < dimension:
        for cluster in clusters:
            if cluster.size < 2:
                continue
            columns = vectors[:, cluster]
            block = columns.conj().T @ product[:, cluster]
            _, rotation = schur(block, output="complex")
            vectors[:, cluster] = columns @ rotation
            product[:, cluster] = product[:, cluster] @ rotation
        diagonal, product, refined = residual_of(vectors, product)
        if refined <= residual_threshold:
            return diagonal, vectors, product, refined

    # Neither cheap route met the threshold; a nonnormal matrix meets the full check.
    triangular, vectors = schur(matrix, output="complex")
    diagonal = np.diag(triangular).copy()
    return (
        diagonal,
        vectors,
        matrix @ vectors,
        float(np.linalg.norm(triangular - np.diag(diagonal), ord="fro")),
    )


def _selection_key(candidate: SectorClosureDiagnostics) -> tuple[float, ...]:
    """Return the lexicographic selection key of a scalar-phase closure.

    The withheld residual leads because it tracks the spectrum; it is zero for
    every candidate of a scan with no withheld moment, which the geometry then
    ranks. The key has no post-discard conservation term: ranking on it improves
    conservation by orders of magnitude but moves the HOMO off its reference.
    """

    return (
        candidate.maximum_withheld_relative_residual,
        -candidate.minimum_correct_node_distance_from_one,
        candidate.wrong_arc_weight,
        candidate.near_singular_weight,
        candidate.inverse_conditioning_penalty,
        0.0 if candidate.phase is None else candidate.phase,
    )


def _ranking_key(candidate: SectorClosureDiagnostics) -> tuple[float, ...]:
    """Return the selection key with clean support ranked ahead of marginal.

    Two values go before :func:`_selection_key`: the band index, so a closure
    inside the ``arc_margin`` band ranks behind every clean one,
    and, inside the band, the wrong-arc weight. Ranking the band by
    discarded weight means a heavier discard never displaces a lighter one,
    so the band width decides only whether a closure is tried, not which.
    """

    if candidate.support_acceptable:
        return (0.0, 0.0, *_selection_key(candidate))
    return (1.0, candidate.wrong_arc_weight, *_selection_key(candidate))


def _support_compression(source: MatrixCayleyMoments) -> dict[str, Any]:
    r"""Return the fields of :class:`NormalizedMatrixCayleyMoments` but ``source``.

    ``C[0]`` is compressed to its positive support and normalized to the identity.
    """

    values = source.values
    zeroth = values[0]
    hermitian = 0.5 * (zeroth + zeroth.conj().T)
    hermiticity_residual = float(np.linalg.norm(zeroth - zeroth.conj().T, ord="fro"))
    zeroth_scale = float(np.linalg.norm(hermitian, ord=2))
    hermiticity_threshold = limits.ABSOLUTE_TOLERANCE + limits.RELATIVE_TOLERANCE * zeroth_scale
    if hermiticity_residual > hermiticity_threshold:
        raise RefusalError(
            "zeroth matrix moment is materially non-Hermitian: "
            f"residual={hermiticity_residual:.3e}, "
            f"threshold={hermiticity_threshold:.3e}. Rebuild the moments with "
            "a larger N_q.",
            kind="block-cmv",
            check="zeroth_hermiticity",
            residual=hermiticity_residual,
            threshold=hermiticity_threshold,
            order=0,
        )

    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    minimum = float(eigenvalues[0])
    positivity_threshold = hermiticity_threshold
    if minimum < -positivity_threshold:
        raise RefusalError(
            "zeroth matrix moment is not positive semidefinite: "
            f"minimum_eigenvalue={minimum:.3e}, "
            f"threshold={-positivity_threshold:.3e}. Rebuild the moments with "
            "a larger N_q.",
            kind="block-cmv",
            check="zeroth_positivity",
            residual=minimum,
            threshold=-positivity_threshold,
            order=0,
        )
    maximum = max(0.0, float(eigenvalues[-1]))
    rank_threshold = limits.ABSOLUTE_TOLERANCE + limits.RELATIVE_TOLERANCE * maximum
    retained_mask = eigenvalues > rank_threshold
    basis = eigenvectors[:, retained_mask]
    retained = eigenvalues[retained_mask]
    factor = basis * np.sqrt(retained)[None, :]
    if retained.size:
        pseudoinverse = (1.0 / np.sqrt(retained))[:, None] * basis.conj().T
    else:
        pseudoinverse = np.zeros((0, source.nphysical), dtype=np.complex128)
    projector = basis @ basis.conj().T
    projected = np.einsum(
        "pa,kab,bq->kpq",
        projector,
        values,
        projector,
        optimize=True,
    )
    support_residuals = np.linalg.norm(values - projected, axis=(1, 2))
    scales = np.linalg.norm(values, axis=(1, 2))
    thresholds = limits.ABSOLUTE_TOLERANCE + limits.RELATIVE_TOLERANCE * np.maximum(
        scales, float(np.linalg.norm(zeroth, ord="fro"))
    )
    failed = np.flatnonzero(support_residuals > thresholds)
    if failed.size:
        order = int(failed[0])
        raise RefusalError(
            "matrix moment leaves the retained support of C[0]: "
            f"order={order}, residual={support_residuals[order]:.3e}, "
            f"threshold={thresholds[order]:.3e}. Rebuild the moments with a larger N_q.",
            kind="block-cmv",
            check="support_leakage",
            residual=float(support_residuals[order]),
            threshold=float(thresholds[order]),
            order=order,
        )

    normalized = np.einsum(
        "ap,kpq,bq->kab",
        pseudoinverse,
        values,
        pseudoinverse.conj(),
        optimize=True,
    )
    identity = np.eye(retained.size, dtype=np.complex128)
    normalization_residual = float(np.linalg.norm(normalized[0] - identity, ord="fro"))
    support_condition = float(retained[-1] / retained[0]) if retained.size else 1.0
    # The residual is roundoff amplified by the support condition, so allow that too.
    root_rank = max(1.0, float(np.sqrt(retained.size)))
    normalization_threshold = max(
        float(limits.ROUNDOFF_TOLERANCE) * root_rank,
        float(np.finfo(np.float64).eps) * root_rank * float(support_condition),
    )
    if normalization_residual > normalization_threshold:
        raise RefusalError(
            "support-compressed zeroth moment is not the identity: "
            f"residual={normalization_residual:.3e}, "
            f"threshold={normalization_threshold:.3e}, "
            f"retained_condition={support_condition:.3e}.",
            kind="block-cmv",
            check="zeroth_identity",
            residual=normalization_residual,
            threshold=float(normalization_threshold),
            order=0,
            condition=support_condition,
        )

    return dict(
        values=_check.readonly_complex(normalized, "normalized values"),
        support_eigenvalues=_check.readonly_real(retained, "support_eigenvalues"),
        support_factor=_check.readonly_complex(factor, "support_factor"),
        support_pseudoinverse=_check.readonly_complex(pseudoinverse, "support_pseudoinverse"),
    )


def _unit_circle_atoms(realization: UnitaryMomentRealization) -> dict[str, Any]:
    r"""Return the checked atoms, as the fields of :class:`BlockCMVSpectrum`."""

    dimension = realization.dimension
    nphysical = realization.normalization.source.nphysical
    if dimension == 0:
        residuals = np.zeros(realization.conserved_order + 1)
        return dict(
            nodes=_check.readonly_complex(np.zeros(0), "nodes"),
            couplings=_check.readonly_complex(np.zeros((nphysical, 0)), "couplings"),
            trace_weights=_check.readonly_real(np.zeros(0), "trace_weights"),
            moment_residuals=_check.readonly_real(residuals, "moment_residuals"),
        )

    matrix_scale = max(1.0, math.sqrt(dimension))
    schur_threshold = limits.ROUNDOFF_TOLERANCE * matrix_scale
    raw_nodes, eigenvectors, image, schur_residual = _unitary_eigendecomposition(
        realization.matrix,
        schur_threshold,
    )
    if schur_residual > schur_threshold:
        raise RefusalError(
            "unitary block-CMV matrix has a material Schur off-diagonal "
            f"residual: residual={schur_residual:.3e}, "
            f"threshold={schur_threshold:.3e}",
            kind="sector",
        )
    radial = np.abs(np.abs(raw_nodes) - 1.0)
    maximum_radial = float(np.max(radial))
    radial_threshold = limits.ROUNDOFF_TOLERANCE * matrix_scale
    if maximum_radial > radial_threshold:
        raise RefusalError(
            "block-CMV Schur nodes are materially off the unit circle: "
            f"residual={maximum_radial:.3e}, threshold={radial_threshold:.3e}",
            kind="sector",
        )
    nodes = raw_nodes / np.abs(raw_nodes)
    order = np.argsort(np.mod(np.angle(nodes), 2.0 * np.pi), kind="stable")
    nodes = nodes[order]
    eigenvectors = eigenvectors[:, order]
    image = image[:, order]
    couplings = realization.coupling @ eigenvectors
    trace_weights = np.sum(np.abs(couplings) ** 2, axis=0).real
    # ``image`` is ``matrix @ eigenvectors``, formed by the decomposition.
    eigendecomposition_residual = float(
        np.linalg.norm(image - eigenvectors * nodes[None, :], ord="fro")
    )
    eigendecomposition_threshold = 4.0 * limits.ROUNDOFF_TOLERANCE * matrix_scale
    if eigendecomposition_residual > eigendecomposition_threshold:
        raise RefusalError(
            "projected block-CMV eigendecomposition is not accurate enough: "
            f"residual={eigendecomposition_residual:.3e}, "
            f"threshold={eigendecomposition_threshold:.3e}",
            kind="sector",
        )
    reconstructed = _atomic_matrix_moments(
        nodes,
        couplings,
        realization.conserved_order,
    )
    target = realization.normalization.source.values
    moment_residuals = np.linalg.norm(reconstructed - target, axis=(1, 2))
    thresholds = realization.physical_moment_acceptance_thresholds(
        base_tolerance=limits.RELATIVE_TOLERANCE,
        absolute_tolerance=limits.ABSOLUTE_TOLERANCE,
    )
    failed = np.flatnonzero(moment_residuals > thresholds)
    if failed.size:
        failed_order = int(failed[0])
        # Records parse ``order=``, the lowest breach; ``breached=`` and ``highest=`` follow.
        raise RefusalError(
            "spectral block-CMV decomposition does not conserve a supplied "
            f"moment: order={failed_order}, "
            f"residual={moment_residuals[failed_order]:.3e}, "
            f"threshold={thresholds[failed_order]:.3e}, "
            f"breached={failed.size}, highest={int(failed[-1])}. Increase "
            "N_q, or lower n_conserved.",
            kind="sector",
        )
    return dict(
        nodes=_check.readonly_complex(nodes, "nodes"),
        couplings=_check.readonly_complex(couplings, "couplings"),
        trace_weights=_check.readonly_real(trace_weights, "trace_weights"),
        moment_residuals=_check.readonly_real(moment_residuals, "moment_residuals"),
    )
