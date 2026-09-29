"""Coupling check, weighted Gram and node refusal of :mod:`cayleygw.screening`."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import RefusalError
from ..validate import ComplexArray, FloatArray, _check


def _sealed_coupling(values: Any) -> FloatArray:
    """Return the coupling as a finite read-only C-contiguous real array.

    An array that is already sealed is returned as it is, which saves a copy of
    the largest array after the DF factors.
    """

    source = np.asarray(values)
    if (
        source.dtype == np.float64
        and source.flags.c_contiguous
        and not source.flags.writeable
        and np.all(np.isfinite(source))
    ):
        return source
    return _check.readonly_real(source, "v_matrix")


def _largest_squared_singular_value(coupling: FloatArray) -> float:
    r"""Return :math:`\sigma_1(V)^2 = \lambda_{\max}(V^\dagger V)`, clamped at zero.

    The ``naux`` Gram eigensolve is several times faster than a thin SVD of a
    tall coupling. Squaring the condition number spoils the small singular
    values but not :math:`\sigma_1`, which agrees with the SVD to 2e-15 even at
    condition 5e17.
    """

    gram = coupling.T @ coupling
    # Remove the roundoff asymmetry before the eigensolve.
    gram = 0.5 * (gram + gram.T)
    return max(float(np.linalg.eigvalsh(gram)[-1]), 0.0)


# Transition rows per block in _weighted_gram; only larger couplings are blocked.
_GRAM_ROW_BLOCK = 16384


def _weighted_gram(
    coupling: FloatArray,
    weights: ComplexArray,
) -> ComplexArray:
    r"""Return :math:`V^\dagger \operatorname{diag}(w) V` as two real products.

    A real ``V`` against complex weights costs half the arithmetic of a complex
    product. The result differs from a complex product by up to about four
    units in the last place.
    """

    rows = coupling.shape[0]
    if rows <= _GRAM_ROW_BLOCK:
        gram = (
            coupling.T @ (weights.real[:, None] * coupling)
        ).astype(np.complex128)
        gram += 1j * (coupling.T @ (weights.imag[:, None] * coupling))
        return gram
    # Bounds the scaled copy w * V; the blocked sum agrees to roundoff, not bitwise.
    naux = coupling.shape[1]
    real_part = np.zeros((naux, naux), dtype=np.float64)
    imaginary_part = np.zeros((naux, naux), dtype=np.float64)
    for start in range(0, rows, _GRAM_ROW_BLOCK):
        block = coupling[start : start + _GRAM_ROW_BLOCK]
        real_part += block.T @ (weights.real[start : start + _GRAM_ROW_BLOCK, None] * block)
        imaginary_part += block.T @ (
            weights.imag[start : start + _GRAM_ROW_BLOCK, None] * block
        )
    gram = real_part.astype(np.complex128)
    gram += 1j * imaginary_part
    return gram


def _raise_node_error(
    kind: str,
    node: complex,
    free_distance: float,
    rpa_distance: float,
    threshold: float,
) -> None:
    """Raise the ``"resolvent-node"`` refusal of a node too close to a pole."""

    raise RefusalError(
        f"resolvent node {node!r} is too close to a {kind}; "
        f"free distance={free_distance:.3e}, RPA distance="
        f"{rpa_distance:.3e}, exclusion threshold={threshold:.3e}. "
        "Increase omega_p, or supply a contour that keeps clear of the "
        "poles.",
        kind="resolvent-node",
        zeta=node,
        free_pole_distance=free_distance,
        rpa_pole_distance=rpa_distance,
        pole_threshold=threshold,
    )
