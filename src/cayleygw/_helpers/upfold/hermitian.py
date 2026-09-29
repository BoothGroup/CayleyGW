"""Hermitian input check of :class:`cayleygw.upfold.UpfoldedDysonHamiltonian`."""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import tolerances as limits
from ..errors import ValidationError
from ..validate import ComplexArray, _check


def _validated_hermitian(
    values: Any,
    name: str,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> ComplexArray:
    """Check that a matrix is square and Hermitian, and return it symmetrized, read-only."""

    matrix = _check.readonly_complex(values, name)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValidationError(f"{name} must be nonempty and square")
    if expected_shape is not None and matrix.shape != expected_shape:
        raise ValidationError(f"{name} must have shape {expected_shape}")
    residual = float(np.linalg.norm(matrix - matrix.conj().T, ord="fro"))
    threshold = limits.ABSOLUTE_TOLERANCE + limits.RELATIVE_TOLERANCE * float(
        np.linalg.norm(matrix, ord="fro")
    )
    if residual > threshold:
        raise ValidationError(
            f"{name} must be Hermitian: residual={residual:.3e}, "
            f"threshold={threshold:.3e}"
        )
    # Symmetrized so eigh sees an exactly Hermitian matrix.
    hermitian = 0.5 * (matrix + matrix.conj().T)
    return _check.readonly_complex(hermitian, name)
