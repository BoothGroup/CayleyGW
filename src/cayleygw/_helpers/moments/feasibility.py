r"""Feasibility checks of a truncated Cayley-moment sequence.

:math:`C^{(0)}, \ldots, C^{(n)}` are the moments of a positive matrix measure
on the unit circle only if their block Toeplitz matrix is positive
semidefinite, with :math:`C^{(-n)} = C^{(n)\dagger}`. The measure lies on one
semicircle only if the semicircle localizer is positive semidefinite too, and
its support is the range of :math:`C^{(0)}`. The checks report and never
repair.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...realization._helpers.toeplitz import _block_toeplitz
from .. import tolerances as limits
from ..types import Sector


def feasibility_violations(positive_moments: Any, sector: Sector) -> tuple[str, ...]:
    """Name every failed Hermiticity, positivity and support check of one sector.

    The Toeplitz matrix and the localizer of degree ``n_max - 1`` are built on
    the normalized retained support of ``C_0``, which is congruent to the
    physical conditions and does not hide a violation in a weak direction.
    Every check uses the shared absolute-plus-relative tolerance.

    Args:
        positive_moments: Moments ``C[0], ..., C[n_max]`` with ``n_max >= 1``; not altered.
        sector: Sets the sign of the localizer (lower or upper semicircle).

    Returns:
        One entry per failed check with its size in tolerances; empty if feasible.
    """

    moments = np.ascontiguousarray(positive_moments, dtype=np.complex128)

    # The retained support of C_0 and the leakage of every moment out of it.
    zeroth = 0.5 * (moments[0] + moments[0].conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(zeroth)
    maximum = max(0.0, float(eigenvalues[-1]))
    rank_threshold = limits.ABSOLUTE_TOLERANCE + limits.RELATIVE_TOLERANCE * maximum
    retained_mask = eigenvalues > rank_threshold
    basis = eigenvectors[:, retained_mask]
    retained = eigenvalues[retained_mask]
    projector = basis @ basis.conj().T
    complement = np.eye(moments.shape[1], dtype=np.complex128) - projector
    left = np.linalg.norm(
        np.einsum("pq,kqr->kpr", complement, moments, optimize=True),
        axis=(1, 2),
    )
    right = np.linalg.norm(
        np.einsum("kpq,qr->kpr", moments, complement, optimize=True),
        axis=(1, 2),
    )
    projected = np.einsum(
        "pa,kab,bq->kpq",
        projector,
        moments,
        projector,
        optimize=True,
    )
    bilateral = np.linalg.norm(moments - projected, axis=(1, 2))
    moment_scales = np.linalg.norm(moments, axis=(1, 2))
    zeroth_scale = float(moment_scales[0])
    support_thresholds = limits.ABSOLUTE_TOLERANCE + limits.RELATIVE_TOLERANCE * np.maximum(
        moment_scales, zeroth_scale
    )
    support_violation = float(
        np.max(np.maximum.reduce((left, right, bilateral)) / support_thresholds)
    )

    # R+ C[n] R+^dagger on the retained support, when there is one.
    if retained.size:
        pseudoinverse = np.ascontiguousarray((1.0 / np.sqrt(retained))[:, None] * basis.conj().T)
        normalized = np.einsum(
            "ap,kpq,bq->kab",
            pseudoinverse,
            moments,
            pseudoinverse.conj(),
            optimize=True,
        )
        prefix = "support-normalized "
    else:
        normalized = moments
        prefix = ""
    toeplitz = _block_toeplitz(np.ascontiguousarray(normalized))
    # Localizer block (i, j) is arc_sign (C[j-i+1] - C[j-i-1]) / 2i: Toeplitz, shifted.
    block = normalized.shape[1]
    localizing = np.ascontiguousarray(
        sector.arc_sign * (toeplitz[:-block, block:] - toeplitz[block:, :-block]) / (2.0j)
    )

    failures: list[str] = []
    for label, matrix in (
        ("zeroth moment", moments[0]),
        (prefix + "block Toeplitz matrix", toeplitz),
        (prefix + "semicircle localizing matrix", localizing),
    ):
        spectrum = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))
        threshold = limits.ABSOLUTE_TOLERANCE + limits.RELATIVE_TOLERANCE * float(
            np.max(np.abs(spectrum))
        )
        hermiticity = float(np.linalg.norm(matrix - matrix.conj().T)) / threshold
        positivity = max(0.0, -float(spectrum[0])) / threshold
        if hermiticity > 1.0:
            failures.append(f"{label} Hermiticity ({hermiticity:.3e} tolerances)")
        if positivity > 1.0:
            failures.append(f"{label} positivity ({positivity:.3e} tolerances)")
    if support_violation > 1.0:
        failures.append(f"zeroth-moment support ({support_violation:.3e} tolerances)")
    return tuple(failures)
