"""Dense direct-RPA and Tamm-Dancoff response problems, and exact Coulomb factors.

The dense solves are the small-system oracles for the contour route. The exact
factors serve any reference that is not density fitted.
"""

from __future__ import annotations

import numpy as np

from .. import tolerances as limits
from ..errors import RefusalError, ValidationError
from ..pyscf import RestrictedMolecularReference
from ..types import Screening
from ..validate import FloatArray


def solve_response(
    reference: RestrictedMolecularReference,
    eri: FloatArray,
    screening: Screening = Screening.RPA,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Diagonalize the restricted-singlet particle-hole response.

    With :math:`g_{ia,jb} = (ia|jb)` and :math:`K = 2g`, direct RPA
    diagonalizes :math:`H = D^2 + 2 D^{1/2} K D^{1/2}` and the TDA
    :math:`A = D + K`. Transitions are ordered occupied-major.

    Args:
        reference: Closed-shell restricted active-orbital reference.
        eri: Active-MO Coulomb tensor ``(pq|rs)`` in chemists' notation.
        screening: Response model; direct RPA by default.

    Returns:
        The gaps :math:`\epsilon_a - \epsilon_i`, the ascending excitation
        energies :math:`\Omega_\nu`, and the amplitudes with transition rows and
        mode columns: ``X + Y`` normalized so that ``(X + Y).T @ (X - Y) = I``
        for direct RPA, the orthonormal ``X`` for the TDA. PySCF's ``td.xy``
        holds one spin component and is smaller by ``sqrt(2)``.

    Raises:
        ValidationError: If the Coulomb matrix is asymmetric or, for the TDA, a
            gap or excitation energy is not safely positive.
        RefusalError: If, for direct RPA, a gap or excitation energy is not
            safely positive.
    """

    rpa = screening is Screening.RPA
    occupied = np.asarray(reference.occupied_positions, dtype=np.int64)
    virtual = np.asarray(reference.virtual_positions, dtype=np.int64)
    gaps = (reference.mo_energy[virtual][None, :] - reference.mo_energy[occupied][:, None]).reshape(
        -1
    )
    coulomb = np.asarray(eri)[np.ix_(occupied, virtual, occupied, virtual)].reshape(
        gaps.size, gaps.size
    )
    symmetry_residual = float(np.max(np.abs(coulomb - coulomb.T)))
    if symmetry_residual > limits.ROUNDOFF_TOLERANCE:
        raise ValidationError(
            f"spatial Coulomb matrix is not symmetric; maximum residual is {symmetry_residual:.3e}"
        )
    kernel = 2.0 * (0.5 * (coulomb + coulomb.T))
    minimum_gap = float(np.min(gaps))
    if minimum_gap <= limits.REFERENCE_TOLERANCE:
        message = (
            f"{screening.label} requires D to be positive definite and well "
            "separated from zero; minimum particle-hole gap is "
            f"{minimum_gap:.3e} Hartree"
        )
        if rpa:
            raise RefusalError(message, kind="rpa-instability")
        raise ValidationError(message)

    if rpa:
        square_root = np.sqrt(gaps)
        squared = np.diag(np.square(gaps))
        squared += 2.0 * square_root[:, None] * kernel * square_root[None, :]
        squared_energies, eigenvectors = np.linalg.eigh(squared)
        minimum_squared = float(squared_energies[0])
        if minimum_squared <= limits.REFERENCE_TOLERANCE**2:
            raise RefusalError(
                "direct-RPA squared matrix has a nonpositive or near-zero "
                "mode; minimum squared excitation is "
                f"{minimum_squared:.3e} Hartree**2",
                kind="rpa-instability",
            )
        energies = np.sqrt(squared_energies)
        amplitudes = square_root[:, None] * eigenvectors / np.sqrt(energies)[None, :]
        return gaps, energies, np.ascontiguousarray(amplitudes)

    a_matrix = kernel
    a_matrix[np.diag_indices(gaps.size)] += gaps
    energies, amplitudes = np.linalg.eigh(a_matrix)
    minimum_eigenvalue = float(energies[0])
    if minimum_eigenvalue <= limits.REFERENCE_TOLERANCE:
        raise ValidationError(
            "the Tamm-Dancoff matrix has a nonpositive or near-zero mode; "
            f"minimum eigenvalue is {minimum_eigenvalue:.3e} Hartree"
        )
    return gaps, energies, np.ascontiguousarray(amplitudes)


def exact_coulomb_factors(eri: FloatArray) -> FloatArray:
    r"""Factor a real four-index MO Coulomb tensor by a dense eigendecomposition.

    Eigenvalues of the pair matrix within the PSD tolerance are dropped as null
    directions; a larger negative one raises rather than being clipped.

    Args:
        eri: ``(pq|rs)`` in Hartree, shape ``(nmo, nmo, nmo, nmo)``.

    Returns:
        Read-only factors with :math:`(pq|rs) = \sum_P L_{P,pq} L_{P,rs}`,
        shape ``(naux, nmo, nmo)``.

    Raises:
        ValidationError: If the pair matrix is materially indefinite.
    """

    nmo = eri.shape[0]
    pair_matrix = np.asarray(eri).reshape(nmo * nmo, nmo * nmo)
    pair_matrix = 0.5 * (pair_matrix + pair_matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(pair_matrix)
    largest = float(np.max(np.abs(eigenvalues)))
    negative_threshold = limits.ABSOLUTE_TOLERANCE + limits.RELATIVE_TOLERANCE * largest
    null_threshold = limits.ROUNDOFF_TOLERANCE * max(1.0, largest)
    minimum_eigenvalue = float(eigenvalues[0])
    if minimum_eigenvalue < -negative_threshold:
        raise ValidationError(
            "MO-pair Coulomb matrix is not positive semidefinite; minimum "
            f"eigenvalue is {minimum_eigenvalue:.3e} Hartree"
        )
    retained = eigenvalues > null_threshold
    factors = (np.sqrt(eigenvalues[retained])[:, None] * eigenvectors[:, retained].T).reshape(
        -1, nmo, nmo
    )
    # eigh's rotation within a degenerate eigenspace can break p <-> q per factor.
    factors = np.ascontiguousarray(0.5 * (factors + factors.swapaxes(1, 2)))
    factors.setflags(write=False)
    return factors
