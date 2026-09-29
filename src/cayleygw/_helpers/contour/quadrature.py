"""Contour geometry checks and node-level gates of :mod:`cayleygw.contour`."""

from __future__ import annotations

import cmath
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import tolerances as limits
from ..cayley import CayleyMap
from ..errors import RefusalError, ValidationError
from ..pyscf import RestrictedMolecularReference
from ..types import Screening, Sector
from ..validate import ComplexArray, IntArray

if TYPE_CHECKING:
    from ...contour import EllipseContour
    from ...screening import ProjectedRPAResolvent


def _raise_lowest_node_failure(
    failures: list[tuple[int, Exception]],
    nodes: ComplexArray,
) -> None:
    """Raise the lowest-index node failure, whatever order the workers hit them in.

    A ``"resolvent-node"`` refusal becomes a ``"contour-node"`` refusal; any
    other error is re-raised as it is.
    """

    index, error = failures[0]
    if not (isinstance(error, RefusalError) and error.kind == "resolvent-node"):
        raise error
    node = nodes[index]
    raise RefusalError(
        f"contour node {index} at {node!r} is prohibited: {error}",
        kind="contour-node",
        diagnostics=error,
        node_index=index,
        zeta=error.zeta,
        free_pole_distance=error.free_pole_distance,
        rpa_pole_distance=error.rpa_pole_distance,
        pole_threshold=error.pole_threshold,
    ) from error


def _confocal_sigma(zeta: complex, center: float, focal: float) -> float:
    """Return the confocal parameter of ``zeta``, zero on the focal segment."""

    return abs(cmath.acosh((zeta - center) / focal).real)


def _equioscillation_sigma_min(
    lower: float,
    center: float,
    focal: float,
    reference: Any,
    mapping: CayleyMap,
    screening: Screening,
) -> tuple[float, str]:
    """Return ``(sigma_min, limiting family)`` over the excluded singularities.

    Every Cayley kernel pole of both sectors is enumerated, which also covers
    frozen cores. For direct RPA the nearest mirror pole is taken at
    ``-Omega_L``, the certified lower bound; the TDA has no mirror family.
    """

    energies = np.asarray(reference.mo_energy, dtype=float)
    kernel = math.inf
    for sector in (Sector.HOLE, Sector.PARTICLE):
        internal = np.asarray(
            reference.occupied_positions if sector is Sector.HOLE else reference.virtual_positions
        )
        shifted = energies[internal] - mapping.center
        if sector is Sector.PARTICLE:
            shifted = -shifted
        for real_part in shifted:
            kernel = min(
                kernel,
                _confocal_sigma(complex(float(real_part), mapping.scale), center, focal),
            )
    candidates = [(kernel, "kernel")]
    if screening is Screening.RPA:
        candidates.append((_confocal_sigma(complex(-lower, 0.0), center, focal), "mirror"))
    return min(candidates, key=lambda item: item[0])


def _validate_enclosure(
    resolvent: ProjectedRPAResolvent,
    contour: EllipseContour,
) -> None:
    """Check the contour strictly encloses the certified bounds, before any solve.

    Direct RPA's resolvent is a function of ``zeta**2``, so its contour must
    also stay right of the imaginary axis or it covers the spectrum twice. The
    TDA inverts ``zeta`` itself and may cross.
    """

    enclosure_points = np.asarray(
        [
            resolvent.spectral_lower_bound,
            resolvent.spectral_upper_bound,
        ]
    )
    scale = max(resolvent.spectral_upper_bound, contour.rightmost)
    geometry_threshold = limits.ABSOLUTE_TOLERANCE + limits.RELATIVE_TOLERANCE * scale
    if resolvent.screening is Screening.RPA and contour.leftmost <= geometry_threshold:
        raise ValidationError(
            "contour reaches or crosses the imaginary axis, so zeta squared is not one-to-one"
        )
    radii = np.asarray(contour.normalized_radius(enclosure_points))
    missed = radii >= 1.0 - limits.ROUNDOFF_TOLERANCE
    if np.any(missed):
        raise ValidationError(
            "contour does not strictly enclose the certified "
            f"{'RPA' if resolvent.screening is Screening.RPA else 'TDA'} "
            "interval endpoints; "
            f"missed energies={enclosure_points[missed].tolist()}"
        )


def _kernel_pole_orbitals(
    reference: RestrictedMolecularReference,
    contour: EllipseContour,
    sector: Sector,
    mapping: CayleyMap,
) -> IntArray:
    """Check a sector's Cayley kernel poles lie outside the contour; return its orbitals.

    Hole poles ``eps_i - mu - i omega_p`` keep a negative real part only if
    ``mu`` is above the HOMO, particle poles ``mu - eps_a + i omega_p`` only if
    it is below the LUMO.
    """

    if sector is Sector.HOLE:
        if not mapping.center > reference.homo_energy:
            raise ValidationError(
                f"hole Cayley-map center {mapping.center:.6f} Ha must lie "
                f"strictly above the reference HOMO at {reference.homo_energy:.6f} Ha"
            )
    elif not mapping.center < reference.lumo_energy:
        raise ValidationError(
            f"particle Cayley-map center {mapping.center:.6f} Ha must lie "
            f"strictly below the reference LUMO at {reference.lumo_energy:.6f} Ha"
        )
    internal = (
        reference.occupied_positions if sector is Sector.HOLE else reference.virtual_positions
    )
    if sector is Sector.HOLE:
        poles = reference.mo_energy[internal] - mapping.center - 1j * mapping.scale
    else:
        poles = mapping.center - reference.mo_energy[internal] + 1j * mapping.scale
    radii = np.asarray(contour.normalized_radius(poles))
    enclosed = radii <= 1.0 + limits.ROUNDOFF_TOLERANCE
    if np.any(enclosed):
        raise ValidationError(
            f"{sector.name.lower()} Cayley kernel has enclosed or boundary "
            f"poles at {poles[enclosed].tolist()}"
        )
    return internal


def _symmetrize_into(
    values: ComplexArray,
    node_index: int,
    out: ComplexArray,
) -> None:
    """Reject a non-symmetric node resolvent, else write its symmetric part to ``out``."""

    residual = float(np.linalg.norm(values - values.T, ord="fro"))
    scale = float(np.linalg.norm(values, ord="fro"))
    threshold = limits.ABSOLUTE_TOLERANCE + limits.RELATIVE_TOLERANCE * scale
    if residual > threshold:
        raise ValidationError(
            "real-V projected resolvent is materially non-symmetric at "
            f"contour node {node_index}: residual={residual:.3e}, "
            f"threshold={threshold:.3e}"
        )
    np.add(values, values.T, out=out)
    out *= 0.5


def _hermitian_zeroth(
    values: ComplexArray,
    position: int,
    label: str,
) -> None:
    """Check, then Hermitianize, the zeroth order of one orbital's moment stack.

    It runs per orbital, before the sum, because defects in separate orbitals
    can cancel in the total.
    """

    matrix = values[0]
    residual = float(np.linalg.norm(matrix - matrix.conj().T, ord="fro"))
    scale = float(np.linalg.norm(matrix, ord="fro"))
    threshold = limits.ABSOLUTE_TOLERANCE + limits.RELATIVE_TOLERANCE * scale
    if residual > threshold:
        raise ValidationError(
            f"paired {label} C_0 is materially non-Hermitian at "
            f"internal position {position}: residual={residual:.3e}, "
            f"threshold={threshold:.3e}"
        )
    values[0] = 0.5 * (matrix + matrix.conj().T)
