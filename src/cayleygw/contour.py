r"""The elliptic contour and the quadrature of the Cayley moments on it.

The auxiliary kernel of internal orbital :math:`x` is

.. math::

   \mathcal{K}^{(n,\lessgtr)}(\epsilon_x)
   = \frac{1}{2\pi i} \oint u(\epsilon_x \mp \zeta)^n \, \Pi(\zeta^2) \, d\zeta ,

with :math:`u` the Cayley map and :math:`\Pi` the projected resolvent of
:mod:`cayleygw.screening`. The ellipse encloses the RPA excitation energies and
excludes the Cayley poles, and the midpoint rule on it converges geometrically
in its confocal parameter.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import numpy as np

from ._helpers.cayley import CayleyMap
from ._helpers.contour.contraction import _contract_moments, _sample_nodes
from ._helpers.contour.quadrature import (
    _equioscillation_sigma_min,
    _kernel_pole_orbitals,
    _validate_enclosure,
)
from ._helpers.errors import ValidationError
from ._helpers.types import Sector
from ._helpers.validate import ComplexArray, FloatArray, _check
from .screening import ProjectedRPAResolvent
from .tools.logger import Stages
from .tools.records.contour import ContourSectorMoments, EquioscillationDiagnostics

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EllipseContour:
    r"""Counterclockwise ellipse :math:`\zeta(\theta) = c + a\cos\theta + i b\sin\theta`.

    Attributes:
        center: Centre :math:`c` on the real axis, in Hartree.
        horizontal_radius: Semi-axis :math:`a` along the real axis, in Hartree.
        vertical_radius: Semi-axis :math:`b` along the imaginary axis, in Hartree.
        equioscillation_diagnostics: How :meth:`from_equioscillation` chose the
            ellipse, if it did.
    """

    center: float
    horizontal_radius: float
    vertical_radius: float
    equioscillation_diagnostics: EquioscillationDiagnostics | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        center = _check.positive_real(self.center, "center")
        horizontal = _check.positive_real(
            self.horizontal_radius,
            "horizontal_radius",
        )
        vertical = _check.positive_real(self.vertical_radius, "vertical_radius")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "horizontal_radius", horizontal)
        object.__setattr__(self, "vertical_radius", vertical)

    @property
    def leftmost(self) -> float:
        """Where the ellipse crosses the real axis on the left."""

        return self.center - self.horizontal_radius

    @property
    def rightmost(self) -> float:
        """Where the ellipse crosses the real axis on the right."""

        return self.center + self.horizontal_radius

    def normalized_radius(self, points: Any) -> float | FloatArray:
        """Return each point's normalized radius: below one inside, above one outside."""

        complex_points = _check.readonly_complex(points, "points")
        values = np.square(
            (complex_points.real - self.center) / self.horizontal_radius
        ) + np.square(complex_points.imag / self.vertical_radius)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 0:
            return float(values)
        values.setflags(write=False)
        return values

    def midpoint_rule(self, n_points: int) -> tuple[ComplexArray, ComplexArray]:
        r"""Return the nodes and weights of the periodic midpoint rule.

        The weights hold :math:`d\zeta/d\theta`, the Cauchy factor
        :math:`(2\pi i)^{-1}` and the direct-RPA residue factor two. A screening
        with residue factor :math:`\rho` uses ``rho * (0.5 * weights)``; halving
        is exact, so direct RPA keeps these weights bit for bit.

        Args:
            n_points: Number of nodes, even and at least two.

        Returns:
            The nodes and the weights.

        Raises:
            ValidationError: If ``n_points`` is odd or below two.
        """

        count = _check.positive_integer(n_points, "n_points", minimum=2)
        if count % 2:
            raise ValidationError(f"n_points must be even; got {count}")
        angles = 2.0 * np.pi * (np.arange(count) + 0.5) / count
        nodes = (
            self.center
            + self.horizontal_radius * np.cos(angles)
            + 1j * self.vertical_radius * np.sin(angles)
        )
        derivatives = -self.horizontal_radius * np.sin(angles) + 1j * self.vertical_radius * np.cos(
            angles
        )
        weights = (2.0 / (1j * count)) * derivatives
        # Conjugate partners by conjugation, not trigonometry, so they pair to the last bit.
        half = count // 2
        partners = count - 1 - np.arange(half)
        nodes[partners] = nodes[:half].conj()
        weights[partners] = weights[:half].conj()
        return nodes, weights

    def trapezoid_rule(self, n_points: int) -> tuple[ComplexArray, ComplexArray]:
        r"""Return the nodes and weights of the periodic trapezoidal rule.

        Its nodes, at :math:`\theta = 2\pi j/N`, contain those of every coarser
        grid, so doubling ``n_points`` adds only the midpoint nodes of the
        coarser grid. The nodes at :math:`\theta = 0` and :math:`\pi` are real
        and their own conjugates; each is stored twice at half weight, so the
        arrays hold ``n_points + 2`` entries paired as the midpoint rule's are.

        Args:
            n_points: Number of distinct nodes, even and at least two.

        Returns:
            The nodes and the weights.

        Raises:
            ValidationError: If ``n_points`` is odd or below two.
        """

        count = _check.positive_integer(n_points, "n_points", minimum=2)
        if count % 2:
            raise ValidationError(f"n_points must be even; got {count}")
        angles = 2.0 * np.pi * np.arange(count // 2 + 1) / count
        nodes = (
            self.center
            + self.horizontal_radius * np.cos(angles)
            + 1j * self.vertical_radius * np.sin(angles)
        )
        derivatives = -self.horizontal_radius * np.sin(angles) + 1j * self.vertical_radius * np.cos(
            angles
        )
        # The end nodes exactly real, since sin(pi) is not zero in floating point.
        nodes[[0, -1]] = self.center + self.horizontal_radius, self.center - self.horizontal_radius
        derivatives[[0, -1]] = 1j * self.vertical_radius, -1j * self.vertical_radius
        weights = (2.0 / (1j * count)) * derivatives
        weights[[0, -1]] *= 0.5
        return (
            np.concatenate((nodes, nodes[::-1].conj())),
            np.concatenate((weights, weights[::-1].conj())),
        )

    @classmethod
    def from_equioscillation(
        cls,
        resolvent: ProjectedRPAResolvent,
        reference: Any,
        mapping: CayleyMap,
    ) -> Self:
        r"""Build the ellipse whose two quadrature errors equioscillate.

        The foci sit on the certified bounds :math:`[\Omega_L, \Omega_U]` and
        the confocal parameter
        :math:`\sigma(\zeta) = |\Re\,\mathrm{arccosh}((\zeta - c)/F)|` is zero on
        that segment. :math:`N` nodes then alias the enclosed excitations at
        :math:`e^{-N\sigma_c}` and the nearest excluded singularity at
        :math:`e^{-N(\sigma_{\min} - \sigma_c)}`, so :math:`\sigma_c =
        \sigma_{\min}/2`. The excluded singularities are the Cayley kernel poles
        of both sectors and, for direct RPA only, the mirror poles at
        :math:`-\Omega_\nu`.

        Args:
            resolvent: Supplies the certified bounds and the screening model.
            reference: Supplies the orbital energies that place the kernel poles.
            mapping: Cayley map whose centre and scale place the kernel poles.

        Returns:
            The contour, with its :class:`EquioscillationDiagnostics`.

        Raises:
            ValidationError: If the bounds coincide or the confocal parameter is
                not positive.
        """

        if not isinstance(resolvent, ProjectedRPAResolvent):
            raise ValidationError("resolvent must be a ProjectedRPAResolvent")
        lower = resolvent.spectral_lower_bound
        upper = resolvent.spectral_upper_bound
        center = 0.5 * (upper + lower)
        focal = 0.5 * (upper - lower)
        if focal <= 0.0:
            # Equal gaps and a vanishing coupling leave sigma undefined.
            raise ValidationError(
                f"the certified bounds coincide at [{lower:.6g}, {upper:.6g}], "
                "so the confocal family is degenerate"
            )

        sigma_min, limited_by = _equioscillation_sigma_min(
            lower, center, focal, reference, mapping, resolvent.screening
        )
        selected = 0.5 * sigma_min
        if selected <= 0.0:
            raise ValidationError(
                "equioscillation produced a non-positive confocal parameter; "
                f"sigma_min is {sigma_min:.6g}"
            )

        return cls(
            center=center,
            horizontal_radius=focal * math.cosh(selected),
            vertical_radius=focal * math.sinh(selected),
            equioscillation_diagnostics=EquioscillationDiagnostics(
                sigma_min=sigma_min,
                sigma_c=selected,
                limiting_singularity=limited_by,
            ),
        )

    def moments(
        self,
        resolvent: ProjectedRPAResolvent,
        mapping: CayleyMap,
        n_max: int,
        n_points: int,
        *,
        rule: Literal["midpoint", "trapezoid"] = "midpoint",
        orbital_block_size: int = 1,
        n_workers: int = 1,
        native_threads: int = 1,
        verbose: int = 0,
    ) -> dict[Sector, ContourSectorMoments]:
        """Integrate both sectors' moments through order ``n_max`` on this contour.

        The resolvent is solved once per conjugate pair of nodes and reused for
        every orbital, sector and order. Orbitals are contracted in blocks and
        folded into one total per sector in orbital order, so the total is
        bit-identical at every worker count.

        Args:
            resolvent: Resolvent from :meth:`ProjectedRPAResolvent.from_adapter`,
                which carries the reference and the factors.
            mapping: Cayley map shared by both sectors.
            n_max: Highest moment order.
            n_points: Number of distinct nodes, even.
            rule: ``"midpoint"``, the rule of a fixed ``n_q``, or ``"trapezoid"``,
                whose nodes nest under doubling for the automatic ladder.
            orbital_block_size: Internal orbitals contracted together.
            n_workers: Python workers over nodes and orbital blocks.
            native_threads: BLAS and LAPACK threads per worker; the lever that scales.
            verbose: ``0`` warnings only, ``1`` the log, ``2`` adds progress counts.

        Returns:
            One :class:`ContourSectorMoments` per :class:`Sector`.

        Raises:
            ValidationError: If ``n_points`` is odd, ``rule`` is unknown, the contour misses the
                certified bounds, crosses the imaginary axis for direct RPA, or
                encloses a kernel pole.
            RefusalError: If a node lies too close to a pole.
        """

        order = _check.nonnegative_integer(n_max, "n_max")
        block_size = _check.positive_integer(orbital_block_size, "orbital_block_size")
        workers = _check.positive_integer(n_workers, "n_workers")
        native = _check.positive_integer(native_threads, "native_threads")
        if not isinstance(mapping, CayleyMap):
            raise ValidationError("mapping must be a CayleyMap")
        internal = {
            sector: _kernel_pole_orbitals(resolvent.reference, self, sector, mapping)
            for sector in Sector
        }
        _validate_enclosure(resolvent, self)
        if rule not in ("midpoint", "trapezoid"):
            raise ValidationError("rule must be 'midpoint' or 'trapezoid'")
        nodes, weights = (
            self.midpoint_rule(n_points) if rule == "midpoint" else self.trapezoid_rule(n_points)
        )
        stages = Stages(verbose, LOGGER)
        projected_working = _sample_nodes(resolvent, nodes, stages, workers, native, int(n_points))
        totals_by_sector = _contract_moments(
            resolvent,
            mapping,
            internal,
            nodes,
            weights,
            projected_working,
            order,
            block_size,
            stages,
            workers,
            native,
        )
        results: dict[Sector, ContourSectorMoments] = {}
        for sector in Sector:
            total = totals_by_sector[sector]
            total.setflags(write=False)
            results[sector] = ContourSectorMoments(moments=total)
        return results
