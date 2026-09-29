r"""The Cayley map of the real frequency axis onto the unit circle.

For a centre :math:`\mu` and a scale :math:`\omega_p > 0`,

.. math::

   u(\omega) = \frac{\omega - \mu + i\omega_p}{\omega - \mu - i\omega_p} .

The hole sector :math:`\omega < \mu` lands on one open semicircle and the
particle sector on the other. :math:`u = 1` is the image of
:math:`\omega \to \pm\infty` and the singular point of the inverse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import tolerances as limits
from .errors import RefusalError, ValidationError
from .validate import ComplexArray, FloatArray, _check


def _finite(values: ArrayLike, dtype: type, name: str) -> np.ndarray:
    """Return ``values`` as a finite array of ``dtype``."""

    try:
        result = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{name} must be finite numbers") from error
    if not np.all(np.isfinite(result)):
        raise ValidationError(f"{name} must be finite numbers")
    return result


@dataclass(frozen=True, slots=True)
class CayleyMap:
    r"""Map real energies to the complex unit circle.

    The map sends ``center`` to ``-1``, ``center + scale`` to ``1j`` and
    ``center - scale`` to ``-1j``; no finite frequency reaches ``u = 1``.

    Attributes:
        center: Centre :math:`\mu` in Hartree, normally inside the HOMO-LUMO gap.
        scale: Scale :math:`\omega_p > 0` in Hartree; larger values spread a
            wider interval around ``u = -1``.

    Raises:
        ValidationError: If ``center`` or ``scale`` is not a finite real, or
            ``scale`` is not positive.

    Example:
        >>> mapping = CayleyMap(center=0.0, scale=2.0)
        >>> mapping.forward(0.0)
        (-1+0j)
        >>> mapping.inverse(1j)
        2.0
    """

    center: float
    scale: float

    def __post_init__(self) -> None:
        center = _check.finite_real(self.center, "center")
        scale = _check.finite_real(self.scale, "scale")
        if scale <= 0.0:
            raise ValidationError("scale must be strictly positive")

        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)

    def forward(self, frequency: ArrayLike) -> complex | ComplexArray:
        """Map finite real frequencies to the unit circle.

        Args:
            frequency: Scalar or array of energies in Hartree.

        Returns:
            The unit-circle points, in the shape of ``frequency``.

        Raises:
            ValidationError: If a frequency is complex or not finite.
        """

        frequencies = _finite(frequency, np.float64, "frequencies")
        scalar = frequencies.ndim == 0
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            offset = frequencies - self.center
            use_direct_ratio = np.abs(offset) <= self.scale
            direct_ratio = offset / self.scale
            reciprocal_ratio = self.scale / offset

            direct_denominator = 1.0 + np.square(direct_ratio)
            reciprocal_denominator = 1.0 + np.square(reciprocal_ratio)
            real_part = np.where(
                use_direct_ratio,
                (np.square(direct_ratio) - 1.0) / direct_denominator,
                (1.0 - np.square(reciprocal_ratio)) / reciprocal_denominator,
            )
            imaginary_part = np.where(
                use_direct_ratio,
                2.0 * direct_ratio / direct_denominator,
                2.0 * reciprocal_ratio / reciprocal_denominator,
            )

        mapped = np.asarray(real_part + 1j * imaginary_part, dtype=np.complex128)
        if not np.all(np.isfinite(mapped)):
            raise ValidationError("Cayley mapping failed to produce finite unit-circle points")
        return complex(mapped.item()) if scalar else mapped

    def inverse(self, point: ArrayLike) -> float | FloatArray:
        """Map unit-circle points back to real energies.

        Points that pass the radial check are normalized to modulus one first,
        which removes the radial roundoff of a large frequency near ``u = 1``.

        Args:
            point: Scalar or array of points on the unit circle.

        Returns:
            The energies in Hartree, in the shape of ``point``.

        Raises:
            ValidationError: If a point is not finite or lies further than
                ``ROUNDOFF_TOLERANCE`` from the unit circle.
            RefusalError: If a point is within ``ABSOLUTE_TOLERANCE``
                of ``u = 1``.
        """

        points = _finite(point, np.complex128, "unit-circle points")
        radial_error = np.abs(np.abs(points) - 1.0)
        if np.any(radial_error > limits.ROUNDOFF_TOLERANCE):
            maximum = float(np.max(radial_error))
            raise ValidationError(
                "inverse Cayley input is not on the unit circle; "
                f"maximum radial residual is {maximum:.3e}"
            )

        normalized_points = points / np.abs(points)
        distance_from_one = np.abs(normalized_points - 1.0)
        if np.any(distance_from_one <= limits.ABSOLUTE_TOLERANCE):
            minimum = float(np.min(distance_from_one))
            raise RefusalError(
                f"inverse Cayley map is singular at u = 1; minimum abs(u - 1) is {minimum:.3e}",
                kind="cayley-singularity",
            )

        angles = np.angle(normalized_points)
        real_values = np.asarray(
            self.center + self.scale / np.tan(0.5 * angles),
            dtype=np.float64,
        )
        return float(real_values.item()) if points.ndim == 0 else real_values

    def powers(self, frequency: ArrayLike, n_max: int) -> ComplexArray:
        """Return ``u(frequency)**n`` for ``n = 0..n_max`` by repeated multiplication.

        Args:
            frequency: Scalar or array of finite energies in Hartree.
            n_max: Highest power.

        Returns:
            Complex array of shape ``(n_max + 1,) + np.shape(frequency)``, whose
            entry zero is exactly one.

        Raises:
            ValidationError: If ``n_max`` is not a nonnegative integer or a
                frequency is invalid.
        """

        order_max = _check.nonnegative_integer(n_max, "n_max")

        mapped = np.asarray(self.forward(frequency), dtype=np.complex128)
        result = np.empty((order_max + 1,) + mapped.shape, dtype=np.complex128)
        result[0] = 1.0 + 0.0j
        for order in range(1, order_max + 1):
            result[order] = result[order - 1] * mapped
        return result

    def sector_sign(self, frequency: ArrayLike) -> int | NDArray[np.int8]:
        """Classify real energies against the map centre.

        Args:
            frequency: Scalar or array of energies in Hartree.

        Returns:
            ``-1`` on the hole (lower-arc) side, ``0`` at the centre and ``+1``
            on the particle (upper-arc) side.

        Raises:
            ValidationError: If a frequency is complex or not finite.
        """

        frequencies = _finite(frequency, np.float64, "frequencies")
        with np.errstate(over="ignore"):
            offset = frequencies - self.center
        signs = np.sign(offset).astype(np.int8)
        return int(signs.item()) if frequencies.ndim == 0 else signs
