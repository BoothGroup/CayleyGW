"""Argument checks shared by every module, and the array type aliases.

Every module checks its arguments through ``_check``, the one
:class:`Validator`. Each check raises
:class:`~cayleygw._helpers.errors.ValidationError` or returns the value: an
array as an independent C-contiguous read-only copy, a scalar as a plain
``int``, ``float`` or ``complex``.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .errors import ValidationError
from .tolerances import Tolerances

BoolArray: TypeAlias = NDArray[np.bool_]
ComplexArray: TypeAlias = NDArray[np.complex128]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


class Validator:
    """Argument checks that raise :class:`ValidationError`."""

    __slots__ = ()

    def readonly_real(self, values: Any, name: str) -> FloatArray:
        """Return ``values`` as an independent finite read-only ``float64`` array."""

        try:
            source = np.asarray(values)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{name} must be numeric") from exc
        if np.iscomplexobj(source):
            raise ValidationError(f"{name} must be real")
        try:
            result = np.array(source, dtype=np.float64, order="C", copy=True)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{name} must be numeric") from exc
        if not np.all(np.isfinite(result)):
            raise ValidationError(f"{name} must contain finite values")
        result.setflags(write=False)
        return result

    def readonly_complex(self, values: Any, name: str) -> ComplexArray:
        """Return ``values`` as an independent finite read-only ``complex128`` array."""

        try:
            result = np.array(values, dtype=np.complex128, order="C", copy=True)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{name} must be numeric") from exc
        if not np.all(np.isfinite(result.real)) or not np.all(np.isfinite(result.imag)):
            raise ValidationError(f"{name} must contain finite values")
        result.setflags(write=False)
        return result

    def readonly_int(self, values: Any, name: str) -> IntArray:
        """Return ``values`` as an independent read-only ``int64`` array."""

        try:
            source = np.asarray(values)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{name} must contain integers") from exc
        if not np.issubdtype(source.dtype, np.integer):
            raise ValidationError(f"{name} must contain integers")
        result = np.array(source, dtype=np.int64, order="C", copy=True)
        result.setflags(write=False)
        return result

    def readonly_bool(self, values: Any, name: str) -> BoolArray:
        """Return ``values`` as an independent read-only boolean array."""

        result = np.array(values, dtype=np.bool_, order="C", copy=True)
        result.setflags(write=False)
        return result

    def tolerances(self, tolerances: Any) -> Tolerances:
        """Return ``tolerances`` if it is a :class:`Tolerances`."""

        if not isinstance(tolerances, Tolerances):
            raise ValidationError("tolerances must be a Tolerances instance")
        return tolerances

    def boolean(self, value: Any, name: str) -> bool:
        """Return ``value`` if it is ``True`` or ``False``."""

        if not isinstance(value, bool):
            raise ValidationError(f"{name} must be boolean")
        return value

    def nonnegative_integer(self, value: Any, name: str) -> int:
        """Return ``value`` as an ``int`` that is at least zero."""

        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValidationError(f"{name} must be a nonnegative integer")
        return int(value)

    def positive_integer(self, value: Any, name: str, *, minimum: int = 1) -> int:
        """Return ``value`` as an ``int`` that is at least ``minimum`` (default 1)."""

        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            if minimum == 1:
                raise ValidationError(f"{name} must be a positive integer")
            raise ValidationError(f"{name} must be an integer of at least {minimum}")
        return int(value)

    def finite_real(self, value: Any, name: str) -> float:
        """Return ``value`` as a finite ``float``."""

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValidationError(f"{name} must be a finite real number")
        return float(value)

    def positive_real(self, value: Any, name: str) -> float:
        """Return ``value`` as a finite ``float`` greater than zero."""

        result = self.finite_real(value, name)
        if result <= 0.0:
            raise ValidationError(f"{name} must be strictly positive")
        return result

    def nonnegative_real(self, value: Any, name: str) -> float:
        """Return ``value`` as a finite ``float`` that is at least zero."""

        result = self.finite_real(value, name)
        if result < 0.0:
            raise ValidationError(f"{name} must be nonnegative")
        return result

    def finite_complex(self, value: Any, name: str) -> complex:
        """Return ``value`` as a finite ``complex`` scalar."""

        if isinstance(value, bool) or np.ndim(value) != 0:
            raise ValidationError(f"{name} must be a finite scalar")
        try:
            result = complex(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{name} must be a finite scalar") from exc
        if not math.isfinite(result.real) or not math.isfinite(result.imag):
            raise ValidationError(f"{name} must be a finite scalar")
        return result


_check = Validator()


__all__ = ["BoolArray", "ComplexArray", "FloatArray", "IntArray", "Validator"]
