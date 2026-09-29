"""Tests of the Cayley map, its powers and its sector signs."""

from __future__ import annotations


import numpy as np
import pytest

from cayleygw import RefusalError, ValidationError
from cayleygw._helpers.cayley import CayleyMap


def test_special_points_and_unit_circle() -> None:
    mapping = CayleyMap(center=1.25, scale=0.75)
    frequencies = np.array(
        [
            mapping.center,
            mapping.center + mapping.scale,
            mapping.center - mapping.scale,
            -1.0e6,
            1.0e6,
        ]
    )
    points = mapping.forward(frequencies)

    np.testing.assert_allclose(points[:3], [-1.0, 1.0j, -1.0j], atol=1.0e-15)
    np.testing.assert_allclose(np.abs(points), 1.0, atol=2.0e-15, rtol=0.0)
    assert points[-1].real > 0.999999
    assert points[0] == pytest.approx(-1.0 + 0.0j)


def test_forward_inverse_round_trip_for_scalar_and_array() -> None:
    mapping = CayleyMap(center=-0.31, scale=0.87)
    frequencies = np.array([-1.0e8, -12.0, -0.31, 0.0, 9.2, 1.0e8])
    recovered = mapping.inverse(mapping.forward(frequencies))
    np.testing.assert_allclose(recovered, frequencies, atol=1.0e-8, rtol=2.0e-8)

    point = mapping.forward(1.75)
    assert isinstance(point, complex)
    recovered_scalar = mapping.inverse(point)
    assert isinstance(recovered_scalar, float)
    assert recovered_scalar == pytest.approx(1.75, abs=1.0e-14)


def test_forward_is_overflow_resistant_at_floating_point_extremes() -> None:
    maximum = np.finfo(np.float64).max
    points = np.array(
        [
            CayleyMap(center=-1.0e308, scale=1.0).forward(maximum),
            CayleyMap(center=1.0e308, scale=1.0).forward(-maximum),
        ]
    )

    assert np.all(np.isfinite(points))
    np.testing.assert_allclose(np.abs(points), 1.0, atol=1.0e-15, rtol=0.0)
    np.testing.assert_allclose(points.real, 1.0, atol=1.0e-15, rtol=0.0)


def test_inverse_special_points() -> None:
    mapping = CayleyMap(center=2.0, scale=3.0)
    assert mapping.inverse(-1.0) == pytest.approx(2.0)
    assert mapping.inverse(1.0j) == pytest.approx(5.0)
    assert mapping.inverse(-1.0j) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("center", "scale"),
    [
        (0.0, 0.0),
        (0.0, -1.0),
        (float("nan"), 1.0),
        (0.0, float("inf")),
        (0.0, True),
        ("zero", 1.0),
    ],
)
def test_map_rejects_invalid_parameters(center: object, scale: object) -> None:
    with pytest.raises(ValidationError):
        CayleyMap(center=center, scale=scale)


@pytest.mark.parametrize(
    "frequency",
    [
        np.inf,
        -np.inf,
        np.nan,
        1.0 + 2.0j,
        [0.0, np.nan],
        "not an energy",
    ],
)
def test_forward_rejects_invalid_frequencies(frequency: object) -> None:
    with pytest.raises(ValidationError):
        CayleyMap(0.0, 1.0).forward(frequency)


def test_inverse_rejects_off_circle_and_singularity() -> None:
    mapping = CayleyMap(0.0, 1.0)
    with pytest.raises(ValidationError, match="unit circle"):
        mapping.inverse(0.5 + 0.0j)
    with pytest.raises(RefusalError):
        mapping.inverse(1.0 + 0.0j)
    with pytest.raises(RefusalError):
        mapping.inverse(np.exp(1.0e-13j))
    with pytest.raises(ValidationError):
        mapping.inverse(np.nan + 1.0j)


def test_recursive_powers_include_order_zero() -> None:
    mapping = CayleyMap(center=0.2, scale=0.9)
    frequencies = np.array([-2.0, 0.2, 4.0])
    points = mapping.forward(frequencies)
    powers = mapping.powers(frequencies, 7)

    assert powers.shape == (8, 3)
    np.testing.assert_array_equal(powers[0], np.ones(3, dtype=complex))
    for order in range(8):
        np.testing.assert_allclose(powers[order], points**order, atol=2.0e-15)
    np.testing.assert_allclose(np.abs(powers), 1.0, atol=3.0e-15)

    scalar_powers = mapping.powers(1.0, 0)
    assert scalar_powers.shape == (1,)
    assert scalar_powers[0] == 1.0


@pytest.mark.parametrize("order", [-1, 1.5, True, "2"])
def test_recursive_powers_reject_invalid_orders(order: object) -> None:
    with pytest.raises(ValidationError):
        CayleyMap(0.0, 1.0).powers(0.0, order)


def test_sector_classification_matches_semicircles() -> None:
    mapping = CayleyMap(center=1.0, scale=2.0)
    frequencies = np.array([-2.0, 1.0 - 1.0e-12, 1.0, 1.0 + 1.0e-12, 4.0])
    signs = mapping.sector_sign(frequencies)
    np.testing.assert_array_equal(signs, [-1, -1, 0, 1, 1])

    outside = np.array([-2.0, 4.0])
    points = mapping.forward(outside)
    assert points[0].imag < 0.0
    assert points[1].imag > 0.0
    assert mapping.sector_sign(-2.0) == -1
    assert mapping.sector_sign(4.0) == 1


#: A water-like spectrum: deep core, frontier gap 0.7, water/6-31G excitation window.
_OCCUPIED = [-20.5, -1.3, -0.7, -0.55, -0.5]
_VIRTUAL = [0.2, 0.3, 1.0, 2.0]
_POTENTIAL = -0.15
_FLOOR = 0.7
_CEILING = 22.4
