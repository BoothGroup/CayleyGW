"""Tests of the ellipse contour and the Cayley-moment quadrature on it.

The H2 and water oracles pair a molecule's projected resolvent with its exact
full-pole moments and a contour padded around the certified interval.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from cayleygw import (
    ExactG0W0SelfEnergy,
    RefusalError,
    Screening,
    Sector,
    ValidationError,
)
from cayleygw import contour as contour_module
from cayleygw._helpers.cayley import CayleyMap
from cayleygw._helpers.contour import contraction as contraction_module
from cayleygw._helpers.pyscf import RestrictedPySCFAdapter
from cayleygw.contour import EllipseContour
from cayleygw.screening import ProjectedRPAResolvent


def _scalar_contour_problem():
    """Return a one-transition problem with its exact positive excitation."""

    gap = 0.8
    coupling = 0.3
    resolvent = ProjectedRPAResolvent([gap], [[coupling]])
    excitation = float(np.sqrt(gap**2 + 2.0 * coupling**2))
    contour = EllipseContour(
        center=excitation,
        horizontal_radius=0.20,
        vertical_radius=0.15,
    )
    return gap, coupling, excitation, resolvent, contour


def _padded_contour(
    resolvent,
    *,
    vertical_radius,
    relative_padding=0.10,
    absolute_padding=1.0e-3,
):
    """Enclose the certified excitation interval, padded horizontally."""

    lower = resolvent.spectral_lower_bound
    upper = resolvent.spectral_upper_bound
    half_width = 0.5 * (upper - lower)
    return EllipseContour(
        center=0.5 * (upper + lower),
        horizontal_radius=half_width + max(relative_padding * half_width, absolute_padding),
        vertical_radius=vertical_radius,
    )


def _resolvent_spectrum(resolvent):
    """Return the positive excitation energies of the resolvent's own operator."""

    coupling = resolvent.v_matrix
    squared = np.diag(np.square(resolvent.particle_hole_gaps)) + 2.0 * coupling @ coupling.T
    return np.sqrt(np.linalg.eigvalsh(squared))


def _exact_oracles(mean_field, **padding):
    """Return the resolvent, the exact full-pole reference and a padded contour."""

    adapter = RestrictedPySCFAdapter(mean_field)
    exact = ExactG0W0SelfEnergy.from_mean_field(mean_field)
    resolvent = ProjectedRPAResolvent.from_adapter(adapter)
    mapping = CayleyMap(
        center=adapter.reference.chemical_potential,
        scale=1.0,
    )
    contour = _padded_contour(resolvent, **padding)
    return adapter, exact, resolvent, mapping, contour


@pytest.fixture(scope="module")
def h2_contour_oracles(h2_rks_hf):
    """Return the H2 oracles, shared by the module."""

    return _exact_oracles(
        h2_rks_hf,
        vertical_radius=0.15,
        relative_padding=0.10,
        absolute_padding=0.20,
    )


def test_trapezoid_nodes_nest_and_pair_their_real_ends_with_themselves() -> None:
    """The 2N grid holds the N grid and its midpoint nodes; the real ends pair at half weight."""

    contour = EllipseContour(1.5, 0.6, 0.2)
    nodes, weights = contour.trapezoid_rule(8)
    assert nodes.size == weights.size == 10
    np.testing.assert_array_equal(nodes[::-1], nodes.conj())
    np.testing.assert_array_equal(weights[::-1], weights.conj())
    for end, value in ((0, 1.5 + 0.6), (4, 1.5 - 0.6)):
        assert nodes[end] == value and weights[end].imag == 0.0
    # The ends carry half their weight twice; the others the midpoint formula at theta = 2 pi j / N.
    np.testing.assert_allclose(weights[[0, 4]] * 2, [2 * 0.2 / 8, -2 * 0.2 / 8], rtol=1e-15)
    fine, _ = contour.trapezoid_rule(16)
    coarse_midpoint, _ = contour.midpoint_rule(8)
    distinct = lambda values: np.unique(np.round(values, 12))
    np.testing.assert_array_equal(
        distinct(fine), distinct(np.concatenate((nodes, coarse_midpoint)))
    )
    with pytest.raises(ValidationError, match="even"):
        contour.trapezoid_rule(7)


@pytest.mark.pyscf
def test_a_trapezoid_doubling_is_the_coarse_rule_and_its_midpoints(h2_contour_oracles) -> None:
    """T_2N = (T_N + M_N) / 2 through the real quadrature, which the automatic ladder uses."""

    _, _, resolvent, mapping, contour = h2_contour_oracles
    fine = contour.moments(resolvent, mapping, 3, 32, rule="trapezoid")
    coarse = contour.moments(resolvent, mapping, 3, 16, rule="trapezoid")
    midpoints = contour.moments(resolvent, mapping, 3, 16)
    for sector in Sector:
        np.testing.assert_allclose(
            fine[sector].moments,
            0.5 * (coarse[sector].moments + midpoints[sector].moments),
            rtol=1e-13,
            atol=1e-15,
        )
    with pytest.raises(ValidationError, match="rule"):
        contour.moments(resolvent, mapping, 3, 16, rule="simpson")


def test_midpoint_nodes_and_weights_follow_counterclockwise_formula() -> None:
    contour = EllipseContour(1.5, 0.6, 0.2)
    nodes, weights = contour.midpoint_rule(8)
    expected_angles = 2.0 * np.pi * (np.arange(8) + 0.5) / 8
    expected_nodes = 1.5 + 0.6 * np.cos(expected_angles) + 0.2j * np.sin(expected_angles)
    expected_weights = (2.0 / 8) * (0.2 * np.cos(expected_angles) + 0.6j * np.sin(expected_angles))
    np.testing.assert_allclose(nodes, expected_nodes, atol=2.0e-16)
    np.testing.assert_allclose(weights, expected_weights, atol=2.0e-16)
    np.testing.assert_array_equal(nodes[::-1], nodes.conj())
    np.testing.assert_array_equal(weights[::-1], weights.conj())
    assert nodes.size == 8


def test_scalar_residue_fixes_orientation_and_cauchy_prefactor() -> None:
    _, coupling, excitation, resolvent, contour = _scalar_contour_problem()
    nodes, weights = contour.midpoint_rule(64)
    integrated = sum(weight * resolvent.woodbury(node) for node, weight in zip(nodes, weights))
    expected = coupling**2 / excitation
    np.testing.assert_allclose(integrated, [[expected]], atol=3.0e-15, rtol=0.0)


@pytest.mark.pyscf
def test_even_rules_solve_one_node_per_conjugate_pair(
    h2_contour_oracles,
    monkeypatch,
) -> None:
    """An even rule solves half its nodes, and an odd rule is refused."""

    _, _, resolvent, mapping, contour = h2_contour_oracles
    original = ProjectedRPAResolvent.woodbury
    calls = 0

    def counted_woodbury(self, zeta):
        nonlocal calls
        calls += 1
        return original(self, zeta)

    monkeypatch.setattr(ProjectedRPAResolvent, "woodbury", counted_woodbury)
    contour.moments(resolvent, mapping, 3, 16)
    assert calls == 8

    calls = 0
    with pytest.raises(ValidationError, match="n_points must be even"):
        contour.moments(resolvent, mapping, 3, 7)
    assert calls == 0


@pytest.mark.pyscf
def test_certified_automatic_bounds_enclose_molecular_spectrum(
    h2_contour_oracles,
) -> None:
    adapter, _, resolvent, mapping, _ = h2_contour_oracles
    contour = EllipseContour.from_equioscillation(resolvent, adapter.reference, mapping)
    energies = _resolvent_spectrum(resolvent)
    assert resolvent.spectral_bound_source == "D_min and spectral-norm upper bound"
    assert resolvent.spectral_lower_bound <= energies[0]
    assert resolvent.spectral_upper_bound >= energies[-1]
    assert np.all(np.asarray(contour.normalized_radius(energies)) < 1.0)
    assert contour.leftmost > 0.0
    # The foci sit on the certified bounds, so the ellipse is wider than tall.
    assert contour.vertical_radius < contour.horizontal_radius
    assert contour.equioscillation_diagnostics is not None
    # The kernel poles have Re < 0; the enclosure check keeps them outside.
    contour_module._validate_enclosure(resolvent, contour)


@pytest.mark.pyscf
def test_contour_moments_match_the_projected_total(h2_contour_oracles) -> None:
    _, oracle, resolvent, mapping, contour = h2_contour_oracles
    n_max = 8
    exact = oracle.cayley_moments(mapping, n_max)
    contour_results = contour.moments(resolvent, mapping, n_max, 128)
    for sector in Sector:
        result = contour_results[sector]
        np.testing.assert_allclose(
            result.moments,
            exact[sector],
            atol=2.0e-14,
            rtol=2.0e-13,
        )
        assert result.moments.shape[0] == n_max + 1
        assert not result.moments.flags.writeable


@pytest.mark.pyscf
def test_conjugate_paired_moments_keep_c0_hermitian_without_hermitianizing_ck(
    h2_contour_oracles,
) -> None:
    """Paired moments match the exact ones through the non-Hermitian orders."""

    _, oracle, resolvent, mapping, contour = h2_contour_oracles
    exact = oracle.cayley_moments(mapping, 5)
    paired = contour.moments(resolvent, mapping, 5, 64)
    for sector in Sector:
        np.testing.assert_allclose(
            paired[sector].moments,
            exact[sector],
            atol=4.0e-15,
            rtol=3.0e-13,
        )
        np.testing.assert_array_equal(
            paired[sector].moments[0],
            paired[sector].moments[0].conj().T,
        )
        assert (
            np.max(np.abs(paired[sector].moments[1] - paired[sector].moments[1].conj().T)) > 1.0e-3
        )


@pytest.mark.pyscf
def test_blocked_conjugate_pairing_reuses_each_projected_resolvent_once(
    h2_contour_oracles,
    monkeypatch,
) -> None:
    """The orbital block size changes neither the solve count nor the moments."""

    _, _, resolvent, mapping, contour = h2_contour_oracles
    original = ProjectedRPAResolvent.woodbury
    calls = 0

    def counted_woodbury(self, zeta):
        nonlocal calls
        calls += 1
        return original(self, zeta)

    monkeypatch.setattr(
        ProjectedRPAResolvent,
        "woodbury",
        counted_woodbury,
    )
    small_blocks = contour.moments(
        resolvent,
        mapping,
        3,
        32,
        orbital_block_size=1,
    )
    assert calls == 16
    calls = 0
    large_blocks = contour.moments(
        resolvent,
        mapping,
        3,
        32,
        orbital_block_size=100,
    )
    assert calls == 16
    for sector in Sector:
        np.testing.assert_allclose(
            small_blocks[sector].moments,
            large_blocks[sector].moments,
            atol=3.0e-15,
            rtol=3.0e-13,
        )


@pytest.mark.pyscf
def test_geometric_convergence_and_zeroth_moment_gate(h2_contour_oracles) -> None:
    _, oracle, resolvent, mapping, contour = h2_contour_oracles
    exact = oracle.cayley_moments(mapping, 6)
    errors: dict[Sector, list[float]] = {sector: [] for sector in Sector}
    zeroth_errors: dict[Sector, list[float]] = {sector: [] for sector in Sector}
    # Padding the certified interval, not the one excitation, costs one doubling.
    for n_points in (32, 64, 128):
        result = contour.moments(resolvent, mapping, 6, n_points)
        for sector in Sector:
            errors[sector].append(float(np.linalg.norm(result[sector].moments - exact[sector])))
            zeroth_errors[sector].append(
                float(np.linalg.norm(result[sector].moments[0] - exact[sector][0]))
            )
    for sector in Sector:
        assert errors[sector][1] < 0.02 * errors[sector][0]
        assert errors[sector][2] < 5.0e-15
        assert zeroth_errors[sector][2] < 4.0e-9


@pytest.mark.pyscf
def test_doubling_estimate_tracks_observed_highest_order_error(
    h2_contour_oracles,
) -> None:
    _, oracle, resolvent, mapping, contour = h2_contour_oracles
    n_max = 6
    exact = oracle.cayley_moments(mapping, n_max)
    coarse = contour.moments(resolvent, mapping, n_max, 32)
    fine = contour.moments(resolvent, mapping, n_max, 64)
    for sector in Sector:
        observed = float(np.linalg.norm(coarse[sector].moments[-1] - exact[sector][-1]))
        fine_error = float(np.linalg.norm(fine[sector].moments[-1] - exact[sector][-1]))
        # The automatic N_q ladder reads this coarse-to-fine change of the top order.
        estimate = float(np.linalg.norm(fine[sector].moments[-1] - coarse[sector].moments[-1]))
        # Reverse triangle inequality: |estimate - observed| <= fine-grid error.
        roundoff = 64.0 * np.finfo(float).eps * max(estimate, observed, 1.0)
        assert abs(estimate - observed) <= fine_error + roundoff
        assert fine_error < 0.02 * observed


@pytest.mark.pyscf
@pytest.mark.parametrize(
    (
        "relative_padding",
        "absolute_padding",
        "vertical_radius",
        "scale",
        "n_max",
    ),
    [
        (0.05, 0.10, 0.08, 0.5, 0),
        (0.10, 0.20, 0.15, 1.0, 4),
        (0.20, 0.30, 0.22, 1.8, 9),
    ],
)
def test_contour_padding_height_map_scale_and_order_variations(
    h2_contour_oracles,
    relative_padding,
    absolute_padding,
    vertical_radius,
    scale,
    n_max,
) -> None:
    adapter, oracle, resolvent, _, _ = h2_contour_oracles
    mapping = CayleyMap(
        center=adapter.reference.chemical_potential,
        scale=scale,
    )
    contour = _padded_contour(
        resolvent,
        vertical_radius=vertical_radius,
        relative_padding=relative_padding,
        absolute_padding=absolute_padding,
    )
    exact = oracle.cayley_moments(mapping, n_max)
    result = contour.moments(resolvent, mapping, n_max, 192)
    for sector in Sector:
        np.testing.assert_allclose(
            result[sector].moments,
            exact[sector],
            atol=3.0e-13,
            rtol=2.0e-12,
        )


@pytest.mark.pyscf
def test_complete_contour_does_not_force_higher_moments_hermitian(
    h2_contour_oracles,
) -> None:
    _, _, resolvent, mapping, contour = h2_contour_oracles
    result = contour.moments(resolvent, mapping, 3, 128)
    for sector in Sector:
        np.testing.assert_allclose(
            result[sector].moments[0],
            result[sector].moments[0].conj().T,
            atol=2.0e-14,
        )
        assert (
            np.max(np.abs(result[sector].moments[1] - result[sector].moments[1].conj().T)) > 1.0e-3
        )


@pytest.mark.pyscf
def test_zero_interaction_limit_gives_zero_contour_moments(h2_rks_hf) -> None:
    interacting = ProjectedRPAResolvent.from_adapter(RestrictedPySCFAdapter(h2_rks_hf))
    resolvent = ProjectedRPAResolvent(
        interacting.particle_hole_gaps,
        np.zeros_like(interacting.v_matrix),
        reference=interacting.reference,
        factors=interacting.factors,
    )
    mapping = CayleyMap(
        center=resolvent.reference.chemical_potential,
        scale=1.0,
    )
    contour = _padded_contour(
        resolvent,
        vertical_radius=0.15,
        relative_padding=0.1,
        absolute_padding=0.02,
    )
    result = contour.moments(resolvent, mapping, 5, 64)
    for sector in Sector:
        np.testing.assert_array_equal(result[sector].moments, 0.0)


@pytest.mark.pyscf
def test_missed_rpa_pole_is_rejected(h2_contour_oracles) -> None:
    _, _, resolvent, mapping, _ = h2_contour_oracles
    energies = _resolvent_spectrum(resolvent)
    contour = EllipseContour(
        center=float(energies[0] + 0.20),
        horizontal_radius=0.05,
        vertical_radius=0.02,
    )
    with pytest.raises(ValidationError, match="does not strictly enclose"):
        contour.moments(resolvent, mapping, 2, 32)


@pytest.mark.pyscf
def test_enclosed_cayley_kernel_pole_is_rejected(h2_contour_oracles) -> None:
    _, _, resolvent, mapping, _ = h2_contour_oracles
    contour = EllipseContour(
        center=0.5,
        horizontal_radius=2.5,
        vertical_radius=2.5,
    )
    with pytest.raises(ValidationError, match="Cayley kernel"):
        contour.moments(resolvent, mapping, 2, 32)


def test_left_half_plane_folding_is_rejected() -> None:
    _, _, _, resolvent, _ = _scalar_contour_problem()
    folding = EllipseContour(
        center=0.5,
        horizontal_radius=0.6,
        vertical_radius=0.1,
    )
    with pytest.raises(ValidationError, match="imaginary axis"):
        contour_module._validate_enclosure(resolvent, folding)


@pytest.mark.pyscf
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_prohibited_node_keeps_its_index_and_diagnostics_under_workers(
    h2_contour_oracles,
    monkeypatch,
    workers,
) -> None:
    """Of two refused nodes, the lowest comes back from a worker with its index and distances."""

    _, _, resolvent, mapping, contour = h2_contour_oracles
    nodes, _ = contour.midpoint_rule(10)
    original = ProjectedRPAResolvent.woodbury

    def rejecting_woodbury(self, zeta):
        if zeta in (nodes[1], nodes[3]):
            raise RefusalError(
                "synthetic prohibited node",
                kind="resolvent-node",
                zeta=complex(zeta),
                free_pole_distance=0.0,
                rpa_pole_distance=1.0,
                pole_threshold=1.0e-10,
            )
        return original(self, zeta)

    monkeypatch.setattr(ProjectedRPAResolvent, "woodbury", rejecting_woodbury)
    with pytest.raises(RefusalError, match="node 1") as error:
        contour.moments(
            resolvent,
            mapping,
            2,
            10,
            n_workers=workers,
            native_threads=1,
        )
    assert error.value.kind == "contour-node"
    assert error.value.diagnostics.kind == "resolvent-node"
    assert error.value.node_index == 1
    assert error.value.zeta == pytest.approx(nodes[1])
    assert error.value.free_pole_distance <= error.value.pole_threshold
    assert error.value.rpa_pole_distance > error.value.pole_threshold


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: EllipseContour(1.0, 0.0, 0.1),
        lambda: EllipseContour(1.0, 0.2, -0.1),
        lambda: EllipseContour(1.0, 0.2, 0.1).midpoint_rule(1),
        lambda: EllipseContour(1.0, 0.2, 0.1).midpoint_rule(7),
    ],
)
def test_invalid_contour_geometry_and_node_counts_are_rejected(
    constructor,
) -> None:
    with pytest.raises(ValidationError):
        constructor()


@pytest.mark.pyscf
@pytest.mark.parametrize("workers", [2, 4, 8])
def test_worker_count_does_not_change_a_single_digit(
    h2_contour_oracles,
    workers,
) -> None:
    """Node results are stored by index, so the worker count changes no bit."""

    _, _, resolvent, mapping, contour = h2_contour_oracles
    arguments = (resolvent, mapping, 3, 32)
    # Threaded BLAS moves about 3 ulp between 1 and 4 threads, so both sides use one.
    serial = contour.moments(*arguments, n_workers=1, native_threads=1)
    parallel = contour.moments(*arguments, n_workers=workers, native_threads=1)
    for sector in Sector:
        assert np.array_equal(
            parallel[sector].moments,
            serial[sector].moments,
        )


@pytest.mark.pyscf
@pytest.mark.parametrize("value", [0, -1, 1.5, "2"])
def test_worker_and_native_thread_counts_are_validated(
    h2_contour_oracles,
    value,
) -> None:
    _, _, resolvent, mapping, contour = h2_contour_oracles
    arguments = (resolvent, mapping, 2, 8)
    with pytest.raises(ValidationError):
        contour.moments(*arguments, n_workers=value)
    with pytest.raises(ValidationError):
        contour.moments(*arguments, native_threads=value)


@pytest.mark.pyscf
@pytest.mark.parametrize("workers", [1, 4])
def test_blocked_contraction_logs_one_stage_per_region(
    h2_contour_oracles,
    workers,
    caplog,
) -> None:
    """One stage line for the node solves and one for the contraction, at any worker count."""

    _, _, resolvent, mapping, contour = h2_contour_oracles
    with caplog.at_level(logging.INFO, logger="cayleygw"):
        contour.moments(
            resolvent,
            mapping,
            2,
            32,
            n_workers=workers,
            native_threads=1,
            verbose=1,
        )
    messages = [record.getMessage() for record in caplog.records]
    assert sum(m.startswith("contour node solves: nodes=32, solved=16") for m in messages) == 1
    assert sum(m.startswith("quadrature and moment contraction: n_max=2") for m in messages) == 1
    assert sum(m.startswith("quadrature and moment contraction: ") for m in messages) == 2


@pytest.fixture(scope="module")
def water_contour_oracles(water_rks_hf):
    """Return the water oracles, which have several orbitals per sector where H2 has one."""

    # Padded less than H2: a tenth of the width would cross the imaginary axis.
    return _exact_oracles(
        water_rks_hf,
        vertical_radius=0.15,
        relative_padding=0.05,
        absolute_padding=0.20,
    )


@pytest.mark.pyscf
@pytest.mark.parametrize("workers", [2, 4])
def test_blocked_accumulation_does_not_depend_on_the_worker_count(
    water_contour_oracles,
    workers,
) -> None:
    """Blocks are folded in orbital order, so the sum is bitwise equal for any worker count."""

    _, _, resolvent, mapping, contour = water_contour_oracles
    arguments = (resolvent, mapping, 3, 32)
    common = {"orbital_block_size": 2, "native_threads": 1}
    serial = contour.moments(*arguments, n_workers=1, **common)
    parallel = contour.moments(*arguments, n_workers=workers, **common)
    for sector in Sector:
        assert np.array_equal(
            parallel[sector].moments,
            serial[sector].moments,
        )


@pytest.mark.pyscf
def test_accumulation_keeps_the_paired_zeroth_moment_gate(
    water_contour_oracles,
    monkeypatch,
) -> None:
    """The zeroth-moment gate runs per orbital, before defects can cancel in the sum."""

    _, _, resolvent, mapping, contour = water_contour_oracles
    original = contraction_module._hermitian_zeroth
    seen: list[int] = []

    def corrupt_second(values, position, label):
        if label.endswith("physical-space"):
            seen.append(position)
            if position == 1:
                values[0, 0, -1] += 1.0e-3
        return original(values, position, label)

    monkeypatch.setattr(contraction_module, "_hermitian_zeroth", corrupt_second)
    with pytest.raises(
        ValidationError,
        match="internal position 1",
    ):
        contour.moments(resolvent, mapping, 2, 32)
    assert seen[:2] == [0, 1]


def test_external_contraction_uses_real_arithmetic_for_real_factors() -> None:
    """Real factors take two real products, which match the complex ``einsum`` to roundoff."""

    from cayleygw._helpers.contour.contraction import contract_external_moments

    rng = np.random.default_rng(7)
    nmo, naux, orders = 9, 13, 4
    factors = rng.standard_normal((nmo, naux))
    auxiliary = rng.standard_normal((orders, naux, naux)) + 1j * rng.standard_normal(
        (orders, naux, naux)
    )
    reference = np.einsum("pP,kPQ,qQ->kpq", factors, auxiliary, factors, optimize=True)
    result = contract_external_moments(factors, auxiliary)
    assert result.shape == (orders, nmo, nmo)
    assert result.dtype == np.complex128
    np.testing.assert_allclose(result, reference, atol=1.0e-13, rtol=1.0e-13)
    # A strided view, as ``orbital_column(...).T`` passes, takes the same route.
    strided = np.ascontiguousarray(factors.T).T
    assert not strided.flags.c_contiguous
    np.testing.assert_allclose(
        contract_external_moments(strided, auxiliary),
        reference,
        atol=1.0e-13,
        rtol=1.0e-13,
    )


@pytest.mark.pyscf
def test_equioscillation_excludes_every_kernel_pole_and_equioscillates(
    h2_contour_oracles,
) -> None:
    """The contour sits at half the nearest excluded singularity, exactly."""

    adapter, exact, resolvent, mapping, _ = h2_contour_oracles
    contour = EllipseContour.from_equioscillation(resolvent, adapter.reference, mapping)
    diagnostics = contour.equioscillation_diagnostics
    assert diagnostics is not None
    assert diagnostics.sigma_c == pytest.approx(0.5 * diagnostics.sigma_min)
    assert diagnostics.limiting_singularity in ("kernel", "mirror")

    # Foci on the certified bounds put the enclosed excitations at sigma = 0.
    half_width = 0.5 * (resolvent.spectral_upper_bound - resolvent.spectral_lower_bound)
    assert contour.horizontal_radius**2 - contour.vertical_radius**2 == (
        pytest.approx(half_width**2)
    )

    energies = _resolvent_spectrum(resolvent)
    assert np.all(np.asarray(contour.normalized_radius(energies)) < 1.0)
    assert contour.leftmost > 0.0
    contour_module._validate_enclosure(resolvent, contour)


@pytest.mark.pyscf
def test_equioscillation_enumerates_poles_rather_than_assuming_d_min(
    water_rhf,
) -> None:
    """The enumerated ``sigma_min`` equals the shortcut that puts the pole at ``-Omega_L/2``."""

    adapter = RestrictedPySCFAdapter(water_rhf)
    mapping = CayleyMap(
        center=adapter.reference.chemical_potential,
        scale=0.5,
    )
    for strategy in ("frobenius", "spectral"):
        resolvent = ProjectedRPAResolvent.from_adapter(adapter, spectral_bound_strategy=strategy)
        contour = EllipseContour.from_equioscillation(resolvent, adapter.reference, mapping)
        sigma_min = contour.equioscillation_diagnostics.sigma_min
        contour_module._validate_enclosure(resolvent, contour)

        centre = contour.center
        focal = math.sqrt(contour.horizontal_radius**2 - contour.vertical_radius**2)
        shortcut = abs(
            np.arccosh(complex(-0.5 * resolvent.spectral_lower_bound - centre, 0.5) / focal).real
        )
        assert shortcut == pytest.approx(sigma_min)


@pytest.mark.pyscf
def test_equioscillation_refuses_a_degenerate_confocal_family(
    h2_contour_oracles,
) -> None:
    """An uncoupled single transition collapses the interval to a point and is refused."""

    adapter, _, _, mapping, _ = h2_contour_oracles
    resolvent = ProjectedRPAResolvent(np.array([0.8]), np.zeros((1, 1)))
    with pytest.raises(ValidationError, match="degenerate"):
        EllipseContour.from_equioscillation(resolvent, adapter.reference, mapping)


@pytest.mark.pyscf
def test_equioscillation_gives_the_tda_no_mirror_family(h2_rks_hf) -> None:
    """For ``p = 1`` there is no ``-Omega_nu`` image, so the kernel pole binds."""

    adapter = RestrictedPySCFAdapter(h2_rks_hf)
    mapping = CayleyMap(
        center=adapter.reference.chemical_potential,
        scale=1.0,
    )
    resolvent = ProjectedRPAResolvent.from_adapter(adapter, screening=Screening.TDA)
    contour = EllipseContour.from_equioscillation(resolvent, adapter.reference, mapping)
    assert contour.equioscillation_diagnostics.limiting_singularity == "kernel"


@pytest.mark.pyscf
def test_the_mirror_family_binds_at_a_large_cayley_scale(
    h2_contour_oracles,
) -> None:
    """At a large enough scale the mirror pole at ``-Omega_L`` is the nearest."""

    adapter, _, resolvent, _, _ = h2_contour_oracles
    mapping = CayleyMap(center=adapter.reference.chemical_potential, scale=20.0)
    record = EllipseContour.from_equioscillation(
        resolvent, adapter.reference, mapping
    ).equioscillation_diagnostics
    assert record.limiting_singularity == "mirror"
    assert record.sigma_c == pytest.approx(0.5 * record.sigma_min)


@pytest.mark.pyscf
def test_the_axis_guard_applies_to_rpa_and_not_to_the_tda(h2_rks_hf) -> None:
    """Crossing the imaginary axis folds ``zeta**2``, so RPA refuses it and the TDA does not."""

    adapter = RestrictedPySCFAdapter(h2_rks_hf)
    mapping = CayleyMap(center=adapter.reference.chemical_potential, scale=1.0)
    for name, screening in (("rpa", Screening.RPA), ("tda", Screening.TDA)):
        resolvent = ProjectedRPAResolvent.from_adapter(adapter, screening=screening)
        lower = resolvent.spectral_lower_bound
        upper = resolvent.spectral_upper_bound
        center = 0.5 * (upper + lower)
        # Crosses the axis but stops short of the kernel poles at Re <= -D_min/2.
        crossing = EllipseContour(
            center=center,
            horizontal_radius=center + 0.25 * lower,
            vertical_radius=0.5 * lower,
        )
        assert crossing.leftmost < 0.0
        if name == "rpa":
            with pytest.raises(ValidationError, match="not one-to-one"):
                contour_module._validate_enclosure(resolvent, crossing)
            continue
        contour_module._validate_enclosure(resolvent, crossing)
        moments = crossing.moments(resolvent, mapping, 2, 32)
        assert all(np.all(np.isfinite(moments[s].moments)) for s in Sector)
