"""Tests of the moment feasibility checks and of ``build_cayley_moments``."""

from __future__ import annotations

import numpy as np
import pytest

import cayleygw._helpers.moments.ladder as moments_module
from cayleygw import ExactG0W0SelfEnergy, Sector, ValidationError, build_cayley_moments
from cayleygw._helpers.cayley import CayleyMap
from cayleygw._helpers.moments.feasibility import feasibility_violations
from cayleygw._helpers.tolerances import RELATIVE_TOLERANCE
from cayleygw.realization._helpers.toeplitz import _block_toeplitz


def _atomic_moments(nodes, residues, n_max):
    """Return matrix moments of a finite positive unit-circle measure."""

    nodes = np.asarray(nodes, dtype=np.complex128)
    residues = np.asarray(residues, dtype=np.complex128)
    return np.einsum(
        "l,klpq->kpq",
        np.ones(nodes.size),
        nodes[None, :, None, None] ** np.arange(n_max + 1)[:, None, None, None]
        * residues[None, :, :, :],
        optimize=True,
    )


def _scalar_moments(node: complex, n_max: int) -> np.ndarray:
    """Return scalar moments for one unit-weight atom."""

    return np.asarray([[[node**order]] for order in range(n_max + 1)])


def test_scalar_toeplitz_has_vandermonde_gram_orientation() -> None:
    node = np.exp(0.37j)
    moments = _scalar_moments(node, 3)
    result = _block_toeplitz(moments)
    vandermonde = node ** np.arange(4)
    expected = vandermonde.conj()[:, None] * vandermonde[None, :]

    np.testing.assert_allclose(result, expected, atol=3.0e-16, rtol=0.0)
    assert np.linalg.eigvalsh(result)[0] > -5.0e-16


def test_block_toeplitz_uses_adjoint_negative_moments() -> None:
    c0 = np.array([[2.0, 0.3j], [-0.3j, 1.0]])
    c1 = np.array([[0.2 + 0.4j, 0.7], [-0.1j, -0.3 + 0.2j]])
    result = _block_toeplitz(np.asarray([c0, c1]))
    expected = np.block([[c0, c1], [c1.conj().T, c0]])

    np.testing.assert_array_equal(result, expected)
    assert not np.allclose(c1, c1.conj().T)


@pytest.mark.parametrize(
    ("sector", "angle"),
    [(Sector.PARTICLE, 0.65), (Sector.HOLE, -0.65)],
)
def test_an_atom_on_its_own_semicircle_is_feasible(sector, angle) -> None:
    moments = _scalar_moments(np.exp(1j * angle), 3)
    assert feasibility_violations(moments, sector) == ()


@pytest.mark.parametrize("node", [1.0 + 0.0j, -1.0 + 0.0j])
@pytest.mark.parametrize("sector", list(Sector))
def test_sector_boundary_atoms_have_zero_localizer(node, sector) -> None:
    moments = _scalar_moments(node, 3)
    assert feasibility_violations(moments, sector) == ()


def test_noncommuting_block_measure_with_repeated_nodes_is_feasible() -> None:
    nodes = np.asarray(
        [np.exp(0.25j), np.exp(0.25j), np.exp(1.1j)],
    )
    vectors = (
        np.array([1.0, 0.2j]),
        np.array([0.3j, 0.8]),
        np.array([0.6, 0.4 + 0.1j]),
    )
    residues = np.asarray([np.outer(vector, vector.conj()) for vector in vectors])
    moments = _atomic_moments(nodes, residues, 4)
    assert feasibility_violations(moments, Sector.PARTICLE) == ()


def test_exact_rank_deficient_support_is_compressed_and_feasible() -> None:
    nodes = np.asarray([np.exp(0.4j), np.exp(1.2j)])
    residues = np.asarray(
        [
            np.diag([0.7, 0.0, 0.0]),
            np.diag([0.3, 0.0, 0.0]),
        ]
    )
    moments = _atomic_moments(nodes, residues, 4)
    assert feasibility_violations(moments, Sector.PARTICLE) == ()


@pytest.mark.parametrize("sector", list(Sector))
def test_zero_measure_has_rank_zero_and_is_feasible(sector) -> None:
    moments = np.zeros((4, 3, 3), dtype=np.complex128)
    assert feasibility_violations(moments, sector) == ()


def test_full_toeplitz_degree_uses_last_supplied_moment() -> None:
    """The Toeplitz check reaches the last moment, beyond the localizer's degree."""

    moments = np.asarray([[[1.0]], [[0.0]], [[1.1]]], dtype=np.complex128)
    violations = feasibility_violations(moments, Sector.PARTICLE)

    assert any(
        item.startswith("support-normalized block Toeplitz matrix positivity")
        for item in violations
    )


def test_support_normalized_localizer_exposes_weak_wrong_arc_direction() -> None:
    """A wrong-arc direction is reported even when its C0 weight is just above the support cut."""

    order = np.arange(4)
    correct_node = np.exp(0.7j)
    wrong_node = np.exp(-0.7j)
    weak_weight = 5.0 * RELATIVE_TOLERANCE
    moments = np.zeros((4, 2, 2), dtype=np.complex128)
    moments[:, 0, 0] = correct_node**order
    moments[:, 1, 1] = weak_weight * wrong_node**order

    violations = feasibility_violations(moments, Sector.PARTICLE)
    assert any(
        item.startswith("support-normalized semicircle localizing matrix positivity")
        for item in violations
    )


def test_corrupted_toeplitz_sequence_fails_for_expected_reason() -> None:
    moments = np.asarray([[[1.0]], [[2.0]], [[1.0]]])
    violations = feasibility_violations(moments, Sector.PARTICLE)

    assert not any("zeroth moment" in item for item in violations)
    assert any("Toeplitz" in item for item in violations)


@pytest.mark.parametrize(
    ("sector", "wrong_angle"),
    [(Sector.PARTICLE, -0.7), (Sector.HOLE, 0.7)],
)
def test_wrong_semicircle_is_toeplitz_positive_but_localizer_negative(
    sector,
    wrong_angle,
) -> None:
    moments = _scalar_moments(np.exp(1j * wrong_angle), 3)
    violations = feasibility_violations(moments, sector)

    assert not any("Toeplitz" in item for item in violations)
    assert any("localizing matrix positivity" in item for item in violations)


def test_higher_moment_outside_c0_support_is_reported() -> None:
    moments = np.zeros((3, 2, 2), dtype=np.complex128)
    moments[0, 0, 0] = 1.0
    moments[1, 0, 0] = 0.5j
    moments[2, 0, 0] = -0.25
    moments[2, 1, 1] = 1.0e-4
    violations = feasibility_violations(moments, Sector.PARTICLE)

    assert any(item.startswith("zeroth-moment support") for item in violations)


def test_nonhermitian_zeroth_moment_is_not_silently_symmetrized() -> None:
    moments = np.zeros((2, 2, 2), dtype=np.complex128)
    moments[0] = np.array([[1.0, 0.1], [0.0, 1.0]])
    violations = feasibility_violations(moments, Sector.PARTICLE)

    assert any(item.startswith("zeroth moment Hermiticity") for item in violations)


@pytest.mark.pyscf
@pytest.mark.parametrize(
    "fixture_name",
    ["h2_rks_hf", "helium_rks_hf", "water_rks_hf"],
)
def test_exact_molecular_pole_moments_pass_localizing_diagnostics(
    fixture_name,
    request,
) -> None:
    mean_field = request.getfixturevalue(fixture_name)
    exact = ExactG0W0SelfEnergy.from_mean_field(mean_field)
    mapping = CayleyMap(
        center=exact.reference.chemical_potential,
        scale=1.0,
    )
    moments = exact.cayley_moments(mapping, n_max=4)

    for sector in Sector:
        assert feasibility_violations(moments[sector], sector) == ()


@pytest.mark.pyscf
def test_converged_contour_moments_recover_localizing_feasibility(h2_rks_hf) -> None:
    common = {"n_conserved": 3, "omega_p": 1.0, "verbose": 0}
    coarse = build_cayley_moments(h2_rks_hf, n_q=8, **common)
    converged = build_cayley_moments(h2_rks_hf, n_q=128, **common)

    for sector, name in ((Sector.HOLE, "hole"), (Sector.PARTICLE, "particle")):
        assert feasibility_violations(getattr(coarse, name).moments, sector)
        assert feasibility_violations(getattr(converged, name).moments, sector) == ()


# build_cayley_moments: its controls, the automatic N_q ladder and workers.


@pytest.mark.pyscf
def test_automatic_n_q_selects_and_records_converged_fine_grid(h2_rhf) -> None:
    moments = build_cayley_moments(
        h2_rhf,
        n_conserved=2,
        n_q="auto",
        n_q_tolerance=1.0e-7,
        omega_p=1.0,
    )
    diagnostics = moments.automatic_n_q_diagnostics
    assert diagnostics is not None
    # 256 depends on the geometry; it is pinned only as a regression guard.
    assert moments.n_q == diagnostics.selected_n_q == 256
    assert diagnostics.initial_n_q == 128
    assert diagnostics.maximum_n_q == 4096
    # The 0.90 vertical fraction converges in one refinement, 128 -> 256.
    assert len(diagnostics.refinements) == 1
    assert all(sector.converged for sector in diagnostics.refinements[-1])
    for refinement in diagnostics.refinements:
        for sector in refinement:
            assert sector.maximum_normalized_error <= 1.0
            assert sector.feasibility_violations == ()


@pytest.mark.pyscf
def test_automatic_n_q_maximum_failure_reports_worst_order(h2_rhf) -> None:
    """A ladder that cannot reach the tolerance names the worst order and the lever."""

    # Below roundoff, so no grid converges whatever the geometry.
    with pytest.raises(ValidationError) as captured:
        build_cayley_moments(
            h2_rhf,
            n_conserved=2,
            n_q="auto",
            n_q_tolerance=1.0e-30,
            omega_p=1.0,
        )
    message = str(captured.value)
    assert "automatic n_q selection did not converge" in message
    assert "initial_n_q=128" in message
    assert "maximum_n_q=4096" in message
    assert "worst moment=" in message
    assert "Loosen n_q_tolerance" in message


@pytest.mark.pyscf
def test_all_electron_hf_automatic_contour_and_n_q_match_df_rpa_oracle(
    hf_pbe_df,
) -> None:
    """All-electron HF's wide core-valence spectrum converges and matches the DF oracle."""

    moments = build_cayley_moments(
        hf_pbe_df,
        n_conserved=5,
        n_q="auto",
        n_q_tolerance=1.0e-7,
        omega_p=1.0,
    )
    contour_diagnostics = moments.contour.equioscillation_diagnostics
    n_q_diagnostics = moments.automatic_n_q_diagnostics
    assert contour_diagnostics is not None
    assert moments.contour.leftmost > 0.0
    assert n_q_diagnostics is not None
    assert moments.n_q == n_q_diagnostics.selected_n_q
    assert moments.n_q <= 2048
    assert all(sector.converged for sector in n_q_diagnostics.refinements[-1])

    assert moments.interaction_backend == "density-fitting"
    factors = moments.resolvent.factors
    fitted_integrals = np.einsum("Ppq,Prs->pqrs", factors, factors, optimize=True)
    exact = ExactG0W0SelfEnergy.from_integrals(
        moments.adapter.reference,
        fitted_integrals,
    ).cayley_moments(moments.mapping, moments.n_max)
    np.testing.assert_allclose(
        moments.hole.moments,
        exact[Sector.HOLE],
        atol=2.0e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        moments.particle.moments,
        exact[Sector.PARTICLE],
        atol=2.0e-7,
        rtol=0.0,
    )


@pytest.mark.pyscf
def test_certified_bound_strategies_avoid_the_exact_rpa_spectrum(
    h2_rhf,
    monkeypatch,
) -> None:
    original_eigvalsh = np.linalg.eigvalsh

    def reject_rpa_diagonalization(*args, **kwargs):
        raise AssertionError("the default contour path diagonalized H")

    monkeypatch.setattr(np.linalg, "eigvalsh", reject_rpa_diagonalization)
    frobenius = build_cayley_moments(
        h2_rhf,
        n_conserved=0,
        n_q=16,
        omega_p=1.0,
        spectral_bound="frobenius",
    )
    monkeypatch.setattr(np.linalg, "eigvalsh", original_eigvalsh)
    spectral = build_cayley_moments(
        h2_rhf,
        n_conserved=0,
        n_q=16,
        omega_p=1.0,
        spectral_bound="spectral",
    )
    gaps = frobenius.resolvent.particle_hole_gaps
    coupling = frobenius.resolvent.v_matrix
    expected_frobenius = np.sqrt(
        np.max(np.square(gaps)) + 2.0 * np.linalg.norm(coupling, ord="fro") ** 2
    )
    expected_spectral = np.sqrt(
        np.max(np.square(gaps)) + 2.0 * np.linalg.norm(coupling, ord=2) ** 2
    )
    assert frobenius.rpa_spectral_bounds.lower == pytest.approx(np.min(gaps))
    assert frobenius.rpa_spectral_bounds.upper == pytest.approx(expected_frobenius)
    assert spectral.rpa_spectral_bounds.upper == pytest.approx(expected_spectral)
    assert spectral.rpa_spectral_bounds.upper <= (frobenius.rpa_spectral_bounds.upper)


@pytest.mark.pyscf
@pytest.mark.parametrize(
    ("keyword", "value", "match"),
    [
        ("n_conserved", -1, "n_conserved"),
        ("n_conserved", True, "n_conserved"),
        ("n_q", 1, "n_q"),
        ("n_q", 7, "n_q must be even"),
        ("n_q", 4.5, "n_q"),
        # The message names both forms n_q accepts.
        ("n_q", "adaptive", "or 'auto'"),
        ("omega_p", 0.0, "omega_p"),
        ("spectral_bound", "ritz", "spectral_bound"),
    ],
)
def test_moment_workflow_rejects_controls_before_expensive_work(
    h2_rhf,
    keyword: str,
    value: object,
    match: str,
) -> None:
    arguments = {"n_conserved": 0, "n_q": 8, "omega_p": 1.0}
    arguments[keyword] = value
    with pytest.raises(ValidationError, match=match):
        build_cayley_moments(h2_rhf, **arguments)


@pytest.mark.pyscf
@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"n_q_tolerance": -1.0}, "n_q_tolerance"),
        ({"n_q_tolerance": 0.0}, "n_q_tolerance"),
    ],
)
def test_automatic_n_q_rejects_invalid_controls_before_expensive_work(
    h2_rhf,
    updates: dict[str, object],
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "n_conserved": 0,
        "n_q": "auto",
        "omega_p": 1.0,
    }
    arguments.update(updates)
    with pytest.raises(ValidationError, match=match):
        build_cayley_moments(h2_rhf, **arguments)


@pytest.mark.pyscf
@pytest.mark.parametrize("workers", [2, 4])
def test_workflow_worker_count_does_not_change_the_moments(
    h2_rhf_df,
    workers,
) -> None:
    """``n_workers`` reaches the node loop through ``build_cayley_moments`` and changes no bit."""

    common = {
        "n_conserved": 2,
        "n_q": 32,
        "omega_p": 1.0,
        "native_threads": 1,
    }
    serial = build_cayley_moments(h2_rhf_df, n_workers=1, **common)
    parallel = build_cayley_moments(h2_rhf_df, n_workers=workers, **common)
    np.testing.assert_array_equal(
        parallel.hole.moments,
        serial.hole.moments,
    )
    np.testing.assert_array_equal(
        parallel.particle.moments,
        serial.particle.moments,
    )


@pytest.mark.pyscf
def test_automatic_n_q_selection_returns_the_converged_grid_moments(
    h2_rhf,
) -> None:
    """The moments returned are those of the grid the ladder stopped at."""

    moments = build_cayley_moments(
        h2_rhf,
        n_conserved=2,
        n_q="auto",
        n_q_tolerance=1.0e-7,
        omega_p=1.0,
    )
    diagnostics = moments.automatic_n_q_diagnostics
    assert diagnostics is not None
    # 256 depends on the geometry; it is pinned only as a regression guard.
    assert moments.n_q == diagnostics.selected_n_q == 256
    assert moments.hole.moments.shape[0] == 4
    # The nested ladder returns the trapezoidal rule at that N_q, every node solved once.
    assert moments.quadrature_rule == "trapezoid" and moments.n_solved == 129
    direct = moments.contour.moments(
        moments.resolvent, moments.mapping, moments.n_max, 256, rule="trapezoid"
    )
    for sector, built in ((Sector.HOLE, moments.hole), (Sector.PARTICLE, moments.particle)):
        np.testing.assert_allclose(built.moments, direct[sector].moments, rtol=1e-12, atol=1e-15)
    fixed = build_cayley_moments(h2_rhf, n_conserved=2, n_q=256, omega_p=1.0, verbose=0)
    assert fixed.quadrature_rule == "midpoint" and fixed.n_solved == 128


@pytest.mark.pyscf
def test_n_q_rejects_an_unknown_selector_by_name(h2_rhf) -> None:
    """A misspelled selector must say what is allowed, not fail as a bad int."""

    with pytest.raises(ValidationError, match="or 'auto'"):
        build_cayley_moments(h2_rhf, n_conserved=1, n_q="latter", omega_p=1.0)


def test_the_defaults_the_regression_reference_assumes_are_pinned() -> None:
    """Pins the defaults the regression reference and examples assume; update both together."""

    import inspect

    from cayleygw.screening import ProjectedRPAResolvent

    def default(callable_, name):
        return inspect.signature(callable_).parameters[name].default

    assert default(build_cayley_moments, "spectral_bound") == ("spectral")
    assert default(ProjectedRPAResolvent.from_adapter, "spectral_bound_strategy") == "spectral"
    assert (
        ProjectedRPAResolvent.__dataclass_fields__["spectral_bound_strategy"].default == "spectral"
    )
    assert moments_module._N_Q_INITIAL == 128
    assert moments_module._N_Q_MAXIMUM == 4096
