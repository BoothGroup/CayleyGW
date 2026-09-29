"""Tests of the screening: the response solve, the projected resolvent and the Coulomb factors.

The direct-RPA and TDA responses are checked against PySCF's dRPA and dTDA, the
Woodbury resolvent against a dense solve, and the TDA moments against the exact
full-pole reference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pyscf import tdscf

from cayleygw import (
    ExactG0W0SelfEnergy,
    RefusalError,
    Screening,
    Sector,
    ValidationError,
    build_cayley_moments,
    build_upfolded_hamiltonian,
    diagonalize_upfolded,
    extract_ip_ea,
)
from cayleygw._helpers import tolerances as tolerances_module
from cayleygw._helpers.cayley import CayleyMap
from cayleygw._helpers.pyscf import RestrictedMolecularReference, RestrictedPySCFAdapter
from cayleygw._helpers.screening.resolvent import _weighted_gram
from cayleygw._helpers.screening.response import exact_coulomb_factors, solve_response
from cayleygw._helpers.tolerances import RELATIVE_TOLERANCE
from cayleygw.contour import EllipseContour
from cayleygw.screening import ProjectedRPAResolvent


OMEGA_P = 1.0
N_MAX = 4
# Water/STO-3G excitations span 1 to 21 Hartree, so the ellipse is eccentric and needs many nodes.
N_Q = 2048


def _exact_tda(mean_field):
    """Return the exact full-pole TDA self-energy for a mean field."""

    return ExactG0W0SelfEnergy.from_mean_field(mean_field, screening=Screening.TDA)


def _mappings(adapter):
    """Return the Cayley map centred in the reference gap."""

    return CayleyMap(
        center=adapter.reference.chemical_potential,
        scale=OMEGA_P,
    )


def _tda_contour(adapter):
    """Return the TDA resolvent and its equioscillation ellipse."""

    resolvent = ProjectedRPAResolvent.from_adapter(
        adapter,
        screening=Screening.TDA,
    )
    contour = EllipseContour.from_equioscillation(
        resolvent,
        adapter.reference,
        _mappings(adapter),
    )
    return resolvent, contour


@pytest.mark.pyscf
def test_screening_contract_values_are_derived_and_documented() -> None:
    assert Screening("rpa") is Screening.RPA
    assert Screening("tda") is Screening.TDA
    assert Screening.RPA.spectral_parameter_power == 2
    assert Screening.TDA.spectral_parameter_power == 1
    assert Screening.RPA.kernel_multiplier == 2.0
    assert Screening.TDA.kernel_multiplier == 1.0
    for model in Screening:
        assert isinstance(model.label, str) and model.label
    with pytest.raises(ValueError):
        Screening("cis")


@pytest.mark.pyscf
def test_tda_and_rpa_couplings_differ_by_the_gap_weight(water_rhf) -> None:
    adapter = RestrictedPySCFAdapter(water_rhf)
    tda = ProjectedRPAResolvent.from_adapter(adapter, screening=Screening.TDA)
    rpa = ProjectedRPAResolvent.from_adapter(adapter)

    assert tda.screening is Screening.TDA
    assert rpa.screening is Screening.RPA
    np.testing.assert_array_equal(tda.particle_hole_gaps, rpa.particle_hole_gaps)
    # Direct RPA weights the coupling by D**(1/2); the TDA does not.
    np.testing.assert_allclose(
        rpa.v_matrix,
        np.sqrt(rpa.particle_hole_gaps)[:, None] * tda.v_matrix,
        atol=3.0e-13,
    )
    _, energies, _ = solve_response(
        adapter.reference,
        adapter.build_full_integrals(),
        Screening.TDA,
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(
            np.diag(tda.particle_hole_gaps) + tda.v_matrix @ tda.v_matrix.T
        ),
        energies,
        atol=3.0e-12,
    )
    with pytest.raises(ValidationError, match="Screening value"):
        ProjectedRPAResolvent.from_adapter(adapter, screening="tda")


@pytest.mark.pyscf
def test_tda_resolvent_matches_the_analytic_scalar() -> None:
    gap = 1.3
    coupling = 0.4
    zeta = 0.8 + 0.2j
    resolvent = ProjectedRPAResolvent(
        np.array([gap]),
        np.array([[coupling]]),
        screening=Screening.TDA,
    )
    expected = coupling**2 / (zeta - gap - coupling**2)
    assert resolvent.woodbury(zeta)[0, 0] == pytest.approx(
        expected,
        abs=2.0e-16,
    )
    with pytest.raises(ValidationError, match="Screening value"):
        ProjectedRPAResolvent(
            np.array([1.3]),
            np.array([[0.4]]),
            screening="tda",
        )


@pytest.mark.pyscf
def test_certified_tda_bounds_enclose_the_exact_spectrum(water_rhf) -> None:
    adapter = RestrictedPySCFAdapter(water_rhf)
    _, energies, _ = solve_response(
        adapter.reference,
        adapter.build_full_integrals(),
        Screening.TDA,
    )
    for strategy in ("frobenius", "spectral"):
        cheap = ProjectedRPAResolvent.from_adapter(
            adapter,
            screening=Screening.TDA,
            spectral_bound_strategy=strategy,
        )
        assert cheap.spectral_lower_bound <= energies[0]
        assert cheap.spectral_upper_bound >= energies[-1]


@pytest.mark.pyscf
def test_tda_contour_moments_match_the_exact_poles(
    water_rhf,
) -> None:
    adapter = RestrictedPySCFAdapter(water_rhf)
    mappings = _mappings(adapter)
    expected = _exact_tda(water_rhf).cayley_moments(mappings, N_MAX)

    resolvent, contour = _tda_contour(adapter)
    obtained = contour.moments(resolvent, mappings, N_MAX, N_Q)

    for sector in Sector:
        np.testing.assert_allclose(
            obtained[sector].moments,
            expected[sector],
            atol=1.0e-11,
        )


@pytest.mark.pyscf
def test_tda_conjugate_pairing_matches_the_exact_moments(
    water_rhf,
) -> None:
    adapter = RestrictedPySCFAdapter(water_rhf)
    mappings = _mappings(adapter)
    expected = _exact_tda(water_rhf).cayley_moments(mappings, 2)

    resolvent, contour = _tda_contour(adapter)
    obtained = contour.moments(resolvent, mappings, 2, N_Q)
    for sector in Sector:
        np.testing.assert_allclose(
            obtained[sector].moments,
            expected[sector],
            atol=1.0e-11,
        )


@pytest.mark.pyscf
def test_exact_tda_self_energy_differs_from_direct_rpa(
    water_rhf,
) -> None:
    adapter = RestrictedPySCFAdapter(water_rhf)
    mappings = _mappings(adapter)

    self_energy = _exact_tda(water_rhf)
    exact = self_energy.cayley_moments(mappings, N_MAX)

    rpa_self_energy = ExactG0W0SelfEnergy.from_mean_field(water_rhf)
    assert not np.allclose(
        np.asarray(exact[Sector.HOLE]),
        np.asarray(rpa_self_energy.cayley_moments(mappings, N_MAX)[Sector.HOLE]),
        atol=1.0e-6,
    )


@pytest.mark.pyscf
def test_workflow_screening_control_reaches_the_exact_tda_poles(
    water_rhf,
) -> None:
    adapter = RestrictedPySCFAdapter(water_rhf)
    expected = _exact_tda(water_rhf).cayley_moments(_mappings(adapter), N_MAX)

    moments = build_cayley_moments(
        water_rhf,
        n_conserved=N_MAX - 1,
        n_q=N_Q,
        omega_p=OMEGA_P,
        screening="tda",
    )
    assert moments.resolvent.screening is Screening.TDA
    enumerated = build_cayley_moments(
        water_rhf,
        n_conserved=N_MAX - 1,
        n_q=N_Q,
        omega_p=OMEGA_P,
        screening=Screening.TDA,
    )
    for sector, obtained in (
        (Sector.HOLE, moments.hole),
        (Sector.PARTICLE, moments.particle),
    ):
        np.testing.assert_allclose(
            obtained.moments,
            expected[sector],
            atol=1.0e-11,
        )
    np.testing.assert_allclose(
        enumerated.hole.moments,
        moments.hole.moments,
        atol=0.0,
        rtol=0.0,
    )

    default = build_cayley_moments(
        water_rhf,
        n_conserved=N_MAX - 1,
        n_q=N_Q,
        omega_p=OMEGA_P,
    )
    assert default.resolvent.screening is Screening.RPA
    assert not np.allclose(
        default.hole.moments,
        moments.hole.moments,
        atol=1.0e-6,
    )

    with pytest.raises(ValidationError, match="screening must be"):
        build_cayley_moments(
            water_rhf,
            n_conserved=0,
            n_q=16,
            omega_p=OMEGA_P,
            screening="cis",
        )


@pytest.mark.pyscf
def test_automatic_contour_and_node_selection_work_for_the_tda(
    water_rhf,
) -> None:
    moments = build_cayley_moments(
        water_rhf,
        n_conserved=1,
        n_q="auto",
        omega_p=OMEGA_P,
        screening="tda",
    )
    diagnostics = moments.automatic_n_q_diagnostics
    assert diagnostics is not None
    assert all(sector.converged for sector in diagnostics.refinements[-1])
    assert diagnostics.selected_n_q == moments.n_q
    assert moments.resolvent.screening is Screening.TDA


@pytest.mark.pyscf
def test_tda_workflow_completes_a_hermitian_upfolded_solution(
    water_rhf,
) -> None:
    results = {}
    for model in ("rpa", "tda"):
        moments = build_cayley_moments(
            water_rhf,
            n_conserved=2,
            n_q=N_Q,
            omega_p=OMEGA_P,
            screening=model,
        )
        hamiltonian = build_upfolded_hamiltonian(
            moments, terminal_selection="scan", terminal_phase_count=64
        )
        spectrum = diagonalize_upfolded(hamiltonian)
        results[model] = extract_ip_ea(spectrum, n_ip=1, n_ea=1)

    for model, ipea in results.items():
        assert ipea.ionization_potentials.size == 1, model
        assert ipea.electron_affinities.size == 1, model
        assert ipea.removal_pole_energies[0] < ipea.addition_pole_energies[0]
    # RPA and TDA are different approximations, so their IPs must differ.
    assert abs(
        results["rpa"].ionization_potentials[0]
        - results["tda"].ionization_potentials[0]
    ) > 1.0e-5


@pytest.mark.pyscf
def test_cauchy_and_screened_weights_are_exactly_consistent() -> None:
    contour = EllipseContour(center=1.0, horizontal_radius=0.5, vertical_radius=0.2)
    _, weights = contour.midpoint_rule(8)

    np.testing.assert_array_equal(
        float(Screening.RPA.spectral_parameter_power) * (0.5 * weights),
        weights,
    )
    np.testing.assert_array_equal(
        float(Screening.TDA.spectral_parameter_power) * (0.5 * weights),
        weights / 2.0,
    )


# Direct-RPA response, checked against PySCF's dRPA.


def _run_pyscf_drpa(mean_field, *, frozen, nstates):
    """Return all converged PySCF restricted direct-RPA modes."""

    direct_rpa = tdscf.dRPA(mean_field, frozen=frozen)
    direct_rpa.nstates = nstates
    direct_rpa.conv_tol = 1.0e-12
    direct_rpa.max_cycle = 100
    direct_rpa.verbose = 0
    direct_rpa.kernel()
    assert np.all(direct_rpa.converged)
    assert direct_rpa.e.shape == (nstates,)
    return direct_rpa


def _pyscf_amplitude_matrices(direct_rpa):
    """Stack PySCF's per-state alpha-spin amplitudes into mode columns."""

    x_amplitudes = np.column_stack(
        [np.asarray(state[0]).reshape(-1) for state in direct_rpa.xy]
    )
    y_amplitudes = np.column_stack(
        [np.asarray(state[1]).reshape(-1) for state in direct_rpa.xy]
    )
    return x_amplitudes, y_amplitudes


def _solve_rpa(adapter):
    """Solve direct RPA on the adapter's own exact four-index tensor."""

    eri = adapter.build_full_integrals()
    return eri, *solve_response(adapter.reference, eri)


def _casida_blocks(reference, eri, gaps):
    """Return the restricted direct-RPA ``A`` and ``B`` built from the tensor."""

    occupied = np.asarray(reference.occupied_positions)
    virtual = np.asarray(reference.virtual_positions)
    kernel = 2.0 * eri[np.ix_(occupied, virtual, occupied, virtual)].reshape(
        gaps.size, gaps.size
    )
    return np.diag(gaps) + kernel, kernel


def _two_orbital_reference(gap):
    """Return a one-transition synthetic reference with the given gap."""

    return RestrictedMolecularReference(
        mo_coeff=np.eye(2),
        mo_energy=np.array([-0.5 * gap, 0.5 * gap]),
        occupied_positions=np.array([0]),
        virtual_positions=np.array([1]),
        chemical_potential=0.0,
        homo_energy=-0.5 * gap,
        lumo_energy=0.5 * gap,
        gap=gap,
    )


@pytest.mark.pyscf
@pytest.mark.parametrize(
    ("fixture_name", "frozen"),
    [
        ("h2_rks_hf", 0),
        ("h2_pbe", 0),
        ("helium_rks_hf", 0),
        ("water_rks_hf", 0),
        ("water_rks_hf", 1),
    ],
)
def test_energies_and_amplitudes_match_pyscf(
    request,
    fixture_name,
    frozen,
) -> None:
    mean_field = request.getfixturevalue(fixture_name)
    adapter = RestrictedPySCFAdapter(mean_field, frozen=frozen)
    _, gaps, energies, x_plus_y = _solve_rpa(adapter)
    pyscf_result = _run_pyscf_drpa(
        mean_field,
        frozen=frozen,
        nstates=gaps.size,
    )
    np.testing.assert_allclose(
        energies,
        pyscf_result.e,
        atol=5.0e-10,
        rtol=0.0,
    )

    pyscf_x, pyscf_y = _pyscf_amplitude_matrices(pyscf_result)
    identity = np.eye(gaps.size)
    np.testing.assert_allclose(
        2.0 * (pyscf_x.T @ pyscf_x - pyscf_y.T @ pyscf_y),
        identity,
        atol=3.0e-10,
        rtol=0.0,
    )
    canonical_plus = math.sqrt(2.0) * (pyscf_x + pyscf_y)
    canonical_minus = math.sqrt(2.0) * (pyscf_x - pyscf_y)

    # These outer products do not see state signs or rotations within degenerate subspaces.
    x_minus_y = x_plus_y * energies[None, :] / gaps[:, None]
    np.testing.assert_allclose(
        x_plus_y @ x_plus_y.T,
        canonical_plus @ canonical_plus.T,
        atol=5.0e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        x_minus_y @ x_minus_y.T,
        canonical_minus @ canonical_minus.T,
        atol=5.0e-10,
        rtol=0.0,
    )


@pytest.mark.pyscf
def test_one_transition_analytic_solution_and_restricted_spin_factor(
    h2_rks_hf,
) -> None:
    adapter = RestrictedPySCFAdapter(h2_rks_hf)
    eri, gaps, energies, x_plus_y = _solve_rpa(adapter)
    assert gaps.size == 1

    gap = gaps[0]
    spatial_coulomb = eri[0, 1, 0, 1]
    expected_energy = math.sqrt(gap * (gap + 4.0 * spatial_coulomb))
    assert energies[0] == pytest.approx(
        expected_energy,
        abs=2.0e-14,
    )
    assert abs(x_plus_y[0, 0]) == pytest.approx(
        math.sqrt(gap / expected_energy),
        abs=2.0e-14,
    )


@pytest.mark.pyscf
def test_rhf_problem_matches_equivalent_exact_exchange_rks(
    h2_rhf,
    h2_rks_hf,
) -> None:
    _, rhf_gaps, rhf_energies, _ = _solve_rpa(RestrictedPySCFAdapter(h2_rhf))
    _, rks_gaps, rks_energies, _ = _solve_rpa(RestrictedPySCFAdapter(h2_rks_hf))

    np.testing.assert_allclose(rhf_gaps, rks_gaps, atol=2.0e-13, rtol=0.0)
    np.testing.assert_allclose(
        rhf_energies,
        rks_energies,
        atol=2.0e-13,
        rtol=0.0,
    )


@pytest.mark.pyscf
def test_squared_rpa_amplitudes_solve_the_casida_problem(water_rks_hf) -> None:
    adapter = RestrictedPySCFAdapter(water_rks_hf, frozen=1)
    eri, gaps, energies, x_plus_y = _solve_rpa(adapter)
    a_matrix, b_matrix = _casida_blocks(adapter.reference, eri, gaps)
    x_minus_y = x_plus_y * energies[None, :] / gaps[:, None]
    x_amplitudes = 0.5 * (x_plus_y + x_minus_y)
    y_amplitudes = 0.5 * (x_plus_y - x_minus_y)

    np.testing.assert_allclose(
        x_plus_y.T @ x_minus_y,
        np.eye(gaps.size),
        atol=2.0e-13,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        a_matrix @ x_amplitudes + b_matrix @ y_amplitudes,
        x_amplitudes * energies[None, :],
        atol=2.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        b_matrix @ x_amplitudes + a_matrix @ y_amplitudes,
        -y_amplitudes * energies[None, :],
        atol=2.0e-12,
        rtol=0.0,
    )


@pytest.mark.pyscf
def test_noninteracting_limit_reduces_to_independent_transitions(
    water_rks_hf,
) -> None:
    adapter = RestrictedPySCFAdapter(water_rks_hf, frozen=1)
    nmo = adapter.reference.nmo
    gaps, energies, x_plus_y = solve_response(
        adapter.reference, np.zeros((nmo, nmo, nmo, nmo))
    )

    np.testing.assert_allclose(
        energies,
        np.sort(gaps),
        atol=2.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        x_plus_y.T @ x_plus_y,
        np.eye(gaps.size),
        atol=2.0e-14,
        rtol=0.0,
    )


@pytest.mark.pyscf
@pytest.mark.parametrize(
    ("gap", "coulomb", "message"),
    [
        (1.0e-12, 0.0, "particle-hole gap"),
        (1.0, -0.30, "squared matrix"),
    ],
)
def test_near_zero_gap_and_unstable_synthetic_problems_fail_explicitly(
    gap,
    coulomb,
    message,
) -> None:
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 1, 0, 1] = coulomb
    with pytest.raises(RefusalError, match=message):
        solve_response(_two_orbital_reference(gap), eri)


@pytest.mark.pyscf
def test_roundoff_sized_coulomb_asymmetry_is_accepted_and_material_is_not(
    monkeypatch,
) -> None:
    reference = RestrictedMolecularReference(
        mo_coeff=np.eye(3),
        mo_energy=np.array([-0.5, 0.5, 1.0]),
        occupied_positions=np.array([0]),
        virtual_positions=np.array([1, 2]),
        chemical_potential=0.0,
        homo_energy=-0.5,
        lumo_energy=0.5,
        gap=1.0,
    )
    eri = np.zeros((3, 3, 3, 3))
    eri[0, 1, 0, 1] = 0.3
    eri[0, 2, 0, 2] = 0.2
    eri[0, 1, 0, 2] = 0.1 + 1.0e-9
    eri[0, 2, 0, 1] = 0.1
    with pytest.raises(ValidationError, match="not symmetric"):
        solve_response(reference, eri)
    monkeypatch.setattr(tolerances_module, "ROUNDOFF_TOLERANCE", 1.0e-8)
    solve_response(reference, eri)


# Tamm-Dancoff response, checked against PySCF's dTDA.


def _run_pyscf_dtda(mean_field, *, frozen, nstates):
    """Return all converged PySCF restricted direct-TDA modes."""

    # tdscf.TDA keeps the exchange kernel; dTDA is the direct-kernel match.
    direct_tda = tdscf.dTDA(mean_field, frozen=frozen)
    direct_tda.nstates = nstates
    direct_tda.conv_tol = 1.0e-12
    direct_tda.max_cycle = 400
    direct_tda.verbose = 0
    direct_tda.kernel()
    assert np.all(direct_tda.converged)
    assert direct_tda.e.shape == (nstates,)
    return direct_tda


def _eigenspace_projector_residual(ours, theirs, energies):
    """Compare amplitudes through rotation-invariant eigenspace projectors."""

    worst = 0.0
    start = 0
    for index in range(1, energies.size + 1):
        if index == energies.size or abs(energies[index] - energies[start]) > 1.0e-8:
            block = slice(start, index)
            residual = float(
                np.max(
                    np.abs(
                        ours[:, block] @ ours[:, block].T
                        - theirs[:, block] @ theirs[:, block].T
                    )
                )
            )
            worst = max(worst, residual)
            start = index
    return worst


def _solve(adapter, screening=Screening.TDA):
    """Solve the response on the adapter's own exact four-index tensor."""

    eri = adapter.build_full_integrals()
    return eri, *solve_response(adapter.reference, eri, screening)


@pytest.mark.pyscf
@pytest.mark.parametrize(
    ("fixture_name", "frozen"),
    [
        ("h2_rks_hf", 0),
        ("helium_rks_hf", 0),
        ("water_rks_hf", 0),
        ("water_rks_hf", 1),
        ("h2_pbe", 0),
    ],
)
def test_energies_and_amplitudes_match_pyscf_dtda(
    request,
    fixture_name,
    frozen,
) -> None:
    mean_field = request.getfixturevalue(fixture_name)
    adapter = RestrictedPySCFAdapter(mean_field, frozen=frozen)
    _, gaps, energies, amplitudes = _solve(adapter)
    pyscf_result = _run_pyscf_dtda(
        mean_field,
        frozen=frozen,
        nstates=gaps.size,
    )

    np.testing.assert_allclose(
        energies,
        np.asarray(pyscf_result.e),
        atol=3.0e-11,
        rtol=0.0,
    )
    pyscf_x = np.column_stack(
        [np.asarray(state[0]).reshape(-1) for state in pyscf_result.xy]
    )
    residual = _eigenspace_projector_residual(
        amplitudes / math.sqrt(2.0),
        pyscf_x,
        np.asarray(pyscf_result.e),
    )
    assert residual < 3.0e-9


@pytest.mark.pyscf
def test_tda_excitations_bound_direct_rpa_from_above(water_rhf) -> None:
    adapter = RestrictedPySCFAdapter(water_rhf)
    _, _, tda, _ = _solve(adapter)
    _, _, rpa, _ = _solve(adapter, Screening.RPA)

    assert np.all(rpa <= tda + 1.0e-12)
    # A physical interaction must separate the two models.
    assert np.max(tda - rpa) > 1.0e-4


@pytest.mark.pyscf
def test_zero_interaction_reproduces_bare_particle_hole_gaps(water_rhf) -> None:
    reference = RestrictedPySCFAdapter(water_rhf).reference
    nmo = reference.nmo
    zero = np.zeros((nmo, nmo, nmo, nmo))
    gaps, tda, _ = solve_response(reference, zero, Screening.TDA)
    _, rpa, _ = solve_response(reference, zero, Screening.RPA)

    np.testing.assert_allclose(tda, np.sort(gaps), atol=2.0e-14)
    np.testing.assert_allclose(tda, rpa, atol=2.0e-14)


@pytest.mark.pyscf
def test_single_transition_matches_the_analytic_tda_energy(h2_rhf) -> None:
    adapter = RestrictedPySCFAdapter(h2_rhf)
    eri, gaps, energies, amplitudes = _solve(adapter)
    assert gaps.size == 1

    gap = float(gaps[0])
    kernel = 2.0 * float(eri[0, 1, 0, 1])
    assert energies[0] == pytest.approx(gap + kernel, abs=2.0e-14)
    assert abs(amplitudes[0, 0]) == pytest.approx(1.0, abs=2.0e-15)


@pytest.mark.pyscf
def test_amplitudes_are_orthonormal_and_energies_positive(water_rhf) -> None:
    _, _, energies, amplitudes = _solve(RestrictedPySCFAdapter(water_rhf))

    np.testing.assert_allclose(
        amplitudes.T @ amplitudes,
        np.eye(energies.size),
        atol=1.0e-12,
    )
    assert energies[0] > 0.0
    assert np.all(np.diff(energies) >= 0.0)


@pytest.mark.pyscf
def test_ungapped_and_asymmetric_inputs_are_rejected() -> None:
    with pytest.raises(ValidationError, match="positive definite"):
        solve_response(_two_orbital_reference(1.0e-12), np.zeros((2, 2, 2, 2)), Screening.TDA)

    reference = RestrictedMolecularReference(
        mo_coeff=np.eye(3),
        mo_energy=np.array([-0.5, 0.5, 1.0]),
        occupied_positions=np.array([0]),
        virtual_positions=np.array([1, 2]),
        chemical_potential=0.0,
        homo_energy=-0.5,
        lumo_energy=0.5,
        gap=1.0,
    )
    asymmetric = np.zeros((3, 3, 3, 3))
    asymmetric[0, 1, 0, 2] = 1.0
    with pytest.raises(ValidationError, match="not symmetric"):
        solve_response(reference, asymmetric, Screening.TDA)


@pytest.mark.pyscf
def test_tda_completes_where_direct_rpa_is_unstable() -> None:
    """With ``-D < K^S < -D/2``, ``D(D + 2 K^S)`` is negative but ``D + K^S`` is positive."""

    reference = _two_orbital_reference(1.0)
    # K^S = 2 * g, so g = -0.35 gives K^S = -0.7 in the required window.
    attractive = np.zeros((2, 2, 2, 2))
    attractive[0, 1, 0, 1] = -0.35

    with pytest.raises(RefusalError):
        solve_response(reference, attractive, Screening.RPA)

    _, energies, _ = solve_response(reference, attractive, Screening.TDA)
    assert energies[0] == pytest.approx(0.3, abs=2.0e-15)


# Woodbury projected resolvent, checked against a dense solve.


def _synthetic_resolvent(**options) -> ProjectedRPAResolvent:
    """Return a deterministic rectangular real projected-RPA problem."""

    gaps = np.array([0.7, 1.1, 1.6, 2.0])
    coupling = np.array(
        [
            [0.20, 0.10],
            [0.05, -0.18],
            [0.30, 0.04],
            [-0.12, 0.22],
        ]
    )
    return ProjectedRPAResolvent(gaps, coupling, **options)


def _response_matrix(resolvent) -> np.ndarray:
    """Return the dense particle-hole operator ``M = D**p + c V V^T``."""

    power = resolvent.screening.spectral_parameter_power
    coupling = resolvent.v_matrix
    return (
        np.diag(np.power(resolvent.particle_hole_gaps, power))
        + resolvent.screening.kernel_multiplier * coupling @ coupling.T
    )


def _dense(resolvent, zeta) -> np.ndarray:
    """Solve ``V^T (zeta**p I - M)^{-1} V`` in particle-hole space."""

    power = resolvent.screening.spectral_parameter_power
    response = _response_matrix(resolvent)
    coefficient = zeta**power * np.eye(response.shape[0]) - response
    return resolvent.v_matrix.T @ np.linalg.solve(coefficient, resolvent.v_matrix)


def _compare(resolvent, zeta) -> float:
    """Return the relative Woodbury-versus-dense difference at ``zeta``."""

    dense = _dense(resolvent, zeta)
    woodbury = resolvent.woodbury(zeta)
    scale = max(1.0, float(np.max(np.abs(dense))), float(np.max(np.abs(woodbury))))
    relative = float(np.max(np.abs(dense - woodbury))) / scale
    assert relative <= RELATIVE_TOLERANCE
    return relative


@pytest.mark.parametrize(
    "zeta",
    [
        0.45 + 0.31j,
        0.45 - 0.31j,
        1.35 + 0.27j,
        1.35 - 0.27j,
        -0.82 + 0.44j,
    ],
)
def test_woodbury_agrees_with_the_dense_solve_at_complex_nodes(zeta) -> None:
    resolvent = _synthetic_resolvent()
    np.testing.assert_allclose(
        _dense(resolvent, zeta),
        resolvent.woodbury(zeta),
        atol=3.0e-15,
        rtol=3.0e-14,
    )
    assert _compare(resolvent, zeta) < 3.0e-15


def test_auxiliary_linear_system_is_explicitly_verified() -> None:
    resolvent = _synthetic_resolvent()
    zeta = 0.93 + 0.36j
    projected = resolvent.woodbury(zeta)
    inverse_free = 1.0 / (
        zeta**2 - np.square(resolvent.particle_hole_gaps)
    )
    q_matrix = resolvent.v_matrix.T @ (
        inverse_free[:, None] * resolvent.v_matrix
    )
    np.testing.assert_allclose(
        (np.eye(q_matrix.shape[0]) - 2.0 * q_matrix) @ projected,
        q_matrix,
        atol=3.0e-16,
        rtol=2.0e-15,
    )


def test_certified_norm_bounds_enclose_the_squared_rpa_eigenspectrum() -> None:
    cheap = _synthetic_resolvent(spectral_bound_strategy="frobenius")
    spectral = _synthetic_resolvent(spectral_bound_strategy="spectral")

    gaps = cheap.particle_hole_gaps
    coupling = cheap.v_matrix
    expected_frobenius = np.sqrt(
        np.max(np.square(gaps)) + 2.0 * np.linalg.norm(coupling, ord="fro") ** 2
    )
    expected_spectral = np.sqrt(
        np.max(np.square(gaps)) + 2.0 * np.linalg.norm(coupling, ord=2) ** 2
    )
    assert cheap.spectral_lower_bound == pytest.approx(np.min(gaps))
    assert cheap.spectral_upper_bound == pytest.approx(expected_frobenius)
    assert cheap.spectral_bound_source == "D_min and Frobenius-norm upper bound"
    assert spectral.spectral_bound_source == "D_min and spectral-norm upper bound"
    assert spectral.spectral_upper_bound == pytest.approx(expected_spectral)
    assert spectral.spectral_upper_bound <= cheap.spectral_upper_bound
    exact_energies = np.sqrt(np.linalg.eigvalsh(_response_matrix(cheap)))
    for bounds in (cheap, spectral):
        assert bounds.spectral_lower_bound <= exact_energies[0]
        assert bounds.spectral_upper_bound >= exact_energies[-1]
    # The strategy moves the enclosure only, never a node's value.
    node = 1.1 + 0.3j
    np.testing.assert_array_equal(cheap.woodbury(node), spectral.woodbury(node))
    with pytest.raises(ValidationError, match="spectral_bound_strategy"):
        _synthetic_resolvent(spectral_bound_strategy="exact")


def test_one_dimensional_case_matches_scalar_analytic_result() -> None:
    gap = 1.3
    coupling = 0.4
    zeta = 0.8 + 0.2j
    resolvent = ProjectedRPAResolvent(
        np.array([gap]),
        np.array([[coupling]]),
    )
    _compare(resolvent, zeta)
    expected = coupling**2 / (zeta**2 - gap**2 - 2.0 * coupling**2)
    assert resolvent.woodbury(zeta)[0, 0] == pytest.approx(
        expected,
        abs=2.0e-16,
    )


def test_rectangular_rank_deficient_and_zero_couplings_are_supported() -> None:
    gaps = np.array([0.6, 0.9, 1.4, 1.8])
    column = np.array([0.1, -0.2, 0.3, 0.05])
    rank_deficient = np.column_stack((column, 2.0 * column, -column))
    resolvent = ProjectedRPAResolvent(gaps, rank_deficient)
    assert resolvent.v_matrix.shape == (4, 3)
    assert np.linalg.matrix_rank(resolvent.v_matrix) == 1
    assert _compare(resolvent, 1.1 + 0.3j) < 2.0e-15

    zero_columns = ProjectedRPAResolvent(gaps, np.zeros((4, 2)))
    np.testing.assert_array_equal(zero_columns.woodbury(1.1 + 0.3j), 0.0)


def test_real_problem_obeys_evenness_and_conjugation_but_not_hermiticity() -> None:
    resolvent = _synthetic_resolvent()
    zeta = 0.93 + 0.37j
    value = resolvent.woodbury(zeta)
    conjugate = resolvent.woodbury(zeta.conjugate())
    negative = resolvent.woodbury(-zeta)
    np.testing.assert_allclose(conjugate, value.conj(), atol=3.0e-16)
    np.testing.assert_allclose(negative, value, atol=3.0e-16)
    np.testing.assert_allclose(value, value.T, atol=3.0e-16)
    assert np.max(np.abs(value - value.conj().T)) > 1.0e-3


def test_auxiliary_basis_rotation_gives_covariant_projected_resolvent() -> None:
    resolvent = _synthetic_resolvent()
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    rotated = ProjectedRPAResolvent(
        resolvent.particle_hole_gaps,
        resolvent.v_matrix @ rotation,
    )
    zeta = 1.24 - 0.41j
    original_value = resolvent.woodbury(zeta)
    rotated_value = rotated.woodbury(zeta)
    np.testing.assert_allclose(
        rotated_value,
        rotation.T @ original_value @ rotation,
        atol=3.0e-16,
        rtol=2.0e-15,
    )


def test_nearly_singular_but_valid_node_still_agrees_with_the_dense_solve() -> None:
    resolvent = _synthetic_resolvent()
    target = float(np.sqrt(np.linalg.eigvalsh(_response_matrix(resolvent))[1]))
    assert _compare(resolvent, target + 1.0e-7j) < RELATIVE_TOLERANCE


def test_solve_reuses_the_factorization_without_changing_the_solution() -> None:
    """One ``getrf`` feeding ``getrs`` must equal a fresh ``solve`` exactly."""

    resolvent = _synthetic_resolvent()
    zeta = 0.93 + 0.36j
    weights = 1.0 / (zeta**2 - np.square(resolvent.particle_hole_gaps))
    q_matrix = _weighted_gram(resolvent.v_matrix, weights)
    coefficient = np.eye(q_matrix.shape[0], dtype=np.complex128) - 2.0 * q_matrix
    assert np.array_equal(
        resolvent.woodbury(zeta),
        np.linalg.solve(coefficient, q_matrix),
    )


def test_prohibited_free_and_interacting_nodes_raise_structured_diagnostics() -> None:
    resolvent = _synthetic_resolvent()
    free_node = float(resolvent.particle_hole_gaps[1])
    with pytest.raises(
        RefusalError,
        match="free particle-hole pole",
    ) as free_error:
        resolvent.woodbury(free_node)
    assert free_error.value.zeta == free_node
    assert (
        free_error.value.free_pole_distance
        <= free_error.value.pole_threshold
    )
    assert free_error.value.rpa_pole_distance >= 0.0

    # Inside the certified interval, a real node may sit on the interacting spectrum.
    rpa_node = float(np.sqrt(np.linalg.eigvalsh(_response_matrix(resolvent))[0]))
    with pytest.raises(
        RefusalError,
        match="interacting RPA pole",
    ) as rpa_error:
        resolvent.woodbury(rpa_node)
    assert rpa_error.value.zeta == rpa_node
    assert (
        rpa_error.value.rpa_pole_distance
        <= rpa_error.value.pole_threshold
    )
    assert "exclusion threshold" in str(rpa_error.value)


@pytest.mark.parametrize(
    ("gaps", "coupling", "message"),
    [
        ([], np.zeros((0, 1)), "nonempty vector"),
        ([0.5, 0.0], np.zeros((2, 1)), "strictly positive"),
        ([0.5, np.nan], np.zeros((2, 1)), "finite"),
        ([0.5, 0.8], np.zeros((3, 1)), "ntransition"),
        ([0.5, 0.8], np.array([[0.1], [np.inf]]), "finite"),
        ([0.5, 0.8], np.array([[0.1 + 0.1j], [0.2]]), "real"),
    ],
)
def test_invalid_resolvent_inputs_are_rejected(gaps, coupling, message) -> None:
    with pytest.raises(ValidationError, match=message):
        ProjectedRPAResolvent(gaps, coupling)


@pytest.mark.parametrize("node", [np.nan, np.inf, True, [1.0, 2.0]])
def test_invalid_nodes_are_rejected(node) -> None:
    with pytest.raises(ValidationError, match="finite scalar"):
        _synthetic_resolvent().woodbury(node)


def test_every_node_agrees_with_the_dense_solve() -> None:
    resolvent = _synthetic_resolvent()
    nodes = [0.5 + 0.2j, 1.0 - 0.3j, -1.3 + 0.4j]
    assert all(_compare(resolvent, node) < 2.0e-15 for node in nodes)


@pytest.mark.pyscf
def test_molecular_resolvent_matches_the_dense_squared_rpa_problem(h2_rks_hf) -> None:
    adapter = RestrictedPySCFAdapter(h2_rks_hf)
    resolvent = ProjectedRPAResolvent.from_adapter(adapter)
    _, energies, _ = solve_response(adapter.reference, adapter.build_full_integrals())
    np.testing.assert_allclose(
        np.sqrt(np.linalg.eigvalsh(_response_matrix(resolvent))),
        energies,
        atol=2.0e-12,
        rtol=0.0,
    )
    nodes = [
        0.8 * energies[0] + 0.25j,
        1.2 * energies[-1] - 0.35j,
    ]
    for node in nodes:
        assert _compare(resolvent, node) < 2.0e-14


def test_arrays_are_defensively_copied_and_read_only() -> None:
    gaps = np.array([0.7, 1.2])
    coupling = np.array([[0.1], [0.2]])
    resolvent = ProjectedRPAResolvent(gaps, coupling)
    gaps[:] = 9.0
    coupling[:] = 9.0
    np.testing.assert_array_equal(resolvent.particle_hole_gaps, [0.7, 1.2])
    np.testing.assert_array_equal(resolvent.v_matrix[:, 0], [0.1, 0.2])
    assert not resolvent.particle_hole_gaps.flags.writeable
    assert not resolvent.v_matrix.flags.writeable


def test_a_sealed_coupling_is_held_without_a_copy() -> None:
    """A sealed C-contiguous V is held as is; a copy is tens of GB on a large molecule."""

    rng = np.random.default_rng(11)
    gaps = rng.uniform(0.5, 2.0, 40)
    coupling = np.ascontiguousarray(0.1 * rng.standard_normal((40, 9)))
    coupling.setflags(write=False)
    resolvent = ProjectedRPAResolvent(gaps, coupling)
    assert resolvent.v_matrix is coupling
    assert resolvent.v_matrix.dtype == np.float64


@pytest.mark.parametrize("zeta", [0.93 + 0.36j, 1.35 - 0.27j, 2.10 + 0.04j])
def test_real_arithmetic_gram_matches_the_complex_product_to_a_few_ulp(
    zeta,
) -> None:
    """The real-split ``Q`` is within 8 ulp of the complex product (measured at most 2.37)."""

    resolvent = _synthetic_resolvent()
    power = resolvent.screening.spectral_parameter_power
    weights = 1.0 / (
        zeta**power - np.power(np.asarray(resolvent.particle_hole_gaps), power)
    )
    coupling = resolvent.v_matrix.astype(np.complex128)
    exact = coupling.conj().T @ (weights[:, None] * coupling)
    split = _weighted_gram(resolvent.v_matrix, weights)
    scale = max(1.0, float(np.max(np.abs(exact))))
    tolerated = 8.0 * np.finfo(np.float64).eps * scale
    assert np.max(np.abs(exact - split)) <= tolerated


def test_the_row_blocked_gram_agrees_with_the_single_block_one() -> None:
    """Above one row block the Gram matches the direct product to roundoff, below it exactly."""

    from cayleygw._helpers.screening import resolvent as resolvent_module

    rng = np.random.default_rng(13)
    rows = resolvent_module._GRAM_ROW_BLOCK + 2500
    coupling = rng.standard_normal((rows, 6))
    weights = rng.standard_normal(rows) + 1j * rng.standard_normal(rows)
    blocked = resolvent_module._weighted_gram(coupling, weights)
    reference = (coupling.T @ (weights.real[:, None] * coupling)).astype(np.complex128)
    reference += 1j * (coupling.T @ (weights.imag[:, None] * coupling))
    np.testing.assert_allclose(blocked, reference, rtol=1.0e-13, atol=1.0e-10)
    small = resolvent_module._weighted_gram(coupling[:100], weights[:100])
    exact = (coupling[:100].T @ (weights.real[:100, None] * coupling[:100])).astype(
        np.complex128
    )
    exact += 1j * (coupling[:100].T @ (weights.imag[:100, None] * coupling[:100]))
    np.testing.assert_array_equal(small, exact)


# Coulomb factorization and the choice of interaction factors.


def _squared_rpa_matrix(reference, eri):
    """Return ``H = D**2 + 2 sqrt(D) (2 g) sqrt(D)`` built from the tensor."""

    occupied = np.asarray(reference.occupied_positions)
    virtual = np.asarray(reference.virtual_positions)
    gaps = (
        reference.mo_energy[virtual][None, :]
        - reference.mo_energy[occupied][:, None]
    ).reshape(-1)
    coulomb = eri[np.ix_(occupied, virtual, occupied, virtual)].reshape(
        gaps.size, gaps.size
    )
    square_root = np.sqrt(gaps)
    return np.diag(np.square(gaps)) + 4.0 * (
        square_root[:, None] * coulomb * square_root[None, :]
    )


@pytest.mark.pyscf
def test_full_coulomb_factorization_reconstructs_integrals_and_is_read_only(
    water_rks_hf,
) -> None:
    eri = RestrictedPySCFAdapter(water_rks_hf).build_full_integrals()
    nmo = eri.shape[0]
    factors = exact_coulomb_factors(eri)
    np.testing.assert_allclose(
        np.einsum("Ppq,Prs->pqrs", factors, factors, optimize=True),
        eri,
        atol=3.0e-13,
        rtol=0.0,
    )
    assert factors.shape[1:] == (nmo, nmo)
    assert 0 < factors.shape[0] <= nmo**2
    np.testing.assert_array_equal(factors, factors.swapaxes(1, 2))
    assert not factors.flags.writeable


@pytest.mark.pyscf
@pytest.mark.parametrize("fixture_name", ["water_rks_hf", "h2_rhf_df"])
def test_resolvent_coupling_reproduces_the_dense_squared_rpa_matrix(
    request,
    fixture_name,
) -> None:
    adapter = RestrictedPySCFAdapter(request.getfixturevalue(fixture_name))
    resolvent = ProjectedRPAResolvent.from_adapter(adapter)
    if resolvent.interaction_backend == "density-fitting":
        factors = resolvent.factors
        eri = np.einsum("Ppq,Prs->pqrs", factors, factors, optimize=True)
        tolerance = 3.0e-13
    else:
        eri = adapter.build_full_integrals()
        tolerance = 2.0e-12
    reconstructed_squared = np.diag(
        np.square(resolvent.particle_hole_gaps)
    ) + 2.0 * resolvent.v_matrix @ resolvent.v_matrix.T

    np.testing.assert_allclose(
        reconstructed_squared,
        _squared_rpa_matrix(adapter.reference, eri),
        atol=tolerance,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.sqrt(np.linalg.eigvalsh(reconstructed_squared)),
        solve_response(adapter.reference, eri)[1],
        atol=tolerance,
        rtol=0.0,
    )


@pytest.mark.pyscf
def test_factorized_adapter_auto_selects_df_without_full_eris(
    h2_rhf_df,
    monkeypatch,
) -> None:
    adapter = RestrictedPySCFAdapter(h2_rhf_df)

    def reject_full_integrals(self):
        raise AssertionError("density-fitted construction requested full ERIs")

    monkeypatch.setattr(
        RestrictedPySCFAdapter,
        "build_full_integrals",
        reject_full_integrals,
    )
    resolvent = ProjectedRPAResolvent.from_adapter(adapter)

    assert resolvent.interaction_backend == "density-fitting"
    assert resolvent.factors.shape == (
        h2_rhf_df.with_df.get_naoaux(),
        adapter.reference.nmo,
        adapter.reference.nmo,
    )


@pytest.mark.pyscf
def test_coulomb_factorization_zero_and_numerical_null_limits() -> None:
    assert exact_coulomb_factors(np.zeros((2, 2, 2, 2))).shape == (0, 2, 2)
    assert exact_coulomb_factors(np.array([[[[-1.0e-12]]]])).shape == (0, 1, 1)


@pytest.mark.pyscf
def test_materially_indefinite_coulomb_tensors_are_rejected() -> None:
    indefinite = np.array([[[[-1.0e-3]]]])
    with pytest.raises(ValidationError, match="positive semidefinite"):
        exact_coulomb_factors(indefinite)


def _synthetic_degenerate_problem():
    """Return a deterministic four-orbital degenerate covariance fixture."""

    rng = np.random.default_rng(9173)
    factors = rng.normal(scale=0.08, size=(6, 4, 4))
    factors = 0.5 * (factors + factors.swapaxes(1, 2))
    integrals = np.einsum("Ppq,Prs->pqrs", factors, factors, optimize=True)
    reference = RestrictedMolecularReference(
        mo_coeff=np.eye(4),
        mo_energy=np.array([-1.0, -1.0, 1.0, 1.0]),
        occupied_positions=np.array([0, 1]),
        virtual_positions=np.array([2, 3]),
        chemical_potential=0.0,
        homo_energy=-1.0,
        lumo_energy=1.0,
        gap=2.0,
    )
    return reference, integrals


def _rotate_degenerate_problem(reference, integrals):
    """Rotate occupied and virtual degenerate subspaces deterministically."""

    occupied_angle = 0.37
    virtual_angle = -0.51
    rotation = np.zeros((4, 4))
    rotation[:2, :2] = [
        [math.cos(occupied_angle), -math.sin(occupied_angle)],
        [math.sin(occupied_angle), math.cos(occupied_angle)],
    ]
    rotation[2:, 2:] = [
        [math.cos(virtual_angle), -math.sin(virtual_angle)],
        [math.sin(virtual_angle), math.cos(virtual_angle)],
    ]
    rotated_integrals = np.einsum(
        "ap,bq,cr,ds,abcd->pqrs",
        rotation,
        rotation,
        rotation,
        rotation,
        integrals,
        optimize=True,
    )
    rotated_reference = RestrictedMolecularReference(
        mo_coeff=reference.mo_coeff @ rotation,
        mo_energy=reference.mo_energy,
        occupied_positions=reference.occupied_positions,
        virtual_positions=reference.virtual_positions,
        chemical_potential=reference.chemical_potential,
        homo_energy=reference.homo_energy,
        lumo_energy=reference.lumo_energy,
        gap=reference.gap,
    )
    return rotated_reference, rotated_integrals, rotation


@pytest.mark.pyscf
def test_moments_are_covariant_under_degenerate_orbital_rotations() -> None:
    reference, integrals = _synthetic_degenerate_problem()
    rotated_reference, rotated_integrals, rotation = _rotate_degenerate_problem(
        reference,
        integrals,
    )
    mapping = CayleyMap(center=0.0, scale=0.73)
    original = ExactG0W0SelfEnergy.from_integrals(reference, integrals).cayley_moments(
        mapping, 8
    )
    rotated = ExactG0W0SelfEnergy.from_integrals(
        rotated_reference, rotated_integrals
    ).cayley_moments(mapping, 8)

    for sector in Sector:
        expected = np.einsum(
            "ap,kab,bq->kpq",
            rotation,
            original[sector],
            rotation,
            optimize=True,
        )
        np.testing.assert_allclose(
            rotated[sector],
            expected,
            atol=2.0e-13,
            rtol=2.0e-12,
        )
