"""The exact self-energy against PySCF, its upfolding, static correction and density fitting."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import scf, tdscf
from pyscf.gw import gw_exact

from cayleygw import (
    ExactG0W0SelfEnergy,
    Screening,
    Sector,
    UpfoldedDysonHamiltonian,
    ValidationError,
    build_cayley_moments,
    diagonalize_upfolded,
)
from cayleygw._helpers.cayley import CayleyMap
from cayleygw._helpers.pyscf import RestrictedPySCFAdapter
from cayleygw._helpers.screening.response import solve_response
from cayleygw.tools.reference import SelfEnergySector

pytestmark = pytest.mark.pyscf


def _pyscf_drpa(mean_field):
    """Solve every restricted direct-RPA mode with tight thresholds."""

    nocc = int(np.count_nonzero(mean_field.mo_occ > 0.0))
    nstates = nocc * (mean_field.mo_occ.size - nocc)
    direct_rpa = tdscf.dRPA(mean_field)
    direct_rpa.nstates = nstates
    direct_rpa.conv_tol = 1.0e-12
    direct_rpa.max_cycle = 100
    direct_rpa.verbose = 0
    direct_rpa.kernel()
    assert np.all(direct_rpa.converged)
    return direct_rpa


def _pyscf_transition_couplings(mean_field, direct_rpa, integrals):
    """Return the transition couplings PySCF's GWExact builds."""

    occupations = np.asarray(mean_field.mo_occ)
    occupied = np.flatnonzero(occupations > 0.0)
    virtual = np.flatnonzero(occupations == 0.0)
    td_xy = 2.0 * np.asarray(direct_rpa.xy)
    td_z = np.sum(td_xy, axis=1).reshape(
        direct_rpa.e.size,
        occupied.size,
        virtual.size,
    )
    nmo = integrals.shape[0]
    transition_integrals = integrals[np.ix_(occupied, virtual, np.arange(nmo), np.arange(nmo))]
    return np.einsum(
        "via,iapq->vpq",
        td_z,
        transition_integrals,
        optimize=True,
    )


@pytest.fixture(scope="module")
def water_exact(water_rks_hf):
    """The exact full-pole self-energy of water, shared within the module."""

    return ExactG0W0SelfEnergy.from_mean_field(water_rks_hf)


def test_sector_cayley_moments_and_read_only_contract() -> None:
    couplings = np.array(
        [
            [1.0 + 0.5j, 0.25 - 0.1j],
            [0.2 - 0.3j, -0.4 + 0.7j],
        ]
    )
    sector = SelfEnergySector(
        sector=Sector.HOLE,
        poles=np.array([-2.0, -0.5]),
        couplings=couplings,
        chemical_potential=0.0,
    )

    explicit_residues = np.stack(
        [np.outer(couplings[:, pole], couplings[:, pole].conj()) for pole in range(2)]
    )
    mapping = CayleyMap(center=0.0, scale=1.25)
    cayley = sector.cayley_moments(mapping, 3)
    points = mapping.forward(sector.poles)
    expected_cayley = np.stack(
        [np.einsum("l,lpq->pq", points**order, explicit_residues) for order in range(4)]
    )
    np.testing.assert_allclose(cayley, expected_cayley, atol=2.0e-15)
    np.testing.assert_allclose(cayley[0], np.sum(explicit_residues, axis=0), atol=2.0e-15)
    assert np.max(np.abs(cayley[1] - cayley[1].conj().T)) > 0.1
    for array in (sector.poles, sector.couplings, cayley):
        assert not array.flags.writeable


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"sector": "<"}, "Sector"),
        ({"poles": [0.2, 0.5]}, "below"),
        ({"couplings": [[1.0], [2.0]]}, "one column"),
        ({"chemical_potential": np.inf}, "finite real"),
    ],
)
def test_sector_invalid_inputs_are_rejected(changes, message) -> None:
    arguments = {
        "sector": Sector.HOLE,
        "poles": [-2.0, -0.5],
        "couplings": [[1.0, 0.5], [0.2, 0.1]],
        "chemical_potential": 0.0,
    }
    arguments.update(changes)
    with pytest.raises(ValidationError, match=message):
        SelfEnergySector(**arguments)


def test_sector_operation_domains_are_checked() -> None:
    sector = SelfEnergySector(
        sector=Sector.PARTICLE,
        poles=[1.0],
        couplings=[[1.0], [0.5]],
        chemical_potential=0.0,
    )
    with pytest.raises(ValidationError, match="nonnegative integer"):
        sector.cayley_moments(CayleyMap(center=0.0, scale=1.0), -1)
    with pytest.raises(ValidationError, match="wrong sector arc"):
        sector.cayley_moments(CayleyMap(center=2.0, scale=1.0), 1)
    with pytest.raises(ValidationError, match="real frequencies"):
        sector.evaluate_time_ordered(1.0 + 0.2j)
    with pytest.raises(ValidationError, match="positive"):
        sector.evaluate_time_ordered(0.0, eta=0.0)


def test_molecular_poles_follow_the_orbital_energies_and_rpa_modes(
    water_rks_hf, water_exact
) -> None:
    reference = water_exact.reference
    integrals = RestrictedPySCFAdapter(water_rks_hf).build_full_integrals()
    _, energies, _ = solve_response(reference, integrals, Screening.RPA)
    occupied = np.asarray(reference.occupied_positions)
    virtual = np.asarray(reference.virtual_positions)

    expected_hole = (reference.mo_energy[occupied, None] - energies[None, :]).reshape(-1)
    expected_particle = (reference.mo_energy[virtual, None] + energies[None, :]).reshape(-1)
    np.testing.assert_allclose(water_exact.hole.poles, expected_hole, atol=0.0)
    np.testing.assert_allclose(
        water_exact.particle.poles,
        expected_particle,
        atol=0.0,
    )
    assert np.all(water_exact.hole.poles < reference.chemical_potential)
    assert np.all(water_exact.particle.poles > reference.chemical_potential)
    assert water_exact.hole.npoles == reference.nocc * energies.size
    assert water_exact.particle.npoles == reference.nvir * energies.size


@pytest.mark.parametrize("fixture_name", ["h2_rks_hf", "h2_pbe"])
def test_time_ordered_matrix_and_quasiparticle_roots_match_pyscf(
    request,
    fixture_name,
) -> None:
    mean_field = request.getfixturevalue(fixture_name)
    adapter = RestrictedPySCFAdapter(mean_field)
    integrals = adapter.build_full_integrals()
    exact = ExactG0W0SelfEnergy.from_mean_field(mean_field)
    direct_rpa = _pyscf_drpa(mean_field)
    pyscf_couplings = _pyscf_transition_couplings(
        mean_field,
        direct_rpa,
        integrals,
    )
    energies = np.asarray(mean_field.mo_energy)
    occupied = np.flatnonzero(mean_field.mo_occ > 0.0)
    virtual = np.flatnonzero(mean_field.mo_occ == 0.0)
    eta = 1.0e-7

    for frequency in (-0.4, 0.2, 1.1):
        denominator_hole = (
            frequency - (energies[occupied][None, :] - direct_rpa.e[:, None]) - 1j * eta
        )
        denominator_particle = (
            frequency - (energies[virtual][None, :] + direct_rpa.e[:, None]) + 1j * eta
        )
        expected = np.einsum(
            "vpi,vi,vqi->pq",
            pyscf_couplings[:, :, occupied],
            1.0 / denominator_hole,
            pyscf_couplings[:, :, occupied],
            optimize=True,
        )
        expected += np.einsum(
            "vpa,va,vqa->pq",
            pyscf_couplings[:, :, virtual],
            1.0 / denominator_particle,
            pyscf_couplings[:, :, virtual],
            optimize=True,
        )
        np.testing.assert_allclose(
            exact.time_ordered_self_energy(frequency, eta=eta),
            expected,
            atol=2.0e-11,
            rtol=0.0,
        )

    pyscf_gw = gw_exact.GWExact(mean_field, tdmf=direct_rpa)
    pyscf_gw.verbose = 0
    pyscf_energies = pyscf_gw.kernel()
    reference_nmo = exact.reference.nmo
    project_energies = np.array(
        [exact.diagonal_quasiparticle_energy(p) for p in range(reference_nmo)]
    )
    assert reference_nmo == pyscf_energies.size
    np.testing.assert_allclose(project_energies, pyscf_energies, atol=2.0e-7, rtol=0.0)


def test_full_matrix_time_ordered_self_energy_matches_pyscf_water(
    water_rks_hf,
    water_exact,
) -> None:
    adapter = RestrictedPySCFAdapter(water_rks_hf)
    integrals = adapter.build_full_integrals()
    direct_rpa = _pyscf_drpa(water_rks_hf)
    pyscf_couplings = _pyscf_transition_couplings(
        water_rks_hf,
        direct_rpa,
        integrals,
    )
    energies = np.asarray(water_rks_hf.mo_energy)
    occupied = np.flatnonzero(water_rks_hf.mo_occ > 0.0)
    virtual = np.flatnonzero(water_rks_hf.mo_occ == 0.0)
    eta = 3.0e-6
    frequency = 0.37
    denominator_hole = frequency - (energies[occupied][None, :] - direct_rpa.e[:, None]) - 1j * eta
    denominator_particle = (
        frequency - (energies[virtual][None, :] + direct_rpa.e[:, None]) + 1j * eta
    )
    expected = np.einsum(
        "vpi,vi,vqi->pq",
        pyscf_couplings[:, :, occupied],
        1.0 / denominator_hole,
        pyscf_couplings[:, :, occupied],
        optimize=True,
    )
    expected += np.einsum(
        "vpa,va,vqa->pq",
        pyscf_couplings[:, :, virtual],
        1.0 / denominator_particle,
        pyscf_couplings[:, :, virtual],
        optimize=True,
    )
    np.testing.assert_allclose(
        water_exact.time_ordered_self_energy(frequency, eta=eta),
        expected,
        atol=2.0e-10,
        rtol=0.0,
    )

    # RPA eigenvectors may differ in sign, so compare their outer products: the residues.
    reference = water_exact.reference
    nstates = direct_rpa.e.size
    screened = np.zeros((reference.nmo, reference.nmo, nstates), dtype=np.complex128)
    screened[:, reference.occupied_positions, :] = water_exact.hole.couplings.reshape(
        reference.nmo, reference.nocc, nstates
    )
    screened[:, reference.virtual_positions, :] = water_exact.particle.couplings.reshape(
        reference.nmo, reference.nvir, nstates
    )
    project_couplings = np.moveaxis(screened, -1, 0)
    for mode in range(direct_rpa.e.size):
        project_vector = project_couplings[mode].reshape(-1)
        pyscf_vector = pyscf_couplings[mode].reshape(-1)
        np.testing.assert_allclose(
            np.outer(project_vector, project_vector.conj()),
            np.outer(pyscf_vector, pyscf_vector.conj()),
            atol=4.0e-10,
            rtol=0.0,
        )


def test_cayley_moments_adjoint_relation_and_nonhermitian_information(
    water_exact,
) -> None:
    mapping = CayleyMap(
        center=water_exact.reference.chemical_potential,
        scale=1.3,
    )
    moments = water_exact.cayley_moments(mapping, 4)
    for sector, measure in (
        (Sector.HOLE, water_exact.hole),
        (Sector.PARTICLE, water_exact.particle),
    ):
        current = moments[sector]
        np.testing.assert_allclose(
            current[0],
            measure.couplings @ measure.couplings.conj().T,
            atol=3.0e-13,
        )
        assert np.max(np.abs(current[1] - current[1].conj().T)) > 1.0e-6


def _exact_upfolded(exact):
    """Return the untruncated upfolded Hamiltonian of the exact poles."""

    return UpfoldedDysonHamiltonian(
        np.diag(exact.reference.mo_energy),
        exact.hole,
        exact.particle,
        static_correction=exact.static_correction,
    )


def test_exact_upfolding_resolvent_spectrum_and_sum_rules(water_exact) -> None:
    from conftest import _upfolded_greens_function, dyson_greens_function

    upfolded = _exact_upfolded(water_exact)
    np.testing.assert_allclose(upfolded.matrix, upfolded.matrix.conj().T, atol=0.0)
    frequencies = np.asarray([-1.3 + 0.4j, 0.2 + 0.7j, 2.0 + 0.5j])
    np.testing.assert_allclose(
        _upfolded_greens_function(upfolded, frequencies),
        dyson_greens_function(upfolded, frequencies),
        atol=2.0e-13,
        rtol=2.0e-13,
    )

    spectrum = diagonalize_upfolded(upfolded, verbose=0)
    np.testing.assert_allclose(
        np.sum(spectrum.orbital_weights, axis=1),
        np.ones(exact_nmo := water_exact.reference.nmo),
        atol=2.0e-14,
    )
    assert spectrum.physical_weights.sum() == pytest.approx(exact_nmo, abs=2.0e-14)
    assert np.all(spectrum.physical_weights >= 0.0)
    grid = np.linspace(-2.0, 2.0, 31)
    total_spectrum = spectrum.spectral_function(grid, broadening=0.05)
    orbital_spectrum = spectrum.spectral_function(
        grid,
        broadening=0.05,
        orbital_position=0,
    )
    assert total_spectrum.shape == grid.shape
    assert orbital_spectrum.shape == grid.shape
    assert np.all(total_spectrum >= 0.0)
    assert np.all(orbital_spectrum >= 0.0)


def test_zero_interaction_removes_dynamic_weights_and_recovers_reference(
    h2_rks_hf,
) -> None:
    adapter = RestrictedPySCFAdapter(h2_rks_hf)
    nmo = adapter.reference.nmo
    exact = ExactG0W0SelfEnergy.from_integrals(
        adapter.reference,
        np.zeros((nmo, nmo, nmo, nmo)),
        static_correction=adapter.build_static_self_energy_correction(),
    )
    np.testing.assert_array_equal(exact.hole.couplings, 0.0)
    np.testing.assert_array_equal(exact.particle.couplings, 0.0)
    np.testing.assert_allclose(exact.static_correction, 0.0, atol=2.0e-14)
    upfolded = _exact_upfolded(exact)
    np.testing.assert_allclose(
        upfolded.matrix[: exact.reference.nmo, : exact.reference.nmo],
        np.diag(exact.reference.mo_energy),
        atol=2.0e-14,
    )
    np.testing.assert_array_equal(
        upfolded.matrix[: exact.reference.nmo, exact.reference.nmo :],
        0.0,
    )


def test_static_correction_hf_limit_and_pbe_starting_potential(h2_rhf, h2_pbe) -> None:
    rhf_adapter = RestrictedPySCFAdapter(h2_rhf)
    np.testing.assert_allclose(
        rhf_adapter.build_static_self_energy_correction(),
        0.0,
        atol=2.0e-14,
    )

    pbe_adapter = RestrictedPySCFAdapter(h2_pbe)
    correction = pbe_adapter.build_static_self_energy_correction()
    density = h2_pbe.make_rdm1(h2_pbe.mo_coeff, h2_pbe.mo_occ)
    v_mf = h2_pbe.get_veff(h2_pbe.mol, density) - h2_pbe.get_j(
        h2_pbe.mol,
        density,
    )
    v_x_hf = -0.5 * h2_pbe.get_k(h2_pbe.mol, density)
    coefficients = pbe_adapter.reference.mo_coeff
    expected = coefficients.T @ (v_x_hf - v_mf) @ coefficients
    np.testing.assert_allclose(correction, expected, atol=2.0e-14)
    assert np.linalg.norm(correction) > 1.0e-3


def test_pbe_static_correction_retains_offdiagonal_and_frozen_projection(
    water_pbe,
) -> None:
    full = RestrictedPySCFAdapter(water_pbe)
    frozen_core = RestrictedPySCFAdapter(water_pbe, frozen=1)
    full_correction = full.build_static_self_energy_correction()
    frozen_correction = frozen_core.build_static_self_energy_correction()
    offdiagonal = full_correction - np.diag(np.diag(full_correction))
    active = np.asarray([1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(frozen_core.reference.mo_energy, water_pbe.mo_energy[active])

    assert np.linalg.norm(offdiagonal) > 1.0e-3
    np.testing.assert_allclose(
        frozen_correction,
        full_correction[np.ix_(active, active)],
        atol=2.0e-14,
        rtol=0.0,
    )
    assert not full_correction.flags.writeable
    assert not frozen_correction.flags.writeable


def test_hartree_fock_static_correction_is_zero_without_touching_the_integrals(
    h2_rhf, monkeypatch
) -> None:
    """An HF reference's correction is identically zero; nothing may rebuild _cderi for it."""

    def forbidden(*args, **kwargs):
        raise AssertionError("get_veff/get_j/get_k must not run for a Hartree-Fock reference")

    for name in ("get_veff", "get_j", "get_k"):
        monkeypatch.setattr(h2_rhf, name, forbidden)
    correction = RestrictedPySCFAdapter(h2_rhf).build_static_self_energy_correction()
    assert correction.shape == (h2_rhf.mo_coeff.shape[1],) * 2
    assert not np.any(correction)


@pytest.fixture(scope="module")
def water_rhf_df(water_rhf):
    """Water/RHF in STO-3G with Weigend density fitting."""

    mean_field = scf.RHF(water_rhf.mol).density_fit(auxbasis="weigend")
    mean_field.conv_tol = 1.0e-12
    mean_field.kernel()
    assert mean_field.converged
    return mean_field


def test_the_density_fitted_reference_is_built_on_the_fitted_integrals(water_rhf_df) -> None:
    """``use_density_fitting`` solves on ``sum_P L[P,pq] L[P,rs]`` and nothing else."""

    adapter = RestrictedPySCFAdapter(water_rhf_df)
    factors = adapter.build_density_fitted_integrals()
    rebuilt = np.einsum("Ppq,Prs->pqrs", factors, factors, optimize=True)
    fitted = ExactG0W0SelfEnergy.from_mean_field(water_rhf_df, use_density_fitting=True)
    on_rebuilt = ExactG0W0SelfEnergy.from_integrals(adapter.reference, rebuilt)
    for sector in (Sector.HOLE, Sector.PARTICLE):
        np.testing.assert_array_equal(
            getattr(fitted, sector.name.lower()).poles,
            getattr(on_rebuilt, sector.name.lower()).poles,
        )
        np.testing.assert_array_equal(
            getattr(fitted, sector.name.lower()).couplings,
            getattr(on_rebuilt, sector.name.lower()).couplings,
        )


def test_density_fitted_moments_converge_to_the_density_fitted_reference_only(
    water_rhf_df,
) -> None:
    """The quadrature error falls to roundoff on the matched reference; the fitting error stays."""

    fitted = ExactG0W0SelfEnergy.from_mean_field(water_rhf_df, use_density_fitting=True)
    exact = ExactG0W0SelfEnergy.from_mean_field(water_rhf_df)
    errors = {}
    for n_q in (64, 256):
        moments = build_cayley_moments(water_rhf_df, n_conserved=3, n_q=n_q, omega_p=1.0, verbose=0)
        built = {Sector.HOLE: moments.hole.moments, Sector.PARTICLE: moments.particle.moments}
        for label, reference in (("fitted", fitted), ("exact", exact)):
            expected = reference.cayley_moments(moments.mapping, moments.n_max)
            errors[label, n_q] = max(
                float(np.max(np.abs(built[sector] - expected[sector]))) for sector in built
            )

    # Measured 3.5e-6 at 64 nodes and 2.2e-15 at 256.
    assert errors["fitted", 256] < 1.0e-13
    assert errors["fitted", 256] < 1.0e-6 * errors["fitted", 64]
    # Measured 2.3e-4 at both: the fitting error, which no N_q removes.
    assert errors["exact", 256] > 1.0e-5
    assert errors["exact", 256] == pytest.approx(errors["exact", 64], rel=0.05)


def test_the_density_fitted_reference_needs_a_density_fitted_mean_field(water_rhf) -> None:
    with pytest.raises(ValidationError, match="with_df"):
        ExactG0W0SelfEnergy.from_mean_field(water_rhf, use_density_fitting=True)
    with pytest.raises(ValidationError, match="use_density_fitting"):
        ExactG0W0SelfEnergy.from_mean_field(water_rhf, use_density_fitting=1)
