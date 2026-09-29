"""The CayleyGW kernel against the four calls it makes, and the end-to-end workflow."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import _exact_upfolded, dyson_greens_function

from cayleygw import (
    CayleyGW,
    ExactG0W0SelfEnergy,
    Sector,
    ValidationError,
    build_cayley_moments,
    build_upfolded_hamiltonian,
    diagonalize_upfolded,
    extract_ip_ea,
)
from cayleygw.moments import G0W0CayleyMoments
from cayleygw.screening import ProjectedRPAResolvent

pytestmark = pytest.mark.pyscf


def test_the_kernel_is_the_four_calls(water_rhf) -> None:
    gw = CayleyGW(water_rhf, n_q=256, verbose=0)
    result = gw.kernel(3)

    moments = build_cayley_moments(water_rhf, n_conserved=3, n_q=256, verbose=0)
    hamiltonian = build_upfolded_hamiltonian(moments, verbose=0)
    expected = extract_ip_ea(
        diagonalize_upfolded(hamiltonian, verbose=0), n_ip=3, n_ea=3, verbose=0
    )

    assert result is gw.result
    assert gw.hamiltonian.dimension == hamiltonian.dimension
    np.testing.assert_array_equal(result.ionization_potentials, expected.ionization_potentials)
    np.testing.assert_array_equal(result.electron_affinities, expected.electron_affinities)
    np.testing.assert_array_equal(result.ip_physical_weights, expected.ip_physical_weights)
    assert result.ionization_potentials.size == 3


def test_a_lower_order_reuses_the_moments_and_a_higher_one_rebuilds(water_rhf) -> None:
    gw = CayleyGW(water_rhf, n_q=256, verbose=0)
    gw.kernel(3)
    built = gw.moments

    lower = gw.kernel(2)
    assert gw.moments is built
    expected = extract_ip_ea(
        diagonalize_upfolded(
            build_upfolded_hamiltonian(built, n_conserved=2, verbose=0), verbose=0
        ),
        n_ip=3,
        n_ea=3,
        verbose=0,
    )
    np.testing.assert_array_equal(lower.ionization_potentials, expected.ionization_potentials)

    gw.kernel(4)
    assert gw.moments is not built
    assert gw.moments.n_conserved == 4


def test_the_kernel_validates_the_order(water_rhf) -> None:
    with pytest.raises(ValidationError, match="n_conserved"):
        CayleyGW(water_rhf, n_q=256, verbose=0).kernel(-1)


# The end-to-end workflow.


@pytest.mark.pyscf
def test_two_call_molecular_workflow_has_complete_provenance(
    molecular_workflow,
) -> None:
    moments, hamiltonian, spectrum = molecular_workflow

    assert isinstance(moments, G0W0CayleyMoments)
    assert moments.n_max == 3
    assert moments.n_q == 32
    assert moments.automatic_n_q_diagnostics is None
    assert moments.mapping.scale == pytest.approx(1.0)
    assert moments.chemical_potential == pytest.approx(moments.adapter.reference.chemical_potential)
    assert isinstance(moments.resolvent, ProjectedRPAResolvent)
    assert moments.interaction_backend == "exact"
    assert moments.spectral_bound == "spectral"
    assert moments.rpa_spectral_bounds.source == ("D_min and spectral-norm upper bound")
    assert moments.conjugate_paired
    assert moments.hole.moments.shape == (4, 2, 2)
    assert moments.particle.moments.shape == (4, 2, 2)
    assert not moments.hole.moments.flags.writeable

    assert hamiltonian.nphysical == 2
    assert hamiltonian.hole.closure_scan is not None
    assert hamiltonian.particle.closure_scan is not None
    assert hamiltonian.dimension <= hamiltonian.nphysical * (2 * moments.n_max + 3)
    assert spectrum.nstates == hamiltonian.dimension
    np.testing.assert_allclose(
        np.sum(spectrum.orbital_weights, axis=1), 1.0, atol=1.0e-14, rtol=0.0
    )


@pytest.mark.pyscf
@pytest.mark.parametrize("mean_field_name", ["h2_rhf_df", "h2_pbe_df"])
def test_density_fitted_restricted_workflow_matches_dense_df_oracle(
    request,
    mean_field_name,
) -> None:
    """DF moments, static correction and Green's function match the fitted exact poles."""

    mean_field = request.getfixturevalue(mean_field_name)
    moments = build_cayley_moments(
        mean_field,
        n_conserved=3,
        n_q=128,
        omega_p=1.0,
    )
    assert moments.interaction_backend == "density-fitting"

    factors = moments.resolvent.factors
    fitted_integrals = np.einsum("Ppq,Prs->pqrs", factors, factors, optimize=True)
    exact = ExactG0W0SelfEnergy.from_integrals(
        moments.adapter.reference,
        fitted_integrals,
        static_correction=(moments.adapter.build_static_self_energy_correction()),
    )
    if mean_field_name == "h2_rhf_df":
        assert np.linalg.norm(exact.static_correction) < 1.0e-9
    else:
        assert np.linalg.norm(exact.static_correction) > 1.0e-3
    exact_moments = exact.cayley_moments(
        moments.mapping,
        moments.n_max,
    )
    np.testing.assert_allclose(
        moments.hole.moments,
        exact_moments[Sector.HOLE],
        atol=2.0e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        moments.particle.moments,
        exact_moments[Sector.PARTICLE],
        atol=2.0e-8,
        rtol=0.0,
    )

    hamiltonian = build_upfolded_hamiltonian(
        moments,
        terminal_selection="scan",
        terminal_phase_count=1,
    )
    np.testing.assert_allclose(
        hamiltonian.static_correction,
        exact.static_correction,
        atol=0.0,
        rtol=0.0,
    )
    frequencies = np.asarray([-1.0 + 0.4j, 0.1 + 0.7j, 1.5 + 0.3j])
    expected = dyson_greens_function(_exact_upfolded(exact), frequencies)
    np.testing.assert_allclose(
        dyson_greens_function(hamiltonian, frequencies),
        expected,
        atol=3.0e-8,
        rtol=0.0,
    )
