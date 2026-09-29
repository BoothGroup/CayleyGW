"""Tests of the restricted PySCF adapter: the reference, the integrals and input checks."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from pyscf import scf

from cayleygw import ValidationError
from cayleygw._helpers.pyscf import RestrictedPySCFAdapter

pytestmark = pytest.mark.pyscf


def _reconstruct(factors):
    """Return ``(pq|rs) = sum_P L[P,p,q] L[P,r,s]``."""

    return np.einsum("Ppq,Prs->pqrs", factors, factors, optimize=True)


def test_h2_reference_metadata_and_default_chemical_potential(h2_rhf) -> None:
    adapter = RestrictedPySCFAdapter(h2_rhf)
    reference = adapter.reference

    assert reference.nmo == 2
    assert reference.nocc == 1
    assert reference.nvir == 1
    assert reference.gap > 0.0
    assert reference.chemical_potential == pytest.approx(
        0.5 * (reference.homo_energy + reference.lumo_energy)
    )
    assert reference.homo_energy < reference.chemical_potential
    assert reference.chemical_potential < reference.lumo_energy
    np.testing.assert_array_equal(reference.mo_energy, h2_rhf.mo_energy)
    np.testing.assert_array_equal(reference.mo_coeff, h2_rhf.mo_coeff)
    np.testing.assert_array_equal(reference.occupied_positions, [0])
    np.testing.assert_array_equal(reference.virtual_positions, [1])

    for array in (
        reference.mo_coeff,
        reference.mo_energy,
        reference.occupied_positions,
        reference.virtual_positions,
    ):
        assert not array.flags.writeable
        assert array.flags.c_contiguous


def test_the_reference_copies_its_input(h2_rhf) -> None:
    adapter = RestrictedPySCFAdapter(h2_rhf)

    original_value = adapter.reference.mo_energy[0]
    h2_rhf.mo_energy[0] += 0.123
    try:
        assert adapter.reference.mo_energy[0] == pytest.approx(original_value)
    finally:
        h2_rhf.mo_energy[0] -= 0.123


def test_rhf_rks_helium_and_water_are_supported(
    h2_rhf,
    h2_rks_hf,
    h2_pbe,
    helium_rhf,
    water_rhf,
) -> None:
    h2_hf = RestrictedPySCFAdapter(h2_rhf).reference
    h2_ks_hf = RestrictedPySCFAdapter(h2_rks_hf).reference
    h2_dft = RestrictedPySCFAdapter(h2_pbe).reference
    helium = RestrictedPySCFAdapter(helium_rhf).reference
    water = RestrictedPySCFAdapter(water_rhf).reference

    assert (h2_hf.nocc, h2_hf.nvir) == (1, 1)
    assert (h2_ks_hf.nocc, h2_ks_hf.nvir) == (1, 1)
    np.testing.assert_allclose(
        h2_ks_hf.mo_energy,
        h2_hf.mo_energy,
        atol=2.0e-12,
        rtol=0.0,
    )
    assert (h2_dft.nocc, h2_dft.nvir) == (1, 1)
    assert (helium.nocc, helium.nvir) == (1, 1)
    assert water.nocc == 5
    assert water.nvir == 2


def test_frozen_core_orbitals(water_rhf) -> None:
    unfrozen = RestrictedPySCFAdapter(water_rhf)
    frozen_core = RestrictedPySCFAdapter(water_rhf, frozen=1)

    assert unfrozen.frozen_indices == ()
    assert frozen_core.frozen_indices == (0,)
    assert frozen_core.reference.nocc == 4
    assert frozen_core.reference.nvir == 2
    np.testing.assert_array_equal(
        frozen_core.reference.mo_energy,
        water_rhf.mo_energy[[1, 2, 3, 4, 5, 6]],
    )
    assert frozen_core.reference.chemical_potential == pytest.approx(
        unfrozen.reference.chemical_potential
    )


def test_full_integrals_match_direct_pyscf_transform_and_symmetry(h2_rhf) -> None:
    adapter = RestrictedPySCFAdapter(h2_rhf)
    full = adapter.build_full_integrals()
    direct = np.asarray(h2_rhf.mol.intor("int2e", aosym="s1")).reshape((h2_rhf.mol.nao_nr(),) * 4)
    coefficients = adapter.reference.mo_coeff
    independently_transformed = np.einsum(
        "up,vq,wr,xs,uvwx->pqrs",
        coefficients,
        coefficients,
        coefficients,
        coefficients,
        direct,
        optimize=True,
    )

    np.testing.assert_allclose(
        full,
        independently_transformed,
        atol=2.0e-13,
        rtol=0.0,
    )
    np.testing.assert_allclose(full, full.swapaxes(0, 1), atol=1e-14)
    np.testing.assert_allclose(full, full.swapaxes(2, 3), atol=1e-14)
    np.testing.assert_allclose(
        full,
        full.transpose(2, 3, 0, 1),
        atol=1e-14,
    )
    assert not full.flags.writeable


def test_three_index_tensor_reconstructs_pyscf_df_exactly(h2_rhf_df) -> None:
    adapter = RestrictedPySCFAdapter(h2_rhf_df)
    factors = adapter.build_density_fitted_integrals()

    direct_df = np.asarray(
        h2_rhf_df.with_df.ao2mo(adapter.reference.mo_coeff, compact=False)
    ).reshape((adapter.reference.nmo,) * 4)

    np.testing.assert_allclose(_reconstruct(factors), direct_df, atol=3.0e-13, rtol=0.0)
    assert factors.shape == (
        h2_rhf_df.with_df.get_naoaux(),
        adapter.reference.nmo,
        adapter.reference.nmo,
    )
    assert np.max(np.abs(factors - factors.swapaxes(1, 2))) < 1.0e-14
    assert not factors.flags.writeable


def test_density_fitting_error_is_quantified_against_full_integrals(h2_rhf_df) -> None:
    adapter = RestrictedPySCFAdapter(h2_rhf_df)
    fitted = adapter.build_density_fitted_integrals()
    full = adapter.build_full_integrals()
    difference = _reconstruct(fitted) - full
    max_abs_error = float(np.max(np.abs(difference)))
    rms_error = float(np.sqrt(np.mean(np.square(difference))))
    relative_frobenius_error = float(np.linalg.norm(difference) / np.linalg.norm(full))

    assert 0.0 < max_abs_error < 3.0e-3
    assert 0.0 < rms_error < max_abs_error
    assert 0.0 < relative_frobenius_error < 2.0e-3


def test_density_fitted_frozen_active_space_matches_pyscf(water_rhf) -> None:
    mean_field = water_rhf.density_fit(auxbasis="weigend")
    adapter = RestrictedPySCFAdapter(mean_field, frozen=1)
    fitted = adapter.build_density_fitted_integrals()
    direct = np.asarray(
        mean_field.with_df.ao2mo(
            adapter.reference.mo_coeff,
            compact=False,
        )
    ).reshape((adapter.reference.nmo,) * 4)

    assert fitted.shape[1] == water_rhf.mo_coeff.shape[1] - 1
    np.testing.assert_allclose(
        _reconstruct(fitted),
        direct,
        atol=4.0e-13,
        rtol=0.0,
    )


def test_unsupported_reference_types_are_rejected(h2_rhf) -> None:
    with pytest.raises(ValidationError, match="RHF and RKS"):
        RestrictedPySCFAdapter(scf.UHF(h2_rhf.mol))
    with pytest.raises(ValidationError, match="RHF and RKS"):
        RestrictedPySCFAdapter(scf.ROHF(h2_rhf.mol))


def test_unconverged_fractional_open_shell_and_gapless_references_are_rejected(
    h2_rhf,
) -> None:
    unconverged = copy.copy(h2_rhf)
    unconverged.converged = False
    with pytest.raises(ValidationError, match="not converged"):
        RestrictedPySCFAdapter(unconverged)

    fractional = copy.copy(h2_rhf)
    fractional.mo_occ = np.array([1.5, 0.5])
    with pytest.raises(ValidationError, match="Fractional|fractional"):
        RestrictedPySCFAdapter(fractional)

    gapless = copy.copy(h2_rhf)
    gapless.mo_energy = h2_rhf.mo_energy.copy()
    gapless.mo_energy[1] = gapless.mo_energy[0]
    with pytest.raises(ValidationError, match="gap is too small"):
        RestrictedPySCFAdapter(gapless)

    inconsistent_spin = copy.copy(h2_rhf)
    inconsistent_spin.mol = copy.copy(h2_rhf.mol)
    inconsistent_spin.mol.spin = 2
    with pytest.raises(ValidationError, match="closed-shell"):
        RestrictedPySCFAdapter(inconsistent_spin)


def test_invalid_frozen(water_rhf) -> None:
    with pytest.raises(ValidationError):
        RestrictedPySCFAdapter(water_rhf, frozen=True)
    with pytest.raises(ValidationError):
        RestrictedPySCFAdapter(water_rhf, frozen=-1)
    with pytest.raises(ValidationError, match="nonnegative integer"):
        RestrictedPySCFAdapter(water_rhf, frozen=[0])
    with pytest.raises(ValidationError, match="leave an occupied orbital"):
        RestrictedPySCFAdapter(water_rhf, frozen=5)


def test_density_fitted_build_and_transform_run_under_the_requested_thread_limit(monkeypatch):
    """``native_threads`` covers both the ``_cderi`` build and the AO-to-MO loop in one region."""

    from contextlib import contextmanager

    from pyscf import gto, scf

    from cayleygw._helpers.pyscf import RestrictedPySCFAdapter
    from cayleygw.tools import parallel as module

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mean_field = scf.RHF(mol).density_fit()
    mean_field.kernel()
    entered: list[int] = []
    built: list[int] = []

    @contextmanager
    def recorder(count):
        entered.append(int(count))
        yield None

    # A test cannot see the thread count, so a recorder stands in for the limiter.
    monkeypatch.setattr(module, "limited_native_threads", recorder)
    adapter = RestrictedPySCFAdapter(mean_field)
    mean_field.with_df._cderi = None
    original_build = mean_field.with_df.build

    def recording_build(*args, **kwargs):
        built.append(len(entered))
        return original_build(*args, **kwargs)

    monkeypatch.setattr(mean_field.with_df, "build", recording_build)
    adapter.build_density_fitted_integrals(native_threads=3)
    assert entered == [3], entered
    assert built == [1], "the _cderi build must run inside the region"
    entered.clear()
    adapter.build_density_fitted_integrals()
    assert entered == [], "the default must leave the ambient pin alone"
