"""Frontier extraction, the spectral function and its plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cayleygw import (
    Sector,
    UpfoldedDysonHamiltonian,
    ValidationError,
    calculate_spectrum,
    diagonalize_upfolded,
    extract_ip_ea,
    plot_spectrum,
    reconstruction_test,
)

from conftest import sector_source


def _source_free_problem() -> UpfoldedDysonHamiltonian:
    """Return a diagonal four-orbital problem that no moments stand behind."""

    energies = np.asarray([-2.0, -0.5, 0.2, 0.8])
    empty = np.empty((energies.size, 0), dtype=np.complex128)
    return UpfoldedDysonHamiltonian(
        np.diag(energies),
        sector_source(Sector.HOLE, [], empty),
        sector_source(Sector.PARTICLE, [], empty),
    )


def test_ip_ea_signs_ordering_counts_and_weight_filter() -> None:
    spectrum = diagonalize_upfolded(_source_free_problem(), verbose=0)
    result = extract_ip_ea(
        spectrum,
        n_ip=2,
        n_ea=2,
        chemical_potential=0.0,
    )

    np.testing.assert_array_equal(result.ip_eigenvalue_indices, [1, 0])
    np.testing.assert_allclose(result.removal_pole_energies, [-0.5, -2.0])
    np.testing.assert_allclose(result.ionization_potentials, [0.5, 2.0])
    np.testing.assert_array_equal(result.ea_eigenvalue_indices, [2, 3])
    np.testing.assert_allclose(result.addition_pole_energies, [0.2, 0.8])
    np.testing.assert_allclose(result.electron_affinities, [-0.2, -0.8])
    assert result.minimum_weight == pytest.approx(1.0e-1)
    assert result.n_occupied_poles == 2
    assert result.n_unoccupied_poles == 2
    assert result.n_boundary_poles == 0

    empty = extract_ip_ea(
        spectrum,
        n_ip=3,
        n_ea=3,
        chemical_potential=0.0,
        minimum_weight=1.1,
    )
    assert empty.ip_eigenvalue_indices.size == 0
    assert empty.ea_eigenvalue_indices.size == 0
    assert empty.n_occupied_poles == 2
    assert empty.n_unoccupied_poles == 2

    boundary_energies = np.asarray([-1.0, 0.0, 1.0])
    boundary_empty = np.empty((3, 0), dtype=np.complex128)
    boundary_problem = UpfoldedDysonHamiltonian(
        np.diag(boundary_energies),
        sector_source(Sector.HOLE, [], boundary_empty),
        sector_source(Sector.PARTICLE, [], boundary_empty),
    )
    boundary = extract_ip_ea(
        diagonalize_upfolded(boundary_problem, verbose=0),
        n_ip=3,
        n_ea=3,
        chemical_potential=0.0,
        minimum_weight=0.0,
    )
    assert boundary.n_occupied_poles == 1
    assert boundary.n_unoccupied_poles == 1
    assert boundary.n_boundary_poles == 1


def test_spectrum_grid_orbital_sum_and_validation() -> None:
    spectrum = diagonalize_upfolded(_source_free_problem(), verbose=0)
    calculated = calculate_spectrum(
        spectrum,
        -4.0,
        4.0,
        n_points=401,
        broadening=0.04,
        orbital_positions=(0, 1, 2, 3),
        chemical_potential=0.0,
    )

    assert calculated.energy_grid.shape == (401,)
    assert calculated.orbital_resolved.shape == (4, 401)
    np.testing.assert_allclose(
        np.sum(calculated.orbital_resolved, axis=0),
        calculated.total,
        atol=0.0,
        rtol=0.0,
    )
    assert np.all(calculated.total >= 0.0)
    assert not calculated.total.flags.writeable

    total_only = calculate_spectrum(
        spectrum,
        -1.0,
        1.0,
        orbital_positions=(),
        chemical_potential=0.0,
    )
    assert total_only.orbital_resolved.shape == (0, 2001)

    with pytest.raises(ValidationError, match="energy_max"):
        calculate_spectrum(
            spectrum,
            1.0,
            1.0,
            chemical_potential=0.0,
        )
    with pytest.raises(ValidationError, match="n_points"):
        calculate_spectrum(
            spectrum,
            -1.0,
            1.0,
            n_points=1,
            chemical_potential=0.0,
        )
    with pytest.raises(ValidationError, match="broadening"):
        calculate_spectrum(
            spectrum,
            -1.0,
            1.0,
            broadening=0.0,
            chemical_potential=0.0,
        )
    with pytest.raises(ValidationError, match="unique"):
        calculate_spectrum(
            spectrum,
            -1.0,
            1.0,
            orbital_positions=(0, 0),
            chemical_potential=0.0,
        )
    with pytest.raises(ValidationError, match="invalid row"):
        calculate_spectrum(
            spectrum,
            -1.0,
            1.0,
            orbital_positions=(4,),
            chemical_potential=0.0,
        )


def test_plot_spectrum_writes_headless_png_with_options(tmp_path: Path, monkeypatch) -> None:
    from matplotlib.colors import to_hex
    from matplotlib.figure import Figure

    colours = []
    original_savefig = Figure.savefig

    def recording_savefig(self, *args, **kwargs):
        colours.extend(to_hex(line.get_color()) for line in self.axes[0].get_lines())
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", recording_savefig)
    spectrum = diagonalize_upfolded(_source_free_problem(), verbose=0)
    calculated = calculate_spectrum(
        spectrum,
        -3.0,
        2.0,
        n_points=201,
        broadening=0.08,
        orbital_positions=(0, 1),
        chemical_potential=0.0,
    )
    destination = plot_spectrum(
        calculated,
        tmp_path / "nested" / "spectrum.png",
        energy_unit="eV",
        relative_to_chemical_potential=True,
        show_orbitals=True,
        title="Synthetic spectrum",
    )

    assert destination == (tmp_path / "nested" / "spectrum.png").resolve()
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert destination.stat().st_size > 10_000
    # No orbital line shares the colour of the total.
    total, *orbitals = colours[:3]
    assert len(orbitals) == 2 and total not in orbitals and orbitals[0] != orbitals[1]

    with pytest.raises(ValidationError, match="energy_unit"):
        plot_spectrum(calculated, tmp_path / "bad.png", energy_unit="joule")
    with pytest.raises(ValidationError, match="show_orbitals"):
        plot_spectrum(calculated, tmp_path / "bad.png", show_orbitals=1)


def test_source_free_helpers_require_explicit_mu_and_reject_reconstruction() -> None:
    problem = _source_free_problem()
    spectrum = diagonalize_upfolded(problem, verbose=0)

    with pytest.raises(ValidationError, match="chemical_potential"):
        extract_ip_ea(spectrum, n_ip=1, n_ea=1)
    with pytest.raises(ValidationError, match="chemical_potential"):
        calculate_spectrum(spectrum, -1.0, 1.0)
    with pytest.raises(ValidationError, match="SectorSelfEnergyRealization"):
        reconstruction_test(problem)
    with pytest.raises(ValidationError, match="Upfolded"):
        diagonalize_upfolded(object())


def test_the_weight_floor_rejects_a_satellite_standing_in_for_a_quasiparticle() -> None:
    """A floor of ``0.1`` keeps a roundoff satellite from passing for the quasiparticle."""

    spectrum = diagonalize_upfolded(_source_free_problem(), verbose=0)
    permissive = extract_ip_ea(
        spectrum, n_ip=2, n_ea=2, chemical_potential=0.0, minimum_weight=1.0e-6
    )
    strict = extract_ip_ea(
        spectrum, n_ip=2, n_ea=2, chemical_potential=0.0, minimum_weight=1.0e-1
    )
    # All poles here carry full weight, so the floor changes nothing.
    np.testing.assert_allclose(
        permissive.removal_pole_energies, strict.removal_pole_energies
    )
    assert permissive.minimum_weight == pytest.approx(1.0e-6)
    assert strict.minimum_weight == pytest.approx(1.0e-1)

    # A floor above every weight returns nothing, not the least-bad pole.
    empty = extract_ip_ea(
        spectrum, n_ip=2, n_ea=2, chemical_potential=0.0, minimum_weight=2.0
    )
    assert len(empty.removal_pole_energies) == 0
    assert len(empty.addition_pole_energies) == 0
