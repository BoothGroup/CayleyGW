"""Records of :mod:`cayleygw.tools.spectrum`: frontier energies and broadened spectra."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..._helpers.validate import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class IPEAResult:
    """Frontier ionization potentials and electron affinities, with their poles.

    The poles are split at ``chemical_potential`` and ordered outwards from it.
    Each ionization potential or electron affinity is minus its pole energy, in
    Hartree.

    Attributes:
        chemical_potential: Energy in Hartree between removal and addition poles.
        minimum_weight: Physical weight a pole needs to be listed.
        n_occupied_poles: Poles below the chemical potential, before the weight filter.
        n_unoccupied_poles: Poles above it, before the weight filter.
        n_boundary_poles: Poles exactly at it.
        ip_eigenvalue_indices: Indices of the removal poles in the ascending spectrum.
        ionization_potentials: ``-removal_pole_energies``.
        removal_pole_energies: Energies of the listed removal poles.
        ip_physical_weights: Physical weights of the listed removal poles.
        ea_eigenvalue_indices: Indices of the addition poles in the ascending spectrum.
        electron_affinities: ``-addition_pole_energies``.
        addition_pole_energies: Energies of the listed addition poles.
        ea_physical_weights: Physical weights of the listed addition poles.
    """

    chemical_potential: float
    minimum_weight: float
    n_occupied_poles: int
    n_unoccupied_poles: int
    n_boundary_poles: int
    ip_eigenvalue_indices: IntArray = field(repr=False)
    ionization_potentials: FloatArray
    removal_pole_energies: FloatArray = field(repr=False)
    ip_physical_weights: FloatArray = field(repr=False)
    ea_eigenvalue_indices: IntArray = field(repr=False)
    electron_affinities: FloatArray
    addition_pole_energies: FloatArray = field(repr=False)
    ea_physical_weights: FloatArray = field(repr=False)


@dataclass(frozen=True, slots=True)
class SpectrumResult:
    """Lorentzian-broadened physical spectrum on an energy grid.

    Attributes:
        energy_grid: Ascending energies in Hartree.
        total: Spectral function traced over the physical orbitals.
        orbital_positions: Physical orbitals in ``orbital_resolved``.
        orbital_resolved: Diagonal spectrum of each of those orbitals, one row each.
        broadening: Lorentzian half-width in Hartree.
        chemical_potential: Energy in Hartree between removal and addition poles.
    """

    energy_grid: FloatArray
    total: FloatArray
    orbital_positions: IntArray = field(repr=False)
    orbital_resolved: FloatArray = field(repr=False)
    broadening: float
    chemical_potential: float
