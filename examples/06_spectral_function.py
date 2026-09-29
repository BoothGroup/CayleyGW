"""The spectral function: orbital-resolved curves, weights and satellites.

Every eigenvector of the upfolded Hamiltonian has a weight on each reference
orbital.  A quasiparticle is the pole with most of an orbital's weight, and
the rest of that weight sits in satellites.  The script runs the kernel, lists
the quasiparticle and satellites of three orbitals from ``gw.spectrum``, and
passes the spectrum to ``calculate_spectrum``, which broadens the poles into a
total and orbital-resolved spectral function, and to ``plot_spectrum``, which
draws it.
"""

from __future__ import annotations

import numpy as np
from pyscf import dft, gto

from cayleygw import CayleyGW, calculate_spectrum, enable_logging, plot_spectrum

# --- controls ---------------------------------------------------------------
ATOM = "C 0 0 0; O 0 0 1.128"  # geometry, Angstrom
BASIS = "cc-pvdz"  # orbital basis
XC = "pbe"  # functional of the reference
N_CONSERVED = 5  # conserved moment orders; satellites need more than the frontier
OMEGA_P = 0.5  # Cayley scale in Hartree
N_Q = "auto"  # contour nodes
ORBITALS = {"HOMO-3": -4, "HOMO": -1, "LUMO": 0}  # offsets from the HOMO (negative) or LUMO
WINDOW_EV = (-45.0, 15.0)  # spectral-function window in eV relative to the chemical potential
BROADENING_EV = 0.3  # Lorentzian half-width in eV
N_POINTS = 2001  # energy grid points
SATELLITE_WEIGHT = 0.01  # smallest orbital weight of a pole listed as a satellite
DEGENERACY = 1.0e-6  # poles closer than this in Hartree are one degenerate pole
PLOT_FILE = "co_spectral_function.png"  # written to the working directory
VERBOSE = 1  # the log on standard error; 0 removes all but warnings, 2 adds progress counts

HARTREE_TO_EV = 27.211386245981


def poles(energies, weights, degeneracy):
    """Merge eigenvalues closer than ``degeneracy`` into one pole with the summed weight."""

    order = np.argsort(energies)
    merged_energies, merged_weights = [], []
    for index in order:
        if merged_energies and energies[index] - merged_energies[-1] < degeneracy:
            merged_weights[-1] += weights[index]
        else:
            merged_energies.append(energies[index])
            merged_weights.append(weights[index])
    return np.asarray(merged_energies), np.asarray(merged_weights)


def main() -> None:
    enable_logging(VERBOSE)
    molecule = gto.M(atom=ATOM, basis=BASIS, unit="Angstrom", verbose=0)
    mean_field = dft.RKS(molecule, xc=XC).density_fit()
    mean_field.kernel()

    gw = CayleyGW(mean_field, omega_p=OMEGA_P, n_q=N_Q, verbose=VERBOSE)
    gw.kernel(N_CONSERVED)  # the frontier quasiparticles are in the log
    spectrum = gw.spectrum  # every eigenvalue, with its weight on each orbital

    reference = gw.moments.adapter.reference
    nocc = reference.nocc
    positions = {
        name: (nocc + offset if offset < 0 else nocc + offset) for name, offset in ORBITALS.items()
    }
    chemical_potential = gw.moments.chemical_potential
    energies = np.asarray(spectrum.energies)
    orbital_weights = np.asarray(spectrum.orbital_weights)  # (n_orbitals, n_states)

    report = []  # printed after the log, as one section
    for name, position in positions.items():
        # Degenerate eigenvectors share an orbital's weight, so a pole is the degenerate group.
        pole_energies, weights = poles(energies, orbital_weights[position], DEGENERACY)
        dominant = int(np.argmax(weights))
        report.append(
            f"{name} (orbital {position}, reference {(reference.mo_energy[position] - chemical_potential) * HARTREE_TO_EV:8.3f} eV):"
            f" quasiparticle at {(pole_energies[dominant] - chemical_potential) * HARTREE_TO_EV:8.3f} eV"
            f" with weight {weights[dominant]:.3f}"
        )
        satellites = [
            index
            for index in np.argsort(-weights)
            if index != dominant and weights[index] >= SATELLITE_WEIGHT
        ]
        for index in satellites:
            report.append(
                f"    satellite at {(pole_energies[index] - chemical_potential) * HARTREE_TO_EV:8.3f} eV"
                f" with weight {weights[index]:.3f}"
            )
        report.append(
            f"    weight of this orbital in poles with weight below {SATELLITE_WEIGHT}: {weights[weights < SATELLITE_WEIGHT].sum():.3f}"
        )

    window = tuple(chemical_potential + edge / HARTREE_TO_EV for edge in WINDOW_EV)
    broadened = calculate_spectrum(
        spectrum,
        window[0],
        window[1],
        n_points=N_POINTS,
        broadening=BROADENING_EV / HARTREE_TO_EV,
        orbital_positions=tuple(positions.values()),
        verbose=VERBOSE,
    )
    grid = np.asarray(broadened.energy_grid)
    total = np.asarray(broadened.total)
    inside = (energies >= window[0]) & (energies <= window[1])
    report.append(
        f"spectral function on {grid.size} points in [{WINDOW_EV[0]}, {WINDOW_EV[1]}] eV:"
        f" integral {np.trapezoid(total, grid):.3f}; the {inside.sum()} poles inside carry"
        f" weight {np.asarray(spectrum.physical_weights)[inside].sum():.3f}"
        " (the Lorentzian tails leave the window)"
    )
    path = plot_spectrum(
        broadened,
        PLOT_FILE,
        energy_unit="ev",
        relative_to_chemical_potential=True,
        show_orbitals=True,
        title=rf"CO/{BASIS} Cayley-moment $G_0W_0$@{XC.upper()} spectral function",
        verbose=VERBOSE,
    )
    report.append(f"plot written to {path}")

    print(
        f"\n{'=' * 78}\nResults: CO/{BASIS} G0W0@{XC.upper()} quasiparticles and satellites;"
        f" energies in eV relative to the chemical potential\n{'=' * 78}"
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
