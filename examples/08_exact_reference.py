"""The exact G0W0 reference of a small molecule and the error of finite orders.

For a small molecule every RPA excitation can be computed, so the self-energy
has one pole per orbital and excitation in each sector, and
``ExactG0W0SelfEnergy.from_mean_field`` keeps them all.  Its exact Cayley
moments measure the contour quadrature of ``gw.moments``, its upfolded
Hamiltonian with every pole measures the moment truncation of one
``CayleyGW`` per conserved order, and its diagonal quasiparticle equation shows
what the off-diagonal self-energy adds.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto, scf

from cayleygw import (
    CayleyGW,
    ExactG0W0SelfEnergy,
    Sector,
    UpfoldedDysonHamiltonian,
    diagonalize_upfolded,
    enable_logging,
    extract_ip_ea,
)

# --- controls ---------------------------------------------------------------
ATOM = "Li 0 0 0; H 0 0 1.595"  # geometry, Angstrom
BASIS = "cc-pvdz"  # orbital basis; no density fitting, so every route shares the exact interaction
ORDERS = (1, 2, 3, 4, 5, 7)  # conserved moment orders to measure against the exact reference
OMEGA_P = 0.5  # Cayley scale in Hartree
N_Q = "auto"  # contour nodes
VERBOSE = 1  # the log on standard error; 0 removes all but warnings, 2 adds progress counts

HARTREE_TO_EV = 27.211386245981


def main() -> None:
    enable_logging(VERBOSE)
    molecule = gto.M(atom=ATOM, basis=BASIS, unit="Angstrom", verbose=0)
    mean_field = scf.RHF(molecule)
    mean_field.kernel()

    # Every RPA excitation, and from them every self-energy pole.
    exact = ExactG0W0SelfEnergy.from_mean_field(mean_field)
    reference = exact.reference
    report = []  # printed after the log, as one section
    report.append(
        f"{reference.nocc * reference.nvir} RPA excitations, {exact.hole.poles.size} hole"
        f" and {exact.particle.poles.size} particle self-energy poles"
    )

    # Every pole coupled to the orbitals; its quasiparticle table is the first in the log.
    exact_hamiltonian = UpfoldedDysonHamiltonian(
        np.diag(reference.mo_energy),
        exact.hole,
        exact.particle,
        static_correction=exact.static_correction,
    )
    exact_spectrum = diagonalize_upfolded(exact_hamiltonian, verbose=VERBOSE)
    exact_charged = extract_ip_ea(
        exact_spectrum,
        n_ip=2,
        n_ea=2,
        chemical_potential=reference.chemical_potential,
        verbose=VERBOSE,
    )
    report.append(f"exact upfolded dimension {exact_hamiltonian.dimension}")

    # The diagonal equation follows one root per orbital from Sigma_pp alone; the full solution has all of Sigma.
    orbital_weights = np.asarray(exact_spectrum.orbital_weights)
    energies = np.asarray(exact_spectrum.energies)
    report.append("")
    report.append("diagonal quasiparticle equation against the full solution:")
    for name, position in (
        ("HOMO", reference.occupied_positions[-1]),
        ("LUMO", reference.virtual_positions[0]),
    ):
        diagonal = exact.diagonal_quasiparticle_energy(position)
        full = energies[int(np.argmax(orbital_weights[position]))]
        report.append(
            f"  {name}: diagonal {diagonal * HARTREE_TO_EV:10.4f} eV,"
            f" full {full * HARTREE_TO_EV:10.4f} eV,"
            f" difference {(full - diagonal) * HARTREE_TO_EV * 1e3:8.3f} meV"
        )

    report.append("")
    report.append("finite conserved order against the exact reference (errors in meV):")
    report.append("  n   N_q   dim   quadrature error      IP      IP(2)       EA      EA(2)")
    for n_conserved in ORDERS:
        gw = CayleyGW(mean_field, omega_p=OMEGA_P, n_q=N_Q, verbose=VERBOSE)
        charged = gw.kernel(n_conserved, n_ip=2, n_ea=2)
        moments = gw.moments
        # The contour moments against the exact ones of the same Cayley map: the quadrature error alone.
        exact_moments = exact.cayley_moments(moments.mapping, moments.n_max)
        quadrature_error = max(
            np.linalg.norm(computed[order] - exact_moments[sector][order])
            / np.linalg.norm(exact_moments[sector][order])
            for sector, computed in (
                (Sector.HOLE, moments.hole.moments),
                (Sector.PARTICLE, moments.particle.moments),
            )
            for order in range(moments.n_max + 1)
        )
        errors = (
            np.concatenate(
                (
                    charged.ionization_potentials - exact_charged.ionization_potentials,
                    charged.electron_affinities - exact_charged.electron_affinities,
                )
            )
            * HARTREE_TO_EV
            * 1e3
        )
        report.append(
            f"  {n_conserved}  {moments.n_q:4d}  {gw.hamiltonian.dimension:4d}"
            f"   {quadrature_error:14.2e}" + "".join(f" {error:9.4f}" for error in errors)
        )

    print(f"\n{'=' * 78}\nResults: LiH/{BASIS} G0W0@RHF against the exact reference\n{'=' * 78}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
