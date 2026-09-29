"""Diagonal quasiparticle energies against PySCF's GW for a matched reference.

PySCF's ``gw_exact`` solves the diagonal quasiparticle equation, in which only
Sigma_pp(omega) enters, with every RPA excitation, and ``gw_cd`` solves it by
contour deformation with density-fitted integrals; ``ExactG0W0SelfEnergy``
solves the same equation.  The kernel's result is the full Dyson solution
with the off-diagonal self-energy, read from ``gw.spectrum`` as the pole with
most of each orbital's weight and measured against the exact self-energy
upfolded with every pole.  All routes start from one mean field and, except
``gw_cd``, one exact four-index interaction.
"""

from __future__ import annotations

import numpy as np
from pyscf import dft, gto, tdscf
from pyscf.gw import gw_cd, gw_exact

from cayleygw import (
    CayleyGW,
    ExactG0W0SelfEnergy,
    UpfoldedDysonHamiltonian,
    diagonalize_upfolded,
    enable_logging,
)


# --- controls ---------------------------------------------------------------
ATOM = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"  # geometry, Angstrom
BASIS = "cc-pvdz"  # orbital basis; no density fitting, which gw_exact does not support
XC = "pbe"  # functional of the shared reference; gw_exact needs a Kohn-Sham object
ORBITAL_OFFSETS = (-1, 0, 1, 2)  # orbitals relative to the HOMO: HOMO-1, HOMO, LUMO, LUMO+1
N_CONSERVED = 5  # conserved moment orders of the Cayley-moment calculation
OMEGA_P = 0.5  # Cayley scale in Hartree
N_Q = "auto"  # contour nodes
VERBOSE = 1  # the log on standard error; 0 removes all but warnings, 2 adds progress counts

HARTREE_TO_EV = 27.211386245981


def main() -> None:
    enable_logging(VERBOSE)
    molecule = gto.M(atom=ATOM, basis=BASIS, unit="Angstrom", verbose=0)
    mean_field = dft.RKS(molecule, xc=XC)
    mean_field.kernel()
    nocc = int(np.count_nonzero(mean_field.mo_occ > 0))
    nvir = mean_field.mo_occ.size - nocc
    positions = [nocc - 1 + offset for offset in ORBITAL_OFFSETS]
    labels = [
        "HOMO" if offset == 0 else ("LUMO" if offset == 1 else (f"HOMO{offset}" if offset < 0 else f"LUMO+{offset - 1}"))
        for offset in ORBITAL_OFFSETS
    ]

    # PySCF: every RPA excitation, then the diagonal equation of gw_exact.
    response = tdscf.dRPA(mean_field)
    response.nstates = nocc * nvir
    response.verbose = 0
    response.kernel()
    pyscf_exact = gw_exact.GWExact(mean_field, tdmf=response)
    pyscf_exact.verbose = 0
    pyscf_exact.kernel(orbs=positions)
    # PySCF: contour deformation; density-fits the interaction on its own.
    pyscf_cd = gw_cd.GWCD(mean_field)
    pyscf_cd.verbose = 0
    pyscf_cd.kernel()

    # cayleygw: the same exact self-energy and its diagonal equation.
    exact = ExactG0W0SelfEnergy.from_mean_field(mean_field)
    diagonal = [exact.diagonal_quasiparticle_energy(position) for position in positions]
    # cayleygw: the exact self-energy upfolded with every pole, the reference for the full solution.
    exact_hamiltonian = UpfoldedDysonHamiltonian(
        np.diag(exact.reference.mo_energy),
        exact.hole,
        exact.particle,
        static_correction=exact.static_correction,
    )
    exact_spectrum = diagonalize_upfolded(exact_hamiltonian, verbose=VERBOSE)
    exact_energies = np.asarray(exact_spectrum.energies)
    exact_weights = np.asarray(exact_spectrum.orbital_weights)

    # cayleygw: the kernel's full upfolded solution from the Cayley moments.
    gw = CayleyGW(mean_field, omega_p=OMEGA_P, n_q=N_Q, verbose=VERBOSE)
    gw.kernel(N_CONSERVED)
    energies = np.asarray(gw.spectrum.energies)
    orbital_weights = np.asarray(gw.spectrum.orbital_weights)

    # The Cayley-moment build, its N_q and its upfolded dimension are in the log.
    print(
        f"\n{'=' * 78}\nResults: H2O/{BASIS} G0W0@{XC.upper()}, {nocc * nvir} RPA excitations;"
        f" quasiparticle energies in eV\n{'=' * 78}"
    )
    print(f"interaction: {gw.moments.interaction_backend} for cayleygw and gw_exact; gw_cd density-fits ({pyscf_cd.with_df.auxbasis})")
    print(f"exact upfolded dimension {exact_hamiltonian.dimension}")
    print()
    print("                       diagonal quasiparticle equation             full Dyson solution")
    print(f"  orbital        {XC.upper():>6}    gw_exact   cayleygw exact W      gw_cd   exact W (weight)   Cayley n={N_CONSERVED} (weight)")
    largest_diagonal = 0.0
    largest_full = 0.0
    for label, position, energy in zip(labels, positions, diagonal):
        dominant = int(np.argmax(orbital_weights[position]))
        exact_dominant = int(np.argmax(exact_weights[position]))
        largest_diagonal = max(largest_diagonal, abs(energy - pyscf_exact.mo_energy[position]))
        largest_full = max(largest_full, abs(energies[dominant] - exact_energies[exact_dominant]))
        print(
            f"  {label:<8} {mean_field.mo_energy[position] * HARTREE_TO_EV:10.3f}"
            f"  {pyscf_exact.mo_energy[position] * HARTREE_TO_EV:10.4f}"
            f"  {energy * HARTREE_TO_EV:16.4f}"
            f"  {pyscf_cd.mo_energy[position] * HARTREE_TO_EV:10.4f}"
            f"  {exact_energies[exact_dominant] * HARTREE_TO_EV:9.4f} ({exact_weights[position, exact_dominant]:.3f})"
            f"  {energies[dominant] * HARTREE_TO_EV:11.4f} ({orbital_weights[position, dominant]:.3f})"
        )
    print()
    print(f"largest difference between gw_exact and the cayleygw diagonal equation: {largest_diagonal * HARTREE_TO_EV * 1e3:.3f} meV")
    print(f"largest difference between the exact and the Cayley-moment full solution: {largest_full * HARTREE_TO_EV * 1e3:.3f} meV")
    print("the diagonal and the full solutions differ by the off-diagonal self-energy, static correction included")


if __name__ == "__main__":
    main()
