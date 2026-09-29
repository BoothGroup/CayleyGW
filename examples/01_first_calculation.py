"""A first Cayley-moment G0W0 calculation: the kernel with its defaults.

``CayleyGW(mf)`` fixes the options that define the self-energy moments, and
``gw.kernel(n)`` builds them by contour quadrature around the RPA excitation
spectrum, realizes orders 0 to ``n`` as poles coupled to the orbitals in one
Hermitian matrix, and returns the ionization potentials and electron
affinities from its eigenvalues.  The log reports each step; the script then
prints the reference HOMO-LUMO gap and the G0W0 correction to it.  Example 02
runs the four steps one at a time.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto, scf

from cayleygw import CayleyGW, enable_logging


# --- controls ---------------------------------------------------------------
ATOM = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"  # geometry, Angstrom
BASIS = "cc-pvdz"  # orbital basis; density fitting picks the matching auxiliary basis
N_CONSERVED = 3  # moment orders C_0..C_3 are conserved exactly by the realization
OMEGA_P = 0.5  # Cayley scale in Hartree: the energy resolution around the chemical potential
N_Q = "auto"  # contour nodes: doubled from 128 until every moment order is converged
N_IP = 3  # removal energies in the log, counted outwards from the chemical potential
N_EA = 3  # addition energies in the log
VERBOSE = 1  # the log on standard error; 0 removes all but warnings, 2 adds progress counts

HARTREE_TO_EV = 27.211386245981


def main() -> None:
    enable_logging(VERBOSE)  # example 11 shows the other styles and the log file

    molecule = gto.M(atom=ATOM, basis=BASIS, unit="Angstrom", verbose=0)
    mean_field = scf.RHF(molecule).density_fit()  # the kernel uses its with_df factors
    mean_field.kernel()

    gw = CayleyGW(mean_field, omega_p=OMEGA_P, n_q=N_Q, verbose=VERBOSE)
    charged = gw.kernel(N_CONSERVED, n_ip=N_IP, n_ea=N_EA)  # the quasiparticle table is in the log

    reference_gap = np.float64(gw.moments.adapter.reference.gap) * HARTREE_TO_EV
    quasiparticle_gap = (
        charged.ionization_potentials[0] - charged.electron_affinities[0]
    ) * HARTREE_TO_EV
    print(f"\n{'=' * 78}\nResults: H2O/{BASIS} G0W0@RHF with Cayley moments\n{'=' * 78}")
    print(f"reference HOMO-LUMO gap       {reference_gap:8.3f} eV")
    print(f"G0W0 correction to the gap    {quasiparticle_gap - reference_gap:+8.3f} eV")


if __name__ == "__main__":
    main()
