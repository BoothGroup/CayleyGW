"""The four calls behind the kernel, and convergence with the conserved order.

``CayleyGW.kernel`` calls ``build_cayley_moments``, ``build_upfolded_hamiltonian``,
``diagonalize_upfolded`` and ``extract_ip_ea`` in turn, and each can be called
alone.  Here the moments are built once at ``n_conserved=7`` and realized at
every lower order: the orders above the requested one are dropped and the next
one up becomes the spare that selects the closure.  The kernel reuses its
moments the same way when called again at a lower order.  The last line checks
the kernel against the four calls at the highest order.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto, scf

from cayleygw import (
    CayleyGW,
    build_cayley_moments,
    build_upfolded_hamiltonian,
    diagonalize_upfolded,
    enable_logging,
    extract_ip_ea,
)

# --- controls ---------------------------------------------------------------
ATOM = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"  # geometry, Angstrom
BASIS = "cc-pvdz"  # orbital basis
N_CONSERVED_BUILT = 7  # highest conserved order; the moments are built once at this order
ORDERS = range(1, N_CONSERVED_BUILT + 1)  # conserved orders re-used from the one build
OMEGA_P = 0.5  # Cayley scale in Hartree
N_Q = "auto"  # contour nodes, converged automatically for the highest order built
VERBOSE = 1  # the log on standard error; 0 removes all but warnings, 2 adds progress counts

HARTREE_TO_EV = 27.211386245981


def main() -> None:
    enable_logging(VERBOSE)
    molecule = gto.M(atom=ATOM, basis=BASIS, unit="Angstrom", verbose=0)
    mean_field = scf.RHF(molecule).density_fit()
    mean_field.kernel()

    # 1. The hole and particle moments C_0..C_8, by contour quadrature.
    moments = build_cayley_moments(
        mean_field,
        n_conserved=N_CONSERVED_BUILT,
        omega_p=OMEGA_P,
        n_q=N_Q,
        verbose=VERBOSE,
    )
    rows = []
    for n_conserved in ORDERS:
        # 2. C_0..C_n realized as poles and couplings; C_{n+1} is the spare that selects the closure.
        hamiltonian = build_upfolded_hamiltonian(moments, n_conserved=n_conserved, verbose=VERBOSE)
        # 3. Every eigenvalue, with its weight on the orbitals.
        spectrum = diagonalize_upfolded(hamiltonian, verbose=VERBOSE)
        # 4. The frontier energies, counted outwards from the chemical potential.
        charged = extract_ip_ea(spectrum, n_ip=1, n_ea=1, verbose=VERBOSE)
        rows.append(
            (
                n_conserved,
                hamiltonian.dimension,
                charged.ionization_potentials[0] * HARTREE_TO_EV,
                charged.electron_affinities[0] * HARTREE_TO_EV,
            )
        )

    # The kernel makes the same four calls, so at the highest order it builds the same moments.
    gw = CayleyGW(mean_field, omega_p=OMEGA_P, n_q=N_Q, verbose=VERBOSE)
    kernel = gw.kernel(N_CONSERVED_BUILT, n_ip=1, n_ea=1)

    print(
        f"\n{'=' * 78}\nResults: H2O/{BASIS} G0W0@RHF, one moment build realized at every lower order\n{'=' * 78}"
    )
    print("  n  orders used   dim      IP (eV)      EA (eV)   dIP (meV)   dEA (meV)")
    previous = None
    for n_conserved, dimension, ip, ea in rows:
        change = (
            "          -           -"
            if previous is None
            else f"{(ip - previous[0]) * 1e3:11.3f} {(ea - previous[1]) * 1e3:11.3f}"
        )
        print(
            f"  {n_conserved}  C_0..C_{n_conserved} + C_{n_conserved + 1}"
            f"  {dimension:4d}  {ip:11.4f}  {ea:11.4f} {change}"
        )
        previous = (ip, ea)
    highest, _, highest_ip, highest_ea = rows[-1]
    print()
    print(f"deviation from the n={highest} result (meV):")
    for n_conserved, _, ip, ea in rows[:-1]:
        print(
            f"  n={n_conserved}: IP {abs(ip - highest_ip) * 1e3:9.3f}"
            f"   EA {abs(ea - highest_ea) * 1e3:9.3f}"
        )
    print(
        f"reference HOMO-LUMO gap: {np.float64(moments.adapter.reference.gap) * HARTREE_TO_EV:.3f} eV"
    )
    print(
        f"gw.kernel({highest}) against the four calls at n={highest}:"
        f" IP {(kernel.ionization_potentials[0] * HARTREE_TO_EV - highest_ip) * 1e3:+.3f} meV,"
        f" EA {(kernel.electron_affinities[0] * HARTREE_TO_EV - highest_ea) * 1e3:+.3f} meV"
    )


if __name__ == "__main__":
    main()
