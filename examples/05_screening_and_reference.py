"""Screening model, mean-field reference and frozen core.

Three options of ``CayleyGW`` fix what is calculated before any numerical
control.  ``screening="tda"`` drops the de-excitation block of the RPA
response, a different approximation to W rather than a cheaper route to the
same one.  A DFT reference brings the static correction ``Sigma_x^HF - v_xc``,
which is zero for Hartree-Fock and is read from
``gw.hamiltonian.static_correction``.  ``frozen`` removes the lowest orbitals
from both the self-energy and the screening.
"""

from __future__ import annotations

import numpy as np
from pyscf import dft, gto, scf

from cayleygw import CayleyGW, enable_logging

# --- controls ---------------------------------------------------------------
ATOM = "N 0 0 0; N 0 0 1.098"  # geometry, Angstrom
BASIS = "cc-pvdz"  # orbital basis
XC = "pbe"  # exchange-correlation functional of the density-functional reference
FROZEN = 2  # lowest occupied orbitals removed from the active space (the two N 1s)
N_CONSERVED = 3  # conserved moment orders
OMEGA_P = 0.5  # Cayley scale in Hartree
N_Q = "auto"  # contour nodes
VERBOSE = 1  # the log on standard error; 0 removes all but warnings, 2 adds progress counts

HARTREE_TO_EV = 27.211386245981


def run(report, mean_field, label, **options):
    """Run one setting's kernel, add its frontier energies to ``report`` and return it."""

    gw = CayleyGW(mean_field, omega_p=OMEGA_P, n_q=N_Q, verbose=VERBOSE, **options)
    charged = gw.kernel(N_CONSERVED, n_ip=2, n_ea=1)
    ips = charged.ionization_potentials * HARTREE_TO_EV
    ea = charged.electron_affinities[0] * HARTREE_TO_EV
    report.append(
        f"  {label:<28} {gw.moments.adapter.reference.nmo:4d} {gw.moments.n_q:5d}"
        f" {ips[0]:10.3f} {ips[1]:10.3f} {ea:10.3f}"
    )
    return gw


def main() -> None:
    enable_logging(VERBOSE)
    molecule = gto.M(atom=ATOM, basis=BASIS, unit="Angstrom", verbose=0)
    hartree_fock = scf.RHF(molecule).density_fit()
    hartree_fock.kernel()
    kohn_sham = dft.RKS(molecule, xc=XC).density_fit()
    kohn_sham.kernel()

    report = []  # printed after the log, as one section
    report.append(f"  {'setting':<28} n_mo   N_q         IP      IP(2)         EA")
    rpa = run(report, hartree_fock, "RHF, RPA screening")
    run(report, hartree_fock, "RHF, TDA screening", screening="tda")
    pbe = run(report, kohn_sham, f"{XC.upper()}, RPA screening")
    run(report, hartree_fock, f"RHF, RPA, frozen={FROZEN}", frozen=FROZEN)

    report.append("")
    report.append("static correction Sigma_x^HF - v_xc in the active orbitals:")
    reference = pbe.moments.adapter.reference
    homo, lumo = reference.occupied_positions[-1], reference.virtual_positions[0]
    for label, gw in (("RHF", rpa), (XC.upper(), pbe)):
        correction = np.real(np.asarray(gw.hamiltonian.static_correction))
        report.append(
            f"  {label:<4} Frobenius norm {np.linalg.norm(correction):9.2e} Ha;"
            f"  HOMO {correction[homo, homo] * HARTREE_TO_EV:8.3f} eV,"
            f"  LUMO {correction[lumo, lumo] * HARTREE_TO_EV:8.3f} eV"
        )
    report.append(
        f"reference gaps: RHF {np.float64(rpa.moments.adapter.reference.gap) * HARTREE_TO_EV:.3f} eV,"
        f" {XC.upper()} {np.float64(reference.gap) * HARTREE_TO_EV:.3f} eV"
    )

    print(
        f"\n{'=' * 78}\nResults: N2/{BASIS} frontier energies in eV by screening, reference and frozen core\n{'=' * 78}"
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
