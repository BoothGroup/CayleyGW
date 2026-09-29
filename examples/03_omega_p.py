"""The Cayley scale omega_p: which energies a few moments resolve.

The Cayley map sends the chemical potential to ``u = -1`` and the energies
``omega_p`` either side of it to ``u = -i`` and ``u = i``; energies far from it
are pressed towards ``u = 1``.  At a low conserved order this decides which
quasiparticles the few poles describe well.  For each ``omega_p`` the kernel
runs at two low orders, and the first electron affinity, the first ionization
potential and two deeper removal energies are compared with a high-order
reference.  Example 04 turns to the contour itself.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto, scf

from cayleygw import CayleyGW, enable_logging

# --- controls ---------------------------------------------------------------
ATOM = "C 0 0 0; O 0 0 1.128"  # geometry, Angstrom
BASIS = "cc-pvdz"  # orbital basis
OMEGA_P_VALUES = (0.125, 0.25, 0.5, 1.0, 2.0)  # Cayley scales in Hartree
ORDERS = (1, 2)  # low conserved orders at which the omega_p dependence is visible
DEEPER_IPS = (
    4,
    5,
)  # further removal energies followed, counted outwards from the chemical potential
REFERENCE_OMEGA_P = 0.5  # Cayley scale of the reference
REFERENCE_N_CONSERVED = 9  # conserved order of the reference
N_Q = "auto"  # contour nodes: doubled from 128 until every moment order is converged
N_Q_TOLERANCE = 1.0e-7  # relative moment change accepted by the automatic doubling
VERBOSE = 1  # the log on standard error; 0 removes all but warnings, 2 adds progress counts

HARTREE_TO_EV = 27.211386245981


def charged_energies(gw, n_conserved):
    """Run the kernel and return the first EA and the removal energies followed here, in eV."""

    charged = gw.kernel(n_conserved, n_ip=max(DEEPER_IPS), n_ea=1)
    removal = charged.ionization_potentials * HARTREE_TO_EV
    return np.array(
        [
            charged.electron_affinities[0] * HARTREE_TO_EV,
            removal[0],
            *(removal[index - 1] for index in DEEPER_IPS),
        ]
    )


def main() -> None:
    enable_logging(VERBOSE)
    molecule = gto.M(atom=ATOM, basis=BASIS, unit="Angstrom", verbose=0)
    mean_field = scf.RHF(molecule).density_fit()
    mean_field.kernel()

    settings = dict(n_q=N_Q, n_q_tolerance=N_Q_TOLERANCE, verbose=VERBOSE)  # shared by every build
    reference = charged_energies(
        CayleyGW(mean_field, omega_p=REFERENCE_OMEGA_P, **settings), REFERENCE_N_CONSERVED
    )
    labels = ["EA", "IP", *(f"IP({index})" for index in DEEPER_IPS)]
    header = "  omega_p   n   N_q" + "".join(f" {label:>9}" for label in labels)
    rule = "  " + "-" * (len(header) - 2)

    report = [  # printed after the log, as one section
        f"errors in meV against n_conserved={REFERENCE_N_CONSERVED} at omega_p={REFERENCE_OMEGA_P}"
        " (IP(k) is the k-th removal energy outwards from the chemical potential):",
        "  reference: "
        + ", ".join(f"{label} {value:.3f} eV" for label, value in zip(labels, reference)),
        header,
    ]
    for omega_p in OMEGA_P_VALUES:
        report.append(rule)
        gw = CayleyGW(mean_field, omega_p=omega_p, **settings)
        for n_conserved in ORDERS:
            # Each order is higher than the last, so each kernel builds its moments again.
            errors = (charged_energies(gw, n_conserved) - reference) * 1e3
            report.append(
                f"  {omega_p:7.3f}   {n_conserved}  {gw.moments.n_q:4d}"
                + "".join(f" {error:9.3f}" for error in errors)
            )

    print(
        f"\n{'=' * 78}\nResults: CO/{BASIS} G0W0@RHF, the Cayley scale omega_p at low conserved order\n{'=' * 78}"
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
