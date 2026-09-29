"""The contour quadrature: the node count and the ellipse the spectral bounds place.

The Cayley moments are sums over ``N_q`` nodes on an ellipse around the RPA
excitation spectrum, and their error falls geometrically with ``N_q``.  One
``CayleyGW`` per setting compares fixed ``n_q`` with ``"auto"``, and the
ellipse at each ``spectral_bound`` is shown with the bounds it rests on.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto, scf

from cayleygw import CayleyGW, RefusalError, build_cayley_moments, enable_logging


# --- controls ---------------------------------------------------------------
ATOM = "C 0 0 0; O 0 0 1.128"  # geometry, Angstrom
BASIS = "cc-pvdz"  # orbital basis
N_CONSERVED = 3  # conserved moment orders in every build below
OMEGA_P = 0.5  # Cayley scale in Hartree
FIXED_N_Q = (64, 128, 256, 512)  # complete-contour node counts for the fixed-grid study
N_Q_TOLERANCE = 1.0e-7  # relative moment change accepted by the automatic doubling
SPECTRAL_BOUNDS = ("frobenius", "spectral")  # enclosures of the RPA spectrum to compare
VERBOSE = 1  # the log on standard error; 0 removes all but warnings, 2 adds progress counts

HARTREE_TO_EV = 27.211386245981


def frontier(gw, n_conserved):
    """Run the kernel and return the first EA and IP, in eV."""

    charged = gw.kernel(n_conserved, n_ip=1, n_ea=1)
    return (
        charged.electron_affinities[0] * HARTREE_TO_EV,
        charged.ionization_potentials[0] * HARTREE_TO_EV,
    )


def main() -> None:
    enable_logging(VERBOSE)
    molecule = gto.M(atom=ATOM, basis=BASIS, unit="Angstrom", verbose=0)
    mean_field = scf.RHF(molecule).density_fit()
    mean_field.kernel()

    settings = dict(n_q_tolerance=N_Q_TOLERANCE, verbose=VERBOSE)  # shared by every build

    report = []  # printed after the log, as one section

    # --- fixed N_q against the automatic doubling ---------------------------
    report.append("fixed contour grids (moment change is the largest relative change of")
    report.append("any hole or particle moment against the previous grid):")
    report.append("   N_q      EA (eV)      IP (eV)   moment change")
    previous = None
    for n_q in FIXED_N_Q:
        gw = CayleyGW(mean_field, n_q=n_q, omega_p=OMEGA_P, **settings)
        try:
            ea, ip = frontier(gw, N_CONSERVED)
            energies = f"{ea:11.4f}  {ip:11.4f}"
        except RefusalError as refusal:
            # A coarse grid's moments are not a positive measure: the realization refuses and names the lever.
            lever = "increase N_q" if "Increase N_q" in str(refusal) else "see the message"
            energies = f"refused, {lever:<20}"
        moments = gw.moments  # kept even when the realization refused
        change = "-"
        if previous is not None:
            change = max(
                np.linalg.norm(current[k] - before[k]) / np.linalg.norm(current[k])
                for current, before in (
                    (moments.hole.moments, previous.hole.moments),
                    (moments.particle.moments, previous.particle.moments),
                )
                for k in range(moments.n_max + 1)
            )
            change = f"{change:.2e}"
        previous = moments
        report.append(f"  {n_q:4d}  {energies}   {change}")

    # The doubling history and the nodes solved are in the log of this build.
    automatic = CayleyGW(mean_field, n_q="auto", omega_p=OMEGA_P, **settings)
    ea, ip = frontier(automatic, N_CONSERVED)
    report.append(f"  auto  {ea:11.4f}  {ip:11.4f}   selected N_q={automatic.moments.n_q}")

    # --- the ellipse the code chose -----------------------------------------
    report.append("")
    report.append("ellipse zeta(theta) = c + a cos(theta) + i b sin(theta), Hartree:")
    for bound in SPECTRAL_BOUNDS:
        moments = automatic.moments
        if bound != moments.spectral_bound:
            # The ellipse needs only the moments, so this build is not realized.
            moments = build_cayley_moments(
                mean_field, n_conserved=N_CONSERVED, n_q="auto", omega_p=OMEGA_P, spectral_bound=bound, **settings
            )
        bounds = moments.rpa_spectral_bounds
        contour = moments.contour
        equioscillation = contour.equioscillation_diagnostics
        report.append(f"  spectral_bound='{bound}':")
        report.append(f"    RPA excitation energies enclosed in [{bounds.lower:.4f}, {bounds.upper:.4f}] ({bounds.source})")
        report.append(
            f"    c={contour.center:.4f}  a={contour.horizontal_radius:.4f}"
            f"  b={contour.vertical_radius:.4f}"
            f"  sigma={equioscillation.sigma_c:.4f} (limit {equioscillation.sigma_min:.4f},"
            f" nearest excluded singularity: {equioscillation.limiting_singularity})"
        )
        report.append(f"    selected N_q={moments.n_q}")

    print(f"\n{'=' * 78}\nResults: CO/{BASIS} G0W0@RHF, contour node count and ellipse\n{'=' * 78}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
