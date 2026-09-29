"""Terminal closure, realization backend, the reconstruction test and the sector diagnostics.

A finite moment sequence leaves the last block of its realization free, and
each sector closes it with the one of ``terminal_phase_count`` phases that best
reproduces the withheld moment.  One ``CayleyGW`` builds the moments once and
each ``gw.kernel`` at the same order realizes them again: ``"restricted"``
selection, which realizes only the phases that can still win, against
``"scan"``, which realizes them all; the ``"block-cmv"`` and ``"toeplitz"``
backends and ``"auto"``, which falls back from the first to the second; and
``reconstruction_test`` on ``gw.hamiltonian``, which rebuilds the conserved
moments from the stored poles.
"""

from __future__ import annotations

import time

from pyscf import gto, scf

from cayleygw import CayleyGW, enable_logging, reconstruction_test


# --- controls ---------------------------------------------------------------
ATOM = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"  # geometry, Angstrom
BASIS = "cc-pvdz"  # orbital basis
N_CONSERVED = 3  # conserved moment orders
OMEGA_P = 0.5  # Cayley scale in Hartree
N_Q = "auto"  # contour nodes
TERMINAL_PHASE_COUNTS = (16, 64)  # candidate phases on the terminal grid
REALIZATIONS = ("block-cmv", "toeplitz", "auto")  # backends to compare
VERBOSE = 1  # the log on standard error; 0 removes all but warnings, 2 adds progress counts

HARTREE_TO_EV = 27.211386245981


def frontier(charged):
    """Return the first IP and EA of a kernel result, in eV."""

    return (
        charged.ionization_potentials[0] * HARTREE_TO_EV,
        charged.electron_affinities[0] * HARTREE_TO_EV,
    )


def main() -> None:
    enable_logging(VERBOSE)
    molecule = gto.M(atom=ATOM, basis=BASIS, unit="Angstrom", verbose=0)
    mean_field = scf.RHF(molecule).density_fit()
    mean_field.kernel()
    gw = CayleyGW(mean_field, omega_p=OMEGA_P, n_q=N_Q, verbose=VERBOSE)
    # The defaults; this kernel builds the moments and every later one at this order reuses them.
    gw.kernel(N_CONSERVED, n_ip=1, n_ea=1)
    report = []  # printed after the log, as one section

    # --- terminal selection -------------------------------------------------
    report.append("terminal closure: candidates realized per sector, selected phase (rad), frontier energies")
    report.append("  selection    phases   hole realized/phase   particle realized/phase      IP (eV)    EA (eV)   seconds")
    for selection in ("restricted", "scan"):
        for count in TERMINAL_PHASE_COUNTS:
            started = time.perf_counter()
            charged = gw.kernel(
                N_CONSERVED,
                terminal_selection=selection,
                terminal_phase_count=count,
                n_ip=1,
                n_ea=1,
            )
            elapsed = time.perf_counter() - started
            ip, ea = frontier(charged)
            columns = []
            for sector in (gw.hamiltonian.hole, gw.hamiltonian.particle):
                columns.append(
                    f"{len(sector.closure_scan.candidates):3d} / {sector.selected_phase:8.4f}"
                )
            report.append(
                f"  {selection:<11} {count:6d}     {columns[0]:<20} {columns[1]:<22}"
                f" {ip:10.4f} {ea:10.4f} {elapsed:9.2f}"
            )

    # --- realization backend and the reconstruction test ---------------------
    report.append("")
    report.append("realization backend: the algorithm each sector ended on, its terminal dimension,")
    report.append("the frontier energies and the reconstruction test of the conserved moments")
    report.append("  requested    hole        particle    terminal      IP (eV)    EA (eV)   reconstruction")
    hamiltonians = {}
    for realization in REALIZATIONS:
        ip, ea = frontier(gw.kernel(N_CONSERVED, realization=realization, n_ip=1, n_ea=1))
        hamiltonian = hamiltonians[realization] = gw.hamiltonian
        reconstruction = reconstruction_test(hamiltonian, verbose=VERBOSE)
        hole, particle = hamiltonian.hole, hamiltonian.particle
        report.append(
            f"  {realization:<12} {hole.closure_scan.realization_algorithm:<11}"
            f" {particle.closure_scan.realization_algorithm:<11}"
            f" {hole.terminal_dimension:3d} / {particle.terminal_dimension:<3d}"
            f"  {ip:10.4f} {ea:10.4f}   {'passed' if reconstruction.passed else 'FAILED'}"
        )

    # --- reading the sector diagnostics ------------------------------------
    report.append("")
    # The poles, closure phase, moment error and marginal flags are in the log's sector table.
    report.append("sector diagnostics of the default realization:")
    for name, sector in (("hole", hamiltonians["auto"].hole), ("particle", hamiltonians["auto"].particle)):
        report.append(
            f"  {name:<9} terminal dimension {sector.terminal_dimension},"
            f" discarded pole weight {sector.discarded_total_weight:.2e}"
            f" (wrong arc {sector.discarded_wrong_arc_weight:.2e}),"
            f" conservation ratio {sector.maximum_conservation_ratio:.2e} of tolerance"
        )

    print(f"\n{'=' * 78}\nResults: H2O/{BASIS} G0W0@RHF, terminal closure and realization backend\n{'=' * 78}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
