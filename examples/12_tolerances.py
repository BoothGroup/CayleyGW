"""Every tuning setting and safeguard, spelled out at its default.

Three tuning settings trade cost against accuracy, or filter what is
reported, and can be changed freely: ``n_q_tolerance`` on ``CayleyGW``,
``terminal_phase_count`` on ``gw.kernel`` and ``minimum_weight`` on
``extract_ip_ea``.  Five safeguards decide whether a realization is trusted;
they travel together in a ``Tolerances`` passed as ``gw.kernel(n,
tolerances=...)``.  The script sets every one explicitly and checks that the
spelled-out defaults reproduce ``gw.kernel(n)``.  The examples README says what
each one guards and lists the constants that stay internal.
"""

from __future__ import annotations

from pyscf import gto, scf

from cayleygw import CayleyGW, Tolerances, enable_logging, extract_ip_ea


# --- controls ---------------------------------------------------------------
ATOM = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"  # geometry, Angstrom
BASIS = "cc-pvdz"  # orbital basis
N_CONSERVED = 3  # conserved moment orders
# Tuning: change freely.
N_Q_TOLERANCE = 1.0e-7  # relative moment change at which n_q="auto" stops doubling
TERMINAL_PHASE_COUNT = 64  # candidate closure phases per sector
MINIMUM_WEIGHT = 0.1  # smallest quasiparticle weight reported as an IP or EA
# Safeguards: read what each guards in the examples README before changing it.
RANK_FLOOR = 1.0e-10  # eigenvalue floor of the rank cuts
MOMENT_CONSERVATION = 1.0e-9  # error allowed on the conserved moments after a discard
CONSERVATION_MARGIN = 2.0  # multiple of it accepted but flagged marginal
ARC_MARGIN = 100.0  # multiple of the negligible weight that wrong-arc poles may carry
POSITIVITY = 1.0e-10  # allowed breach of moment positivity
VERBOSE = 1  # the log on standard error; 0 removes all but warnings, 2 adds progress counts

HARTREE_TO_EV = 27.211386245981


def main() -> None:
    enable_logging(VERBOSE)
    molecule = gto.M(atom=ATOM, basis=BASIS, unit="Angstrom", verbose=0)
    mean_field = scf.RHF(molecule).density_fit()
    mean_field.kernel()

    tolerances = Tolerances(
        rank_floor=RANK_FLOOR,
        moment_conservation=MOMENT_CONSERVATION,
        conservation_margin=CONSERVATION_MARGIN,
        arc_margin=ARC_MARGIN,
        positivity=POSITIVITY,
    )
    gw = CayleyGW(mean_field, n_q_tolerance=N_Q_TOLERANCE, verbose=VERBOSE)
    gw.kernel(N_CONSERVED, terminal_phase_count=TERMINAL_PHASE_COUNT, tolerances=tolerances)
    # The kernel reads its IPs and EAs at the default weight; minimum_weight belongs to extract_ip_ea.
    spelled_out = extract_ip_ea(gw.spectrum, n_ip=3, n_ea=3, minimum_weight=MINIMUM_WEIGHT, verbose=0)

    # The same order again with every default: the kernel reuses the moments.
    defaults = gw.kernel(N_CONSERVED)
    difference = max(
        abs(a - b)
        for a, b in zip(
            (*spelled_out.ionization_potentials, *spelled_out.electron_affinities),
            (*defaults.ionization_potentials, *defaults.electron_affinities),
        )
    )

    print(f"\n{'=' * 78}\nResults: H2O/{BASIS} G0W0@RHF with every setting spelled out\n{'=' * 78}")
    print(f"safeguards      {tolerances}")
    print(f"minimum_weight  {MINIMUM_WEIGHT}")
    print(
        f"against gw.kernel({N_CONSERVED}) with the defaults: largest IP or EA difference "
        f"{difference * HARTREE_TO_EV * 1000.0:.3f} meV"
    )


if __name__ == "__main__":
    main()
