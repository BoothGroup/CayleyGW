"""A larger molecule: threads, workers, orbital block size, timing and memory.

A template for a workstation or cluster job: benzene in def2-SVP with the six
carbon 1s orbitals frozen runs in under a minute on one thread.  The cost is
in the contour node solves and the moment contraction, and grows with the
square of the auxiliary basis per node.  ``native_threads`` is the BLAS and
LAPACK thread count of every stage and the lever that scales, ``n_workers``
runs contour nodes and closure candidates in Python workers, and
``contour_orbital_block_size`` trades peak memory for larger products.  The
memory estimate prints before anything expensive runs, and ``LOG_FILE``
follows every stage as it happens.
"""

from __future__ import annotations

import time

from pyscf import dft, gto

from cayleygw import CayleyGW, enable_logging

# --- controls ---------------------------------------------------------------
ATOM = """
C  1.3970  0.0000 0; C  0.6985  1.2098 0; C -0.6985  1.2098 0;
C -1.3970  0.0000 0; C -0.6985 -1.2098 0; C  0.6985 -1.2098 0;
H  2.4810  0.0000 0; H  1.2405  2.1486 0; H -1.2405  2.1486 0;
H -2.4810  0.0000 0; H -1.2405 -2.1486 0; H  1.2405 -2.1486 0
"""  # benzene, Angstrom
BASIS = "def2-svp"  # orbital basis; density fitting picks the matching auxiliary basis
XC = "pbe"  # functional of the reference
FROZEN = 6  # lowest occupied orbitals frozen (the six C 1s)
N_CONSERVED = 5  # conserved moment orders
OMEGA_P = 0.5  # Cayley scale in Hartree
N_Q = "auto"  # contour nodes; the estimate below assumes ESTIMATE_N_Q
ESTIMATE_N_Q = 512  # node count assumed by the memory estimate
NATIVE_THREADS = "auto"  # BLAS/LAPACK threads per worker; "auto" shares the cores this job may use
N_WORKERS = 1  # Python workers over contour nodes and orbital blocks
CONTOUR_ORBITAL_BLOCK_SIZE = 4  # internal orbitals contracted at once
DRY_RUN = False  # True prints the sizes and the memory estimate and stops
VERBOSE = 1  # the log on standard error; 0 removes all but warnings, 2 adds progress counts
LOG_FILE = "large_molecule.log"  # every stage and result, appended in the working directory


def memory_estimate(n_mo, n_aux, n_q, n_conserved, block_size):
    """Return the largest arrays of the calculation in bytes, by name.

    The shapes are the ones the package allocates; the peak of a run is about
    their sum plus the LAPACK workspace of the final eigensolve.
    """

    n_max = n_conserved + 1
    dimension = n_mo * (2 * n_conserved + 3)  # physical block plus two full-rank sectors
    return {
        "density-fitted MO integrals (n_aux, n_mo, n_mo), real": n_aux * n_mo**2 * 8,
        "projected resolvent at N_q/2 nodes (N_q/2, n_aux, n_aux), complex": (n_q // 2)
        * n_aux**2
        * 16,
        "auxiliary moments of one orbital block, both conjugate halves (2 block, n_max+1, n_aux, n_aux), complex": 2
        * block_size
        * (n_max + 1)
        * n_aux**2
        * 16,
        "upfolded Hamiltonian at its largest dimension (dimension, dimension), complex": dimension
        ** 2
        * 16,
    }


def main() -> None:
    enable_logging(VERBOSE, log_file=LOG_FILE)
    molecule = gto.M(atom=ATOM, basis=BASIS, unit="Angstrom", verbose=0)
    mean_field = dft.RKS(molecule, xc=XC).density_fit()
    # The three-centre integrals are built here and reused by the SCF, so the sizes are known first.
    mean_field.with_df.build()
    n_mo = molecule.nao - FROZEN
    n_aux = mean_field.with_df.get_naoaux()
    n_max = N_CONSERVED + 1
    # Printed before anything expensive runs, so that it can stop the job.
    print(f"{'=' * 78}\nBefore the run: the sizes that set the cost\n{'=' * 78}")
    print(
        f"{molecule.natm} atoms, {molecule.nao} basis functions, {n_mo} active orbitals, {n_aux} auxiliary functions, moments C_0..C_{n_max}"
    )
    print(f"memory estimate at N_q={ESTIMATE_N_Q}, block size {CONTOUR_ORBITAL_BLOCK_SIZE}:")
    estimate = memory_estimate(n_mo, n_aux, ESTIMATE_N_Q, N_CONSERVED, CONTOUR_ORBITAL_BLOCK_SIZE)
    for name, size in estimate.items():
        print(f"  {size / 2**20:9.1f} MiB  {name}")
    print(f"  {sum(estimate.values()) / 2**20:9.1f} MiB  sum")
    if DRY_RUN:
        return

    started = time.perf_counter()
    mean_field.kernel()
    mean_field_seconds = time.perf_counter() - started
    gw = CayleyGW(
        mean_field,
        omega_p=OMEGA_P,
        n_q=N_Q,
        frozen=FROZEN,
        native_threads=NATIVE_THREADS,
        n_workers=N_WORKERS,
        contour_orbital_block_size=CONTOUR_ORBITAL_BLOCK_SIZE,
        verbose=VERBOSE,
    )
    # The selected N_q, the upfolded dimension, the frontier energies and the time of each step are in the log.
    gw.kernel(N_CONSERVED, n_ip=1, n_ea=1)

    print(f"\n{'=' * 78}\nResults: {molecule.natm} atoms, {BASIS}, G0W0@{XC.upper()}\n{'=' * 78}")
    print(f"mean field: {mean_field_seconds:.1f} s")
    print(f"every stage with its sizes and wall time: {LOG_FILE}")


if __name__ == "__main__":
    main()
