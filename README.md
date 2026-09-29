# cayleygw

Cayley-moment $G_0W_0$ for closed-shell molecules on top of PySCF.

`cayleygw` computes the one-shot $G_0W_0$ self-energy of a closed-shell
molecule from moments of its spectral measure, taken after a Cayley transform
that maps the poles of each sector, below or above the chemical potential,
onto the unit circle.  A finite set of self-energy poles coupled to the
orbitals is obtained from a Hermitian upfolded Hamiltonian. The methods 
implemented in this package have been described in the following papers:

> M. K. Allen and G. H. Booth, *Full-frequency GW from Cayley-transformed
> self-energy moments*, [arXiv:2609.29271](https://arxiv.org/abs/2609.29271)
> (2026).

## Install

```
pip install .
```

Python 3.11 or later, NumPy, SciPy, PySCF 2.7 or later, threadpoolctl and
rich.
`pip install ".[plot]"` adds matplotlib for `plot_spectrum`;
`pip install ".[test]"` adds pytest and matplotlib for the test suite.

## QuickStart example

```python
from pyscf import gto, scf
from cayleygw import CayleyGW

mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", basis="cc-pvdz")
mf = scf.RHF(mol).density_fit().run()
# The kernel logs its options, results and timings on standard error.
result = CayleyGW(mf).kernel(3)
print(result.ionization_potentials * 27.2114)  # 12.17, 14.42, 18.56 eV
print(result.electron_affinities * 27.2114)  # -4.70, -6.64, -20.34 eV
```

For finer control the kernel is four calls, which can be used alone or in
other combinations; example 02 takes it apart:

```python
from cayleygw import (
    build_cayley_moments,
    build_upfolded_hamiltonian,
    diagonalize_upfolded,
    extract_ip_ea,
)

moments = build_cayley_moments(mf, n_conserved=3, omega_p=0.5, n_q="auto")
hamiltonian = build_upfolded_hamiltonian(moments)
spectrum = diagonalize_upfolded(hamiltonian)
result = extract_ip_ea(spectrum, n_ip=3, n_ea=3)
```

Everything inside the package is in Hartree.

## The controls that matter

`CayleyGW(mf, ...)` takes the options that define the moments, and
`gw.kernel(n_conserved, ...)` the ones that realize them; the build functions
take the same names.  The kernel keeps what it built as `gw.moments`,
`gw.hamiltonian`, `gw.spectrum` and `gw.result`.

### The constructor: what defines the moments

**`omega_p`.**  The scale of the Cayley map, in Hartree, centred at the
chemical potential $\mu$, the middle of the HOMO-LUMO gap; the default is 0.5
Hartree.  Energies are spread such that $\mu$ maps to $-1$ and
$\mu \pm \omega_p$ to $\pm i$, and energies tend to $1$ as they approach
$\pm\infty$.

**`n_q` and `n_q_tolerance`.**  `n_q` is the number of nodes $N_q$ on an
ellipse enclosing the positive RPA excitation spectrum; it must be even, and a
fixed count uses the periodic midpoint rule.  `n_q="auto"` starts at 128 nodes
and doubles the complete contour until every moment order in both sectors
changes by less than `n_q_tolerance` (relative, `1e-7`).  It uses the
trapezoidal rule instead, whose nodes nest, so each doubling solves only the
new nodes; the two rules are equally accurate at the same $N_q$.

**Screening.**  The random phase approximation (`screening="rpa"`), the
default, or the Tamm-Dancoff approximation (`"tda"`).

**`frozen`.**  The number of lowest occupied orbitals to freeze.  Frozen
orbitals leave both the self-energy and the screening.

**`spectral_bound`.**  How the RPA excitation energies are bounded to place
the ellipse.  `"spectral"`, the default, uses the exact two-norm of the
coupling at about the cost of one node; `"frobenius"` uses its Frobenius norm,
which is cheaper but overshoots more as the system grows and slows the quadrature.

### `kernel()`: how they are realized

**`n_conserved`.**  The realization reproduces the self-energy moments $C_0$
to $C_{n_\mathrm{conserved}}$, and one more, $C_{n_\mathrm{conserved}+1}$, is
built to choose the terminal closure.  A later `kernel` at the same or a lower
order reuses `gw.moments`; a higher order builds them again.

**Terminal closure.**  Defaults to `terminal_selection="restricted"`, an
optimized minimization around a single fitting moment.  The alternative,
`"scan"`, evaluates a uniform grid over the unit circle.
`terminal_phase_count` sets the number of phases on the grid, 64 by default.

**Realization.**  `realization="auto"` runs the block-CMV recursion while its
checks pass and falls back to the block-Toeplitz Gram completion.  CMV tends
to perform better for smaller moment orders ($n \lesssim 5$), while Toeplitz
performs better for larger orders ($n \gtrsim 7$).

**`tolerances`.**  The safeguards of the realization, held within `Tolerances`:
`rank_floor`, the eigenvalue floor of the rank cuts (`1e-10`);
`moment_conservation` and its `conservation_margin`; `arc_margin`; and
`positivity`.  The defaults are the tested ones, and
[examples/README.md](examples/README.md#tolerances) says what each guards.  A
sector that cannot be realized raises a `RefusalError` that names the larger
rank floors to try, as in `gw.kernel(3, tolerances=Tolerances(rank_floor=3e-10))`.

**`n_ip` and `n_ea`.**  How many ionization potentials and electron
affinities the result holds and the log tabulates, counted outwards from the
chemical potential; three of each by default.

## The reference route for small systems

For a molecule whose RPA problem can be solved completely, the self-energy has
an explicit pole representation, one pole per orbital and excitation in each
sector, and `ExactG0W0SelfEnergy` in `cayleygw.tools.reference` provides it.
It gives the exact Cayley moments, which test the contour quadrature; the
exact upfolded Hamiltonian with every pole, which tests the moment truncation;
and the diagonal quasiparticle equation, which agrees with PySCF's `gw_exact`
to below a microelectronvolt.  By default it uses the exact four-index
integrals.  With `use_density_fitting=True` it rebuilds them from the mean
field's density-fitting factors, $(pq|rs) = \sum_P L_{P,pq} L_{P,rs}$, so it
carries the same fitting error as moments built from that mean field and the
comparison measures only the quadrature and the realization.

```python
import numpy as np
from cayleygw import ExactG0W0SelfEnergy, UpfoldedDysonHamiltonian

exact = ExactG0W0SelfEnergy.from_mean_field(mf, use_density_fitting=True)
exact_moments = exact.cayley_moments(moments.mapping, moments.n_max)
untruncated = UpfoldedDysonHamiltonian(
    np.diag(exact.reference.mo_energy),
    exact.hole,
    exact.particle,
    static_correction=exact.static_correction,
)
```

Examples 08 and 10 measure the finite-order results and PySCF's `gw_exact`
and `gw_cd` against it.

## Scope and limitations

- Restricted closed-shell molecular references from PySCF, RHF or RKS, with
  real orbitals.  PBE is exercised in the tests and examples; other functionals
  go through the same PySCF potential and are not separately benchmarked.
  Unrestricted, open-shell, fractionally occupied and periodic references are
  not supported.
- One-shot $G_0W_0$: no self-consistency in either $G$ or $W$.
- Direct RPA or direct Tamm-Dancoff screening.
- Density fitting is assumed for production; the exact four-index interaction
  is for small systems and the reference route.
- Threads and Python workers on one machine; no MPI.

## Examples

Twelve scripts in [`examples/`](./examples/README.md): the first runs the kernel,
the second takes it apart into its four calls, and each later one adds one
idea.  Each opens with what it shows and its named controls.  The log they
print is described in [`examples/logging.md`](./examples/logging.md).

## Tests

```
pip install ".[test]"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest
```

The suite takes about three minutes on one core and runs every example.  The
tests mirror the package: a `tests/test_<module>.py` for each main module,
`tests/realization/`, `tests/tools/` and `tests/_helpers/` for the modules
below them, and `tests/test_layout.py`, which holds the package layout itself
as tests.  `tests/test_regression.py` runs the regression protocol, water,
carbon monoxide and nitrogen from RHF and PBE with both screenings at three
conserved orders and the exact route for LiH and $\mathrm{H_2}$, against a
stored reference at a stated tolerance.  Quantities that move with the BLAS
build or the core count, such as the number of poles a rank cut keeps, are
compared with a stored single-threaded reference through `tests/drift.py` and
reported, never gated.  Both files say how to regenerate their references.

## Advanced setup

**Threads and workers.**  `native_threads` is the BLAS and LAPACK thread count
inside every stage and the lever that scales.  The default, `"auto"`, uses the
thread count set in `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` or
`MKL_NUM_THREADS` if one is set, and otherwise the cores the process may run
on, which respects `taskset`, SLURM and container limits; the cores are shared
among the `n_workers`.  `n_workers` (default 1) evaluates independent contour
nodes and closure candidates in Python workers.  An integer pins either
exactly.  The thread count moves results at roundoff, which can occasionally
change how many poles a rank cut keeps, so pin both to compare runs across
machines.  On laptops with a few fast and many slow cores, fewer threads than
`"auto"` picks can be faster.  `contour_orbital_block_size` trades peak
memory in the moment contraction for larger matrix products.  Example 09
prints the sizes and a memory estimate before running.

The package pins its own BLAS threads to `native_threads`, but PySCF's mean
field runs before it at whatever count the environment sets.  To make a run
reproducible end to end, set the count before Python starts; `"auto"` then
uses the same number:

```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Citing

If you use `cayleygw`, please cite the paper:

> M. K. Allen and G. H. Booth, *Full-frequency GW from Cayley-transformed
> self-energy moments*, [arXiv:2609.29271](https://arxiv.org/abs/2609.29271)
> (2026).

`CITATION.cff` carries the same reference in machine-readable form.

## License

Apache License 2.0; see [`LICENSE`](./LICENSE).

## Authors

Marcus K. Allen and George H. Booth, Department of Physics, King's College
London.
