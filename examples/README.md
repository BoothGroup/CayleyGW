# Examples

Twelve scripts.  The first runs a whole calculation with the `CayleyGW` kernel,
the second takes the kernel apart into its four calls, and each later one adds
one idea.  Every script starts with a short description of what it shows and
a block of named controls with a one-line comment each; edit the controls and
run it again.  Each runs in under a minute on one core except 09, which is a
template for a larger job.  Run one from anywhere:

    python examples/01_first_calculation.py

Energies inside the package are Hartree; the scripts print eV.  Every script
runs with the log on standard error and prints its results after it, as one
section; `VERBOSE = 0` removes the log, and example 11 shows its styles.
[`logging.md`](./logging.md) describes what the log shows and the log file.

| # | Script | Adds | System |
|---|---|---|---|
| 01 | [`01_first_calculation.py`](./01_first_calculation.py) | The kernel with its defaults, `CayleyGW(mf).kernel(3)`; the ionization potentials, electron affinities and the log. | H2O / cc-pVDZ |
| 02 | [`02_the_parts.py`](./02_the_parts.py) | The four calls behind the kernel; one build at `n_conserved=7` realized at every lower order; the kernel checked against the calls. | H2O / cc-pVDZ |
| 03 | [`03_omega_p.py`](./03_omega_p.py) | The Cayley scale `omega_p`: the first EA and IPs at two low conserved orders against a high-order reference, for five scales. | CO / cc-pVDZ |
| 04 | [`04_contour_quadrature.py`](./04_contour_quadrature.py) | Fixed `n_q` against `"auto"`; the ellipse and the spectral bounds it rests on. | CO / cc-pVDZ |
| 05 | [`05_screening_and_reference.py`](./05_screening_and_reference.py) | RPA against TDA screening; RHF against PBE and the static correction; a frozen core. | N2 / cc-pVDZ |
| 06 | [`06_spectral_function.py`](./06_spectral_function.py) | Quasiparticle weights and satellites from `gw.spectrum`; the broadened total and orbital-resolved spectral function; plotting. | CO / cc-pVDZ |
| 07 | [`07_closure_and_realization.py`](./07_closure_and_realization.py) | One build realized again by each kernel: `restricted` against `scan` closure; `block-cmv`, `toeplitz` and `auto`; the reconstruction test; the sector diagnostics. | H2O / cc-pVDZ |
| 08 | [`08_exact_reference.py`](./08_exact_reference.py) | The exact G0W0 reference with every RPA pole; the quadrature and truncation errors of finite orders. | LiH / cc-pVDZ |
| 09 | [`09_large_molecule.py`](./09_large_molecule.py) | Threads, workers, orbital block size, the timing summary and a memory estimate; a template for a larger job, not run in CI. | benzene / def2-SVP |
| 10 | [`10_pyscf_comparison.py`](./10_pyscf_comparison.py) | Diagonal quasiparticle energies against PySCF's `gw_exact` and `gw_cd` for one reference. | H2O / cc-pVDZ |
| 11 | [`11_log_styles.py`](./11_log_styles.py) | One calculation under the standard and the bare log: what each shows and leaves out. | H2O / cc-pVDZ |
| 12 | [`12_tolerances.py`](./12_tolerances.py) | Every tuning setting and safeguard below, spelled out at its default; the same result as `gw.kernel(3)`. | H2O / cc-pVDZ |

`tests/test_examples.py` runs every script; the template runs there on a
small molecule.

## Tolerances

A handful of thresholds decide the calculation.  The tuning settings trade
cost against accuracy, or filter what is reported, and can be changed freely.
The safeguards decide whether a realization is trusted: loosening one can
admit a wrong answer and tightening one can refuse a good one, so change them
to study a refusal rather than to silence it.  Example 12 sets every one.

### Tuning

Each is an argument of the call it tunes: `CayleyGW(mf, n_q_tolerance=...)`,
`gw.kernel(n, terminal_phase_count=...)` and
`extract_ip_ea(gw.spectrum, minimum_weight=...)`.

| Setting | Default | What it changes |
|---|---|---|
| `n_q_tolerance` | `1e-7` | The relative change in every moment order at which `n_q="auto"` stops doubling the contour nodes.  Tighter costs more nodes; looser stops sooner. |
| `terminal_phase_count` | `64` | Candidate closure phases per sector.  More costs time and refines the closure. |
| `minimum_weight` | `0.1` | The smallest quasiparticle weight reported as an ionization potential or electron affinity.  It only filters what is reported. |

### Safeguards

They travel together in a `Tolerances`, passed as
`gw.kernel(n, tolerances=Tolerances(rank_floor=3e-10))`; fields left out keep
their defaults.

| Safeguard | Default | What it decides |
|---|---|---|
| `rank_floor` | `1e-10` | The eigenvalue floor of the rank cuts in the block-Toeplitz Gram matrix and the block-CMV defects.  Too low keeps roundoff as signal; too high drops real directions.  A refusal names the next floors to try, and the response is not monotone. |
| `moment_conservation` | `1e-9` | The error allowed on the conserved moments, relative to `1 + ‖C_n‖`, after unsupported poles are discarded.  Looser accepts a realization that no longer reproduces its moments. |
| `conservation_margin` | `2` | The multiple of `moment_conservation` accepted but flagged as marginal, so that roundoff from the BLAS summation order does not flip the decision. |
| `arc_margin` | `100` | The multiple of the negligible weight, `1e-12 + 1e-9 ×` the total weight, that poles on the wrong arc may carry and still be discarded, flagged as marginal.  Looser discards weight that is real. |
| `positivity` | `1e-10` | The breach of moment positivity allowed in the block-Toeplitz Gram matrix and the block-CMV Schur steps.  It refuses moments whose quadrature error breaks positivity, mostly at a small fixed `N_q`; looser admits them. |

### Internal

These cover checks that never decide a result, or steps of the closure
search.  They are constants in `cayleygw/_helpers/tolerances.py` and are not
meant to be changed.

| Constant | Value | What it covers |
|---|---|---|
| `ABSOLUTE_TOLERANCE` + `RELATIVE_TOLERANCE` | `1e-12 + 1e-9·scale` | The resolvent and contour guards, the moment feasibility checks, the zeroth-moment and support checks, the moment check of each realized spectrum, the Dyson Hermiticity check and the negligible weight under `arc_margin`. |
| `ROUNDOFF_TOLERANCE` | `1e-10` | Identities exact up to roundoff: unitarity, the unit circle, symmetry and contour enclosure; also the wrong-arc test, the normalized zeroth moment and the Coulomb null cut. |
| `REFERENCE_TOLERANCE` | `1e-8` | The mean field's occupations and gap, and the stability of the exact reference. |
| `BLOCK_CMV_MOMENT_TOLERANCE` | `1e-9` | The moments each backend rebuilds before any discard, relative to `1 + ‖C_n‖`. |
| `SECTOR_REDISTRIBUTION` | `0.5` | The largest zeroth-moment deficit the congruence may restore after a discard. |
| `PHASE_REFINE_STEPS` | `8` | Bisection steps of the restricted closure search. |
