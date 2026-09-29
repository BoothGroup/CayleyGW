# Logging

The log is on by default.  Through the logger named `cayleygw` it draws on
standard error a banner with the package versions, the options and sizes of
each main call, a spinner naming the stage that is running, one line per
doubling of `N_q`, a summary panel with the stage times when each main call
ends, and the quasiparticle table from `extract_ip_ea`.  `verbose=0` silences
one call or one `CayleyGW` and `enable_logging(0)` all of them but warnings;
`verbose=2` with `enable_logging(2)` adds progress counts to the spinner.  On a
stream that is not a terminal the spinner is left out and nothing is coloured.
This is the run of [`01_first_calculation.py`](./01_first_calculation.py) (water, cc-pVDZ, RHF),
abridged:

```
build_cayley_moments                                                     options
            Options                        Sizes
          Option        Value              Space   Size
───────────────────────────────    ──────────────────────
     n_conserved            3           orbitals     24
             n_q         auto           occupied      5
   n_q_tolerance        1e-07            virtual     19
         omega_p          0.5        transitions     95
       screening   direct RPA
...

N_q   128 -> 256    error/tolerance = 6.257e-03                        converged

╭─────────────────── build_cayley_moments ────────────────────╮
│  Moments C_0..C_4 converged at N_q = 256 in 0.24 s.         │
│                     Contour                                 │
│             Nodes N_q        256 (129 solved)               │
│        Ellipse centre               12.772207               │
│  ...                                                        │
│                           Timings                           │
│                                   Stage     Time   Share    │
│                        automatic N_q x2   0.18 s   75.8%    │
│    quadrature and moment contraction x2   0.09 s   38.2%    │
│  ...                                                        │
╰─────────────────────────────────────────────────────────────╯
...
│      Sector     Backend   Poles   Closure   Moment error   Status    │
│        Hole   block-cmv      96   1.118pi       4.20e-14   clean     │
│    Particle   block-cmv      96   0.895pi       1.85e-14   clean     │
...
│    Excitation   Energy (eV)   Energy (Ha)   QP weight   Dominant orbitals    │
│          IP 0       12.1664    0.44710706      0.9514   HOMO (100%)          │
│          EA 0       -4.6991   -0.17268757      0.9887   LUMO (100%)          │
```

The error of each `N_q` doubling is the largest relative change of any moment
order between the two grids, in units of `n_q_tolerance`; below one the
moments have converged, and the doubling is accepted once they also pass the
positivity checks.  The closure is the terminal phase each sector was closed
with, in units of pi (`natural` where the moments terminated on their own),
and a status other than `clean` names the tolerance band or fallback the
sector was accepted through.

## Bare logging

`enable_logging(style="bare")` draws the same view with only the key
results: the options, the sizes, the selected `N_q`, the sector table and the
quasiparticle energies, with the total time of each call in its panel.  It
leaves out the doublings before the selected `N_q`, the ellipse and the stage
times.  Example 11 runs one calculation under both styles.

## Log file

`enable_logging(log_file="run.log")` appends a plain-text log to `run.log`
beside either style.  It opens with the versions, the command, the host and
the native libraries with their thread counts, and holds a timestamped line
when each stage starts, with its sizes, and when it ends, with its wall time.
It also holds every option, and each result in full: the moment norms and the
`N_q` history per sector, the closure, the residuals and the discarded weight
of each sector, every orbital in a quasiparticle, and the reconstruction error
of every order.  `verbose=2` adds progress counts.  A batch job is followed
through this file, since the console draws no spinner there; example 09 writes
one.  The file of the same run, abridged, without its timestamps and the lines
where a stage ends:

```
reference preparation
build_cayley_moments: options                          # every option, then the sizes
density-fitted interaction factors: n_mo=24            # 24 active orbitals,
AO-to-MO transform: n_aux=116, n_mo=24                 # 116 auxiliary functions
transition-factor assembly
spectral bounds and resolvent: n_mo=24, n_transitions=95, screening=direct RPA
contour selection                                      # the ellipse from the certified bounds
automatic N_q: N_q=128, n_max=4                        # first budget: 128 nodes, moments C_0..C_4
contour node solves: nodes=128, solved=65, n_aux=116   # conjugate pairs and the two real nodes
quadrature and moment contraction: n_max=4, n_orbitals=24
automatic N_q: N_q=256, n_max=4
contour node solves: nodes=128, solved=64, n_aux=116   # only the new nodes; the 128 are reused
...
automatic N_q feasibility (hole)                       # the positivity checks of the last grid
automatic N_q feasibility (particle)
Automatic N_q refinement 128 -> 256: maximum normalized moment error 6.257e-03; converged=True
build_cayley_moments: result                           # N_q history, moment norms, stage times
build_upfolded_hamiltonian: options
realizing 5 moment orders with n_conserved 3: 1 spare moment
sector realizations: n_conserved=3, realization=auto
hole realization: rank_floor=1.0e-10, n_conserved=3
hole block-cmv realization: n_conserved=3, rank=24     # the backend that passed its checks
hole terminal candidates: phase_count=64, refinement=restricted
hole poles and couplings
particle realization: rank_floor=1.0e-10, n_conserved=3    # the same for the particle sector
...
upfolded assembly
build_upfolded_hamiltonian: result                     # closure, residuals and discards per sector
upfolded diagonalization: dimension=216                # 24 orbitals + 96 + 96 poles
pole classification                                    # extract_ip_ea
extract_ip_ea: result                                  # every orbital of each quasiparticle
```

A refusal is a `RefusalError` whose message says what failed, with the
offending quantity, and what to change: usually `N_q`, `omega_p`,
`n_conserved`, `terminal_phase_count` or the rank floor in `tolerances`.
