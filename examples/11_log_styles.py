"""The two console log styles, standard and bare, and the log file.

``enable_logging`` chooses the style of the log on standard error.
``"standard"``, the default, shows a banner with the package versions, the
options and sizes of each call, one line per doubling of ``N_q``, a summary
panel with the stage times and the quasiparticle energies.  ``"bare"`` keeps
the options, the sizes and the results, and gives each call's total time only.
On a terminal a spinner names the running stage.  ``log_file`` adds a
plain-text file beside either style, with a timestamped line per stage and
every option and result in full.  The same kernel runs once per style, so only
the log differs; both runs append to one file.
"""

from __future__ import annotations

from pyscf import gto, scf

from cayleygw import CayleyGW, enable_logging


# --- controls ---------------------------------------------------------------
ATOM = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"  # geometry, Angstrom
BASIS = "cc-pvdz"  # orbital basis
N_CONSERVED = 3  # conserved moment orders
N_Q = "auto"  # contour nodes: the standard log shows every doubling, the bare one the last
STYLES = ("standard", "bare")  # log styles to run the calculation under, in order
VERBOSE = 1  # the log on standard error; 0 removes all but warnings, 2 adds progress counts
N_IP = 3  # removal energies in the quasiparticle table
N_EA = 3  # addition energies in the quasiparticle table
LOG_FILE = "log_styles.log"  # appended in the working directory; None writes no file


def main() -> None:
    molecule = gto.M(atom=ATOM, basis=BASIS, unit="Angstrom", verbose=0)
    mean_field = scf.RHF(molecule).density_fit()
    mean_field.kernel()

    for style in STYLES:
        # Replaces the previous style's handlers and reopens the file; each style's log opens with the banner.
        enable_logging(VERBOSE, style=style, log_file=LOG_FILE)
        # A new CayleyGW per style, since a second kernel on one would reuse its moments.
        CayleyGW(mean_field, n_q=N_Q, verbose=VERBOSE).kernel(N_CONSERVED, n_ip=N_IP, n_ea=N_EA)

    print(f"\n{'=' * 78}\nResults: H2O/{BASIS}, one calculation under each log style\n{'=' * 78}")
    print("styles, in the order their logs appear above: " + ", ".join(repr(style) for style in STYLES))
    if LOG_FILE is not None:
        print(f"log file, holding every run: {LOG_FILE}")


if __name__ == "__main__":
    main()
