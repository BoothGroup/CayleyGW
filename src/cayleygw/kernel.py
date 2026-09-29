"""The CayleyGW kernel: a whole Cayley-moment G0W0 calculation in one call."""

from __future__ import annotations

from typing import Any, Literal

from ._helpers.tolerances import DEFAULT_TOLERANCES, Tolerances
from ._helpers.types import Screening
from ._helpers.validate import _check
from .moments import G0W0CayleyMoments, build_cayley_moments
from .tools.records.spectrum import IPEAResult
from .tools.records.upfold import DysonSpectrum
from .tools.spectrum import extract_ip_ea
from .upfold import UpfoldedDysonHamiltonian, build_upfolded_hamiltonian, diagonalize_upfolded


class CayleyGW:
    """Cayley-moment G0W0 for a closed-shell PySCF mean field.

    The constructor fixes what defines the moments; :meth:`kernel` realizes them
    at a conserved order and returns the frontier energies. The moments are
    kept, so a later kernel at the same or a lower order reuses them and only a
    higher order builds them again.

    Args:
        mean_field: Converged restricted PySCF mean field (RHF or RKS).
        omega_p: Scale of the Cayley map in Hartree.
        n_q: Contour nodes, even, or ``"auto"`` to double them, reusing each, until the
            moments converge.
        n_q_tolerance: Relative moment change at which ``"auto"`` stops.
        screening: ``"rpa"`` or ``"tda"``.
        frozen: Number of lowest occupied orbitals to freeze.
        spectral_bound: ``"spectral"`` or ``"frobenius"`` bound on the RPA spectrum.
        n_workers: Python workers over contour nodes and closure candidates.
        native_threads: BLAS and LAPACK threads per worker, or ``"auto"`` to share the cores.
        contour_orbital_block_size: Orbitals contracted together; memory against speed.
        verbose: ``0`` warnings only, ``1`` the log, ``2`` adds progress counts.

    Attributes:
        moments: The Cayley moments of the last build.
        hamiltonian: The upfolded Hamiltonian of the last kernel.
        spectrum: Its eigenvalues and physical weights.
        result: The ionization potentials and electron affinities of the last kernel.
    """

    def __init__(
        self,
        mean_field: Any,
        *,
        omega_p: float = 0.5,
        n_q: int | Literal["auto"] = "auto",
        n_q_tolerance: float = 1.0e-7,
        screening: Screening | Literal["rpa", "tda"] = Screening.RPA,
        frozen: int = 0,
        spectral_bound: Literal["spectral", "frobenius"] = "spectral",
        n_workers: int = 1,
        native_threads: int | Literal["auto"] = "auto",
        contour_orbital_block_size: int = 1,
        verbose: int = 1,
    ) -> None:
        self.mean_field = mean_field
        self.moment_options = dict(
            omega_p=omega_p,
            n_q=n_q,
            n_q_tolerance=n_q_tolerance,
            screening=screening,
            frozen=frozen,
            spectral_bound=spectral_bound,
            n_workers=n_workers,
            native_threads=native_threads,
            contour_orbital_block_size=contour_orbital_block_size,
        )
        self.verbose = verbose
        self.moments: G0W0CayleyMoments | None = None
        self.hamiltonian: UpfoldedDysonHamiltonian | None = None
        self.spectrum: DysonSpectrum | None = None
        self.result: IPEAResult | None = None

    def kernel(
        self,
        n_conserved: int,
        *,
        terminal_selection: Literal["restricted", "scan"] = "restricted",
        terminal_phase_count: int | None = None,
        realization: Literal["auto", "block-cmv", "toeplitz"] = "auto",
        tolerances: Tolerances = DEFAULT_TOLERANCES,
        n_ip: int = 3,
        n_ea: int = 3,
    ) -> IPEAResult:
        """Run the calculation at one conserved order.

        Args:
            n_conserved: Highest moment order the realization conserves.
            terminal_selection: ``"restricted"`` or ``"scan"`` choice of the closure.
            terminal_phase_count: Candidate closure phases; 64 by default.
            realization: ``"auto"``, ``"block-cmv"`` or ``"toeplitz"``.
            tolerances: The safeguards of the realization; a refusal names larger rank floors
                to try.
            n_ip: Ionization potentials to return.
            n_ea: Electron affinities to return.

        Returns:
            The first ``n_ip`` ionization potentials and ``n_ea`` electron
            affinities in Hartree, with their weights.

        Raises:
            ValidationError: If an argument or a moment option is invalid.
            RefusalError: If a sector cannot be realized within ``tolerances``.
        """

        order = _check.nonnegative_integer(n_conserved, "n_conserved")
        if self.moments is None or self.moments.n_conserved < order:
            self.moments = build_cayley_moments(
                self.mean_field, n_conserved=order, verbose=self.verbose, **self.moment_options
            )
        self.hamiltonian = build_upfolded_hamiltonian(
            self.moments,
            n_conserved=order,
            terminal_selection=terminal_selection,
            terminal_phase_count=terminal_phase_count,
            realization=realization,
            tolerances=tolerances,
            n_workers=self.moment_options["n_workers"],
            native_threads=self.moment_options["native_threads"],
            verbose=self.verbose,
        )
        self.spectrum = diagonalize_upfolded(self.hamiltonian, verbose=self.verbose)
        self.result = extract_ip_ea(self.spectrum, n_ip=n_ip, n_ea=n_ea, verbose=self.verbose)
        return self.result


__all__ = ["CayleyGW"]
