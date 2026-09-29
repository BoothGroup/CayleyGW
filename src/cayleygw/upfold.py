r"""The upfolded Dyson Hamiltonian: realize the sector moments, assemble, diagonalize.

Each sector's Cayley moments are realized as poles :math:`d` and couplings
:math:`W`, and the two sectors are assembled around the physical block
:math:`h_0 + \Sigma_\infty` into

.. math::

   H = \begin{pmatrix}
     h_0 + \Sigma_\infty & W^{<} & W^{>} \\
     W^{<\dagger} & d^{<} & 0 \\
     W^{>\dagger} & 0 & d^{>}
   \end{pmatrix},

whose resolvent on the physical block is the Green's function. Its eigenvalues
are the charged poles and their weight on the physical block tells
quasiparticles from satellites. All energies are in Hartree.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np

from ._helpers.errors import ValidationError
from ._helpers.tolerances import DEFAULT_TOLERANCES, Tolerances
from ._helpers.types import Sector
from ._helpers.upfold.hermitian import _validated_hermitian
from ._helpers.upfold.reconstruction import _sector_reconstruction
from ._helpers.upfold.sectors import _checked_settings, _log_settings, _realize_sectors
from ._helpers.validate import ComplexArray, _check
from .moments import G0W0CayleyMoments
from .realization.base import RealizationAlgorithm
from .realization.sector import SectorSelfEnergyRealization
from .tools.logger import Stages, summarized
from .tools.parallel import limited_native_threads, resolve_native_threads
from .tools.records.upfold import (
    DysonProblem,
    DysonSpectrum,
    MomentReconstructionResult,
    SectorMomentReconstruction,
)

TerminalSelection: TypeAlias = Literal["scan", "restricted"]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class UpfoldedDysonHamiltonian(DysonProblem):
    r"""Hermitian upfolded Hamiltonian of a hole and a particle pole sum.

    Build it with :func:`build_upfolded_hamiltonian`, or directly from any two
    sectors. A Hermiticity residual within the shared tolerance is
    symmetrized away and a larger one refused. No pole, coupling or weight is
    clipped.

    Attributes:
        reference_operator: Hermitian physical one-body operator :math:`h_0`.
        hole: Hole sector, an object with ``sector``, ``poles`` and ``couplings``
            of shape (physical dimension, poles); it may have no poles.
        particle: Particle sector, with the same attributes.
        static_correction: Hermitian static self-energy :math:`\Sigma_\infty`;
            ``None`` means zero.
        tolerances: Moment-conservation threshold of the reconstruction test.
        native_threads: Threads for :meth:`diagonalize`, or ``None`` for the
            ambient count. :func:`build_upfolded_hamiltonian` sets the count it
            ran with.
        mo_indices: Index of each physical orbital in ``mf.mo_energy``, or ``None``
            to number them from 0. :func:`build_upfolded_hamiltonian` sets them,
            so the log names the mean-field orbitals past a frozen core.

    Raises:
        ValidationError: If an operator is not square and Hermitian, a sector
            has the wrong label or coupling shape, ``tolerances`` has the
            wrong type, or ``mo_indices`` is not one index per orbital.
    """

    reference_operator: ComplexArray = field(repr=False)
    hole: Any
    particle: Any
    static_correction: ComplexArray | None = field(default=None, repr=False)
    tolerances: Tolerances = field(
        default=DEFAULT_TOLERANCES,
        repr=False,
        compare=False,
    )
    native_threads: int | None = field(default=None, repr=False, compare=False)
    mo_indices: tuple[int, ...] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tolerances, Tolerances):
            raise ValidationError("tolerances must be a Tolerances instance")
        reference = _validated_hermitian(self.reference_operator, "reference_operator")
        if self.static_correction is None:
            static_input = np.zeros_like(reference)
        else:
            static_input = self.static_correction
        static = _validated_hermitian(
            static_input,
            "static_correction",
            expected_shape=reference.shape,
        )
        for label, source, sector in (
            ("hole", self.hole, Sector.HOLE),
            ("particle", self.particle, Sector.PARTICLE),
        ):
            if getattr(source, "sector", None) is not sector:
                raise ValidationError(f"{label} source must carry Sector.{sector.name}")
            poles = getattr(source, "poles", None)
            if np.ndim(poles) != 1 or np.shape(getattr(source, "couplings", None)) != (
                reference.shape[0],
                np.size(poles),
            ):
                raise ValidationError(
                    f"{label} couplings must have shape (physical dimension, number of poles)"
                )
        if self.mo_indices is not None:
            indices = tuple(
                _check.nonnegative_integer(index, "mo_indices entry") for index in self.mo_indices
            )
            if len(indices) != reference.shape[0] or len(set(indices)) != len(indices):
                raise ValidationError("mo_indices must name each physical orbital once")
            object.__setattr__(self, "mo_indices", indices)
        object.__setattr__(self, "reference_operator", reference)
        object.__setattr__(self, "static_correction", static)

    @property
    def nphysical(self) -> int:
        """Number of physical orbitals, the size of the physical block."""

        return int(self.reference_operator.shape[0])

    @property
    def nhole(self) -> int:
        """Number of hole auxiliary states, one per hole pole."""

        return int(np.size(self.hole.poles))

    @property
    def nparticle(self) -> int:
        """Number of particle auxiliary states, one per particle pole."""

        return int(np.size(self.particle.poles))

    @property
    def dimension(self) -> int:
        """Dimension of :attr:`matrix`: the physical, hole and particle states."""

        return self.nphysical + self.nhole + self.nparticle

    @property
    def physical_slice(self) -> slice:
        """Rows and columns of the physical block in :attr:`matrix`."""

        return slice(0, self.nphysical)

    @property
    def hole_slice(self) -> slice:
        """Rows and columns of the hole auxiliary block in :attr:`matrix`."""

        return slice(self.nphysical, self.nphysical + self.nhole)

    @property
    def particle_slice(self) -> slice:
        """Rows and columns of the particle auxiliary block in :attr:`matrix`."""

        return slice(self.nphysical + self.nhole, self.dimension)

    @property
    def physical_matrix(self) -> ComplexArray:
        """The physical block ``reference_operator + static_correction`` as a read-only array."""

        return _check.readonly_complex(
            self.reference_operator + self.static_correction,
            "physical_matrix",
        )

    @property
    def matrix(self) -> ComplexArray:
        """The full Hermitian matrix, read-only, ordered physical, hole, particle."""

        # Filled in place: the largest array of a run, and np.diag would copy each block.
        result = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        result[self.physical_slice, self.physical_slice] = self.physical_matrix
        result[self.physical_slice, self.hole_slice] = self.hole.couplings
        result[self.physical_slice, self.particle_slice] = self.particle.couplings
        result[self.hole_slice, self.physical_slice] = self.hole.couplings.conj().T
        result[self.particle_slice, self.physical_slice] = self.particle.couplings.conj().T
        np.fill_diagonal(result[self.hole_slice, self.hole_slice], self.hole.poles)
        np.fill_diagonal(
            result[self.particle_slice, self.particle_slice],
            self.particle.poles,
        )
        result.setflags(write=False)
        return result

    @classmethod
    def from_moments(
        cls,
        moments: G0W0CayleyMoments,
        *,
        n_conserved: int | None = None,
        terminal_selection: TerminalSelection = "restricted",
        terminal_phase_count: int | None = None,
        n_workers: int = 1,
        native_threads: int | Literal["auto"] = "auto",
        realization: RealizationAlgorithm = "auto",
        tolerances: Tolerances = DEFAULT_TOLERANCES,
        verbose: int = 1,
    ) -> UpfoldedDysonHamiltonian:
        """Realize the moments and assemble the Hamiltonian.

        See :func:`build_upfolded_hamiltonian`.
        """

        if not isinstance(moments, G0W0CayleyMoments):
            raise ValidationError("moments must be a G0W0CayleyMoments result")
        stages = Stages(_check.nonnegative_integer(verbose, "verbose"), LOGGER)
        settings = _checked_settings(
            moments,
            n_conserved=n_conserved,
            terminal_selection=terminal_selection,
            terminal_phase_count=terminal_phase_count,
            n_workers=n_workers,
            native_threads=native_threads,
            realization=realization,
            tolerances=tolerances,
        )
        _log_settings(
            stages,
            moments,
            settings,
            realization=realization,
            terminal_selection=terminal_selection,
            n_workers=n_workers,
        )
        if stages.verbose >= 1:
            LOGGER.info(
                "realizing %d moment orders with n_conserved %d: 1 spare moment",
                settings.n_max + 1,
                settings.n_conserved,
            )
        realized = _realize_sectors(
            SectorSelfEnergyRealization.realize,
            moments,
            settings,
            realization=realization,
            n_workers=n_workers,
            verbose=verbose,
            stages=stages,
        )

        assembly_context = stages.stage("upfolded assembly")
        with assembly_context:
            return cls(
                reference_operator=np.diag(moments.adapter.reference.mo_energy),
                static_correction=(
                    moments.static_correction
                    if moments.static_correction is not None
                    else moments.adapter.build_static_self_energy_correction()
                ),
                hole=realized[Sector.HOLE],
                particle=realized[Sector.PARTICLE],
                tolerances=settings.tolerances,
                # Carried so the Dyson solve runs at this count, not an ambient 1.
                native_threads=settings.threads,
                mo_indices=moments.adapter.active_indices,
            )

    def diagonalize(
        self,
        *,
        verbose: int = 1,
        native_threads: int | Literal["auto"] | None = None,
    ) -> DysonSpectrum:
        """Diagonalize this Hamiltonian into its spectrum; see :func:`diagonalize_upfolded`."""

        stages = Stages(_check.nonnegative_integer(verbose, "verbose"), LOGGER)
        threads = (
            self.native_threads
            if native_threads is None
            else resolve_native_threads(native_threads)
        )
        with stages.stage("upfolded diagonalization", dimension=self.dimension):
            with nullcontext() if threads is None else limited_native_threads(threads):
                energies, eigenvectors = np.linalg.eigh(self.matrix)
            # Sealed so the spectrum shares them instead of copying the eigenvectors.
            energies = np.ascontiguousarray(energies, dtype=np.float64)
            energies.setflags(write=False)
            eigenvectors.setflags(write=False)
            return DysonSpectrum(energies=energies, eigenvectors=eigenvectors, problem=self)

    def reconstruction_test(
        self,
        *,
        n_max: int | None = None,
        verbose: int = 1,
    ) -> MomentReconstructionResult:
        """Rebuild the Cayley moments from this Hamiltonian; see :func:`reconstruction_test`."""

        stages = Stages(_check.nonnegative_integer(verbose, "verbose"), LOGGER)
        context = stages.stage("moment reconstruction")
        with context:
            reconstructions = {}
            for sector, source in (
                (Sector.HOLE, self.hole),
                (Sector.PARTICLE, self.particle),
            ):
                if not isinstance(source, SectorSelfEnergyRealization):
                    raise ValidationError(
                        "moment reconstruction needs sectors from "
                        "SectorSelfEnergyRealization.realize, which "
                        "build_upfolded_hamiltonian makes"
                    )
                reconstructions[sector] = _sector_reconstruction(self, sector, n_max)
            result = MomentReconstructionResult(
                hole=reconstructions[Sector.HOLE],
                particle=reconstructions[Sector.PARTICLE],
            )
        stages.result("reconstruction_test", result)
        return result


@summarized("build_upfolded_hamiltonian", LOGGER)
def build_upfolded_hamiltonian(
    moments: G0W0CayleyMoments,
    *,
    n_conserved: int | None = None,
    terminal_selection: TerminalSelection = "restricted",
    terminal_phase_count: int | None = None,
    n_workers: int = 1,
    native_threads: int | Literal["auto"] = "auto",
    realization: RealizationAlgorithm = "auto",
    tolerances: Tolerances = DEFAULT_TOLERANCES,
    verbose: int = 1,
) -> UpfoldedDysonHamiltonian:
    """Realize the Cayley moments of both sectors and assemble the upfolded Hamiltonian.

    Each sector is realized as a pole sum whose closure is chosen against the
    spare moment. The physical block is ``diag(mo_energy) + Sigma_x - v_MF``
    with ``v_MF = get_veff() - get_j()``, so the static term is zero for
    Hartree-Fock and removes the exchange-correlation potential for DFT.

    Args:
        moments: Result of :func:`build_cayley_moments`.
        n_conserved: Highest moment order to conserve; the order of the build by
            default. A lower order reuses the build with order ``n_conserved + 1``
            as the spare, a higher one is refused.
        terminal_selection: ``"restricted"`` realizes only the closure phases
            that can still win on the spare moment; it is never worse than the
            full grid. ``"scan"`` realizes every phase on the grid.
        terminal_phase_count: Closure phases on the grid; 64 by default.
        n_workers: Python workers over the two sectors and the closure candidates.
        native_threads: BLAS and LAPACK threads for the realization and the later
            diagonalization, or ``"auto"`` to share the cores among ``n_workers``.
        realization: ``"auto"`` uses block-CMV and falls back to block-Toeplitz
            when a construction, support or conservation check fails;
            ``"block-cmv"`` and ``"toeplitz"`` force one backend.
        tolerances: The safeguards: rank floor, conservation, arc margin and
            positivity. A refusal names the larger rank floors to try.
        verbose: ``0`` warnings only, ``1`` the log, ``2`` adds progress counts.

    Returns:
        The Hamiltonian, with the realized sectors as ``hole`` and ``particle``.

    Raises:
        ValidationError: If an argument is invalid or ``n_conserved`` exceeds
            the build.
        RefusalError: If a sector cannot be realized within ``tolerances``.
    """

    return UpfoldedDysonHamiltonian.from_moments(
        moments,
        n_conserved=n_conserved,
        terminal_selection=terminal_selection,
        terminal_phase_count=terminal_phase_count,
        n_workers=n_workers,
        native_threads=native_threads,
        realization=realization,
        tolerances=tolerances,
        verbose=verbose,
    )


def diagonalize_upfolded(
    hamiltonian: UpfoldedDysonHamiltonian,
    *,
    verbose: int = 1,
    native_threads: int | Literal["auto"] | None = None,
) -> DysonSpectrum:
    """Diagonalize an upfolded Hamiltonian and return its full spectrum.

    The eigenvalues are the poles of the Green's function; their weight on the
    physical block tells quasiparticles from satellites.

    Args:
        hamiltonian: From :func:`build_upfolded_hamiltonian` or built directly.
        verbose: ``0`` warnings only, ``1`` the log, ``2`` adds progress counts.
        native_threads: BLAS and LAPACK threads for the eigensolve, or ``"auto"``
            for the available cores. ``None`` uses the Hamiltonian's own
            ``native_threads``, which :func:`build_upfolded_hamiltonian` sets to
            the count of the calculation, so a driver that pins
            ``OMP_NUM_THREADS=1`` does not run this cubic solve on one core.

    Returns:
        The ascending eigenvalues, the eigenvectors and the physical weights.

    Raises:
        ValidationError: If ``hamiltonian`` is not an :class:`UpfoldedDysonHamiltonian`.
    """

    if not isinstance(hamiltonian, UpfoldedDysonHamiltonian):
        raise ValidationError("hamiltonian must be an UpfoldedDysonHamiltonian")
    return hamiltonian.diagonalize(verbose=verbose, native_threads=native_threads)


def reconstruction_test(
    hamiltonian: UpfoldedDysonHamiltonian,
    *,
    n_max: int | None = None,
    verbose: int = 1,
) -> MomentReconstructionResult:
    r"""Rebuild the Cayley moments from an upfolded Hamiltonian and compare them.

    Each sector's moments are recomputed from the poles and couplings in the
    matrix, :math:`C_k = \sum_\ell w_\ell u_\ell^k w_\ell^\dagger` with
    :math:`u_\ell` the Cayley image of pole :math:`d_\ell`, and compared with
    the moments the sector was realized from, order by order.

    Args:
        hamiltonian: From :func:`build_upfolded_hamiltonian`, whose sectors keep
            the moments and the Cayley map.
        n_max: Highest order to compare; every retained order by default.
            Orders above the conserved one are reported but do not count
            towards ``passed``.
        verbose: ``0`` warnings only, ``1`` the log, ``2`` adds progress counts.

    Returns:
        The absolute and relative errors and the thresholds of every hole and
        particle order.

    Raises:
        ValidationError: If ``hamiltonian`` is not an :class:`UpfoldedDysonHamiltonian`,
            its sectors were not realized from moments, or ``n_max`` exceeds
            the retained orders.
    """

    if not isinstance(hamiltonian, UpfoldedDysonHamiltonian):
        raise ValidationError("hamiltonian must be an UpfoldedDysonHamiltonian")
    return hamiltonian.reconstruction_test(n_max=n_max, verbose=verbose)


__all__ = [
    "MomentReconstructionResult",
    "SectorMomentReconstruction",
    "TerminalSelection",
    "build_upfolded_hamiltonian",
    "diagonalize_upfolded",
    "reconstruction_test",
    "DysonSpectrum",
    "UpfoldedDysonHamiltonian",
]
