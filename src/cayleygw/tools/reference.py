r"""Exact full-pole :math:`G_0W_0` from the complete response spectrum.

With every excitation :math:`\Omega_\nu` of the response problem, the
self-energy is a pole sum with hole poles at :math:`\epsilon_i - \Omega_\nu` and
particle poles at :math:`\epsilon_a + \Omega_\nu`, and its Cayley moments are
exact. It is the reference every approximate route is checked against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import newton

from .._helpers.cayley import CayleyMap
from .._helpers.errors import ValidationError
from .._helpers.pyscf import RestrictedMolecularReference, RestrictedPySCFAdapter
from .._helpers.screening.response import solve_response
from .._helpers.types import Screening, Sector
from .._helpers.validate import ComplexArray, FloatArray, _check


@dataclass(frozen=True, slots=True)
class SelfEnergySector:
    r"""Poles and couplings of one exact self-energy sector.

    The residues :math:`w_\lambda w_\lambda^\dagger` are positive semidefinite by
    construction. Arrays are copied read-only and nothing is clipped.

    Attributes:
        sector: Hole or particle.
        poles: Pole energies :math:`d_\lambda` in Hartree.
        couplings: Coupling columns :math:`w_\lambda`, shape (nphysical, npoles).
        chemical_potential: Energy in Hartree; hole poles lie below it and
            particle poles above.

    Raises:
        ValidationError: If a shape is wrong or a pole is on the wrong side of
            the chemical potential.
    """

    sector: Sector
    poles: FloatArray = field(repr=False)
    couplings: ComplexArray = field(repr=False)
    chemical_potential: float

    def __post_init__(self) -> None:
        if not isinstance(self.sector, Sector):
            raise ValidationError("sector must be a Sector value")
        poles = _check.readonly_real(self.poles, "poles")
        couplings = _check.readonly_complex(self.couplings, "couplings")
        if poles.ndim != 1 or poles.size == 0:
            raise ValidationError("poles must be a nonempty vector")
        if couplings.ndim != 2 or couplings.shape[0] == 0:
            raise ValidationError("couplings must have shape (nphysical, npoles)")
        if couplings.shape[1] != poles.size:
            raise ValidationError("couplings must contain one column per pole")
        chemical_potential = _check.finite_real(self.chemical_potential, "chemical_potential")
        if self.sector is Sector.HOLE and np.any(poles >= chemical_potential):
            raise ValidationError(
                "every hole self-energy pole must lie below the chemical potential"
            )
        if self.sector is Sector.PARTICLE and np.any(poles <= chemical_potential):
            raise ValidationError(
                "every particle self-energy pole must lie above the chemical potential"
            )

        object.__setattr__(self, "poles", poles)
        object.__setattr__(self, "couplings", couplings)
        object.__setattr__(self, "chemical_potential", chemical_potential)

    @property
    def nphysical(self) -> int:
        """Number of physical orbitals, the rows of :attr:`couplings`."""

        return int(self.couplings.shape[0])

    @property
    def npoles(self) -> int:
        """Number of poles in this sector, the columns of :attr:`couplings`."""

        return int(self.poles.size)

    def cayley_moments(self, mapping: CayleyMap, n_max: int) -> ComplexArray:
        """Exact Cayley moments of this sector up to order ``n_max``.

        Args:
            mapping: Cayley map; every pole must fall on this sector's arc.
            n_max: Highest moment order.

        Returns:
            Read-only ``C[k, p, q]`` of shape ``(n_max + 1, nphysical, nphysical)``.

        Raises:
            ValidationError: If ``mapping`` is not a Cayley map or a pole falls on
                the other sector's arc.
        """

        if not isinstance(mapping, CayleyMap):
            raise ValidationError("mapping must be a CayleyMap object")
        order = _check.nonnegative_integer(n_max, "n_max")
        signs = np.asarray(mapping.sector_sign(self.poles))
        if np.any(signs != self.sector.arc_sign):
            raise ValidationError(
                "Cayley-map center places at least one pole on the wrong sector arc"
            )
        powers = mapping.powers(self.poles, order)
        result = np.einsum(
            "pl,kl,ql->kpq",
            self.couplings,
            powers,
            self.couplings.conj(),
            optimize=True,
        )
        result.setflags(write=False)
        return result

    def evaluate_time_ordered(
        self,
        frequency: ArrayLike,
        *,
        eta: float = 1.0e-8,
    ) -> ComplexArray:
        """Evaluate this sector on the real axis with the time-ordered shift.

        Hole denominators are ``omega - pole - 1j*eta`` and particle ones
        ``omega - pole + 1j*eta``, as in :mod:`pyscf.gw.gw_exact`.

        Args:
            frequency: Real energies in Hartree, a scalar or an array.
            eta: Broadening in Hartree.

        Returns:
            The read-only self-energy matrix at each frequency.

        Raises:
            ValidationError: If a frequency is complex or ``eta`` is not positive.
        """

        broadening = _check.positive_real(eta, "eta")
        values = _check.readonly_complex(frequency, "frequency")
        if np.any(values.imag != 0.0):
            raise ValidationError("time-ordered evaluation requires real frequencies")
        shift = -1j * broadening if self.sector is Sector.HOLE else 1j * broadening
        denominator = values.real[..., None] - self.poles + shift
        result = np.einsum(
            "pl,...l,ql->...pq",
            self.couplings,
            1.0 / denominator,
            self.couplings.conj(),
            optimize=True,
        )
        result = np.asarray(result, dtype=np.complex128)
        if values.ndim == 0:
            result = result.reshape(self.nphysical, self.nphysical)
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class ExactG0W0SelfEnergy:
    r"""Exact :math:`G_0W_0` self-energy of a restricted mean field, with every pole kept.

    Its Cayley moments are exact, so it is the reference for small systems. It
    builds the full four-index integrals, exact or from density fitting, and
    solves the full response problem, for RPA or TDA screening. Build it with
    :meth:`from_mean_field`.

    Attributes:
        reference: Active restricted molecular reference.
        hole: Hole sector with every pole.
        particle: Particle sector with every pole.
        static_correction: Hermitian exact exchange minus the starting potential.
    """

    reference: RestrictedMolecularReference
    hole: SelfEnergySector
    particle: SelfEnergySector
    static_correction: ComplexArray = field(repr=False)

    @classmethod
    def from_integrals(
        cls,
        reference: RestrictedMolecularReference,
        eri: ArrayLike,
        *,
        screening: Screening = Screening.RPA,
        static_correction: ArrayLike | None = None,
    ) -> Self:
        r"""Build every self-energy pole and coupling from the full electron-repulsion integrals.

        The coupling is :math:`W_{pq,\nu} = \sqrt{2} \sum_{ia} (pq|ia) T_{ia,\nu}`,
        with :math:`T = X + Y` for RPA and :math:`T = X` for TDA. For RPA this
        equals PySCF's ``2 * (X + Y)``, whose amplitudes are smaller by
        :math:`\sqrt{2}`.

        Args:
            reference: Active restricted molecular reference.
            eri: Full active-orbital electron-repulsion integrals.
            screening: RPA or TDA response; RPA by default.
            static_correction: Hermitian exact exchange minus the starting
                potential; zero by default, which is right only for Hartree-Fock
                or a synthetic test.

        Returns:
            The exact self-energy.

        Raises:
            ValidationError: If a pole falls on the wrong side of the chemical
                potential.
        """

        eri = np.ascontiguousarray(eri, dtype=np.float64)
        _, energies, amplitudes = solve_response(reference, eri, screening)
        occupied = np.asarray(reference.occupied_positions)
        virtual = np.asarray(reference.virtual_positions)
        transition_integrals = eri[
            :, :, np.repeat(occupied, virtual.size), np.tile(virtual, occupied.size)
        ]
        screened = math.sqrt(2.0) * np.einsum(
            "pqt,tn->pqn",
            transition_integrals,
            amplitudes,
            optimize=True,
        )
        hole_poles = (reference.mo_energy[occupied, None] - energies[None, :]).reshape(-1)
        particle_poles = (reference.mo_energy[virtual, None] + energies[None, :]).reshape(-1)
        hole_couplings = screened[:, occupied, :].reshape(reference.nmo, -1)
        particle_couplings = screened[:, virtual, :].reshape(reference.nmo, -1)
        hole = SelfEnergySector(
            sector=Sector.HOLE,
            poles=hole_poles,
            couplings=hole_couplings,
            chemical_potential=reference.chemical_potential,
        )
        particle = SelfEnergySector(
            sector=Sector.PARTICLE,
            poles=particle_poles,
            couplings=particle_couplings,
            chemical_potential=reference.chemical_potential,
        )
        if static_correction is None:
            static_correction = np.zeros((reference.nmo, reference.nmo))
        return cls(
            reference=reference,
            hole=hole,
            particle=particle,
            static_correction=_check.readonly_complex(static_correction, "static_correction"),
        )

    @classmethod
    def from_mean_field(
        cls,
        mean_field: Any,
        *,
        frozen: int = 0,
        screening: Screening = Screening.RPA,
        use_density_fitting: bool = False,
    ) -> Self:
        r"""Build the exact self-energy of a PySCF mean field.

        Args:
            mean_field: Converged restricted PySCF mean field (RHF or RKS) with
                real orbitals.
            frozen: Number of lowest occupied orbitals to freeze.
            screening: ``Screening.RPA`` or ``Screening.TDA``. TDA drops the
                de-excitation block: a different approximation, not a refinement.
            use_density_fitting: Build the integrals from the mean field's
                density-fitting factors, :math:`(pq|rs) = \sum_P L_{P,pq} L_{P,rs}`,
                so the reference carries the same fitting error as moments built
                from that mean field. It still stores ``nmo**4`` numbers.

        Returns:
            The exact self-energy, with PySCF's static exchange minus the
            starting potential.

        Raises:
            ValidationError: If the mean field is not a supported restricted
                reference, an argument has the wrong type, or
                ``use_density_fitting`` is asked of a mean field without
                ``with_df``.
        """

        adapter = RestrictedPySCFAdapter(mean_field, frozen=frozen)
        if not isinstance(screening, Screening):
            raise ValidationError("screening must be a Screening value")
        if _check.boolean(use_density_fitting, "use_density_fitting"):
            if getattr(mean_field, "with_df", None) is None:
                raise ValidationError(
                    "use_density_fitting needs a mean field with with_df; "
                    "call mean_field.density_fit() first"
                )
            factors = adapter.build_density_fitted_integrals()
            eri = np.einsum("Ppq,Prs->pqrs", factors, factors, optimize=True)
        else:
            eri = adapter.build_full_integrals()
        return cls.from_integrals(
            adapter.reference,
            eri,
            screening=screening,
            static_correction=adapter.build_static_self_energy_correction(),
        )

    def cayley_moments(
        self,
        mapping: CayleyMap,
        n_max: int,
    ) -> dict[Sector, ComplexArray]:
        """Exact hole and particle Cayley moments up to order ``n_max``.

        Args:
            mapping: Cayley map of both sectors.
            n_max: Highest moment order.

        Returns:
            A dict from :class:`Sector` to a read-only array of shape
            ``(n_max + 1, nmo, nmo)``.
        """

        return {
            Sector.HOLE: self.hole.cayley_moments(mapping, n_max),
            Sector.PARTICLE: self.particle.cayley_moments(mapping, n_max),
        }

    def time_ordered_self_energy(
        self,
        frequency: ArrayLike,
        *,
        eta: float = 1.0e-8,
    ) -> ComplexArray:
        """Evaluate both sectors on the real axis with the time-ordered shifts.

        Args:
            frequency: Real energies in Hartree, a scalar or an array.
            eta: Broadening in Hartree.

        Returns:
            The read-only self-energy matrix at each frequency.
        """

        result = self.hole.evaluate_time_ordered(
            frequency,
            eta=eta,
        ) + self.particle.evaluate_time_ordered(frequency, eta=eta)
        result.setflags(write=False)
        return result

    def diagonal_quasiparticle_energy(
        self,
        orbital_position: int,
        *,
        eta: float = 1.0e-8,
        tolerance: float = 1.0e-10,
        max_iterations: int = 100,
    ) -> float:
        """Solve PySCF's diagonal quasiparticle equation for one orbital by Newton's method.

        It follows :mod:`pyscf.gw.gw_exact` and finds one root; diagonalizing
        the upfolded Hamiltonian gives every satellite as well.

        Args:
            orbital_position: Active orbital index, from zero.
            eta: Time-ordered broadening in Hartree.
            tolerance: Newton convergence threshold in Hartree.
            max_iterations: Most Newton iterations.

        Returns:
            The quasiparticle energy in Hartree.

        Raises:
            ValidationError: If the orbital is out of range or Newton does not
                reach a finite root.
        """

        orbital = _check.nonnegative_integer(orbital_position, "orbital_position")
        if orbital >= self.reference.nmo:
            raise ValidationError("orbital_position is out of range")
        newton_tolerance = _check.positive_real(tolerance, "tolerance")
        iterations = _check.positive_integer(max_iterations, "max_iterations")
        reference_energy = float(self.reference.mo_energy[orbital])
        static = float(self.static_correction[orbital, orbital].real)

        def quasiparticle(omega: float) -> float:
            sigma = self.time_ordered_self_energy(float(omega), eta=eta)
            return float(omega - reference_energy - sigma[orbital, orbital].real - static)

        try:
            root = newton(
                quasiparticle,
                reference_energy,
                tol=newton_tolerance,
                maxiter=iterations,
            )
        except RuntimeError as error:
            raise ValidationError(
                f"diagonal quasiparticle root did not converge for orbital {orbital}"
            ) from error
        if not math.isfinite(float(root)):
            raise ValidationError("diagonal quasiparticle solver returned a non-finite root")
        return float(root)
