"""Cayley moments of the G0W0 self-energy, integrated around a contour.

Orders 0 to ``n_conserved + 1`` are built for the hole and the particle sector.
The realization conserves the first ``n_conserved + 1`` of them and uses the
spare order ``n_conserved + 1`` to choose its terminal closure. Energies are in
Hartree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Literal

from ._helpers.cayley import CayleyMap
from ._helpers.moments.build import _checked_options, _log_options
from ._helpers.moments.ladder import _automatic_n_q_moments
from ._helpers.pyscf import RestrictedPySCFAdapter
from ._helpers.types import Screening, Sector
from ._helpers.validate import FloatArray
from .contour import EllipseContour
from .screening import ProjectedRPAResolvent
from .tools.logger import Stages, summarized
from .tools.parallel import resolve_native_threads
from .tools.records.contour import ContourSectorMoments, SpectralBounds
from .tools.records.moments import AutomaticNQDiagnostics, AutomaticNQSectorDiagnostics


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class G0W0CayleyMoments:
    """Hole and particle Cayley moments and what they were built from.

    Attributes:
        adapter: The validated PySCF reference.
        resolvent: Projected resolvent the contour sampled, with the gaps, the
            coupling and the three-index factors.
        rpa_spectral_bounds: Certified interval of the RPA excitation energies.
        spectral_bound: ``"spectral"`` or ``"frobenius"``, how that interval was bounded.
        contour: The ellipse the quadrature ran on.
        mapping: Cayley map shared by both sectors.
        hole: Hole-sector moments through ``n_max``.
        particle: Particle-sector moments through ``n_max``.
        n_conserved: Highest order the realization conserves.
        n_q: Contour nodes used.
        automatic_n_q_diagnostics: History of the ``N_q`` ladder; ``None`` for fixed ``n_q``.
        static_correction: :math:`v_x^{HF} - v_{MF}` in the active MO basis, built
            while the density-fitting tensor was in memory; ``None`` makes the
            realization ask the adapter.
    """

    adapter: RestrictedPySCFAdapter = field(repr=False, compare=False)
    resolvent: ProjectedRPAResolvent = field(repr=False, compare=False)
    rpa_spectral_bounds: SpectralBounds
    spectral_bound: Literal["frobenius", "spectral"]
    contour: EllipseContour
    mapping: CayleyMap
    hole: ContourSectorMoments = field(repr=False)
    particle: ContourSectorMoments = field(repr=False)
    n_conserved: int
    n_q: int
    automatic_n_q_diagnostics: AutomaticNQDiagnostics | None = None
    static_correction: FloatArray | None = field(default=None, repr=False, compare=False)

    @property
    def chemical_potential(self) -> float:
        """Centre of the Cayley map in Hartree."""

        return self.adapter.reference.chemical_potential

    @property
    def n_max(self) -> int:
        """Highest moment order built, ``n_conserved + 1``."""

        return self.n_conserved + 1

    @property
    def conjugate_paired(self) -> bool:
        """Always ``True``: the contour nodes are solved in conjugate pairs."""

        return True

    @property
    def quadrature_rule(self) -> str:
        """``"trapezoid"`` for ``n_q="auto"``, whose nodes nest, else ``"midpoint"``."""

        return "midpoint" if self.automatic_n_q_diagnostics is None else "trapezoid"

    @property
    def n_solved(self) -> int:
        """Resolvent solves of the final grid: one per conjugate pair and real node."""

        return self.n_q // 2 + (1 if self.quadrature_rule == "trapezoid" else 0)

    @property
    def interaction_backend(self) -> str:
        """How the interaction factors were built: ``"density-fitting"`` or ``"exact"``."""

        return self.resolvent.interaction_backend

    @classmethod
    def build(
        cls,
        mean_field: Any,
        *,
        n_conserved: int,
        n_q: int | Literal["auto"],
        omega_p: float = 0.5,
        n_q_tolerance: float = 1.0e-7,
        frozen: int = 0,
        spectral_bound: Literal["frobenius", "spectral"] = "spectral",
        contour_orbital_block_size: int = 1,
        n_workers: int = 1,
        native_threads: int | Literal["auto"] = "auto",
        screening: Screening | Literal["rpa", "tda"] = Screening.RPA,
        verbose: int = 1,
    ) -> G0W0CayleyMoments:
        """Build the moments; arguments and result as for :func:`build_cayley_moments`."""

        options = _checked_options(
            n_conserved=n_conserved,
            n_q=n_q,
            n_q_tolerance=n_q_tolerance,
            verbose=verbose,
            omega_p=omega_p,
            spectral_bound=spectral_bound,
            contour_orbital_block_size=contour_orbital_block_size,
            screening=screening,
        )
        native_threads = resolve_native_threads(native_threads, n_workers)
        stages = Stages(options.verbose, LOGGER)
        with stages.stage("reference preparation"):
            adapter = RestrictedPySCFAdapter(mean_field, frozen=frozen)
        _log_options(
            stages,
            mean_field,
            adapter,
            options,
            spectral_bound=spectral_bound,
            n_workers=n_workers,
            native_threads=native_threads,
        )
        mapping = CayleyMap(
            center=adapter.reference.chemical_potential,
            scale=options.omega_p,
        )
        # Built before the DF factors, so the realization never needs the mean field.
        static_correction = adapter.build_static_self_energy_correction()
        resolvent = ProjectedRPAResolvent.from_adapter(
            adapter,
            screening=options.screening,
            spectral_bound_strategy=spectral_bound,
            verbose=verbose,
            native_threads=native_threads,
        )
        rpa_bounds = SpectralBounds(
            lower=resolvent.spectral_lower_bound,
            upper=resolvent.spectral_upper_bound,
            source=resolvent.spectral_bound_source,
        )
        # The ellipse does not depend on N_q, so every rung of the ladder shares it.
        with stages.stage("contour selection"):
            contour = EllipseContour.from_equioscillation(
                resolvent,
                adapter.reference,
                mapping,
            )

        def evaluate(nodes: int, rule: str = "midpoint") -> dict[Sector, ContourSectorMoments]:
            return contour.moments(
                resolvent,
                mapping,
                options.order,
                nodes,
                rule=rule,
                orbital_block_size=options.orbital_block_size,
                n_workers=n_workers,
                native_threads=native_threads,
                verbose=verbose,
            )

        if options.automatic_n_q:
            sectors, automatic_diagnostics = _automatic_n_q_moments(
                evaluate,
                options.order,
                tolerance=options.n_q_rtol,
                verbose=verbose,
            )
            selected_n_q = automatic_diagnostics.selected_n_q
        else:
            assert options.count is not None
            sectors = evaluate(options.count)
            automatic_diagnostics = None
            selected_n_q = options.count
        return cls(
            adapter=adapter,
            resolvent=resolvent,
            rpa_spectral_bounds=rpa_bounds,
            spectral_bound=spectral_bound,
            contour=contour,
            mapping=mapping,
            hole=sectors[Sector.HOLE],
            particle=sectors[Sector.PARTICLE],
            n_conserved=options.conserved,
            n_q=selected_n_q,
            automatic_n_q_diagnostics=automatic_diagnostics,
            static_correction=static_correction,
        )


@summarized("build_cayley_moments", LOGGER)
def build_cayley_moments(
    mean_field: Any,
    *,
    n_conserved: int,
    n_q: int | Literal["auto"],
    omega_p: float = 0.5,
    n_q_tolerance: float = 1.0e-7,
    frozen: int = 0,
    spectral_bound: Literal["frobenius", "spectral"] = "spectral",
    contour_orbital_block_size: int = 1,
    n_workers: int = 1,
    native_threads: int | Literal["auto"] = "auto",
    screening: Screening | Literal["rpa", "tda"] = Screening.RPA,
    verbose: int = 1,
) -> G0W0CayleyMoments:
    r"""Build the hole and particle Cayley moments by contour quadrature.

    The auxiliary kernel is integrated by the midpoint rule around an ellipse
    that encloses the RPA excitation energies and excludes the Cayley poles, then
    contracted with the three-index factors. Orders 0 to ``n_conserved + 1`` are
    built; the last is the spare that chooses the closure.

    Args:
        mean_field: Converged restricted PySCF mean field (RHF or RKS). One from
            ``mean_field.density_fit()`` uses its ``with_df`` factors and never
            forms ``(pq|rs)``; any other uses the exact four-index integrals.
        n_conserved: Highest moment order the realization conserves.
        n_q: Contour nodes, even, for the midpoint rule; or ``"auto"`` to double
            them from 128 up to 4096 until the moments converge and pass the
            feasibility checks, on the trapezoidal rule so each doubling reuses
            every node already solved.
        omega_p: Scale of the Cayley map in Hartree.
        n_q_tolerance: Relative change per order,
            :math:`\lVert C_{fine} - C_{coarse}\rVert \le tol \lVert C_{fine}\rVert`,
            at which ``"auto"`` stops.
        frozen: Number of lowest occupied orbitals to freeze.
        spectral_bound: ``"spectral"`` bounds the excitation energies with the
            exact two-norm of the coupling, at about one node's cost;
            ``"frobenius"`` uses its Frobenius norm, which overshoots more as the
            system grows and slows the quadrature.
        contour_orbital_block_size: Orbitals contracted together; memory against speed.
        n_workers: Python workers over contour nodes and orbital blocks. Results
            do not depend on it at a fixed ``native_threads``.
        native_threads: BLAS and LAPACK threads per worker, or ``"auto"`` to share
            the cores; see :func:`~cayleygw.tools.parallel.resolve_native_threads`.
        screening: ``"rpa"`` for direct RPA or ``"tda"`` for the Tamm-Dancoff
            approximation, which drops the de-excitation block and gives
            different energies.
        verbose: ``0`` warnings only, ``1`` the log, ``2`` adds progress counts.

    Returns:
        The moments of both sectors with the reference, resolvent, contour and
        Cayley map they were built from.

    Raises:
        ValidationError: If an argument is invalid, the contour geometry is
            refused, or ``"auto"`` does not converge by 4096 nodes.
        RefusalError: If a contour node lies too close to a pole.
    """

    return G0W0CayleyMoments.build(
        mean_field,
        n_conserved=n_conserved,
        n_q=n_q,
        omega_p=omega_p,
        n_q_tolerance=n_q_tolerance,
        frozen=frozen,
        spectral_bound=spectral_bound,
        contour_orbital_block_size=contour_orbital_block_size,
        n_workers=n_workers,
        native_threads=native_threads,
        screening=screening,
        verbose=verbose,
    )


__all__ = [
    "AutomaticNQDiagnostics",
    "AutomaticNQSectorDiagnostics",
    "G0W0CayleyMoments",
    "build_cayley_moments",
]
