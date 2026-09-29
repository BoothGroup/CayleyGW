"""The projected resolvent of the particle-hole response at a contour node.

Every node costs one solve of auxiliary dimension through the Woodbury
identity. The resolvent also holds the certified bounds on the excitation
energies that place the contour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Literal, Self

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from ._helpers import tolerances as limits
from ._helpers.errors import ValidationError
from ._helpers.pyscf import RestrictedMolecularReference, RestrictedPySCFAdapter
from ._helpers.screening.resolvent import (
    _largest_squared_singular_value,
    _raise_node_error,
    _sealed_coupling,
    _weighted_gram,
)
from ._helpers.screening.response import exact_coulomb_factors
from ._helpers.types import Screening
from ._helpers.validate import ComplexArray, FloatArray, _check
from .tools.logger import Stages

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, eq=False)
class ProjectedRPAResolvent:
    r"""Projected resolvent of a particle-hole response, solved by the Woodbury identity.

    It evaluates :math:`V^\dagger(\zeta^p I - M)^{-1} V` at complex nodes
    :math:`\zeta`, where both screening models give :math:`M = D^p + c\,VV^\dagger`
    with eigenvalues :math:`\Omega_\nu^p`. Direct RPA has :math:`p = c = 2` and
    :math:`V = D^{1/2}(L^S)^\dagger`; the TDA has :math:`p = c = 1` and
    :math:`V = (L^S)^\dagger`. :math:`V` may be rank deficient, since no
    auxiliary inverse is formed.

    Attributes:
        particle_hole_gaps: Diagonal of :math:`D`, strictly positive, in Hartree.
        v_matrix: Real coupling :math:`V`, shape ``(ntransition, naux)``.
        spectral_bound_strategy: ``"spectral"`` bounds with the exact largest
            singular value of :math:`V`, at about one node's cost;
            ``"frobenius"`` with its Frobenius norm, which overshoots more as
            the system grows and slows the quadrature.
        screening: Response model, which sets :math:`p` and :math:`c`.
        reference: Active reference, set by :meth:`from_adapter`; else ``None``.
        factors: Three-index factors :math:`L_{P,pq}`, shape ``(naux, nmo, nmo)``,
            set by :meth:`from_adapter`; else ``None``.
        interaction_backend: ``"density-fitting"`` or ``"exact"``, set by
            :meth:`from_adapter`; else ``None``.
        spectral_lower_bound: Certified lower bound on the excitation energies, in Hartree.
        spectral_upper_bound: Certified upper bound on the excitation energies, in Hartree.
        spectral_bound_source: How the bounds were obtained.
    """

    particle_hole_gaps: FloatArray = field(repr=False)
    v_matrix: FloatArray = field(repr=False)
    spectral_bound_strategy: Literal["frobenius", "spectral"] = "spectral"
    screening: Screening = Screening.RPA
    reference: RestrictedMolecularReference | None = field(default=None, repr=False)
    factors: FloatArray | None = field(default=None, repr=False)
    interaction_backend: str | None = None
    spectral_lower_bound: float = field(init=False)
    spectral_upper_bound: float = field(init=False)
    spectral_bound_source: str = field(init=False)
    _diagonal: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.spectral_bound_strategy not in ("frobenius", "spectral"):
            raise ValidationError(
                "spectral_bound_strategy must be 'frobenius' or 'spectral'"
            )
        if not isinstance(self.screening, Screening):
            raise ValidationError("screening must be a Screening value")
        gaps = _check.readonly_real(self.particle_hole_gaps, "particle_hole_gaps")
        coupling = _sealed_coupling(self.v_matrix)
        if gaps.ndim != 1 or gaps.size == 0:
            raise ValidationError(
                "particle_hole_gaps must be a nonempty vector"
            )
        if np.any(gaps <= 0.0):
            raise ValidationError(
                "particle_hole_gaps must be strictly positive"
            )
        if coupling.ndim != 2 or coupling.shape[0] != gaps.size:
            raise ValidationError(
                "v_matrix must have shape (ntransition, naux)"
            )
        power = self.screening.spectral_parameter_power
        multiplier = self.screening.kernel_multiplier
        diagonal = _check.readonly_real(np.power(gaps, power), "response diagonal")

        # Weyl bounds M = D**p + c V V^dagger; the p-th root maps back to energies.
        lower_bound = float(np.min(gaps))
        maximum_diagonal = float(np.max(diagonal))
        if self.spectral_bound_strategy == "frobenius":
            norm_squared = float(np.sum(np.abs(coupling) ** 2))
            bound_source = "D_min and Frobenius-norm upper bound"
        else:
            # sigma_1(V)**2 from the naux-by-naux Gram, not an SVD of V.
            norm_squared = _largest_squared_singular_value(coupling)
            bound_source = "D_min and spectral-norm upper bound"
        upper_bound = maximum_diagonal + multiplier * norm_squared
        if power == 2:
            upper_bound = math.sqrt(upper_bound)
        object.__setattr__(self, "particle_hole_gaps", gaps)
        object.__setattr__(self, "v_matrix", coupling)
        object.__setattr__(self, "_diagonal", diagonal)
        object.__setattr__(self, "spectral_lower_bound", lower_bound)
        object.__setattr__(self, "spectral_upper_bound", upper_bound)
        object.__setattr__(self, "spectral_bound_source", bound_source)

    @classmethod
    def from_adapter(
        cls,
        adapter: RestrictedPySCFAdapter,
        *,
        screening: Screening = Screening.RPA,
        spectral_bound_strategy: Literal["frobenius", "spectral"] = "spectral",
        verbose: int = 0,
        native_threads: int | None = None,
    ) -> Self:
        r"""Build the resolvent of a reference without solving the response eigenproblem.

        Only the gaps, the three-index factors and the coupling
        :math:`V^S = D^{(p-1)/2}(L^S)^\dagger` with :math:`L^S = \sqrt{2}\,L` are
        formed. A mean field from PySCF's ``density_fit()`` supplies its
        ``with_df`` factors; any other takes the exact four-index route.

        Args:
            adapter: Real closed-shell restricted reference.
            screening: Response model; direct RPA by default.
            spectral_bound_strategy: ``"spectral"`` or ``"frobenius"``; see the class.
            verbose: ``0`` warnings only, ``1`` the log, ``2`` adds progress counts.
            native_threads: BLAS threads for the density-fitted transform;
                ``None`` keeps the process setting.

        Returns:
            The resolvent, carrying the reference and factors the contour reads.

        Raises:
            ValidationError: If ``screening`` is not a :class:`Screening`.
        """

        if not isinstance(screening, Screening):
            raise ValidationError("screening must be a Screening value")
        stages = Stages(verbose, LOGGER)
        reference = adapter.reference
        density_fitted = getattr(adapter.mean_field, "with_df", None) is not None
        with stages.stage(
            "density-fitted interaction factors"
            if density_fitted
            else "exact four-index interaction factors",
            n_mo=reference.nmo,
        ):
            if density_fitted:
                factors = adapter.build_density_fitted_integrals(
                    native_threads=native_threads,
                    verbose=verbose,
                )
            else:
                factors = exact_coulomb_factors(adapter.build_full_integrals())
        with stages.stage("transition-factor assembly"):
            occupied = np.asarray(reference.occupied_positions)
            virtual = np.asarray(reference.virtual_positions)
            gaps = (
                reference.mo_energy[virtual][None, :]
                - reference.mo_energy[occupied][:, None]
            ).reshape(-1)
            transition_factors = factors[
                :,
                np.repeat(occupied, virtual.size),
                np.tile(virtual, occupied.size),
            ]
            # Direct RPA's squared matrix D^2 + 2 D^1/2 K D^1/2 puts D^1/2 in V.
            gap_weight = (
                np.sqrt(gaps)[:, None]
                if screening is Screening.RPA
                else 1.0
            )
            # C order: the bits of every BLAS product downstream depend on the layout.
            v_matrix = np.ascontiguousarray(
                gap_weight * (math.sqrt(2.0) * transition_factors).T
            )
            v_matrix.setflags(write=False)
        with stages.stage(
            "spectral bounds and resolvent",
            n_mo=reference.nmo,
            n_transitions=reference.nocc * reference.nvir,
            screening=screening.label,
        ):
            return cls(
                gaps,
                v_matrix,
                spectral_bound_strategy=spectral_bound_strategy,
                screening=screening,
                reference=reference,
                factors=factors,
                interaction_backend=(
                    "density-fitting" if density_fitted else "exact"
                ),
            )

    def woodbury(self, zeta: Any) -> ComplexArray:
        r"""Solve the projected resolvent at one node in the auxiliary space.

        Forms :math:`Q = V^\dagger(\zeta^p I - D^p)^{-1} V` and solves
        :math:`(I - cQ)S = Q` without an explicit inverse.

        Args:
            zeta: Finite complex node in Hartree.

        Returns:
            The projected resolvent :math:`S`, shape ``(naux, naux)``.

        Raises:
            RefusalError: If the node is too close to a free pole or to the
                certified interval of interacting poles, or the auxiliary
                system is singular.
        """

        node = _check.finite_complex(zeta, "zeta")
        power = self.screening.spectral_parameter_power
        squared = node**power
        free_distance = float(np.min(np.abs(squared - self._diagonal)))
        lower_squared = self.spectral_lower_bound**power
        upper_squared = self.spectral_upper_bound**power
        if squared.real < lower_squared:
            rpa_distance = abs(squared - lower_squared)
        elif squared.real > upper_squared:
            rpa_distance = abs(squared - upper_squared)
        else:
            rpa_distance = abs(squared.imag)
        threshold = limits.ABSOLUTE_TOLERANCE + limits.RELATIVE_TOLERANCE * max(
            abs(squared),
            float(np.max(self._diagonal)),
            upper_squared,
        )
        label = "RPA" if self.screening is Screening.RPA else "TDA"
        if free_distance <= threshold:
            _raise_node_error(
                "free particle-hole pole", node, free_distance, rpa_distance, threshold
            )
        if rpa_distance <= threshold:
            _raise_node_error(
                f"interacting {label} pole", node, free_distance, rpa_distance, threshold
            )
        inverse_free = 1.0 / (squared - self._diagonal)
        q_matrix = _weighted_gram(self.v_matrix, inverse_free)
        coefficient = (
            np.eye(q_matrix.shape[0], dtype=np.complex128)
            - self.screening.kernel_multiplier * q_matrix
        )
        factorization = lu_factor(coefficient, check_finite=False)
        if not np.all(np.diagonal(factorization[0])):
            _raise_node_error(
                "singular auxiliary system", node, free_distance, rpa_distance, threshold
            )
        # C order: the symmetry gate downstream sums its Frobenius norms in memory order.
        return np.ascontiguousarray(
            lu_solve(factorization, q_matrix, check_finite=False)
        )
