"""The restricted closed-shell reference taken from a PySCF mean field.

The adapter supplies the active orbital energies and occupations, the Coulomb
integrals as density-fitted three-index factors (or as the exact four-index
tensor for small checks), and the static self-energy of the upfolded
Hamiltonian.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pyscf import ao2mo, lib, scf

from . import tolerances as limits
from .errors import ValidationError
from .validate import FloatArray, IntArray, _check

LOGGER = logging.getLogger(__name__)


def _sealed(values: Any, dtype: type = np.float64) -> NDArray[Any]:
    """Return an independent C-contiguous read-only copy of ``values``."""

    array = np.array(values, dtype=dtype, order="C", copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class RestrictedMolecularReference:
    """Validated active-orbital data of a closed-shell PySCF reference.

    Arrays are independent, C-contiguous and read-only.

    Attributes:
        mo_coeff: Real AO-to-active-MO coefficients, shape ``(nao, nmo)``.
        mo_energy: Active MO energies in Hartree, shape ``(nmo,)``.
        occupied_positions: Positions of the occupied orbitals in the active arrays.
        virtual_positions: Positions of the virtual orbitals in the active arrays.
        chemical_potential: Cayley centre in Hartree, the HOMO-LUMO midgap.
        homo_energy: HOMO energy of the unfrozen reference in Hartree.
        lumo_energy: LUMO energy of the unfrozen reference in Hartree.
        gap: HOMO-LUMO gap of the unfrozen reference in Hartree.
    """

    mo_coeff: FloatArray = field(repr=False)
    mo_energy: FloatArray = field(repr=False)
    occupied_positions: IntArray = field(repr=False)
    virtual_positions: IntArray = field(repr=False)
    chemical_potential: float
    homo_energy: float
    lumo_energy: float
    gap: float

    @property
    def nmo(self) -> int:
        """Number of active spatial orbitals."""

        return int(self.mo_coeff.shape[1])

    @property
    def nocc(self) -> int:
        """Number of active doubly occupied orbitals."""

        return int(self.occupied_positions.size)

    @property
    def nvir(self) -> int:
        """Number of active virtual orbitals."""

        return int(self.virtual_positions.size)


class RestrictedPySCFAdapter:
    """Validate a restricted closed-shell PySCF mean field and copy its active data.

    Open-shell, generalized, periodic and complex-orbital references are not
    supported. The active data are copied, so later changes to the PySCF object
    do not reach :attr:`reference`.

    Args:
        mean_field: Converged molecular RHF or RKS with real orbitals and
            occupations of zero or two.
        frozen: Number of lowest occupied orbitals to freeze.

    Attributes:
        mean_field: The PySCF mean field.
        frozen_indices: PySCF MO indices left out of the active space.
        active_indices: PySCF MO index of each active orbital, in order.
        reference: The validated :class:`RestrictedMolecularReference`.

    Raises:
        ValidationError: If the reference is not a converged closed-shell RHF or
            RKS, the gap is too small, or ``frozen`` is not a count that leaves
            an occupied orbital.
    """

    def __init__(self, mean_field: Any, *, frozen: int = 0) -> None:
        if not isinstance(mean_field, scf.hf.RHF) or isinstance(mean_field, scf.rohf.ROHF):
            raise ValidationError("only molecular RHF and RKS references are supported")
        if int(mean_field.mol.spin) != 0:
            raise ValidationError("a closed-shell molecule with mol.spin == 0 is required")
        if not bool(getattr(mean_field, "converged", False)):
            raise ValidationError("the PySCF mean-field calculation is not converged")

        coefficients = np.asarray(mean_field.mo_coeff, dtype=np.float64)
        energies = np.asarray(mean_field.mo_energy, dtype=np.float64)
        occupations = np.asarray(mean_field.mo_occ, dtype=np.float64)
        nmo = coefficients.shape[1]
        occupied_mask = np.isclose(
            occupations,
            2.0,
            rtol=0.0,
            atol=limits.REFERENCE_TOLERANCE,
        )
        virtual_mask = np.isclose(
            occupations,
            0.0,
            rtol=0.0,
            atol=limits.REFERENCE_TOLERANCE,
        )
        if not np.all(occupied_mask | virtual_mask):
            raise ValidationError("fractional or singly occupied orbitals are not supported")
        occupied_indices = np.flatnonzero(occupied_mask)
        virtual_indices = np.flatnonzero(virtual_mask)
        if occupied_indices.size == 0 or virtual_indices.size == 0:
            raise ValidationError("the reference must contain occupied and virtual orbitals")

        homo_energy = float(np.max(energies[occupied_indices]))
        lumo_energy = float(np.min(energies[virtual_indices]))
        gap = lumo_energy - homo_energy
        if gap <= limits.REFERENCE_TOLERANCE:
            raise ValidationError(
                f"the restricted HOMO-LUMO gap is too small; gap={gap:.3e} Hartree"
            )

        count = _check.nonnegative_integer(frozen, "frozen")
        if count >= occupied_indices.size:
            raise ValidationError(
                f"frozen must leave an occupied orbital; got {count} of {occupied_indices.size}"
            )
        ordered_occupied = occupied_indices[np.argsort(energies[occupied_indices], kind="stable")]
        frozen_indices = tuple(int(index) for index in ordered_occupied[:count])
        active_mask = np.ones(nmo, dtype=bool)
        active_mask[list(frozen_indices)] = False
        active_indices = np.flatnonzero(active_mask)
        active_occupied = occupied_mask[active_indices]
        active_virtual = virtual_mask[active_indices]

        self.mean_field = mean_field
        self.frozen_indices: tuple[int, ...] = frozen_indices
        self.active_indices: tuple[int, ...] = tuple(int(index) for index in active_indices)
        self.reference = RestrictedMolecularReference(
            mo_coeff=_sealed(coefficients[:, active_indices]),
            mo_energy=_sealed(energies[active_indices]),
            occupied_positions=_sealed(np.flatnonzero(active_occupied), np.int64),
            virtual_positions=_sealed(np.flatnonzero(active_virtual), np.int64),
            chemical_potential=0.5 * (homo_energy + lumo_energy),
            homo_energy=homo_energy,
            lumo_energy=lumo_energy,
            gap=gap,
        )

    def build_density_fitted_integrals(
        self,
        *,
        native_threads: int | None = None,
        verbose: int = 0,
    ) -> FloatArray:
        r"""Build the density-fitted three-index factors in the active MOs.

        The mean field's ``with_df`` Cholesky vectors are unpacked and
        transformed one auxiliary block at a time into a preallocated tensor, so
        storage is ``O(naux * nmo**2)`` and no four-index tensor is formed.

        Args:
            native_threads: BLAS threads for the ``_cderi`` build and the AO-to-MO
                transform; ``None`` keeps the process setting. Factors built at
                different counts are not bit-identical.
            verbose: ``0`` warnings only, ``1`` the log, ``2`` adds progress
                counts of the transform.

        Returns:
            Read-only :math:`L_{P,pq}` with
            :math:`(pq|rs) \approx \sum_P L_{P,pq} L_{P,rs}`, shape
            ``(naux, nmo, nmo)``.
        """

        from ..tools.logger import Stages
        from ..tools.parallel import limited_native_threads

        stages = Stages(verbose, LOGGER)
        df_object = self.mean_field.with_df
        coefficients = self.reference.mo_coeff
        nmo = self.reference.nmo
        # The thread count also covers the three-centre integrals and the metric Cholesky.
        with (
            nullcontext() if native_threads is None else limited_native_threads(int(native_threads))
        ):
            if getattr(df_object, "_cderi", None) is None:
                with stages.stage("three-centre integrals"):
                    df_object.build()
            naux = int(df_object.get_naoaux())
            factor_store = np.empty((naux, nmo, nmo), dtype=np.float64)
            offset = 0
            with stages.stage("AO-to-MO transform", n_aux=naux, n_mo=nmo):
                for packed_block in df_object.loop():
                    ao_block = lib.unpack_tril(np.asarray(packed_block))
                    stop = offset + int(ao_block.shape[0])
                    factor_store[offset:stop] = np.einsum(
                        "Puv,up,vq->Ppq",
                        ao_block,
                        coefficients,
                        coefficients,
                        optimize=True,
                    )
                    offset = stop
                    # Counted in auxiliary functions: PySCF sizes its blocks by memory.
                    stages.progress("AO-to-MO transform", offset, naux)
        factor_store.setflags(write=False)
        return factor_store

    def build_full_integrals(self) -> FloatArray:
        """Build the full active-MO Coulomb tensor ``(pq|rs)`` in chemists' notation.

        Storage scales as ``nmo**4``, so this is for small references and tests.

        Returns:
            Read-only tensor in Hartree, shape ``(nmo, nmo, nmo, nmo)``.
        """

        nmo = self.reference.nmo
        transformed = ao2mo.kernel(
            self.mean_field.mol,
            self.reference.mo_coeff,
            compact=False,
        )
        return _sealed(np.asarray(transformed).reshape(nmo, nmo, nmo, nmo))

    def build_static_self_energy_correction(self) -> FloatArray:
        r"""Build the static self-energy :math:`v_x^{HF} - v_{MF}` in the active MO basis.

        As in :mod:`pyscf.gw.gw_exact`, ``v_mf = get_veff() - get_j()`` and
        ``v_x = -0.5 * get_k(dm)``, with the density of every occupied orbital,
        frozen ones included. For a Hartree-Fock mean field it returns zeros
        without evaluating anything.

        Returns:
            Read-only real symmetric matrix; zero to SCF precision for
            ``RKS(xc="hf")``.
        """

        mean_field = self.mean_field
        coefficients = self.reference.mo_coeff
        if not hasattr(mean_field, "xc"):
            # Zero for Hartree-Fock; evaluating it could rebuild a released _cderi tensor.
            nactive = int(coefficients.shape[1])
            return _sealed(np.zeros((nactive, nactive)))
        density = mean_field.make_rdm1(mean_field.mo_coeff, mean_field.mo_occ)
        mean_field_potential = mean_field.get_veff(mean_field.mol, density) - mean_field.get_j(
            mean_field.mol, density
        )
        exact_exchange = -0.5 * mean_field.get_k(mean_field.mol, density)
        correction = coefficients.T @ (exact_exchange - mean_field_potential) @ coefficients
        correction = np.asarray(correction, dtype=np.float64)
        return _sealed(0.5 * (correction + correction.T))
