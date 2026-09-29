"""Node solves, quadrature and contraction of :meth:`EllipseContour.moments`."""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

from ...tools.parallel import evaluate_by_index
from ..types import Sector
from ..validate import ComplexArray, FloatArray, IntArray
from .quadrature import _hermitian_zeroth, _raise_lowest_node_failure, _symmetrize_into

if TYPE_CHECKING:
    from ...screening import ProjectedRPAResolvent
    from ...tools.logger import Stages
    from ..cayley import CayleyMap


def contract_external_moments(
    external_factors: FloatArray,
    auxiliary: ComplexArray,
) -> ComplexArray:
    r"""Return :math:`C_k = B A_k B^\dagger` for every order :math:`k`.

    ``external_factors`` is the real ``(nmo, naux)`` :math:`B` and
    ``auxiliary`` the C-contiguous complex ``(n_max + 1, naux, naux)`` stack
    :math:`A_k`. A real :math:`B` against a complex :math:`A_k` is two real
    products per side, half the arithmetic of a complex one; the first reads
    :math:`A_k` through its real view without a copy. The result differs from
    a complex product by a few units in the last place.
    """

    factors = np.ascontiguousarray(external_factors, dtype=np.float64)
    orders, naux, _ = auxiliary.shape
    interleaved = auxiliary.view(np.float64).reshape(orders, naux, 2 * naux)
    half = np.matmul(factors, interleaved).view(np.complex128)
    real_part = np.matmul(np.ascontiguousarray(half.real), factors.T)
    imaginary_part = np.matmul(np.ascontiguousarray(half.imag), factors.T)
    result = np.empty(real_part.shape, dtype=np.complex128)
    result.real = real_part
    result.imag = imaginary_part
    return result


def _sample_nodes(
    resolvent: ProjectedRPAResolvent,
    nodes: ComplexArray,
    stages: Stages,
    workers: int,
    native: int,
    n_points: int,
) -> ComplexArray:
    """Solve the projected resolvent at one node of each conjugate pair of ``n_points`` nodes.

    Returns the node axis ``(nodes // 2, naux, naux)``.
    """

    node_count = nodes.size
    representative_indices = tuple(range(node_count // 2))
    local_node_count = len(representative_indices)
    naux = resolvent.factors.shape[0]
    # A real coupling makes partners conjugate, so contract_block folds each one in algebraically.
    projected_working = np.empty(
        (local_node_count, naux, naux),
        dtype=np.complex128,
    )
    # One stage for the whole sweep, so its time is wall clock under parallel workers.
    sampling = stages.stage(
        "contour node solves",
        nodes=n_points,
        solved=local_node_count,
        n_aux=naux,
    )
    sampling.__enter__()
    # Stream each result into its node's slot; collecting them first holds three node axes.

    def _store(node_index: int, evaluation: ComplexArray) -> None:
        _symmetrize_into(evaluation, node_index, projected_working[node_index])

    # A running count, so a job killed mid-sweep still shows how far it got.
    sampled_nodes = 0

    def _note_node(_index: int) -> None:
        nonlocal sampled_nodes
        sampled_nodes += 1
        stages.progress("contour node solves", sampled_nodes, local_node_count)

    _, failures = evaluate_by_index(
        lambda index: resolvent.woodbury(nodes[index]),
        representative_indices,
        n_workers=workers,
        native_threads=native,
        sink=_store,
        on_complete=_note_node,
    )
    if failures:
        _raise_lowest_node_failure(failures, nodes)
    sampling.__exit__(None, None, None)
    return projected_working


def _contract_moments(
    resolvent: ProjectedRPAResolvent,
    mapping: CayleyMap,
    internal: dict[Sector, IntArray],
    nodes: ComplexArray,
    weights: ComplexArray,
    projected_working: ComplexArray,
    order: int,
    block_size: int,
    stages: Stages,
    workers: int,
    native: int,
) -> dict[Sector, ComplexArray]:
    """Integrate and contract every orbital block into one total per sector."""

    node_count = nodes.size
    local_node_count = projected_working.shape[0]
    naux = resolvent.factors.shape[0]
    reference = resolvent.reference
    order_count = order + 1
    quadrature_weights = float(resolvent.screening.spectral_parameter_power) * (
        0.5 * weights
    )
    projected_matrix = projected_working.reshape(
        projected_working.shape[0],
        naux * naux,
    )
    moment_shape = (order_count, reference.nmo, reference.nmo)
    # One running total per sector; a per-orbital stack is the largest array at scale.
    totals_by_sector: dict[Sector, ComplexArray] = {
        sector: np.zeros(moment_shape, dtype=np.complex128)
        for sector in Sector
    }
    parked: dict[Sector, dict[int, ComplexArray]] = {
        sector: {} for sector in Sector
    }
    next_start = {sector: 0 for sector in Sector}
    fold_lock = Lock()

    def fold_block(
        sector: Sector,
        start: int,
        contributions: ComplexArray,
    ) -> None:
        """Add a block to its sector's total once every earlier block is in.

        Parking blocks until their predecessors arrive and adding one orbital
        at a time keeps the total bit-identical at every worker count and
        block size.
        """

        with fold_lock:
            parked[sector][start] = contributions
            while next_start[sector] in parked[sector]:
                for contribution in parked[sector].pop(next_start[sector]):
                    totals_by_sector[sector] += contribution
                next_start[sector] += block_size

    # One work list over both sectors, so every block is dispatched at once.
    work_units = [
        (sector, start)
        for sector in Sector
        for start in range(0, internal[sector].size, block_size)
    ]

    def contract_block(unit_index: int) -> int:
        """Quadrature and contraction for one orbital block."""

        sector, start = work_units[unit_index]
        sector_internal = internal[sector]
        sign = -1.0 if sector is Sector.HOLE else 1.0
        stop = min(start + block_size, sector_internal.size)
        orbital_block = sector_internal[start:stop]

        frequencies = (
            reference.mo_energy[orbital_block, None]
            + sign * nodes[None, :]
        )
        offsets = frequencies - mapping.center
        cayley_values = (offsets + 1j * mapping.scale) / (
            offsets - 1j * mapping.scale
        )
        powers = np.empty(
            (stop - start, order_count, node_count),
            dtype=np.complex128,
        )
        powers[:, 0] = 1.0
        for moment_order in range(order):
            powers[:, moment_order + 1] = (
                powers[:, moment_order] * cayley_values
            )
        power_rows = (stop - start) * order_count
        # Partner n-1-r has matrix conj(S_r): its half is conj(sum conj(p) S_r).
        upper = slice(0, local_node_count)
        lower = np.arange(
            node_count - 1,
            node_count - 1 - local_node_count,
            -1,
        )
        upper_powers = (
            powers[:, :, upper]
            * quadrature_weights[None, None, upper]
        )
        lower_powers = (
            powers[:, :, lower] * quadrature_weights[None, None, lower]
        )
        stacked_powers = np.concatenate(
            (
                upper_powers.reshape(power_rows, local_node_count),
                lower_powers.reshape(
                    power_rows,
                    local_node_count,
                ).conj(),
            ),
            axis=0,
        )
        stacked_flat = np.matmul(stacked_powers, projected_matrix)
        partner_flat = stacked_flat[power_rows:]
        np.conjugate(partner_flat, out=partner_flat)
        auxiliary_flat = stacked_flat[:power_rows]
        auxiliary_flat += partner_flat
        del stacked_powers, partner_flat
        auxiliary_block = auxiliary_flat.reshape(
            stop - start,
            order_count,
            naux,
            naux,
        )
        for position in range(auxiliary_block.shape[0]):
            _hermitian_zeroth(
                auxiliary_block[position],
                position,
                "blocked auxiliary-space",
            )

        block_contributions = np.empty(
            (stop - start,) + moment_shape, dtype=np.complex128
        )
        for local_position, orbital in enumerate(orbital_block):
            external_factors = resolvent.factors[:, :, orbital].T
            contribution = contract_external_moments(
                external_factors,
                auxiliary_block[local_position],
            )
            # Gate each orbital's C_0 before the sum can hide it.
            _hermitian_zeroth(
                contribution,
                start + local_position,
                "blocked physical-space",
            )
            block_contributions[local_position] = contribution
        fold_block(sector, start, block_contributions)
        return int(stop - start)

    with stages.stage(
        "quadrature and moment contraction",
        n_max=order,
        n_orbitals=sum(internal[sector].size for sector in Sector),
    ):
        _, block_failures = evaluate_by_index(
            contract_block,
            range(len(work_units)),
            n_workers=workers,
            native_threads=native,
        )
        if block_failures:
            raise block_failures[0][1]
    return totals_by_sector
