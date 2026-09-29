"""Tests of the Toeplitz backend: its realization, reclosure and unitarity certificate."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cayleygw import RefusalError, Sector, ValidationError
from cayleygw._helpers.cayley import CayleyMap
from cayleygw._helpers.tolerances import DEFAULT_TOLERANCES
from cayleygw.realization import toeplitz
from cayleygw.realization._helpers.base import _hermitian_square
from cayleygw.realization._helpers.toeplitz import _UnitarityCertificate
from cayleygw.realization.block_cmv import BlockCMVRealization
from cayleygw.realization.sector import SectorSelfEnergyRealization
from cayleygw.realization.toeplitz import ToeplitzRealization


def _atomic_moments(nodes, residues, n_max):
    """Return the moments of a positive matrix-valued atomic measure."""

    return np.asarray(
        [
            np.einsum(
                "m,mij->ij",
                np.asarray(nodes) ** order,
                residues,
                optimize=True,
            )
            for order in range(n_max + 1)
        ]
    )


def _high_order_stress_moments(n_max=30):
    """Return a clustered, strongly weighted matrix measure near termination."""

    generator = np.random.default_rng(851)
    block = 4
    atom_count = 60
    angles = np.linspace(0.02, np.pi - 0.02, atom_count) ** 1.3 / (np.pi - 0.02) ** 0.3
    nodes = np.exp(1.0j * angles)
    factors = generator.normal(size=(atom_count, block, 2)) + 1.0j * (
        generator.normal(size=(atom_count, block, 2))
    )
    factors *= np.geomspace(1.0, 1.0e-5, atom_count)[:, None, None]
    residues = np.einsum("mai,mbi->mab", factors, factors.conj(), optimize=True)
    residues /= np.trace(np.sum(residues, axis=0)).real / block
    return _atomic_moments(nodes, residues, n_max)


def test_toeplitz_avoids_inverse_defect_operations() -> None:
    source = inspect.getsource(toeplitz)

    assert "np.linalg.inv" not in source
    assert "np.linalg.pinv" not in source
    assert "inverse_roots" not in source


def test_toeplitz_conserves_noncommuting_moments_and_is_unitary() -> None:
    generator = np.random.default_rng(410)
    nodes = np.exp(1.0j * np.asarray([0.17, 0.73, 1.41, 2.28, 3.36, 4.51, 5.42]))
    factors = generator.normal(size=(nodes.size, 2, 2)) + 1.0j * (
        generator.normal(size=(nodes.size, 2, 2))
    )
    residues = np.einsum("mij,mkj->mik", factors, factors.conj(), optimize=True)
    moments = _atomic_moments(nodes, residues, 5)

    result = ToeplitzRealization.realize(moments)

    assert result.dimension == 12
    assert not result.terminal_from_moments
    np.testing.assert_allclose(result.reconstructed_moments, moments, atol=8.0e-14)
    np.testing.assert_allclose(
        result.matrix.conj().T @ result.matrix,
        np.eye(result.dimension),
        atol=5.0e-15,
    )
    assert result.diagnostics.terminal_dimension == 2
    assert result.diagnostics.determined_shift_residual < 8.0e-14


def test_physical_moments_are_a_congruence_of_the_normalized_ones() -> None:
    """``B U**k B.H`` is the support congruence of ``E.H U**k E``, and matches direct powers."""

    generator = np.random.default_rng(77)
    nodes = np.exp(1.0j * np.asarray([0.21, 0.94, 1.77, 2.65, 3.81, 5.02]))
    factors = generator.normal(size=(nodes.size, 3, 3)) + 1.0j * (
        generator.normal(size=(nodes.size, 3, 3))
    )
    residues = np.einsum("mij,mkj->mik", factors, factors.conj(), optimize=True)
    order = 4
    realization = ToeplitzRealization.realize(_atomic_moments(nodes, residues, order))
    support = np.asarray(realization.normalization.support_factor)

    normalized = realization.normalized_matrix_moments(order)
    physical = realization.matrix_moments(order)
    np.testing.assert_allclose(physical, support @ normalized @ support.conj().T, atol=1.0e-12)

    # Check against explicit powers of the matrix as well.
    matrix = np.asarray(realization.matrix)
    coupling = np.asarray(realization.coupling)
    power = np.eye(matrix.shape[0], dtype=np.complex128)
    for index in range(order + 1):
        np.testing.assert_allclose(
            physical[index], coupling @ power @ coupling.conj().T, atol=1.0e-11
        )
        power = power @ matrix


def test_hermitian_square_agrees_with_the_general_product() -> None:
    """The Hermitian BLAS kernel gives the general product, exactly Hermitian."""

    from cayleygw.realization._helpers.base import _hermitian_square

    generator = np.random.default_rng(2024)
    values = generator.normal(size=(37, 53)) + 1.0j * generator.normal(size=(37, 53))
    result = _hermitian_square(values)

    np.testing.assert_allclose(result, values.conj().T @ values, atol=1.0e-12)
    # Exactly Hermitian, which the general product is only to roundoff.
    assert np.array_equal(result, result.conj().T)

    # BLAS returns an uninitialized buffer for an empty inner dimension, so pin the guard.
    empty = _hermitian_square(np.zeros((0, 4), dtype=np.complex128))
    assert empty.shape == (4, 4)
    assert not empty.any()


def test_toeplitz_detects_clear_natural_rank_termination() -> None:
    nodes = np.exp(1.0j * np.asarray([0.3, 1.5, 4.4]))
    vectors = np.asarray([[1.0, 0.0, 0.0], [0.4, 0.9, 0.0], [0.3j, 0.7, 0.0]])
    residues = np.einsum("mi,mj->mij", vectors, vectors.conj(), optimize=True)
    moments = _atomic_moments(nodes, residues, 8)

    result = ToeplitzRealization.realize(moments)

    assert result.dimension == 3
    assert result.normalization.rank == 2
    assert result.terminal_from_moments
    assert np.max(result.moment_residuals) < 1.0e-14


def test_toeplitz_deflates_tolerance_bounded_ambiguous_null_direction() -> None:
    """A near-boundary input is deflated with a finite-rank backward error, not a dense ridge."""

    moments = np.asarray([[[1.0]], [[1.0 + 5.0e-11]]])

    result = ToeplitzRealization.realize(moments)

    assert result.diagnostics.numerical_rank < 2 * result.normalization.rank
    assert result.dimension == 1
    assert result.diagnostics.gram_residual > 0.0
    strict = 1.0e-9 * (1.0 + np.linalg.norm(moments, axis=(1, 2)))
    accepted = result.physical_moment_acceptance_thresholds()
    assert np.all(accepted > strict)
    assert np.max(result.moment_residuals) < np.max(accepted)


def test_cached_terminal_reclosure_changes_only_unconstrained_moments() -> None:
    moments = np.asarray([np.eye(2), 0.2j * np.eye(2)])
    first = ToeplitzRealization.realize(moments)
    swap = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    second = first.reclose(swap)

    np.testing.assert_allclose(first.matrix_moments(1), moments, atol=3.0e-15)
    np.testing.assert_allclose(second.matrix_moments(1), moments, atol=3.0e-15)
    assert np.linalg.norm(first.matrix_moments(2)[2] - second.matrix_moments(2)[2]) > 0.1
    assert first.diagnostics is second.diagnostics
    assert first._fixed_matrix is second._fixed_matrix


def test_high_order_stress_extends_beyond_inverse_defect_failure() -> None:
    moments = _high_order_stress_moments(30)

    with pytest.raises(RefusalError, match="contraction ball"):
        BlockCMVRealization.realize(moments[:16])
    stable = ToeplitzRealization.realize(moments)

    assert stable.conserved_order == 30
    assert np.max(stable.moment_residuals) < 3.0e-11
    assert stable.unitarity_residual < 5.0e-13

    automatic = SectorSelfEnergyRealization.realize(
        moments[:16],
        Sector.PARTICLE,
        CayleyMap(center=0.0, scale=1.0),
        phase_count=1,
    )
    assert automatic.closure_scan.requested_realization_algorithm == "auto"
    assert automatic.closure_scan.realization_algorithm == "toeplitz"


def test_sector_default_is_guarded_and_both_backends_are_selectable() -> None:
    node = 1.0j
    moments = np.asarray([[[1.0 + 0.0j]], [[node]]])
    mapping = CayleyMap(center=0.0, scale=1.0)
    default = SectorSelfEnergyRealization.realize(
        moments,
        Sector.PARTICLE,
        mapping,
        n_conserved=0,
        phase_count=8,
    )
    legacy = SectorSelfEnergyRealization.realize(
        moments,
        Sector.PARTICLE,
        mapping,
        n_conserved=0,
        phase_count=8,
        realization_algorithm="block-cmv",
    )

    assert default.closure_scan.requested_realization_algorithm == "auto"
    assert default.closure_scan.realization_algorithm == "block-cmv"
    assert default.selected_phase == pytest.approx(1.5 * np.pi)
    assert legacy.closure_scan.realization_algorithm == "block-cmv"
    assert legacy.selected_phase == pytest.approx(1.5 * np.pi)
    with pytest.raises(ValidationError, match="realization_algorithm"):
        SectorSelfEnergyRealization.realize(
            moments,
            Sector.PARTICLE,
            mapping,
            realization_algorithm="unknown",
        )


def _sector_arguments(moments, **overrides):
    """Return the positional and keyword arguments of a small particle closure scan."""

    arguments = {
        "sector": Sector.PARTICLE,
        "phase_count": 8,
    }
    arguments.update(overrides)
    sector = arguments.pop("sector")
    return (moments, sector), arguments


def test_this_route_reports_no_contraction_ball_use_structurally() -> None:
    """Toeplitz has no choice coefficient, so its contraction ratio is 0 where block-CMV fails."""

    moments = _high_order_stress_moments(30)[:16]
    with pytest.raises(RefusalError, match="contraction ball"):
        BlockCMVRealization.realize(moments)

    stable = ToeplitzRealization.realize(moments)
    assert stable.maximum_contraction_ratio == 0.0


def test_auto_is_bit_for_bit_the_forced_first_backend() -> None:
    """``auto`` is bit for bit the backend it tries first, so recorded results reproduce."""

    node = 1.0j
    moments = np.asarray([[[1.0 + 0.0j]], [[node]]])
    mapping = CayleyMap(center=0.0, scale=1.0)
    keywords = dict(n_conserved=0, phase_count=8)

    default = SectorSelfEnergyRealization.realize(moments, Sector.PARTICLE, mapping, **keywords)
    legacy = SectorSelfEnergyRealization.realize(
        moments, Sector.PARTICLE, mapping, **keywords, realization_algorithm="block-cmv"
    )

    assert default.closure_scan.requested_realization_algorithm == "auto"
    assert default.closure_scan.realization_algorithm == "block-cmv"
    assert default.selected_phase == legacy.selected_phase
    np.testing.assert_array_equal(default.poles, legacy.poles)
    np.testing.assert_array_equal(default.couplings, legacy.couplings)


def test_unknown_realization_algorithm_names_every_accepted_value() -> None:
    node = 1.0j
    moments = np.asarray([[[1.0 + 0.0j]], [[node]]])
    positional, keywords = _sector_arguments(moments)

    with pytest.raises(ValidationError, match="toeplitz"):
        SectorSelfEnergyRealization.scan_closures(
            *positional, **keywords, realization_algorithm="unknown"
        )


def test_the_endpoint_blocks_are_square_only_outside_the_band() -> None:
    """The ``(t, r)`` endpoint block is square only outside the band, where no search runs."""

    stored = np.load(
        Path(__file__).resolve().parents[1] / "fixtures" / "magnesium-monoxide-K11-hole.npz",
        allow_pickle=True,
    )
    values = stored["values"]
    # Measured at 1e-12: the default floor already deflates two directions at n_conserved 6.
    shipped = replace(DEFAULT_TOLERANCES, rank_floor=1.0e-12)
    banded = ToeplitzRealization.realize(values[:9], tolerances=shipped).diagnostics
    unbanded = ToeplitzRealization.realize(values[:7], tolerances=shipped).diagnostics

    assert 0 < banded.terminal_dimension < 32
    assert unbanded.terminal_dimension == 32


def test_a_positive_smallest_eigenvalue_is_no_breach_and_raises_no_alarm(caplog) -> None:
    """The breach is signed, so a well-conditioned Toeplitz matrix logs no breach warning."""

    import logging

    rng = np.random.default_rng(11)
    rank, atoms, n_max = 3, 12, 5
    nodes = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, atoms))
    weights = []
    for _ in range(atoms):
        factor = rng.normal(size=(rank, rank)) + 1j * rng.normal(size=(rank, rank))
        weights.append(factor @ factor.conj().T)
    moments = np.asarray(
        [sum(node**k * w for node, w in zip(nodes, weights)) for k in range(n_max + 1)]
    )
    from cayleygw.realization._helpers.toeplitz import _block_toeplitz
    from cayleygw.realization.base import UnitaryMomentRealization

    normalized = UnitaryMomentRealization.normalize(moments)
    eigenvalues = np.linalg.eigvalsh(_block_toeplitz(np.asarray(normalized.values)))
    assert eigenvalues[0] > 1e-4 * eigenvalues[-1]
    with caplog.at_level(logging.WARNING, logger="cayleygw.realization.toeplitz"):
        realization = ToeplitzRealization.realize(moments)
    assert realization.moment_residuals.max() < 1e-10
    assert not [r for r in caplog.records if "breach" in r.getMessage().lower()]


def test_the_gram_thread_count_is_reachable_and_changes_no_result(monkeypatch) -> None:
    """``gram_native_threads`` reaches the eigensolve, defaults to 1 and changes no result."""

    stored = np.load(
        Path(__file__).resolve().parents[1] / "fixtures" / "lithium-hydride-K3-hole.npz",
        allow_pickle=True,
    )
    values = stored["values"]

    pinned = ToeplitzRealization.realize(values, gram_native_threads=1)
    default = ToeplitzRealization.realize(values)
    np.testing.assert_array_equal(pinned.matrix, default.matrix)

    threaded = ToeplitzRealization.realize(values, gram_native_threads=4)
    assert threaded.diagnostics.numerical_rank == pinned.diagnostics.numerical_rank

    # A closure scan passes its own ``native_threads`` to the Gram build.
    built = []
    original = toeplitz.build_gram_spectrum

    def recording(normalized_values, *, native_threads=1):
        built.append(native_threads)
        return original(normalized_values, native_threads=native_threads)

    monkeypatch.setattr(toeplitz, "build_gram_spectrum", recording)
    SectorSelfEnergyRealization.scan_closures(
        values,
        Sector.HOLE,
        phase_count=1,
        native_threads=3,
        realization_algorithm="toeplitz",
    )
    assert built == [3]


# The O(t**3) unitarity residual of each reclose ``M(T) = F + L T.H R`` of one base.


def _unitary(generator, size):
    """Return a random unitary from the QR of a complex Gaussian matrix."""

    matrix, _ = np.linalg.qr(
        generator.normal(size=(size, size)) + 1.0j * generator.normal(size=(size, size))
    )
    return matrix


def _dense_residual(matrix):
    """Return ``||M.H M - I||_F`` from the dense square."""

    square = _hermitian_square(matrix)
    square[np.diag_indices_from(square)] -= 1.0
    return float(np.linalg.norm(square, ord="fro"))


@pytest.mark.parametrize(
    "dimension, terminal_dimension, defect",
    [(60, 7, 1.0e-9), (120, 30, 1.0e-6), (80, 1, 1.0e-12), (50, 50, 1.0e-8)],
)
def test_closed_form_matches_the_dense_residual_on_a_perturbed_family(
    dimension, terminal_dimension, defect
):
    """With small defects, as the guard requires, the closed form tracks the dense residual."""

    generator = np.random.default_rng(7)
    left_full = _unitary(generator, dimension)
    right_full = _unitary(generator, dimension)
    keep = dimension - terminal_dimension
    fixed = left_full[:, :keep] @ right_full[:, :keep].conj().T
    fixed = fixed + defect * generator.normal(size=(dimension, dimension))
    left = left_full[:, keep:] + defect * generator.normal(size=(dimension, terminal_dimension))
    right = right_full[:, keep:].conj().T + defect * generator.normal(
        size=(terminal_dimension, dimension)
    )
    certificate = _UnitarityCertificate.build(fixed, left, right)
    terminals = [
        np.eye(terminal_dimension, dtype=np.complex128),
        np.exp(1.0j * 1.234) * np.eye(terminal_dimension, dtype=np.complex128),
        _unitary(generator, terminal_dimension),
    ]
    for terminal in terminals:
        matrix = fixed + left @ terminal.conj().T @ right
        dense = _dense_residual(matrix)
        closed = certificate.residual(terminal)
        assert closed == pytest.approx(dense, rel=1.0e-6)
    assert certificate.scale <= 10.0 * defect * dimension


def _stress_moments(n_max, block=4, atom_count=40, seed=851):
    """Return the moments of a rank-two-per-atom measure on the upper semicircle."""

    generator = np.random.default_rng(seed)
    angles = np.linspace(0.05, np.pi - 0.05, atom_count)
    nodes = np.exp(1.0j * angles)
    factors = generator.normal(size=(atom_count, block, 2)) + 1.0j * (
        generator.normal(size=(atom_count, block, 2))
    )
    residues = np.einsum("mai,mbi->mab", factors, factors.conj(), optimize=True)
    return np.asarray(
        [
            np.einsum("m,mij->ij", nodes**order, residues, optimize=True)
            for order in range(n_max + 1)
        ]
    )


def test_a_reclosed_realization_reports_the_certified_residual():
    """A reclosed candidate reports the closed-form residual, and the base keeps the dense one."""

    base = ToeplitzRealization.realize(_stress_moments(6))
    assert base.diagnostics.terminal_dimension > 0
    assert base._unitarity_certificate is not None
    assert base.unitarity_residual == _dense_residual(base.matrix)
    generator = np.random.default_rng(3)
    dimension = base.diagnostics.terminal_dimension
    for terminal in (
        np.exp(0.7j) * np.eye(dimension, dtype=np.complex128),
        np.exp(-2.1j) * np.eye(dimension, dtype=np.complex128),
        _unitary(generator, dimension),
    ):
        candidate = base.reclose(terminal)
        dense = _dense_residual(candidate.matrix)
        # Both are roundoff-sized, so they must agree absolutely, far inside the gate.
        assert abs(candidate.unitarity_residual - dense) <= 1.0e-12
        assert candidate.unitarity_residual < 1.0e-10
        # The certificate leaves the delivered arrays untouched.
        assert candidate._unitarity_certificate is base._unitarity_certificate
        assert np.array_equal(candidate.coupling, base.coupling)


def test_a_naturally_terminated_realization_carries_no_certificate():
    """With no terminal freedom there is no family to certify."""

    moments = _stress_moments(2, block=1, atom_count=2)
    realization = ToeplitzRealization.realize(moments)
    if realization.diagnostics.terminal_dimension == 0:
        assert realization._unitarity_certificate is None
        assert realization.reclose(None) is realization
