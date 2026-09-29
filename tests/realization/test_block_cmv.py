"""Tests of the matrix moments, their normalization and the block-CMV realization."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cayleygw import RefusalError, ValidationError
from cayleygw._helpers.tolerances import DEFAULT_TOLERANCES, ROUNDOFF_TOLERANCE
from cayleygw.realization import block_cmv as block_cmv_module
from cayleygw.realization._helpers import block_cmv as block_cmv_helpers
from cayleygw.realization._helpers.base import _unitarity_residual
from cayleygw.realization.base import (
    MatrixCayleyMoments,
    UnitaryMomentRealization,
)
from cayleygw.realization.block_cmv import BlockCMVRealization


def _defect_ranks(realization):
    """Return the retained defect ranks of the moment-determined Schur steps."""

    return tuple(step.defect_rank for step in realization._parameters.steps)


def _matrix_atomic_moments(nodes, residues, n_max):
    """Return moments of a finite positive matrix-valued atomic measure."""

    nodes = np.asarray(nodes, dtype=np.complex128)
    residues = np.asarray(residues, dtype=np.complex128)
    return np.asarray(
        [
            np.einsum("m,mij->ij", nodes**order, residues, optimize=True)
            for order in range(n_max + 1)
        ]
    )


def _full_rank_noncommuting_measure(n_max=4):
    """Return a deterministic noncommuting positive matrix measure."""

    generator = np.random.default_rng(410)
    nodes = np.exp(1.0j * np.asarray([0.17, 0.73, 1.41, 2.28, 3.36, 4.51, 5.42]))
    factors = generator.normal(size=(nodes.size, 2, 2)) + 1.0j * generator.normal(
        size=(nodes.size, 2, 2)
    )
    residues = np.einsum("mij,mkj->mik", factors, factors.conj(), optimize=True)
    return nodes, residues, _matrix_atomic_moments(nodes, residues, n_max)


def test_realization_package_has_no_moment_producer_imports() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "cayleygw" / "realization"
    forbidden = {"pyscf", "reference_state", "response", "contour", "reference"}

    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append(node.module)
        assert not any(part in forbidden for imported in imports for part in imported.split(".")), (
            f"{path.name} imports a moment-producing layer: {imports}"
        )


def test_matrix_moment_container_is_source_agnostic_and_read_only() -> None:
    values = np.asarray([np.eye(2), 0.2j * np.eye(2)])
    sequence = MatrixCayleyMoments(values)
    values[0, 0, 0] = 9.0

    assert sequence.n_max == 1
    assert sequence.nphysical == 2
    assert sequence.values[0, 0, 0] == pytest.approx(1.0)
    assert not sequence.values.flags.writeable
    with pytest.raises(ValueError):
        sequence.values[0, 0, 0] = 2.0


def test_normalization_recovers_support_factor_and_identity() -> None:
    nodes = np.exp(1.0j * np.asarray([0.3, 1.2, 4.4]))
    vectors = np.asarray([[1.0, 0.0, 0.0], [0.3, 0.8, 0.0], [0.2j, 0.7, 0.0]])
    residues = np.einsum("mi,mj->mij", vectors, vectors.conj(), optimize=True)
    moments = _matrix_atomic_moments(nodes, residues, n_max=4)
    normalized = UnitaryMomentRealization.normalize(moments)

    assert normalized.rank == 2
    assert normalized.support_factor.shape == (3, 2)
    np.testing.assert_allclose(
        normalized.support_factor @ normalized.support_factor.conj().T,
        moments[0],
        atol=2.0e-15,
    )
    np.testing.assert_allclose(normalized.values[0], np.eye(2), atol=8.0e-16)
    assert np.linalg.norm(normalized.values[0] - np.eye(2)) < 8.0e-16
    projector = normalized.support_factor @ normalized.support_pseudoinverse
    support_residuals = np.linalg.norm(
        moments - projector @ moments @ projector.conj().T, axis=(1, 2)
    )
    assert np.max(support_residuals) < 2.0e-15


def test_first_block_parameter_fixes_adjoint_and_selector_conventions() -> None:
    _, _, moments = _full_rank_noncommuting_measure(n_max=2)
    normalized = UnitaryMomentRealization.normalize(moments)
    realization = BlockCMVRealization.realize(moments)
    step = realization._parameters.steps[0]

    np.testing.assert_allclose(
        step.choice_parameter,
        normalized.values[1],
        atol=7.0e-16,
    )
    np.testing.assert_allclose(
        step.verblunsky_coefficient,
        normalized.values[1].conj().T,
        atol=7.0e-16,
    )
    np.testing.assert_allclose(realization.matrix[:2, :2], normalized.values[1], atol=7.0e-16)


def test_nonnormal_julia_rotation_keeps_left_and_right_defects_distinct() -> None:
    alpha = np.asarray([[0.15 + 0.1j, 0.42 - 0.08j], [0.03j, -0.21 + 0.17j]])
    step = block_cmv_helpers._step_from_choice(alpha.conj().T, 0, DEFAULT_TOLERANCES)
    rotation = step.rotation
    roots = np.sqrt(step.defect_eigenvalues)
    left_defect = (step.left_defect_basis * roots[None, :]) @ step.left_defect_basis.conj().T
    right_defect = (step.right_defect_basis * roots[None, :]) @ step.right_defect_basis.conj().T

    expected_left_squared = np.eye(2) - alpha.conj().T @ alpha
    expected_right_squared = np.eye(2) - alpha @ alpha.conj().T
    np.testing.assert_allclose(
        left_defect @ left_defect,
        expected_left_squared,
        atol=8.0e-16,
    )
    np.testing.assert_allclose(
        right_defect @ right_defect,
        expected_right_squared,
        atol=8.0e-16,
    )
    assert np.linalg.norm(left_defect - right_defect) > 1.0e-2
    np.testing.assert_allclose(rotation.conj().T @ rotation, np.eye(4), atol=9.0e-16)
    stored = step.verblunsky_coefficient
    for contraction in (
        np.eye(2) - stored.conj().T @ stored,
        np.eye(2) - stored @ stored.conj().T,
    ):
        hermitian = 0.5 * (contraction + contraction.conj().T)
        assert np.min(np.linalg.eigvalsh(hermitian)) > 0.0


def test_noncommuting_positive_measure_is_conserved_by_block_cmv() -> None:
    _, _, moments = _full_rank_noncommuting_measure(n_max=5)
    assert np.linalg.norm(moments[1] @ moments[2] - moments[2] @ moments[1]) > 1.0

    result = BlockCMVRealization.realize(moments)

    assert result.dimension == 12
    assert result.normalization.rank == 2
    assert _defect_ranks(result) == (2, 2, 2, 2, 2)
    assert not result.terminal_from_moments
    np.testing.assert_allclose(result.matrix, result._prefix.left @ result._prefix.right)
    np.testing.assert_allclose(
        result.selector.conj().T @ result.selector,
        np.eye(2),
        atol=2.0e-15,
    )
    np.testing.assert_allclose(result.reconstructed_moments, moments, atol=5.0e-14)
    np.testing.assert_allclose(result.matrix_moments(5), moments, atol=5.0e-14)
    np.testing.assert_allclose(
        result.normalized_matrix_moments(5),
        result.normalization.values,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        result.matrix.conj().T @ result.matrix,
        np.eye(result.dimension),
        atol=2.0e-15,
    )
    assert np.max(result.moment_residuals) < 5.0e-14
    steps = result._parameters.steps
    assert max(_unitarity_residual(step.rotation) for step in steps) < 2.0e-15
    alphas = [step.verblunsky_coefficient for step in steps]
    alphas.append(result.terminal_unitary)
    for alpha in alphas:
        for contraction in (
            np.eye(alpha.shape[0]) - alpha.conj().T @ alpha,
            np.eye(alpha.shape[0]) - alpha @ alpha.conj().T,
        ):
            hermitian = 0.5 * (contraction + contraction.conj().T)
            assert np.min(np.linalg.eigvalsh(hermitian)) > -2.0e-15


def test_rank_deflation_reaches_minimal_three_atom_dimension() -> None:
    nodes = np.exp(1.0j * np.asarray([0.3, 1.5, 4.4]))
    vectors = np.asarray([[1.0, 0.0, 0.0], [0.4, 0.9, 0.0], [0.3j, 0.7, 0.0]])
    residues = np.einsum("mi,mj->mij", vectors, vectors.conj(), optimize=True)
    moments = _matrix_atomic_moments(nodes, residues, n_max=8)

    result = BlockCMVRealization.realize(moments)

    assert result.normalization.rank == 2
    assert result.dimension == 3
    assert _defect_ranks(result) == (1, 0)
    assert result.terminal_from_moments
    np.testing.assert_allclose(result.reconstructed_moments, moments, atol=3.0e-15)
    np.testing.assert_allclose(
        np.sort(np.mod(np.angle(np.linalg.eigvals(result.matrix)), 2.0 * np.pi)),
        np.sort(np.mod(np.angle(nodes), 2.0 * np.pi)),
        atol=2.0e-15,
    )


def test_one_atom_matrix_measure_terminates_at_first_step() -> None:
    node = np.exp(0.71j)
    zeroth = np.asarray([[2.0, 0.3], [0.3, 1.0]])
    moments = np.asarray([zeroth * node**order for order in range(7)])
    result = BlockCMVRealization.realize(moments)

    assert result.dimension == 2
    assert _defect_ranks(result) == (0,)
    assert result.terminal_from_moments
    np.testing.assert_allclose(result.matrix, node * np.eye(2), atol=7.0e-16)
    np.testing.assert_allclose(result.matrix_moments(6), moments, atol=3.0e-15)


def test_later_moment_inconsistent_with_natural_termination_is_rejected() -> None:
    node = np.exp(0.4j)
    normalized = np.asarray(
        [
            np.eye(2),
            node * np.eye(2),
            (node**2 + 1.0e-3) * np.eye(2),
        ]
    )

    with pytest.raises(RefusalError, match="does not conserve a supplied"):
        BlockCMVRealization.realize(normalized)


def test_terminal_unitary_changes_only_unconstrained_moments() -> None:
    _, _, moments = _full_rank_noncommuting_measure(n_max=3)
    first = BlockCMVRealization.realize(moments)
    swap = np.asarray([[0.0, 1.0j], [1.0j, 0.0]])
    second = first.reclose(swap)

    np.testing.assert_allclose(first.matrix_moments(3), moments, atol=3.0e-14)
    np.testing.assert_allclose(second.matrix_moments(3), moments, atol=3.0e-14)
    assert np.linalg.norm(first.matrix_moments(4)[-1] - second.matrix_moments(4)[-1]) > 1.0


def test_zero_measure_has_zero_dimensional_minimal_realization() -> None:
    moments = np.zeros((5, 3, 3), dtype=np.complex128)
    result = BlockCMVRealization.realize(moments)

    assert result.normalization.rank == 0
    assert result.dimension == 0
    assert result.coupling.shape == (3, 0)
    assert result.selector.shape == (0, 0)
    np.testing.assert_array_equal(result.matrix_moments(7), np.zeros((8, 3, 3)))


@pytest.mark.parametrize(
    ("moments", "match"),
    [
        ([], "shape"),
        (np.ones((2, 2)), "shape"),
        (np.ones((2, 2, 3)), "shape"),
        (np.asarray([[[np.nan]]]), "finite"),
    ],
)
def test_invalid_matrix_moment_shapes_and_values_are_rejected(moments, match) -> None:
    with pytest.raises(ValidationError, match=match):
        MatrixCayleyMoments(moments)


def test_invalid_zeroth_support_and_schur_contraction_are_rejected() -> None:
    nonhermitian = np.asarray([[[1.0, 0.2j], [0.0, 1.0]]])
    with pytest.raises(RefusalError, match="non-Hermitian"):
        UnitaryMomentRealization.normalize(nonhermitian)

    indefinite = np.asarray([np.diag([1.0, -0.1])])
    with pytest.raises(RefusalError, match="positive semidefinite"):
        UnitaryMomentRealization.normalize(indefinite)

    support_leak = np.asarray(
        [
            np.diag([1.0, 0.0]),
            np.asarray([[0.2, 0.1], [0.1, 0.0]]),
        ]
    )
    with pytest.raises(RefusalError, match="leaves the retained support"):
        UnitaryMomentRealization.normalize(support_leak)

    with pytest.raises(RefusalError, match="contraction ball") as raised:
        BlockCMVRealization.realize(np.asarray([np.eye(2), 1.01 * np.eye(2)]))
    # The contraction-ball refusal names its gate, since its remedy is a backend switch.
    assert raised.value.kind == "block-cmv"
    assert raised.value.check == "schur_contraction"


def test_zeroth_moment_gates_carry_their_numbers_as_attributes() -> None:
    """The ``C[0]`` gates carry their residual, threshold and order as attributes too."""

    cases = (
        (np.asarray([[[1.0, 0.2j], [0.0, 1.0]]]), "zeroth_hermiticity", 0),
        (np.asarray([np.diag([1.0, -0.1])]), "zeroth_positivity", 0),
        (
            np.asarray(
                [
                    np.diag([1.0, 0.0]),
                    np.asarray([[0.2, 0.1], [0.1, 0.0]]),
                ]
            ),
            "support_leakage",
            1,
        ),
    )
    for moments, check, order in cases:
        with pytest.raises(RefusalError) as raised:
            UnitaryMomentRealization.normalize(moments)
        error = raised.value
        assert error.kind == "block-cmv"
        assert error.check == check
        assert error.order == order
        assert error.residual is not None and error.threshold is not None
        # The attributes are the numbers the message quotes.
        assert f"{error.residual:.3e}" in str(error) or (
            f"{error.residual:.3e}" == f"{error.threshold:.3e}"
        )


def test_a_refusal_without_a_named_gate_still_constructs() -> None:
    """The diagnostic fields of a refusal are optional."""

    error = RefusalError("something else went wrong")
    assert error.kind is None
    assert error.diagnostics is None
    assert error.check is None
    assert error.residual is None
    assert error.threshold is None
    assert error.order is None


def test_invalid_terminal_tolerance_and_requested_order_are_rejected() -> None:
    _, _, moments = _full_rank_noncommuting_measure(n_max=2)
    result = BlockCMVRealization.realize(moments)
    with pytest.raises(ValidationError, match="shape"):
        result.reclose(np.eye(3))
    with pytest.raises(RefusalError, match="not unitary"):
        result.reclose(0.8 * np.eye(2))
    with pytest.raises(ValidationError, match="Tolerances"):
        BlockCMVRealization.realize(moments, tolerances=None)
    with pytest.raises(ValidationError, match="nonnegative integer"):
        result.matrix_moments(-1)
    with pytest.raises(ValidationError, match="nonnegative integer"):
        result.normalized_matrix_moments(True)


def test_roundoff_singular_values_are_projected_only_within_tolerance() -> None:
    choice = np.diag([1.0 + 2.0e-9, 0.2])
    step = block_cmv_helpers._step_from_choice(
        choice, 0, replace(DEFAULT_TOLERANCES, positivity=1.0e-8)
    )

    assert step.defect_rank == 1
    projected = np.linalg.svd(step.choice_parameter, compute_uv=False)
    assert projected[0] == pytest.approx(1.0)
    assert np.linalg.norm(step.choice_parameter - choice) == pytest.approx(2.0e-9)


def test_an_excess_over_the_ball_is_projected_within_the_tolerance_and_refused_past_it() -> None:
    """A noise-scale excess over the ball is projected; one past the tolerance is refused."""

    within = 0.5 * DEFAULT_TOLERANCES.positivity
    normalized = np.asarray([np.eye(2), np.diag([1.0 + within, 0.2])])

    realization = BlockCMVRealization.realize(normalized)
    step = realization._parameters.steps[0]

    assert step.contraction_ratio == pytest.approx(0.5, rel=1.0e-6)
    # The coefficient is projected onto the ball.
    projected = np.linalg.svd(step.choice_parameter, compute_uv=False)
    assert projected[0] == pytest.approx(1.0)
    assert realization.maximum_contraction_ratio == pytest.approx(0.5, rel=1.0e-6)

    past = np.asarray([np.eye(2), np.diag([1.0 + 5.0 * DEFAULT_TOLERANCES.positivity, 0.2])])
    with pytest.raises(RefusalError, match="contraction ball"):
        BlockCMVRealization.realize(past)


def test_a_clean_contraction_stays_inside_the_ball() -> None:
    _, _, moments = _full_rank_noncommuting_measure(n_max=3)
    realization = BlockCMVRealization.realize(moments)

    assert realization.maximum_contraction_ratio <= 1.0
    assert all(step.contraction_ratio <= 1.0 for step in realization._parameters.steps)


def test_no_positivity_tolerance_can_admit_a_material_violation() -> None:
    """However loose the positivity tolerance, the moment reconstruction check refuses a material excess."""  # noqa: E501

    node = np.exp(0.4j)
    residue = np.asarray([[1.0, 0.2j], [-0.2j, 0.5]])
    single_atom = np.asarray([node**order * residue for order in range(3)])
    excess = 1.0e-6
    outside = single_atom * ((1.0 + excess) ** np.arange(3))[:, None, None]

    with pytest.raises(RefusalError, match="conserve"):
        BlockCMVRealization.realize(
            outside, tolerances=replace(DEFAULT_TOLERANCES, positivity=1.0e2)
        )


def test_all_public_realization_arrays_are_read_only() -> None:
    _, _, moments = _full_rank_noncommuting_measure(n_max=3)
    result = BlockCMVRealization.realize(moments)
    arrays = [
        result.normalization.source.values,
        result.normalization.values,
        result.normalization.support_eigenvalues,
        result.normalization.support_factor,
        result.normalization.support_pseudoinverse,
        result.terminal_unitary,
        result.matrix,
        result.selector,
        result.coupling,
        result.reconstructed_moments,
        result.moment_residuals,
        result.matrix_moments(4),
        result.normalized_matrix_moments(4),
    ]
    assert all(not array.flags.writeable for array in arrays)


def test_reclose_block_cmv_reproduces_a_full_realization_exactly() -> None:
    """``reclose`` skips only terminal-independent work, so it matches a fresh build to roundoff."""

    _, _, moments = _full_rank_noncommuting_measure(n_max=3)
    truncated = MatrixCayleyMoments(moments[:3])
    canonical = BlockCMVRealization.realize(truncated)
    dimension = canonical.terminal_unitary.shape[0]
    assert dimension > 0, "this fixture must not rank-terminate"

    for phase in (0.0, 0.7, 2.9, 4.4):
        terminal = np.exp(1.0j * phase) * np.eye(dimension, dtype=np.complex128)
        # The full assembly of this closure, without the cached prefix.
        expected = block_cmv_module.BlockCMVRealization(
            **block_cmv_module._realize_block_cmv_from_prefix(
                canonical.normalization,
                canonical._parameters,
                terminal,
                DEFAULT_TOLERANCES,
                None,
            )
        )
        actual = canonical.reclose(terminal)
        for name in (
            "matrix",
            "coupling",
            "selector",
            "terminal_unitary",
            "reconstructed_moments",
        ):
            np.testing.assert_allclose(
                np.asarray(getattr(actual, name)),
                np.asarray(getattr(expected, name)),
                rtol=0.0,
                atol=1.0e-13,
                err_msg=f"{name} differs at phase={phase}",
            )
        np.testing.assert_allclose(actual.moment_residuals, expected.moment_residuals, atol=1.0e-13)


def test_reclose_block_cmv_returns_the_fixed_closure_after_termination() -> None:
    """Rank-terminated data admits one closure, so a scan has nothing to vary."""

    nodes = np.exp(1.0j * np.asarray([0.4, 2.1, 4.7]))
    factors = np.random.default_rng(11).normal(size=(3, 2, 2))
    residues = np.einsum("mij,mkj->mik", factors, factors, optimize=True)
    moments = _matrix_atomic_moments(nodes, residues, 6)
    canonical = BlockCMVRealization.realize(MatrixCayleyMoments(moments))
    assert canonical.terminal_from_moments

    other = np.exp(1.0j * 1.3) * np.eye(
        max(canonical.terminal_unitary.shape[0], 1), dtype=np.complex128
    )
    assert canonical.reclose(other) is canonical


def test_physical_moments_are_a_congruence_of_the_normalized_ones() -> None:
    """``B U**k B.H`` is the support congruence of ``E0.H U**k E0``, and matches direct powers."""

    _, _, moments = _full_rank_noncommuting_measure(n_max=4)
    realization = BlockCMVRealization.realize(MatrixCayleyMoments(moments))
    support = np.asarray(realization.normalization.support_factor)
    order = 4

    normalized = realization.normalized_matrix_moments(order)
    physical = realization.matrix_moments(order)
    np.testing.assert_allclose(
        physical,
        support @ normalized @ support.conj().T,
        atol=1.0e-12,
    )

    matrix = np.asarray(realization.matrix)
    coupling = np.asarray(realization.coupling)
    power = np.eye(matrix.shape[0], dtype=np.complex128)
    for index in range(order + 1):
        np.testing.assert_allclose(
            physical[index], coupling @ power @ coupling.conj().T, atol=1.0e-11
        )
        power = power @ matrix


def test_assembly_certification_falls_back_to_exact_residuals() -> None:
    """A blockwise bound over the threshold falls back to the exact residuals, not a refusal."""

    from cayleygw.realization._helpers.block_cmv import _certify_assembled_factors

    unitary = np.eye(4, dtype=np.complex128)
    # A bound far above threshold, on factors that are exactly unitary.
    _certify_assembled_factors(unitary, unitary, unitary, 1.0, 1.0, 1.0e-10, 4)

    # A factor that is not unitary is still refused, and named.
    not_unitary = np.eye(4, dtype=np.complex128)
    not_unitary[0, 0] = 2.0
    with pytest.raises(RefusalError, match="assembled left"):
        _certify_assembled_factors(not_unitary, unitary, not_unitary, 9.0, 0.0, 1.0e-10, 4)


def test_banded_assembly_matches_the_dense_product() -> None:
    """The banded ``L M`` product equals the dense one for block-diagonal factors on tiled spans."""

    from cayleygw.realization._helpers.block_cmv import _block_diagonal_product

    generator = np.random.default_rng(606)

    def unitary(size: int) -> np.ndarray:
        raw = generator.normal(size=(size, size)) + 1.0j * generator.normal(size=(size, size))
        left_vectors, _, right_adjoint = np.linalg.svd(raw)
        return left_vectors @ right_adjoint

    # Blocks sized as the recursion sizes them: rotation j spans blocks j, j+1.
    dimensions = [3, 3, 3, 3, 3, 3]
    dimension = sum(dimensions)
    offsets = np.cumsum([0, *dimensions])
    left = np.eye(dimension, dtype=np.complex128)
    right = np.eye(dimension, dtype=np.complex128)
    spans: tuple[list[tuple[int, int]], list[tuple[int, int]]] = ([], [])
    for index in range(len(dimensions)):
        start = int(offsets[index])
        stop = int(offsets[min(index + 2, len(dimensions))])
        (left if index % 2 == 0 else right)[start:stop, start:stop] = unitary(stop - start)
        spans[index % 2].append((start, stop))

    banded = _block_diagonal_product(left, right, tuple(spans[0]), tuple(spans[1]), dimension)
    np.testing.assert_allclose(banded, left @ right, atol=1.0e-14)


def _conditioned_zeroth(rank: int, condition: float) -> np.ndarray:
    """Return a Hermitian positive-definite ``C_0`` of the given rank and condition number."""

    eigenvalues = np.geomspace(1.0 / condition, 1.0, rank)
    generator = np.random.default_rng(0)
    basis, _ = np.linalg.qr(
        generator.normal(size=(rank, rank)) + 1.0j * generator.normal(size=(rank, rank))
    )
    matrix = (basis * eigenvalues) @ basis.conj().T
    return np.asarray([0.5 * (matrix + matrix.conj().T)])


def test_the_constant_zeroth_identity_threshold_refuses_conditioning_alone() -> None:
    """A valid ill-conditioned ``C_0`` exceeds the constant threshold and still normalizes."""

    moments = _conditioned_zeroth(50, 3.0e7)
    # ``P`` carries ``1/sqrt(lambda_min)``, so the threshold scales with the conditioning.
    normalized = UnitaryMomentRealization.normalize(moments)
    assert normalized.rank == 50
    residual = np.linalg.norm(normalized.values[0] - np.eye(50))
    assert residual > ROUNDOFF_TOLERANCE * np.sqrt(50.0)


def test_the_conditioning_term_cannot_tighten_a_well_conditioned_gate() -> None:
    """The threshold is the larger of the constant and the roundoff scale, so it never tightens."""

    epsilon = float(np.finfo(np.float64).eps)
    root = np.sqrt(50.0)
    constant = ROUNDOFF_TOLERANCE * root
    benign = max(constant, epsilon * root * 7.33e3)
    assert benign == constant

    # A matrix at that conditioning normalizes.
    moments = _conditioned_zeroth(50, 7.33e3)
    assert UnitaryMomentRealization.normalize(moments).rank == 50
