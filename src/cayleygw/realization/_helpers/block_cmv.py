"""Private helpers of :mod:`cayleygw.realization.block_cmv`."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._helpers import tolerances as limits
from ..._helpers.errors import RefusalError
from ..._helpers.tolerances import Tolerances
from ..._helpers.validate import ComplexArray, _check
from ...tools.records.realization import BlockSchurParameters, BlockSchurStep
from .base import _canonical_terminal_unitary, _conserved_fields, _unitarity_residual

if TYPE_CHECKING:
    from ...tools.records.realization import NormalizedMatrixCayleyMoments


def _initial_choice_series(moments: ComplexArray) -> ComplexArray:
    r"""Return the reflected matrix Schur series through the determined order.

    It solves ``Theta(z) @ (F(z) + I) = (F(z) - I) / z`` coefficient by
    coefficient, keeping the product order of a noncommuting measure.
    """

    maximum_order = moments.shape[0] - 1
    rank = moments.shape[1]
    if maximum_order == 0:
        return np.empty((0, rank, rank), dtype=np.complex128)
    result = np.empty((maximum_order, rank, rank), dtype=np.complex128)
    denominator = np.empty_like(result)
    denominator[0] = 2.0 * np.eye(rank, dtype=np.complex128)
    if maximum_order > 1:
        denominator[1:] = 2.0 * moments[1:maximum_order]
    numerator = 2.0 * moments[1:]
    for order in range(maximum_order):
        coefficient = numerator[order].copy()
        for previous in range(order):
            coefficient -= result[previous] @ denominator[order - previous]
        result[order] = 0.5 * coefficient
    return result


def _step_from_choice(
    choice_parameter: Any,
    index: int,
    tolerances: Tolerances,
) -> BlockSchurStep:
    """Project a choice coefficient onto the contraction ball and build its Julia rotation.

    A choice coefficient has ``sigma_max <= 1`` exactly, so any excess is
    quadrature error amplified by the support normalization, and its size
    changes with the BLAS summation order. An excess within
    ``tolerances.positivity`` is projected onto the ball and recorded as
    ``contraction_ratio``; the per-closure moment check must still pass.
    """

    raw = _check.readonly_complex(choice_parameter, "choice_parameter")
    left_vectors, raw_singular_values, right_adjoint = np.linalg.svd(
        raw,
        full_matrices=True,
    )
    maximum = float(np.max(raw_singular_values))
    excess = max(0.0, maximum - 1.0)
    contraction_ratio = excess / tolerances.positivity
    if contraction_ratio > 1.0:
        raise RefusalError(
            f"block-Schur coefficient {index} lies outside the contraction "
            f"ball: maximum_singular_value={maximum:.16e}, "
            f"excess={excess:.3e}, positivity={tolerances.positivity:.3e}. "
            "Use realization='toeplitz' or a larger N_q.",
            kind="block-cmv",
            check="schur_contraction",
        )
    singular_values = np.minimum(raw_singular_values, 1.0)
    defect_squared = np.maximum(0.0, 1.0 - singular_values**2)
    largest_defect = float(np.max(defect_squared))
    rank_threshold = tolerances.rank_floor + tolerances.rank_floor * largest_defect
    retained = defect_squared > rank_threshold
    # Discarded defect directions go to exactly one, so the reduced rotation is unitary.
    singular_values = singular_values.copy()
    singular_values[~retained] = 1.0
    defect_squared = np.maximum(0.0, 1.0 - singular_values**2)
    choice = (left_vectors * singular_values[None, :]) @ right_adjoint
    right_vectors = right_adjoint.conj().T
    defect_values = defect_squared[retained]
    defect_roots = np.sqrt(defect_values)
    left_basis = left_vectors[:, retained]
    right_basis = right_vectors[:, retained]

    top_right = left_basis * defect_roots[None, :]
    bottom_left = defect_roots[:, None] * right_basis.conj().T
    bottom_right = -right_basis.conj().T @ choice.conj().T @ left_basis
    rotation = np.block(
        [
            [choice, top_right],
            [bottom_left, bottom_right],
        ]
    )
    unitarity_residual = _unitarity_residual(rotation)
    threshold = limits.ROUNDOFF_TOLERANCE * max(1.0, np.sqrt(rotation.shape[0]))
    if unitarity_residual > threshold:
        raise RefusalError(
            "rank-deflated Julia rotation is not unitary within tolerance: "
            f"index={index}, residual={unitarity_residual:.3e}, "
            f"threshold={threshold:.3e}",
            kind="block-cmv",
        )
    return BlockSchurStep(
        choice_parameter=_check.readonly_complex(choice, "choice_parameter"),
        left_defect_basis=_check.readonly_complex(left_basis, "left_defect_basis"),
        right_defect_basis=_check.readonly_complex(right_basis, "right_defect_basis"),
        defect_eigenvalues=_check.readonly_real(defect_values, "defect_eigenvalues"),
        rotation=_check.readonly_complex(rotation, "rotation"),
        contraction_ratio=contraction_ratio,
    )


def _next_choice_series(
    series: ComplexArray,
    step: BlockSchurStep,
) -> ComplexArray:
    """Apply one step of the operator Schur recursion in the defect bases."""

    remaining = series.shape[0] - 1
    rank = step.defect_rank
    if remaining == 0:
        return np.empty((0, rank, rank), dtype=np.complex128)
    roots = np.sqrt(step.defect_eigenvalues)
    inverse_roots = 1.0 / roots
    left_inverse = inverse_roots[:, None] * step.left_defect_basis.conj().T
    right_inverse = step.right_defect_basis * inverse_roots[None, :]
    mobius = np.empty((remaining, rank, rank), dtype=np.complex128)
    for order in range(remaining):
        mobius[order] = left_inverse @ series[order + 1] @ right_inverse

    coupling = (
        step.right_defect_basis.conj().T @ step.choice_parameter.conj().T @ step.left_defect_basis
    )
    result = np.empty_like(mobius)
    # Shmul'yan: with A(z) = D_left+ (Theta - Gamma) / z D_right+, X = A + z A Gamma.H X.
    for order in range(remaining):
        coefficient = mobius[order].copy()
        for left_order in range(order):
            right_order = order - 1 - left_order
            coefficient += mobius[left_order] @ coupling @ result[right_order]
        result[order] = coefficient
    return result


def _schur_parameters(
    normalized_moments: ComplexArray,
    tolerances: Tolerances,
) -> BlockSchurParameters:
    r"""Return the rank-deflated Schur steps of normalized moments, ``C[0] = I``.

    The first step's choice parameter is ``C[1]``.
    """

    steps: list[BlockSchurStep] = []
    series = _initial_choice_series(normalized_moments)
    while series.shape[0]:
        step = _step_from_choice(series[0], len(steps), tolerances)
        steps.append(step)
        if step.terminated:
            break
        series = _next_choice_series(series, step)
    return BlockSchurParameters(
        steps=tuple(steps),
        terminated=bool(steps) and steps[-1].terminated,
        initial_dimension=int(normalized_moments.shape[1]),
    )


@dataclass(frozen=True, slots=True)
class _AssemblyPrefix:
    """The part of an ``L M`` assembly that the terminal block cannot change.

    The closure is one diagonal block, ``[start, stop)``, of one factor, so
    the rest of both factors and of their product is the same for every
    candidate of a phase scan. The ``*_residual_square`` fields hold the
    squared unitarity residuals of each factor's fixed blocks, so a candidate
    adds only its own block's term.
    """

    left: ComplexArray = field(repr=False)
    right: ComplexArray = field(repr=False)
    matrix: ComplexArray = field(repr=False)
    start: int
    stop: int
    in_left: bool
    left_residual_square: float
    right_residual_square: float


def _block_diagonal_residual_square(
    factor: ComplexArray,
    spans: tuple[tuple[int, int], ...],
) -> float:
    """Return the summed squared unitarity residual of the given diagonal blocks.

    For a block-diagonal factor this sum is the whole residual; rows outside
    every span are identity rows and add zero.
    """

    total = 0.0
    for start, stop in spans:
        block = factor[start:stop, start:stop]
        identity = np.eye(stop - start, dtype=np.complex128)
        total += float(np.linalg.norm(block.conj().T @ block - identity) ** 2)
    return total


def _block_diagonal_product(
    left: ComplexArray,
    right: ComplexArray,
    left_spans: tuple[tuple[int, int], ...],
    right_spans: tuple[tuple[int, int], ...],
    total_dimension: int,
) -> ComplexArray:
    r"""Return ``left @ right`` for the two block-diagonal ``L M`` factors.

    Each Julia rotation spans blocks ``j`` and ``j + 1``, so the even ones
    tile ``[0, D)`` and the odd ones tile ``[offsets[1], D)``, and the product
    is block five-diagonal. The banded product needs ``8 / (n_max + 1)**2`` of
    the dense arithmetic. It differs from the dense product by ``2.5e-16`` on
    order-one entries, because BLAS sums in an order set by the operand
    shape, and ``matrix`` is the realization, so the poles move at roundoff.
    """

    result = np.zeros((total_dimension, total_dimension), dtype=np.complex128)
    for start, stop in sorted(left_spans):
        # The blocks of ``right`` these rows reach are adjacent, so one column range.
        lowest, highest = start, stop
        for begin, end in right_spans:
            if begin < stop and end > start:
                lowest = min(lowest, begin)
                highest = max(highest, end)
        result[start:stop, lowest:highest] = (
            left[start:stop, start:stop] @ right[start:stop, lowest:highest]
        )
    return result


def _certify_assembled_factors(
    left: ComplexArray,
    right: ComplexArray,
    matrix: ComplexArray,
    left_residual_square: float,
    right_residual_square: float,
    threshold: float,
    total_dimension: int,
) -> None:
    """Accept an ``L M`` assembly on the blockwise residual bound, else on exact residuals.

    Each factor's unitarity residual is a sum over its own blocks, and the
    product ``P = L M`` has ``||P.H P - I|| <= ||L.H L - I|| + ||M.H M - I||``,
    so the blockwise sum checks all three without a dense product. If the
    bound fails the exact residuals are computed, so nothing the dense check
    rejects is accepted.
    """

    residual = math.sqrt(max(left_residual_square, 0.0)) + math.sqrt(
        max(right_residual_square, 0.0)
    )
    if residual <= threshold:
        return
    identity = np.eye(total_dimension, dtype=np.complex128)
    for label, value in (("left", left), ("right", right), ("full", matrix)):
        exact = float(np.linalg.norm(value.conj().T @ value - identity, ord="fro"))
        if exact > threshold:
            raise RefusalError(
                f"assembled {label} block-CMV factor is not unitary: "
                f"residual={exact:.3e}, threshold={threshold:.3e}",
                kind="block-cmv",
            )


def _assemble_block_cmv(
    parameters: BlockSchurParameters,
    terminal_unitary: Any,
    tolerances: Tolerances,
    prefix: _AssemblyPrefix | None = None,
) -> tuple[
    tuple[BlockSchurStep, ...],
    ComplexArray,
    ComplexArray,
    ComplexArray,
    ComplexArray,
    bool,
]:
    """Complete the choice sequence with the terminal block and assemble ``L M``.

    ``prefix``, from :func:`_assembly_prefix` for these ``parameters``, reuses
    the terminal-independent part of an earlier assembly.
    """

    steps = list(parameters.steps)
    if parameters.terminated:
        terminal = steps[-1].verblunsky_coefficient
        terminal_from_moments = True
    else:
        current_dimension = parameters.initial_dimension if not steps else steps[-1].defect_rank
        terminal = _canonical_terminal_unitary(
            terminal_unitary,
            current_dimension,
        )
        terminal_step = _step_from_choice(
            terminal.conj().T,
            len(steps),
            tolerances,
        )
        if not terminal_step.terminated:
            raise RefusalError(
                "canonical terminal coefficient retained a nonzero defect rank",
                kind="block-cmv",
            )
        steps.append(terminal_step)
        terminal_from_moments = False

    complete = tuple(steps)
    if not complete or not complete[-1].terminated:
        raise RefusalError(
            "finite block-CMV sequence lacks a terminal unitary",
            kind="block-cmv",
        )
    block_dimensions = [parameters.initial_dimension]
    block_dimensions.extend(step.defect_rank for step in complete[:-1])
    total_dimension = int(sum(block_dimensions))
    offsets = np.cumsum([0, *block_dimensions])
    threshold = limits.ROUNDOFF_TOLERANCE * max(1.0, np.sqrt(total_dimension))
    if prefix is None:
        left = np.eye(total_dimension, dtype=np.complex128)
        right = np.eye(total_dimension, dtype=np.complex128)
        spans: tuple[list[tuple[int, int]], list[tuple[int, int]]] = ([], [])
        for index, step in enumerate(complete):
            if step.input_dimension != block_dimensions[index]:
                raise RefusalError(
                    "choice-sequence defect dimensions are inconsistent: "
                    f"index={index}, expected={block_dimensions[index]}, "
                    f"actual={step.input_dimension}",
                    kind="block-cmv",
                )
            start = int(offsets[index])
            stop = start + step.rotation.shape[0]
            factor = left if index % 2 == 0 else right
            factor[start:stop, start:stop] = step.rotation
            spans[index % 2].append((start, stop))
        matrix = _block_diagonal_product(
            left, right, tuple(spans[0]), tuple(spans[1]), total_dimension
        )
        # Blockwise: three dense products would cost ``(n_max + 1)**2 / 8`` times more.
        _certify_assembled_factors(
            left,
            right,
            matrix,
            _block_diagonal_residual_square(left, tuple(spans[0])),
            _block_diagonal_residual_square(right, tuple(spans[1])),
            threshold,
            total_dimension,
        )
    else:
        # Only the closure's block changes: overwrite it in a copy and redo its band.
        terminal_step = complete[-1]
        start, stop = prefix.start, prefix.stop
        if terminal_step.rotation.shape != (stop - start, stop - start):
            raise RefusalError(
                "cached block-CMV assembly does not match this closure: "
                f"expected={(stop - start, stop - start)}, "
                f"actual={terminal_step.rotation.shape}",
                kind="block-cmv",
            )
        matrix = prefix.matrix.copy()
        if prefix.in_left:
            left = prefix.left.copy()
            right = prefix.right
            left[start:stop, start:stop] = terminal_step.rotation
            matrix[start:stop, :] = terminal_step.rotation @ right[start:stop, :]
        else:
            left = prefix.left
            right = prefix.right.copy()
            right[start:stop, start:stop] = terminal_step.rotation
            matrix[:, start:stop] = left[:, start:stop] @ terminal_step.rotation
        # The moved factor's term is the cached fixed-block sum plus the closure's block.
        terminal_square = _block_diagonal_residual_square(
            left if prefix.in_left else right,
            ((start, stop),),
        )
        left_square = prefix.left_residual_square
        right_square = prefix.right_residual_square
        if prefix.in_left:
            left_square += terminal_square
        else:
            right_square += terminal_square
        _certify_assembled_factors(
            left,
            right,
            matrix,
            left_square,
            right_square,
            threshold,
            total_dimension,
        )
    return (
        complete,
        _check.readonly_complex(terminal, "terminal_unitary"),
        _check.readonly_complex(left, "left_factor"),
        _check.readonly_complex(right, "right_factor"),
        _check.readonly_complex(matrix, "matrix"),
        terminal_from_moments,
    )


def _assembly_prefix(
    parameters: BlockSchurParameters,
    complete: tuple[BlockSchurStep, ...],
    left: ComplexArray,
    right: ComplexArray,
    matrix: ComplexArray,
) -> _AssemblyPrefix | None:
    """Return the terminal-independent part of an ``L M`` assembly, or ``None``.

    It is ``None`` if the moments terminate naturally, which fixes the
    terminal block.
    """

    if parameters.terminated:
        return None
    block_dimensions = [parameters.initial_dimension]
    block_dimensions.extend(step.defect_rank for step in complete[:-1])
    offsets = np.cumsum([0, *block_dimensions])
    index = len(complete) - 1
    start = int(offsets[index])
    stop = start + complete[-1].rotation.shape[0]
    in_left = index % 2 == 0
    fixed_left, fixed_right = [], []
    for position, step in enumerate(complete[:-1]):
        begin = int(offsets[position])
        end = begin + step.rotation.shape[0]
        (fixed_left if position % 2 == 0 else fixed_right).append((begin, end))
    return _AssemblyPrefix(
        left=_check.readonly_complex(left, "cached left_factor"),
        right=_check.readonly_complex(right, "cached right_factor"),
        matrix=_check.readonly_complex(matrix, "cached matrix"),
        start=start,
        stop=stop,
        in_left=in_left,
        left_residual_square=_block_diagonal_residual_square(left, tuple(fixed_left)),
        right_residual_square=_block_diagonal_residual_square(right, tuple(fixed_right)),
    )


def _realize_block_cmv_from_prefix(
    normalization: NormalizedMatrixCayleyMoments,
    parameters: BlockSchurParameters,
    terminal_unitary: Any,
    tolerances: Tolerances,
    prefix: _AssemblyPrefix | None,
) -> dict[str, Any]:
    """Assemble and check one closure of a prepared recursion, and return its fields.

    :meth:`BlockCMVRealization.realize` (``prefix=None``) and
    :meth:`BlockCMVRealization.reclose` both call it, so they enforce the
    same contracts.
    """

    rank = normalization.rank
    (
        complete_steps,
        terminal,
        left,
        right,
        matrix,
        terminal_from_moments,
    ) = _assemble_block_cmv(parameters, terminal_unitary, tolerances, prefix)
    dimension = matrix.shape[0]
    selector = np.zeros((dimension, rank), dtype=np.complex128)
    selector[:rank] = np.eye(rank, dtype=np.complex128)
    selector = _check.readonly_complex(selector, "selector")
    scales = np.linalg.norm(normalization.source.values, axis=(1, 2))
    normalized_scales = np.linalg.norm(normalization.values, axis=(1, 2))
    fields = _conserved_fields(
        normalization,
        matrix,
        selector,
        limits.BLOCK_CMV_MOMENT_TOLERANCE * (1.0 + scales),
        limits.BLOCK_CMV_MOMENT_TOLERANCE * (1.0 + normalized_scales),
        "finite block-CMV realization",
    )
    if prefix is None:
        prefix = _assembly_prefix(parameters, complete_steps, left, right, matrix)
    return dict(
        normalization=normalization,
        terminal_unitary=terminal,
        terminal_from_moments=terminal_from_moments,
        matrix=matrix,
        selector=selector,
        maximum_contraction_ratio=max(
            (step.contraction_ratio for step in complete_steps), default=0.0
        ),
        _parameters=parameters,
        _prefix=prefix,
        **fields,
    )
