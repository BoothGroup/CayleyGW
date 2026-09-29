"""Records of :mod:`cayleygw.moments`: the history of the automatic ``N_q`` ladder."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..._helpers.types import Sector
from ..._helpers.validate import FloatArray


@dataclass(frozen=True, slots=True)
class AutomaticNQSectorDiagnostics:
    """One sector's errors and feasibility result for one ``N_q`` doubling.

    An order passes when ``error <= tolerance * fine_norm``. The feasibility
    checks run only after both sectors pass the error test.

    Attributes:
        sector: Hole or particle.
        coarse_n_q: Nodes of the coarse grid.
        fine_n_q: Nodes of the fine grid, twice the coarse.
        absolute_errors: ``||C_fine - C_coarse||`` per order.
        relative_errors: The same divided by ``||C_fine||``.
        acceptance_thresholds: ``tolerance * ||C_fine||`` per order.
        feasibility_violations: Failed feasibility checks of the fine moments;
            empty if they passed or did not run.
    """

    sector: Sector
    coarse_n_q: int
    fine_n_q: int
    absolute_errors: FloatArray = field(repr=False)
    relative_errors: FloatArray = field(repr=False)
    acceptance_thresholds: FloatArray = field(repr=False)
    feasibility_violations: tuple[str, ...] = ()

    @property
    def moment_errors_converged(self) -> bool:
        """Whether every order meets its error threshold."""

        return bool(np.all(self.absolute_errors <= self.acceptance_thresholds))

    @property
    def converged(self) -> bool:
        """Whether the errors and the feasibility checks both pass."""

        return self.moment_errors_converged and not self.feasibility_violations

    @property
    def normalized_errors(self) -> FloatArray:
        """Each absolute error divided by its threshold."""

        result = np.divide(
            self.absolute_errors,
            self.acceptance_thresholds,
            out=np.zeros_like(self.absolute_errors),
            where=self.acceptance_thresholds > 0.0,
        )
        result[(self.acceptance_thresholds == 0.0) & (self.absolute_errors > 0.0)] = np.inf
        result.setflags(write=False)
        return result

    @property
    def worst_order(self) -> int:
        """Order with the largest normalized error."""

        return int(np.argmax(self.normalized_errors))

    @property
    def maximum_normalized_error(self) -> float:
        """Largest normalized error over the orders."""

        return float(np.max(self.normalized_errors))


@dataclass(frozen=True, slots=True)
class AutomaticNQDiagnostics:
    """History of one converged automatic ``N_q`` selection.

    Attributes:
        initial_n_q: First node count of the ladder.
        maximum_n_q: Largest node count the ladder allows.
        tolerance: Relative error tolerance of every order and sector.
        refinements: One ``(hole, particle)`` pair per doubling, ending at the
            selected grid.
    """

    initial_n_q: int
    maximum_n_q: int
    tolerance: float
    refinements: tuple[tuple[AutomaticNQSectorDiagnostics, AutomaticNQSectorDiagnostics], ...]

    @property
    def selected_n_q(self) -> int:
        """Node count the ladder stopped at."""

        return self.refinements[-1][0].fine_n_q
