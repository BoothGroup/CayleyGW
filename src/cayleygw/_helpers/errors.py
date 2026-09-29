"""The three exceptions :mod:`cayleygw` raises."""


class CayleyGWError(Exception):
    """Base class of every error that :mod:`cayleygw` raises on purpose."""


class ValidationError(CayleyGWError, ValueError):
    """Raised when an argument breaks its documented contract or lies outside the domain."""


class RefusalError(CayleyGWError, ArithmeticError):
    """Raised when a computation ran and refused to deliver its result.

    Every attribute but ``kind`` is ``None`` where the raising site does not
    record it; the message carries the same numbers.

    Attributes:
        kind: Stage that refused: ``"rpa-instability"``, ``"resolvent-node"``,
            ``"contour-node"``, ``"cayley-singularity"``, ``"block-cmv"`` or
            ``"sector"``. Test this, not the message, to catch one refusal.
        diagnostics: The closure scan of a sector refusal, or the refusal a
            wrapping site converted.
        check: Block-CMV gate that refused: ``"zeroth_hermiticity"``,
            ``"zeroth_positivity"``, ``"support_leakage"``, ``"zeroth_identity"``
            or ``"schur_contraction"``.
        residual: Quantity the gate measured.
        threshold: Value it was measured against.
        order: Moment order the gate was reading, where the gate is per order.
        condition: Condition number of the retained support.
        zeta: The prohibited resolvent node.
        free_pole_distance: Distance of the node from the nearest free particle-hole pole.
        rpa_pole_distance: Distance of the node from the interval of interacting poles.
        pole_threshold: Distance below which a node is prohibited.
        node_index: Index of the failed contour node.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str | None = None,
        diagnostics: object | None = None,
        check: str | None = None,
        residual: float | None = None,
        threshold: float | None = None,
        order: int | None = None,
        condition: float | None = None,
        zeta: complex | None = None,
        free_pole_distance: float | None = None,
        rpa_pole_distance: float | None = None,
        pole_threshold: float | None = None,
        node_index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.diagnostics = diagnostics
        self.check = check
        self.residual = residual
        self.threshold = threshold
        self.order = order
        self.condition = condition
        self.zeta = zeta
        self.free_pole_distance = free_pole_distance
        self.rpa_pole_distance = rpa_pole_distance
        self.pole_threshold = pole_threshold
        self.node_index = node_index
