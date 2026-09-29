"""The two enums every module shares: the self-energy sector and the screening model."""

from __future__ import annotations

from enum import Enum, unique


@unique
class Sector(str, Enum):
    """Hole or particle sector of the self-energy, each built and realized apart.

    Attributes:
        HOLE: Lesser sector, ``"<"``: poles below the chemical potential, on the
            lower Cayley semicircle.
        PARTICLE: Greater sector, ``">"``: poles above the chemical potential, on
            the upper Cayley semicircle.
    """

    HOLE = "<"
    PARTICLE = ">"

    @property
    def arc_sign(self) -> int:
        """Return ``-1`` for the lower arc and ``+1`` for the upper arc."""

        return -1 if self is Sector.HOLE else 1


@unique
class Screening(str, Enum):
    r"""Particle-hole response model of the screened interaction.

    Both models build, from the positive gaps :math:`D` and the restricted
    singlet kernel :math:`K^S = (L^S)^\dagger L^S`, an operator
    :math:`M = D^p + c\,VV^\dagger` with eigenvalues :math:`\Omega_\nu^p`. The
    screened couplings follow from one amplitude matrix. TDA is another
    approximation to :math:`W`, not a refinement of RPA. It is stable for any
    gapped restricted reference, since :math:`A \succeq D \succ 0`, while RPA
    also needs its squared problem to stay positive definite.

    Attributes:
        RPA: Direct RPA, :math:`A = D + K^S` and :math:`B = K^S`: :math:`p = 2`,
            :math:`V = D^{1/2}(L^S)^\dagger`, :math:`c = 2`, amplitude :math:`X + Y`.
        TDA: Tamm-Dancoff, :math:`B = 0`: :math:`p = 1`, :math:`V = (L^S)^\dagger`,
            :math:`c = 1`, amplitude :math:`X`.
    """

    RPA = "rpa"
    TDA = "tda"

    @property
    def spectral_parameter_power(self) -> int:
        r"""Return the power :math:`p` of :math:`\Omega_\nu` in the eigenvalues of :math:`M`.

        2 for RPA, whose squared problem has eigenvalues :math:`\Omega_\nu^2`, and 1 for TDA.
        """

        return 2 if self is Screening.RPA else 1

    @property
    def kernel_multiplier(self) -> float:
        r"""Return the multiplier :math:`c` of :math:`VV^\dagger` in :math:`M`.

        2 for RPA, whose retained combination is :math:`A + B = D + 2K^S`, and 1 for TDA.
        """

        return 2.0 if self is Screening.RPA else 1.0

    @property
    def label(self) -> str:
        """Return the name used in messages: ``"direct RPA"`` or ``"Tamm-Dancoff"``."""

        return "direct RPA" if self is Screening.RPA else "Tamm-Dancoff"
