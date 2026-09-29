"""Cayley-moment :math:`G_0W_0` for molecules on top of PySCF.

The self-energy of a closed-shell mean field is represented by a few of its
Cayley moments, computed by contour quadrature and realized as a pole sum. The
quasiparticle energies are the eigenvalues of the upfolded Hamiltonian.
:class:`CayleyGW` runs the whole calculation; the steps are
:func:`build_cayley_moments`, :func:`build_upfolded_hamiltonian`,
:func:`diagonalize_upfolded` and :func:`extract_ip_ea`.
:class:`ExactG0W0SelfEnergy` keeps every pole and is the reference for small
systems.
"""

from ._helpers.version import __version__
from ._helpers.tolerances import Tolerances
from ._helpers.types import Screening, Sector
from ._helpers.errors import CayleyGWError, RefusalError, ValidationError
from .tools.reference import ExactG0W0SelfEnergy
from .tools.logger import enable_logging
from .moments import build_cayley_moments
from .kernel import CayleyGW
from .upfold import (
    UpfoldedDysonHamiltonian,
    build_upfolded_hamiltonian,
    diagonalize_upfolded,
    reconstruction_test,
)
from .tools.spectrum import (
    calculate_spectrum,
    extract_ip_ea,
    plot_spectrum,
)

enable_logging()  # the standard console view is on from import

__all__ = [
    "CayleyGW",
    "enable_logging",
    "CayleyGWError",
    "ExactG0W0SelfEnergy",
    "RefusalError",
    "Screening",
    "Sector",
    "Tolerances",
    "UpfoldedDysonHamiltonian",
    "ValidationError",
    "__version__",
    "build_cayley_moments",
    "build_upfolded_hamiltonian",
    "calculate_spectrum",
    "diagonalize_upfolded",
    "extract_ip_ea",
    "plot_spectrum",
    "reconstruction_test",
]
