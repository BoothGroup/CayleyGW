r"""Frontier energies and spectral functions of a Dyson spectrum.

The spectral function is
:math:`A(\omega) = -\frac{1}{\pi}\,\mathrm{Im}\,\mathrm{Tr}\,G(\omega + i\eta)`.
Energies are in Hartree, except on the axis of :func:`plot_spectrum`, which can
be in eV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Sequence, TypeAlias

import numpy as np

from .._helpers.errors import ValidationError
from .._helpers.tools.logging import HARTREE_TO_EV
from .._helpers.tools.spectrum import _chemical_potential_from_problem
from .._helpers.validate import _check
from .logger import Stages
from .records.spectrum import IPEAResult, SpectrumResult
from .records.upfold import DysonSpectrum

LOGGER = logging.getLogger(__name__)


EnergyUnit: TypeAlias = Literal["hartree", "ev"]


def extract_ip_ea(
    spectrum: DysonSpectrum,
    *,
    n_ip: int,
    n_ea: int,
    chemical_potential: float | None = None,
    minimum_weight: float = 1.0e-1,
    verbose: int = 1,
) -> IPEAResult:
    """Return the ionization potentials and electron affinities nearest the chemical potential.

    Poles with at least ``minimum_weight`` physical weight are ordered outwards
    from the chemical potential. Those below give ``IP = -E`` and those above
    ``EA = -E``, so a negative EA is an unbound addition pole.

    Args:
        spectrum: Result of :func:`diagonalize_upfolded`.
        n_ip: Most ionization potentials to return; fewer if fewer poles pass
            the weight filter.
        n_ea: Most electron affinities to return; fewer if fewer poles pass.
        chemical_potential: Energy in Hartree between removal and addition
            poles; by default the center of the Cayley map of the realization.
        minimum_weight: Physical weight a pole needs to be listed; ``0`` keeps
            every pole. The default ``0.1`` keeps out satellites of negligible
            weight, whose positions are set by roundoff and which would
            otherwise be reported as the frontier, several eV off.
        verbose: ``0`` warnings only, ``1`` the log, ``2`` adds progress counts.

    Returns:
        The energies in Hartree, with their eigenvalue indices, pole energies
        and physical weights, and the pole counts before the weight filter.

    Raises:
        ValidationError: If ``spectrum`` is not a :class:`DysonSpectrum`, a count
            or the weight is negative, or there is no chemical potential to use.
    """

    if not isinstance(spectrum, DysonSpectrum):
        raise ValidationError("spectrum must be a DysonSpectrum")
    stages = Stages(_check.nonnegative_integer(verbose, "verbose"), LOGGER)
    with stages.stage("pole classification"):
        ip_count = _check.nonnegative_integer(n_ip, "n_ip")
        ea_count = _check.nonnegative_integer(n_ea, "n_ea")
        threshold = _check.nonnegative_real(minimum_weight, "minimum_weight")
        mu = (
            _chemical_potential_from_problem(spectrum.problem)
            if chemical_potential is None
            else _check.finite_real(chemical_potential, "chemical_potential")
        )
        occupied = spectrum.energies < mu
        unoccupied = spectrum.energies > mu
        n_occupied = int(np.count_nonzero(occupied))
        n_unoccupied = int(np.count_nonzero(unoccupied))
        n_boundary = spectrum.nstates - n_occupied - n_unoccupied

        eligible = spectrum.physical_weights >= threshold
        removal = np.flatnonzero(occupied & eligible)
        addition = np.flatnonzero(unoccupied & eligible)
        removal_distances = mu - spectrum.energies[removal]
        addition_distances = spectrum.energies[addition] - mu
        removal = removal[np.argsort(removal_distances, kind="stable")][:ip_count]
        addition = addition[np.argsort(addition_distances, kind="stable")][:ea_count]
        removal_energies = spectrum.energies[removal]
        addition_energies = spectrum.energies[addition]
        result = IPEAResult(
            chemical_potential=mu,
            minimum_weight=threshold,
            n_occupied_poles=n_occupied,
            n_unoccupied_poles=n_unoccupied,
            n_boundary_poles=n_boundary,
            ip_eigenvalue_indices=removal,
            ionization_potentials=-removal_energies,
            removal_pole_energies=removal_energies,
            ip_physical_weights=spectrum.physical_weights[removal],
            ea_eigenvalue_indices=addition,
            electron_affinities=-addition_energies,
            addition_pole_energies=addition_energies,
            ea_physical_weights=spectrum.physical_weights[addition],
        )
    stages.result("extract_ip_ea", result, spectrum=spectrum)
    return result


def calculate_spectrum(
    spectrum: DysonSpectrum,
    energy_min: float,
    energy_max: float,
    *,
    n_points: int = 2001,
    broadening: float = 0.05,
    orbital_positions: Sequence[int] | None = None,
    chemical_potential: float | None = None,
    verbose: int = 1,
) -> SpectrumResult:
    """Evaluate the broadened physical spectral function on a uniform grid.

    Each pole adds a Lorentzian of half-width ``broadening`` times its physical
    weight.

    Args:
        spectrum: Result of :func:`diagonalize_upfolded`.
        energy_min: Lower end of the grid in Hartree.
        energy_max: Upper end of the grid in Hartree, above ``energy_min``.
        n_points: Grid points, both ends included.
        broadening: Lorentzian half-width in Hartree.
        orbital_positions: Physical orbitals whose diagonal spectra are also
            computed; none by default.
        chemical_potential: Energy in Hartree between removal and addition
            poles, kept for the plot; by default the center of the Cayley map.
        verbose: ``0`` warnings only, ``1`` the log, ``2`` adds progress counts.

    Returns:
        The grid, the total spectrum, the orbital spectra and the chemical
        potential.

    Raises:
        ValidationError: If ``spectrum`` is not a :class:`DysonSpectrum`, the
            range is empty, an orbital is repeated or out of range, or there is
            no chemical potential to use.
    """

    if not isinstance(spectrum, DysonSpectrum):
        raise ValidationError("spectrum must be a DysonSpectrum")
    stages = Stages(_check.nonnegative_integer(verbose, "verbose"), LOGGER)
    with stages.stage("spectral function", n_points=n_points):
        lower = _check.finite_real(energy_min, "energy_min")
        upper = _check.finite_real(energy_max, "energy_max")
        if upper <= lower:
            raise ValidationError("energy_max must exceed energy_min")
        count = _check.positive_integer(n_points, "n_points", minimum=2)
        eta = _check.positive_real(broadening, "broadening")
        orbitals = np.array(
            [
                _check.nonnegative_integer(orbital, "orbital_positions entry")
                for orbital in (() if orbital_positions is None else orbital_positions)
            ],
            dtype=np.int64,
        )
        if np.unique(orbitals).size != orbitals.size:
            raise ValidationError("orbital_positions must be unique")
        if np.any(orbitals >= spectrum.nphysical):
            raise ValidationError("orbital_positions contains an invalid row")
        mu = (
            _chemical_potential_from_problem(spectrum.problem)
            if chemical_potential is None
            else _check.finite_real(chemical_potential, "chemical_potential")
        )
        grid = np.linspace(lower, upper, count)
        total = spectrum.spectral_function(grid, broadening=eta)
        components = np.array(
            [
                spectrum.spectral_function(grid, broadening=eta, orbital_position=int(orbital))
                for orbital in orbitals
            ],
            dtype=np.float64,
        ).reshape(orbitals.size, count)
        return SpectrumResult(
            energy_grid=grid,
            total=total,
            orbital_positions=orbitals,
            orbital_resolved=components,
            broadening=eta,
            chemical_potential=mu,
        )


def plot_spectrum(
    spectrum: SpectrumResult,
    filename: str | Path,
    *,
    energy_unit: EnergyUnit = "ev",
    relative_to_chemical_potential: bool = True,
    show_orbitals: bool = False,
    title: str | None = None,
    verbose: int = 1,
) -> Path:
    r"""Plot a spectrum to a file whose suffix sets the format, such as PNG, PDF or SVG.

    Args:
        spectrum: Result of :func:`calculate_spectrum`.
        filename: Destination; missing parent directories are created.
        energy_unit: ``"ev"`` or ``"hartree"`` for the horizontal axis; the
            density is scaled so its integral does not change.
        relative_to_chemical_potential: If true, plot :math:`\omega - \mu` and
            mark zero; otherwise plot the absolute energy and mark :math:`\mu`.
        show_orbitals: Also plot the orbital spectra of :func:`calculate_spectrum`.
        title: Figure title; none by default.
        verbose: ``0`` warnings only, ``1`` the log, ``2`` adds progress counts.

    Returns:
        The absolute path of the written file.

    Raises:
        ValidationError: If an option is invalid, Matplotlib is not installed,
            or the file cannot be written.
    """

    stages = Stages(_check.nonnegative_integer(verbose, "verbose"), LOGGER)
    with stages.stage("spectrum plot"):
        if not isinstance(spectrum, SpectrumResult):
            raise ValidationError("spectrum must be a SpectrumResult")
        if not isinstance(filename, (str, Path)) or not str(filename):
            raise ValidationError("filename must be a nonempty path")
        unit = str(energy_unit).lower()
        if unit not in ("hartree", "ev"):
            raise ValidationError("energy_unit must be 'hartree' or 'ev'")
        relative = _check.boolean(relative_to_chemical_potential, "relative_to_chemical_potential")
        _check.boolean(show_orbitals, "show_orbitals")
        if title is not None and not isinstance(title, str):
            raise ValidationError("title must be a string or None")

        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
        except ImportError as error:
            raise ValidationError(
                "plot_spectrum requires Matplotlib; install cayleygw[plot]"
            ) from error

        factor = HARTREE_TO_EV if unit == "ev" else 1.0
        offset = spectrum.chemical_potential if relative else 0.0
        x_values = (spectrum.energy_grid - offset) * factor
        total = spectrum.total / factor

        figure = Figure(figsize=(7.0, 4.0), constrained_layout=True)
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(1, 1, 1)
        axis.plot(x_values, total, color="C0", linewidth=1.6, label="Total")
        axis.fill_between(x_values, total, color="C0", alpha=0.20, linewidth=0.0)
        if show_orbitals:
            for index, (position, values) in enumerate(
                zip(spectrum.orbital_positions, spectrum.orbital_resolved / factor, strict=True)
            ):
                # C0 is the total, so the orbitals cycle through C1 to C9.
                axis.plot(
                    x_values,
                    values,
                    color=f"C{1 + index % 9}",
                    linewidth=1.2,
                    linestyle="--",
                    label=f"MO {int(position)}",
                )
            if spectrum.orbital_positions.size:
                axis.legend(frameon=False)
        marker = 0.0 if relative else spectrum.chemical_potential * factor
        axis.axvline(marker, color="0.45", linewidth=0.9, linestyle=":")
        axis.set_xlim(float(x_values[0]), float(x_values[-1]))
        axis.set_ylim(bottom=0.0)
        unit_label = "eV" if unit == "ev" else "Ha"
        if relative:
            axis.set_xlabel(rf"Energy $\omega-\mu$ ({unit_label})")
        else:
            axis.set_xlabel(rf"Energy $\omega$ ({unit_label})")
        axis.set_ylabel(rf"$A(\omega)$ ({unit_label}$^{{-1}}$)")
        if title is not None:
            axis.set_title(title)
        axis.grid(axis="x", color="0.88", linewidth=0.6)

        destination = Path(filename).expanduser().resolve()
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(destination, dpi=200)
        except (OSError, ValueError) as error:
            raise ValidationError(f"could not save spectrum plot to {destination}") from error
        finally:
            figure.clear()
        return destination


__all__ = [
    "EnergyUnit",
    "IPEAResult",
    "SpectrumResult",
    "calculate_spectrum",
    "extract_ip_ea",
    "plot_spectrum",
]
