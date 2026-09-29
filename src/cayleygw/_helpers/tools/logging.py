"""Private helpers of the logger and the console view; the log file shares some."""

from __future__ import annotations

import math
import subprocess
import threading
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import numpy as np
from rich import box
from rich.console import Group, RenderableType
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from ...tools.logger import Stages

#: Hartree to eV, CODATA 2022.
HARTREE_TO_EV = 27.211386245981

#: Packages whose versions the banner and the log file header list.
PACKAGES = ("numpy", "scipy", "pyscf", "threadpoolctl")


def _table(title: str | None = None, **options: Any) -> Table:
    """Return a table in the house style: a rule under the header, no frame."""

    options.setdefault("box", box.SIMPLE)
    options.setdefault("show_edge", False)
    options.setdefault("padding", (0, 2))
    options.setdefault("collapse_padding", True)
    options.setdefault("title_style", "bold")
    options.setdefault("header_style", "")
    return Table(title=title, **options)


def _duration(seconds: float) -> str:
    """Format a wall time in s, min or h."""

    if seconds < 60.0:
        return f"{seconds:.2f} s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds / 3600.0:.2f} h"


def _version(name: str) -> str | None:
    """Return the installed version of ``name``, ``None`` if it is not installed."""

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _stage_rows(rows: Any, total: float, *, merge: bool) -> list[tuple[str, float, float]]:
    """Return ``(stage, seconds, share)``; ``merge`` sums repeats, longest first."""

    if merge:
        merged: dict[str, list[float]] = {}
        for name, seconds in rows:
            entry = merged.setdefault(name, [0.0, 0])
            entry[0] += seconds
            entry[1] += 1
        ranked = sorted(merged.items(), key=lambda item: item[1][0], reverse=True)
        rows = [(name if n == 1 else f"{name} x{int(n)}", s) for name, (s, n) in ranked]
    return [(name, s, 0.0 if total <= 0.0 else 100.0 * s / total) for name, s in rows]


def _frontier(result: Any) -> list[tuple[str, float, float, int]]:
    """Return ``(label, energy, weight, state)`` for every IP, then every EA."""

    sides = (
        (
            "IP",
            result.ionization_potentials,
            result.ip_physical_weights,
            result.ip_eigenvalue_indices,
        ),
        (
            "EA",
            result.electron_affinities,
            result.ea_physical_weights,
            result.ea_eigenvalue_indices,
        ),
    )
    return [
        (f"{name} {position}", *row)
        for name, *columns in sides
        for position, row in enumerate(zip(*columns))
    ]


def _git_hash() -> str | None:
    """Return the short commit of a source checkout, ``None`` for an install."""

    git = Path(__file__).resolve().parents[4] / ".git"
    if not git.exists():
        return None
    try:
        completed = subprocess.run(
            ["git", f"--git-dir={git}", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _value(value: Any) -> str:
    """Format one option or size for a table cell."""

    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (tuple, list)):
        return escape(", ".join(str(item) for item in value)) or "none"
    if value is None:
        return "none"
    return escape(str(getattr(value, "value", value)))


def _timings(rows: tuple[tuple[str, float], ...], total: float) -> Table:
    """Return the stage times, repeated stages summed, longest first."""

    table = _table("Timings")
    table.add_column("Stage", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Share", justify="right")
    for label, seconds, share in _stage_rows(rows, total, merge=True)[:8]:
        table.add_row(label, _duration(seconds), f"{share:.1f}%")
    table.add_section()
    table.add_row("total wall time", _duration(total), "")
    return table


def _moments_summary(moments: Any, total: float, bare: bool) -> list[RenderableType]:
    """Return the line and contour table of the moments summary panel."""

    orders = f"C_0..C_{moments.n_max}"
    if moments.automatic_n_q_diagnostics is None:
        message = f"Moments {orders} built at N_q = {moments.n_q} in {_duration(total)}."
    else:
        message = (
            f"Moments {orders} [good]converged[/] at N_q = {moments.n_q} in {_duration(total)}."
        )
    contour = moments.contour
    bounds = moments.rpa_spectral_bounds
    table = _table("Contour")
    table.add_column("Quantity", justify="right")
    table.add_column("Value (Ha)", justify="right", style="output")
    table.add_row("Nodes N_q", f"{moments.n_q} ({moments.n_solved} solved)")
    if not bare:
        table.add_row("Ellipse centre", f"{contour.center:.6f}")
        table.add_row(
            "Semi-axes",
            f"{contour.horizontal_radius:.6f}, {contour.vertical_radius:.6f}",
        )
    table.add_row("RPA excitations", f"{bounds.lower:.6f} to {bounds.upper:.6f}")
    table.add_row("Chemical potential", f"{moments.chemical_potential:.6f}")
    return [message, table]


def _upfolded_summary(hamiltonian: Any, total: float, bare: bool) -> list[RenderableType]:
    """Return the line and sector table of the upfolded summary panel."""

    message = (
        f"Upfolded Hamiltonian of dimension {hamiltonian.dimension} built in {_duration(total)}."
    )
    table = _table("Sectors")
    table.add_column("Sector", justify="right")
    table.add_column("Backend", justify="right")
    table.add_column("Poles", justify="right")
    table.add_column("Closure", justify="right")
    table.add_column("Moment error", justify="right")
    table.add_column("Status", justify="left")
    for sector in (hamiltonian.hole, hamiltonian.particle):
        name, poles = sector.sector.name.capitalize(), str(sector.poles.size)
        if not hasattr(sector, "closure_scan"):
            # Built through the low-level interface: poles, but no realization record.
            table.add_row(name, "", poles, "", "", "")
            continue
        phase = sector.selected_phase
        closure = "natural" if phase is None else f"{phase / math.pi:.3f}pi"
        style = "good" if sector.maximum_conservation_ratio <= 1.0 else "ok"
        table.add_row(
            name,
            sector.closure_scan.realization_algorithm,
            poles,
            closure,
            f"[{style}]{sector.maximum_conserved_moment_residual:.2e}[/]",
            _flags(sector),
        )
    return [message, table]


def _flags(sector: Any) -> str:
    """Name the bands and fallbacks a sector was accepted through."""

    flags = [
        label
        for label, raised in (
            ("marginal support", sector.support_is_marginal),
            ("marginal conservation", sector.conservation_is_marginal),
            (f"closure rank {sector.closure_rank}", sector.closure_rank > 0),
        )
        if raised
    ]
    if not flags:
        return "[good]clean[/]"
    return "[ok]" + ", ".join(flags) + "[/]"


def _quasiparticles(event: dict) -> RenderableType:
    """Return the quasiparticle panel of ``extract_ip_ea``."""

    result, spectrum = event["value"], event["spectrum"]
    labels = _orbital_labels(spectrum, result.chemical_potential)
    table = _table()
    table.add_column("Excitation", justify="right")
    table.add_column("Energy (eV)", justify="right", style="output")
    table.add_column("Energy (Ha)", justify="right")
    table.add_column("QP weight", justify="right")
    table.add_column("Dominant orbitals", justify="left")
    for number, (label, energy, weight, index) in enumerate(_frontier(result)):
        table.add_row(
            label,
            f"{energy * HARTREE_TO_EV:.4f}",
            f"{energy:.8f}",
            f"{weight:.4f}",
            _dominant(spectrum.orbital_weights[:, index], labels),
            end_section=number + 1 == result.ionization_potentials.size,
        )
    homo, lumo = _frontier_orbitals(spectrum, result.chemical_potential)
    parts: list[RenderableType] = [
        "Excitations count outwards from the gap.\n"
        f"HOMO and LUMO below are mean-field orbitals {homo} and {lumo}.",
        "",
        table,
    ]
    if result.ionization_potentials.size and result.electron_affinities.size:
        gap = result.ionization_potentials[0] - result.electron_affinities[0]
        parts.extend(("", f"Quasiparticle gap: [output]{gap * HARTREE_TO_EV:.4f}[/] eV"))
    return Panel(Group(*parts), title="Quasiparticle energies", padding=(1, 2), expand=False)


def _orbital_labels(spectrum: Any, chemical_potential: float) -> list[str]:
    """Name the physical orbitals HOMO, HOMO-1, ..., LUMO, LUMO+1, ..."""

    energies = np.real(np.diag(spectrum.problem.reference_operator))
    labels = [""] * energies.size
    occupied = np.flatnonzero(energies < chemical_potential)
    virtual = np.flatnonzero(energies >= chemical_potential)
    for rank, index in enumerate(occupied[np.argsort(-energies[occupied])]):
        labels[index] = "HOMO" if rank == 0 else f"HOMO-{rank}"
    for rank, index in enumerate(virtual[np.argsort(energies[virtual])]):
        labels[index] = "LUMO" if rank == 0 else f"LUMO+{rank}"
    return labels


def _frontier_orbitals(spectrum: Any, chemical_potential: float) -> tuple[int, int]:
    """Return the mean-field indices of the HOMO and LUMO, counting a frozen core."""

    energies = np.real(np.diag(spectrum.problem.reference_operator))
    occupied = np.flatnonzero(energies < chemical_potential)
    virtual = np.flatnonzero(energies >= chemical_potential)
    homo = int(occupied[np.argmax(energies[occupied])])
    lumo = int(virtual[np.argmin(energies[virtual])])
    indices = spectrum.problem.mo_indices
    return (homo, lumo) if indices is None else (indices[homo], indices[lumo])


def _dominant(weights: np.ndarray, labels: list[str]) -> str:
    """Name the at most three orbitals above 10% of a state's weight."""

    total = float(np.sum(weights))
    if total <= 0.0:
        return ""
    order = np.argsort(weights)[::-1][:3]
    return ", ".join(
        f"{labels[index]} ({100.0 * weights[index] / total:.0f}%)"
        for index in order
        if weights[index] > 0.1 * total
    )


def _reconstruction(event: dict) -> RenderableType:
    """Return the verdict line of ``reconstruction_test``."""

    result = event["value"]
    style, verdict = ("good", "passed") if result.passed else ("bad", "failed")
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right", style="comment")
    grid.add_row(
        "Moment reconstruction: largest relative error "
        f"[{style}]{result.maximum_conserved_relative_error:.3e}[/]",
        verdict,
    )
    return grid


_SUMMARIES: dict[str, Callable[[Any, float, bool], list[RenderableType]]] = {
    "build_cayley_moments": _moments_summary,
    "build_upfolded_hamiltonian": _upfolded_summary,
}

_RESULTS: dict[str, Callable[[dict], RenderableType]] = {
    "extract_ip_ea": _quasiparticles,
    "reconstruction_test": _reconstruction,
}


# The recorders collecting nested stages, outermost first.
_ACTIVE: list["Stages"] = []
_ACTIVE_LOCK = threading.Lock()


_Function = TypeVar("_Function", bound=Callable[..., Any])
