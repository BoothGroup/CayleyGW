"""Private helpers of :mod:`cayleygw.tools.logfile`: the header and the plain-text blocks."""

from __future__ import annotations

import datetime
import logging
import math
import os
import platform
import sys
import time
from typing import Any, Callable

import numpy as np

from ..version import __version__
from .logging import (
    HARTREE_TO_EV,
    PACKAGES,
    _duration,
    _frontier,
    _frontier_orbitals,
    _git_hash,
    _orbital_labels,
    _stage_rows,
    _version,
)

_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


class _Formatter(logging.Formatter):
    """One timestamped line per record, with its values written out beneath."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        event = getattr(record, "cayleygw", None)
        describe = None if event is None else _BLOCKS.get(event["event"])
        text = message if describe is None else describe(event, message)
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        prefix = f"{stamp}.{int(record.msecs):03d} {record.levelname:<7} "
        if record.threadName != "MainThread":
            text = f"[{record.threadName}] {text}"
        lines = text.splitlines() or [""]
        indent = " " * len(prefix)
        return "\n".join([prefix + lines[0]] + [indent + line for line in lines[1:]])


def _header() -> str:
    """Describe the run: versions, command, host and native thread setup."""

    opened = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    commit = _git_hash()
    title = f"cayleygw {__version__}" + ("" if commit is None else f" (git {commit})")
    versions = [f"{name} {_version(name) or 'not installed'}" for name in (*PACKAGES, "rich")]
    cpus = f"{os.cpu_count()} CPUs"
    if hasattr(os, "sched_getaffinity"):
        cpus += f", {len(os.sched_getaffinity(0))} available to this process"
    rows = [
        ("command", " ".join([sys.executable, *sys.argv])),
        ("directory", os.getcwd()),
        ("python", platform.python_version()),
        ("host", f"{platform.node()}, {platform.platform()}, {cpus}"),
        ("packages", ", ".join(versions)),
        (
            "environment",
            " ".join(f"{name}={os.environ.get(name, 'unset')}" for name in _THREAD_VARIABLES),
        ),
    ]
    try:
        from threadpoolctl import threadpool_info

        for library in threadpool_info():
            rows.append(
                (
                    "native",
                    f"{library.get('internal_api')} {library.get('version')} "
                    f"({library.get('user_api')}), {library.get('num_threads')} "
                    f"threads, {library.get('filepath')}",
                )
            )
    except Exception as error:  # a missing or broken native library is itself worth a line
        rows.append(("native", f"not inspected: {error}"))
    width = max(len(key) for key, _ in rows)
    lines = ["=" * 78, f"{title}: log opened {opened}"]
    lines.extend(f"  {key:<{width}}  {value}" for key, value in rows)
    return "\n".join(lines) + "\n"


def _pairs(values: dict[str, Any]) -> list[str]:
    """Return one aligned ``key  value`` line per entry."""

    if not values:
        return []
    width = max(len(key) for key in values)
    return [f"  {key:<{width}}  {_text(value)}" for key, value in values.items()]


def _text(value: Any) -> str:
    """Format one option or size; an enum by its value, an empty tuple as none."""

    if isinstance(value, tuple) and not value:
        return "none"
    return str(getattr(value, "value", value))


def _options(event: dict, message: str) -> str:
    """Write every option and size of a main call."""

    title = event["title"]
    lines = [f"{title}: options"] + _pairs({**event["options"], **event["details"]})
    lines += [f"{title}: sizes"] + _pairs(event["sizes"])
    return "\n".join(lines)


def _summary(event: dict, message: str) -> str:
    """Write the result of a main call and every stage time, in order."""

    result, title, total = event["result"], event["title"], event["total"]
    rows = _stage_rows(event["rows"], total, merge=False)
    width = max([len(title)] + [len(name) for name, _, _ in rows]) + 2
    lines = [message, f"  {'stage':<{width}}{'seconds':>10}  share"]
    lines += [
        f"  {name:<{width}}{seconds:>10.2f}  {share:5.1f}%" for name, seconds, share in rows
    ]
    lines.append(f"  {'total wall time':<{width}}{total:>10.2f}")
    times = "\n".join(lines)
    if result is None:
        return f"{title}: failed after {_duration(total)}\n{times}"
    describe = _SUMMARIES.get(title)
    return times if describe is None else describe(result) + "\n" + times


def _result(event: dict, message: str) -> str:
    """Write a result that has a renderer, else the plain message."""

    describe = _RESULTS.get(event["title"])
    return message if describe is None else describe(event)


def _moments(moments: Any) -> str:
    """Write the contour, the ``N_q`` history and the moment norms of a build."""

    contour = moments.contour
    bounds = moments.rpa_spectral_bounds
    lines = [
        "build_cayley_moments: result",
        f"  moments C_0..C_{moments.n_max}, conserved through "
        f"C_{moments.n_conserved}, 1 spare",
        f"  N_q {moments.n_q}, {moments.n_solved} nodes solved ({moments.quadrature_rule} rule, conjugate pairs)",
        f"  ellipse centre {contour.center:.10g} Ha, semi-axes "
        f"{contour.horizontal_radius:.10g} and {contour.vertical_radius:.10g} Ha",
    ]
    diagnostics = contour.equioscillation_diagnostics
    if diagnostics is not None:
        lines.append(f"  equioscillation sigma_min {diagnostics.sigma_min:.10g}")
    lines += [
        f"  RPA excitations {bounds.lower:.10g} to {bounds.upper:.10g} Ha "
        f"({bounds.source}, certified)",
        f"  chemical potential {moments.chemical_potential:.10g} Ha, omega_p "
        f"{moments.mapping.scale:.10g} Ha",
    ]
    history = moments.automatic_n_q_diagnostics
    if history is not None:
        lines.append(
            f"  automatic N_q from {history.initial_n_q} to at most "
            f"{history.maximum_n_q}, tolerance {history.tolerance:.3e}"
        )
        for refinement in history.refinements:
            sectors = []
            for sector in refinement:
                described = (
                    f"{sector.sector.name.lower()} worst C_{sector.worst_order} "
                    f"{sector.maximum_normalized_error:.3e}"
                )
                if sector.feasibility_violations:
                    described += " (" + ", ".join(sector.feasibility_violations) + ")"
                sectors.append(described)
            converged = all(sector.converged for sector in refinement)
            verdict = "converged" if converged else "not converged"
            lines.append(
                f"    {refinement[0].coarse_n_q} -> {refinement[0].fine_n_q}: "
                + "; ".join(sectors)
                + f"; {verdict}"
            )
    lines.append("  Frobenius norm of each moment order")
    lines.append(f"    {'order':>5}  {'hole':>12}  {'particle':>12}")
    for order in range(moments.n_max + 1):
        lines.append(
            f"    {order:>5}  {np.linalg.norm(moments.hole.moments[order]):>12.6e}  "
            f"{np.linalg.norm(moments.particle.moments[order]):>12.6e}"
        )
    return "\n".join(lines)


def _upfolded(hamiltonian: Any) -> str:
    """Write the dimension and, per sector, the closure, residuals and discards."""

    lines = [
        "build_upfolded_hamiltonian: result",
        f"  dimension {hamiltonian.dimension} = {hamiltonian.nphysical} orbitals + "
        f"{hamiltonian.nhole} hole poles + {hamiltonian.nparticle} particle poles",
    ]
    for sector in (hamiltonian.hole, hamiltonian.particle):
        name = sector.sector.name.lower()
        if not hasattr(sector, "closure_scan"):
            lines.append(f"  {name}: {sector.poles.size} poles, no realization record")
            continue
        scan = sector.closure_scan
        phase = sector.selected_phase
        closure = "natural termination" if phase is None else f"phase {phase / math.pi:.6f} pi"
        lines += [
            f"  {name}: {scan.realization_algorithm} (requested "
            f"{scan.requested_realization_algorithm}), {sector.poles.size} poles",
            f"    closure {closure}, candidate {sector.closure_rank} in preference "
            f"order of {len(scan.candidates)} realized, terminal dimension "
            f"{sector.terminal_dimension}",
        ]
        residuals = ", ".join(
            f"C_{order} {value:.3e}" for order, value in enumerate(sector.moment_residuals)
        )
        lines += [
            f"    moment residuals {residuals}",
            f"    largest conserved residual {sector.maximum_conserved_moment_residual:.3e}, "
            f"{sector.maximum_conservation_ratio:.3g} of its threshold",
            f"    discarded weight {sector.discarded_total_weight:.3e} (wrong arc "
            f"{sector.discarded_wrong_arc_weight:.3e}, near singular "
            f"{sector.discarded_near_singular_weight:.3e}); zeroth weight "
            f"redistributed {sector.zeroth_weight_redistributed} "
            f"({sector.redistributed_zeroth_weight:.3e})",
            f"    marginal support {sector.support_is_marginal}, conservation "
            f"{sector.conservation_is_marginal}",
        ]
    return "\n".join(lines)


def _quasiparticles(event: dict) -> str:
    """Write every frontier energy with each orbital above 1% of its weight."""

    result, spectrum = event["value"], event["spectrum"]
    labels = _orbital_labels(spectrum, result.chemical_potential)
    homo, lumo = _frontier_orbitals(spectrum, result.chemical_potential)
    lines = [
        "extract_ip_ea: result",
        f"  chemical potential {result.chemical_potential:.10g} Ha, minimum weight "
        f"{result.minimum_weight:g}",
        f"  HOMO and LUMO below are mean-field orbitals {homo} and {lumo}",
        f"  poles: {result.n_occupied_poles} below, {result.n_unoccupied_poles} above, "
        f"{result.n_boundary_poles} at the chemical potential",
    ]
    for label, energy, weight, index in _frontier(result):
        orbital = spectrum.orbital_weights[:, index]
        total = float(np.sum(orbital))
        shares = ", ".join(
            f"{labels[i]} {100.0 * orbital[i] / total:.1f}%"
            for i in np.argsort(orbital)[::-1]
            if total > 0.0 and orbital[i] > 0.01 * total
        )
        lines.append(
            f"  {label}: {energy * HARTREE_TO_EV:.6f} eV "
            f"({energy:.10f} Ha), weight {weight:.6f}, state {index}; {shares}"
        )
    if result.ionization_potentials.size and result.electron_affinities.size:
        gap = result.ionization_potentials[0] - result.electron_affinities[0]
        lines.append(f"  quasiparticle gap {gap * HARTREE_TO_EV:.6f} eV ({gap:.10f} Ha)")
    return "\n".join(lines)


def _reconstruction(event: dict) -> str:
    """Write the reconstruction error of every order in both sectors."""

    result = event["value"]
    lines = [
        f"reconstruction_test: {'passed' if result.passed else 'failed'}; largest "
        f"conserved error {result.maximum_conserved_absolute_error:.3e} absolute, "
        f"{result.maximum_conserved_relative_error:.3e} relative"
    ]
    for sector in (result.hole, result.particle):
        for order, (absolute, relative, threshold) in enumerate(
            zip(sector.absolute_errors, sector.relative_errors, sector.acceptance_thresholds)
        ):
            role = "conserved" if order <= sector.conserved_order else "withheld"
            lines.append(
                f"  {sector.sector.name.lower()} C_{order}: {absolute:.3e} absolute, "
                f"{relative:.3e} relative, threshold {threshold:.3e} ({role})"
            )
    return "\n".join(lines)


_BLOCKS: dict[str, Callable[[dict, str], str]] = {
    "options": _options,
    "summary": _summary,
    "result": _result,
}

_SUMMARIES: dict[str, Callable[[Any], str]] = {
    "build_cayley_moments": _moments,
    "build_upfolded_hamiltonian": _upfolded,
}

_RESULTS: dict[str, Callable[[dict], str]] = {
    "extract_ip_ea": _quasiparticles,
    "reconstruction_test": _reconstruction,
}
