r"""The regression protocol, run at a tolerance against a stored reference.

Every case runs through the public API at the protocol's settings
(``omega_p=0.5``, ``N_q=512``, the Frobenius enclosure, ``restricted`` closure,
one spare order) and is compared with ``tests/regression_reference.json``. What
holds on any machine is asserted: node counts, the contour geometry, and the
frontier energies and weights within the tolerances below. What moves with the
BLAS, the core count or the LAPACK version (pole counts, the backend chosen,
the terminal phase) goes to :mod:`drift`, which reports it and never gates.

Regenerate the reference on one machine, single-threaded and serial, after a
change meant to move the numbers::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        CAYLEYGW_REGRESSION_CAPTURE=1 pytest -n 0 tests/test_regression.py

Capture rewrites the file case by case and asserts nothing. It must not run
under xdist, where workers would overwrite each other's cases.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import drift
import numpy as np
import pytest
from pyscf import dft, gto, scf

from cayleygw import (
    ExactG0W0SelfEnergy,
    Screening,
    Sector,
    Tolerances,
    UpfoldedDysonHamiltonian,
    build_cayley_moments,
    build_upfolded_hamiltonian,
    diagonalize_upfolded,
    extract_ip_ea,
    reconstruction_test,
)
from cayleygw._helpers.cayley import CayleyMap
from cayleygw.realization.sector import SectorSelfEnergyRealization

pytestmark = pytest.mark.pyscf

REFERENCE_PATH = Path(__file__).resolve().parent / "regression_reference.json"
CAPTURE = bool(os.environ.get("CAYLEYGW_REGRESSION_CAPTURE"))

#: Weighted realized pole energies in Hartree; machines move them far less, regressions more.
REALIZED_ENERGY = 1.0e-6
#: Physical weights of realized poles.
REALIZED_WEIGHT = 1.0e-6
#: Quantities reached by sums and eigensolves only: mu, the bounds, the ellipse, exact poles.
DETERMINISTIC = 1.0e-8
#: Relative agreement of moment norms between machines.
MOMENT_NORM = 1.0e-9
#: Contour moments against the exact Cayley moments at ``N_q=512``.
QUADRATURE = 1.0e-7

OMEGA_P = 0.5
N_Q = 512
SPARE = 1
CONSERVED_ORDERS = (2, 5, 8)
MOMENT_SETTINGS = dict(
    n_q=N_Q,
    omega_p=OMEGA_P,
    contour_orbital_block_size=1,
    spectral_bound="frobenius",
    n_workers=1,
    native_threads=1,
)
UPFOLD_SETTINGS = dict(
    terminal_selection="restricted",
    realization="auto",
    n_workers=1,
    native_threads=1,
)
#: Both rank floors stay exercised: ``default`` is the library's, ``library`` a tighter one.
RANK_FLOORS = {
    "default": 1.0e-10,
    "library": 1.0e-12,
}
MOLECULES = {
    "H2O": "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
    "CO": "C 0 0 0; O 0 0 1.128",
    "N2": "N 0 0 0; N 0 0 1.098",
    "LiH": "Li 0 0 0; H 0 0 1.5949",
    "H2": "H 0 0 0; H 0 0 0.74",
}
CONTOUR_CASES = [
    (name, reference, screening, n_conserved)
    for name in ("H2O", "CO", "N2")
    for reference in ("rhf", "pbe")
    for screening in ("rpa", "tda")
    for n_conserved in CONSERVED_ORDERS
]
EXACT_CASES = {
    "exact/LiH/sto-3g/rpa": ("LiH", "sto-3g", "rpa", 0.5, 6, (1, 2, 3, 4, 5)),
    "exact/LiH/sto-3g/tda": ("LiH", "sto-3g", "tda", 0.5, 6, (1, 2, 3, 4, 5)),
    "exact/LiH/6-31g/rpa": ("LiH", "6-31g", "rpa", 1.0, 6, (2, 4)),
    "exact/H2/6-31g/rpa": ("H2", "6-31g", "rpa", 1.0, 6, (1, 2, 3)),
}


# Reference file.


def _load_reference() -> dict:
    """Return the stored reference, or an empty one."""

    if not REFERENCE_PATH.exists():
        return {"metadata": {}, "cases": {}}
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def _store_case(key: str, summary: dict) -> None:
    """Write one case into the reference file."""

    reference = _load_reference()
    reference["metadata"] = {
        "note": "One single-threaded machine's answer to the regression "
        "protocol; energies in Hartree.",
        **drift.machine_description(),
    }
    reference["cases"][key] = summary
    REFERENCE_PATH.write_text(
        json.dumps(reference, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


def _expected(key: str) -> dict:
    """Return the stored case, or fail if it has not been captured."""

    cases = _load_reference()["cases"]
    if key not in cases:
        pytest.fail(f"{key} has no reference; capture one (see the module docstring)")
    return cases[key]


# Comparison.


def _close(label: str, got, want, tolerance: float, *, relative: bool = False) -> None:
    """Assert ``got`` matches ``want`` within ``tolerance``, absolute or relative."""

    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    assert got.shape == want.shape, f"{label}: shape {got.shape} against {want.shape}"
    if got.size == 0:
        return
    gap = np.max(np.abs(got - want))
    scale = np.max(np.abs(want)) if relative else 1.0
    assert gap <= tolerance * max(scale, 1e-300), (
        f"{label}: largest gap {gap:.3e} exceeds {tolerance:.1e}"
        f"{' relative' if relative else ''}; got {got.tolist()} against {want.tolist()}"
    )


def _sector_summary(realization) -> dict:
    """Return the facts of one realized sector that the reference stores."""

    scan = realization.closure_scan
    phase = realization.selected_phase
    return {
        "algorithm": scan.realization_algorithm,
        "npoles": int(realization.poles.size),
        "terminal_dimension": int(realization.terminal_dimension),
        "selected_phase": None if phase is None else float(phase),
        "candidates_evaluated": len(scan.candidates),
        "maximum_conservation_ratio": float(realization.maximum_conservation_ratio),
        "discarded_total_weight": float(realization.discarded_total_weight),
    }


def _record_sector(prefix: str, got: dict) -> None:
    """Report the machine-sensitive facts of one realized sector."""

    for name in (
        "algorithm",
        "npoles",
        "terminal_dimension",
        "candidates_evaluated",
        "maximum_conservation_ratio",
    ):
        drift.record(f"{prefix}.{name}", got[name])
    if got["selected_phase"] is not None:
        drift.record(f"{prefix}.selected_phase", got["selected_phase"])


def _frontier(energies, weights, mu: float, *, count: int = 3, minimum_weight: float = 0.1) -> dict:
    """Return the ``count`` states nearest ``mu`` on each side that carry enough weight."""

    energies = np.asarray(energies)
    weights = np.asarray(weights)
    keep = weights >= minimum_weight
    below = np.flatnonzero(keep & (energies < mu))
    above = np.flatnonzero(keep & (energies > mu))
    below = below[np.argsort(-energies[below])][:count]
    above = above[np.argsort(energies[above])][:count]
    return {
        "below": [float(x) for x in energies[below]],
        "below_weight": [float(x) for x in weights[below]],
        "above": [float(x) for x in energies[above]],
        "above_weight": [float(x) for x in weights[above]],
    }


def _compare_frontier(label: str, got: dict, want: dict, energy: float, weight: float) -> None:
    """Assert the frontier energies and weights on both sides of ``mu`` agree."""

    for side in ("below", "above"):
        _close(f"{label} {side}", got[side], want[side], energy)
        _close(f"{label} {side} weight", got[f"{side}_weight"], want[f"{side}_weight"], weight)


# Mean fields, shared within one process.


@functools.lru_cache(maxsize=None)
def _mean_field(name: str, basis: str, reference: str, density_fit: bool):
    """Return a converged mean field, built once per process."""

    molecule = gto.M(atom=MOLECULES[name], basis=basis, unit="Angstrom", verbose=0)
    if reference == "rhf":
        mf = scf.RHF(molecule)
    else:
        mf = dft.RKS(molecule)
        mf.xc = reference
        mf.grids.level = 3
    if density_fit:
        mf = mf.density_fit()
    mf.conv_tol = 1.0e-12
    mf.kernel()
    assert mf.converged, (name, basis, reference)
    return mf


# Contour cases.


def _contour_summary(name: str, reference: str, screening: str, n_conserved: int) -> dict:
    """Build one contour case, realize it at both rank floors and summarize it."""

    mf = _mean_field(name, "cc-pvdz", reference, True)
    moments = build_cayley_moments(
        mf, n_conserved=n_conserved, screening=screening, **MOMENT_SETTINGS
    )
    contour = moments.contour
    bounds = moments.rpa_spectral_bounds
    summary = {
        "n_q": int(moments.n_q),
        "n_max": int(moments.n_max),
        "n_resolvent_evaluations": int(
            moments.n_q // 2 if moments.conjugate_paired else moments.n_q
        ),
        "chemical_potential": float(moments.chemical_potential),
        "contour": [
            float(contour.center),
            float(contour.horizontal_radius),
            float(contour.vertical_radius),
        ],
        "spectral_bounds": [float(bounds.lower), float(bounds.upper)],
        "static_correction_norm": float(np.linalg.norm(moments.static_correction)),
        "hole_moment_norms": [float(np.linalg.norm(block)) for block in moments.hole.moments],
        "particle_moment_norms": [
            float(np.linalg.norm(block)) for block in moments.particle.moments
        ],
        "upfold": {},
    }
    for label, rank_floor in RANK_FLOORS.items():
        hamiltonian = build_upfolded_hamiltonian(
            moments,
            n_conserved=n_conserved,
            tolerances=Tolerances(rank_floor=rank_floor),
            **UPFOLD_SETTINGS,
        )
        spectrum = diagonalize_upfolded(hamiltonian, native_threads=1)
        charged = extract_ip_ea(spectrum, n_ip=3, n_ea=3)
        check = reconstruction_test(hamiltonian)
        summary["upfold"][label] = {
            "dimension": int(hamiltonian.dimension),
            "hole": _sector_summary(hamiltonian.hole),
            "particle": _sector_summary(hamiltonian.particle),
            "ip": [float(x) for x in charged.ionization_potentials],
            "ip_weight": [float(x) for x in charged.ip_physical_weights],
            "ea": [float(x) for x in charged.electron_affinities],
            "ea_weight": [float(x) for x in charged.ea_physical_weights],
            "reconstruction_passed": bool(check.passed),
            "max_conserved_relative_error": float(check.maximum_conserved_relative_error),
        }
    return summary


@pytest.mark.parametrize(
    "name, reference, screening, n_conserved",
    CONTOUR_CASES,
    ids=[f"{n}-{r}-{s}-n{k}" for n, r, s, k in CONTOUR_CASES],
)
def test_contour_case(name, reference, screening, n_conserved) -> None:
    key = f"contour/{name}/cc-pvdz/{reference}/{screening}/n{n_conserved}"
    got = _contour_summary(name, reference, screening, n_conserved)
    if CAPTURE:
        _store_case(key, got)
        return
    want = _expected(key)
    for field in ("n_q", "n_max", "n_resolvent_evaluations"):
        assert got[field] == want[field], f"{key}: {field} {got[field]} against {want[field]}"
    _close(
        f"{key} chemical potential",
        got["chemical_potential"],
        want["chemical_potential"],
        DETERMINISTIC,
    )
    _close(f"{key} contour", got["contour"], want["contour"], DETERMINISTIC)
    _close(f"{key} spectral bounds", got["spectral_bounds"], want["spectral_bounds"], DETERMINISTIC)
    _close(
        f"{key} static correction",
        got["static_correction_norm"],
        want["static_correction_norm"],
        DETERMINISTIC,
    )
    for sector in ("hole", "particle"):
        _close(
            f"{key} {sector} moment norms",
            got[f"{sector}_moment_norms"],
            want[f"{sector}_moment_norms"],
            MOMENT_NORM,
            relative=True,
        )
    for label in RANK_FLOORS:
        got_entry = got["upfold"][label]
        want_entry = want["upfold"][label]
        prefix = f"{key}/{label}"
        assert got_entry["reconstruction_passed"], f"{prefix}: the reconstruction test failed"
        assert len(got_entry["ip"]) == len(want_entry["ip"]), (
            f"{prefix}: {len(got_entry['ip'])} removal poles against {len(want_entry['ip'])}"
        )
        assert len(got_entry["ea"]) == len(want_entry["ea"]), (
            f"{prefix}: {len(got_entry['ea'])} addition poles against {len(want_entry['ea'])}"
        )
        _close(f"{prefix} IP", got_entry["ip"], want_entry["ip"], REALIZED_ENERGY)
        _close(f"{prefix} EA", got_entry["ea"], want_entry["ea"], REALIZED_ENERGY)
        _close(
            f"{prefix} IP weight", got_entry["ip_weight"], want_entry["ip_weight"], REALIZED_WEIGHT
        )
        _close(
            f"{prefix} EA weight", got_entry["ea_weight"], want_entry["ea_weight"], REALIZED_WEIGHT
        )
        drift.record(f"{prefix}.dimension", got_entry["dimension"])
        drift.record(
            f"{prefix}.max_conserved_relative_error", got_entry["max_conserved_relative_error"]
        )
        for sector in ("hole", "particle"):
            _record_sector(f"{prefix}.{sector}", got_entry[sector])


# Exact cases.


def _exact_summary(name, basis, screening, omega_p, n_max, realize_orders) -> dict:
    """Summarize one exact case: poles, the untruncated spectrum, each realized order."""

    mf = _mean_field(name, basis, "rhf", False)
    exact = ExactG0W0SelfEnergy.from_mean_field(mf, screening=Screening(screening))
    mu = float(exact.reference.chemical_potential)
    summary: dict = {
        "chemical_potential": mu,
        "static_correction_norm": float(np.linalg.norm(exact.static_correction)),
    }
    measures = {Sector.HOLE: exact.hole, Sector.PARTICLE: exact.particle}
    for sector, measure in measures.items():
        poles = np.asarray(measure.poles)
        zeroth_moment = measure.couplings @ measure.couplings.conj().T
        summary[sector.name.lower()] = {
            "npoles": int(measure.npoles),
            "pole_range": [float(poles.min()), float(poles.max())],
            "zeroth_moment_norm": float(np.linalg.norm(zeroth_moment)),
        }
    mapping = CayleyMap(center=mu, scale=omega_p)
    exact_moments = exact.cayley_moments(mapping, n_max)
    summary["cayley_moment_norms"] = {
        sector.name.lower(): [float(np.linalg.norm(block)) for block in exact_moments[sector]]
        for sector in (Sector.HOLE, Sector.PARTICLE)
    }
    untruncated = diagonalize_upfolded(
        UpfoldedDysonHamiltonian(
            reference_operator=np.diag(exact.reference.mo_energy),
            static_correction=exact.static_correction,
            hole=exact.hole,
            particle=exact.particle,
        ),
        native_threads=1,
    )
    summary["untruncated"] = {
        "nstates": int(untruncated.energies.size),
        "frontier": _frontier(untruncated.energies, untruncated.physical_weights, mu),
    }
    summary["realized"] = {}
    # Exact moments of finitely many poles end the Schur recursion on the ball; roundoff
    # lands past it.
    exact_tolerances = Tolerances(positivity=10.0 * Tolerances().positivity)
    for n_conserved in realize_orders:
        sectors = {}
        for sector in (Sector.HOLE, Sector.PARTICLE):
            sectors[sector] = SectorSelfEnergyRealization.realize(
                measures[sector].cayley_moments(mapping, n_conserved + SPARE),
                sector,
                mapping,
                n_conserved=n_conserved,
                phase_refinement="restricted",
                realization_algorithm="auto",
                tolerances=exact_tolerances,
            )
        problem = UpfoldedDysonHamiltonian(
            reference_operator=np.diag(exact.reference.mo_energy),
            static_correction=exact.static_correction,
            hole=sectors[Sector.HOLE],
            particle=sectors[Sector.PARTICLE],
        )
        spectrum = diagonalize_upfolded(problem, native_threads=1)
        summary["realized"][str(n_conserved)] = {
            "dimension": int(problem.dimension),
            "hole": _sector_summary(sectors[Sector.HOLE]),
            "particle": _sector_summary(sectors[Sector.PARTICLE]),
            "frontier": _frontier(spectrum.energies, spectrum.physical_weights, mu),
        }
    contour = build_cayley_moments(
        mf,
        n_conserved=n_max - 1,
        screening=screening,
        **{**MOMENT_SETTINGS, "omega_p": omega_p},
    )
    summary["contour_vs_exact"] = {
        sector.name.lower(): max(
            float(
                np.linalg.norm(built[order] - exact_moments[sector][order])
                / max(np.linalg.norm(exact_moments[sector][order]), 1e-300)
            )
            for order in range(n_max + 1)
        )
        for sector, built in (
            (Sector.HOLE, contour.hole.moments),
            (Sector.PARTICLE, contour.particle.moments),
        )
    }
    return summary


@pytest.mark.parametrize("key", list(EXACT_CASES), ids=[k.replace("/", "-") for k in EXACT_CASES])
def test_exact_case(key) -> None:
    got = _exact_summary(*EXACT_CASES[key])
    if CAPTURE:
        _store_case(key, got)
        return
    want = _expected(key)
    _close(
        f"{key} chemical potential",
        got["chemical_potential"],
        want["chemical_potential"],
        DETERMINISTIC,
    )
    _close(
        f"{key} static correction",
        got["static_correction_norm"],
        want["static_correction_norm"],
        DETERMINISTIC,
    )
    for sector in ("hole", "particle"):
        assert got[sector]["npoles"] == want[sector]["npoles"], f"{key}: {sector} pole count"
        _close(
            f"{key} {sector} pole range",
            got[sector]["pole_range"],
            want[sector]["pole_range"],
            DETERMINISTIC,
        )
        _close(
            f"{key} {sector} zeroth moment",
            got[sector]["zeroth_moment_norm"],
            want[sector]["zeroth_moment_norm"],
            DETERMINISTIC,
        )
        _close(
            f"{key} {sector} Cayley moment norms",
            got["cayley_moment_norms"][sector],
            want["cayley_moment_norms"][sector],
            MOMENT_NORM,
            relative=True,
        )
        assert got["contour_vs_exact"][sector] < QUADRATURE, (
            f"{key}: {sector} contour moments {got['contour_vs_exact'][sector]:.2e} "
            f"from the exact ones at N_q={N_Q}"
        )
        drift.record(f"{key}.{sector}.contour_vs_exact", got["contour_vs_exact"][sector])
    assert got["untruncated"]["nstates"] == want["untruncated"]["nstates"]
    _compare_frontier(
        f"{key} untruncated",
        got["untruncated"]["frontier"],
        want["untruncated"]["frontier"],
        DETERMINISTIC,
        DETERMINISTIC,
    )
    assert set(got["realized"]) == set(want["realized"])
    for order, got_entry in got["realized"].items():
        prefix = f"{key}/n{order}"
        _compare_frontier(
            prefix,
            got_entry["frontier"],
            want["realized"][order]["frontier"],
            REALIZED_ENERGY,
            REALIZED_WEIGHT,
        )
        drift.record(f"{prefix}.dimension", got_entry["dimension"])
        for sector in ("hole", "particle"):
            _record_sector(f"{prefix}.{sector}", got_entry[sector])
