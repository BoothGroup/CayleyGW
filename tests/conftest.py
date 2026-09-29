"""Shared fixtures and helpers of the test suite.

The session fixtures are converged PySCF mean fields of small molecules, run on
one thread so the stored references reproduce. The helpers build a bare sector
source, the Green's function by the Schur complement and by the full upfolded
resolvent, and the untruncated upfolded Hamiltonian of exact poles.
``molecular_workflow`` runs the H2 build once per module. At session end the
terminal summary lists what :mod:`drift` recorded.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

# Set before NumPy loads: the references are single-threaded and "auto" reads it.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
from pyscf import dft, gto, lib, scf

from cayleygw import (
    UpfoldedDysonHamiltonian,
    build_cayley_moments,
    build_upfolded_hamiltonian,
    diagonalize_upfolded,
)

import drift


def pytest_terminal_summary(terminalreporter) -> None:
    """List the machine-sensitive quantities against their reference, or rewrite it."""

    observed = drift.observed()
    if not observed:
        return
    if os.environ.get("CAYLEYGW_DRIFT_CAPTURE"):
        written = drift.write_reference()
        terminalreporter.write_line(f"drift reference rewritten: {written}")
        return
    terminalreporter.write_sep("-", "machine-sensitive quantities")
    for key, value in sorted(observed.items()):
        reference = drift._REFERENCE.get("values", {}).get(key)
        shown = f"{value:.6g}" if isinstance(value, float) else repr(value)
        if reference is None:
            terminalreporter.write_line(f"  {key}: {shown} (no reference)")
        else:
            base = f"{reference:.6g}" if isinstance(reference, float) else repr(reference)
            terminalreporter.write_line(f"  {key}: {shown}   reference {base}")


@pytest.fixture(scope="session", autouse=True)
def single_threaded_pyscf() -> None:
    """Run PySCF on one thread so the small molecular tests are deterministic."""

    previous = lib.num_threads()
    lib.num_threads(1)
    try:
        yield
    finally:
        lib.num_threads(previous)


@pytest.fixture(scope="session")
def h2_rhf():
    """H2/RHF in STO-3G at equilibrium: one occupied and one virtual orbital."""

    molecule = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1.0e-12
    mean_field.kernel()
    assert mean_field.converged
    return mean_field


@pytest.fixture(scope="session")
def h2_rhf_df(h2_rhf):
    """H2/RHF with Weigend density fitting."""

    mean_field = scf.RHF(h2_rhf.mol).density_fit(auxbasis="weigend")
    mean_field.conv_tol = 1.0e-12
    mean_field.kernel()
    assert mean_field.converged
    return mean_field


@pytest.fixture(scope="session")
def h2_pbe():
    """H2/PBE in STO-3G at equilibrium."""

    molecule = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mean_field = dft.RKS(molecule)
    mean_field.xc = "pbe"
    mean_field.grids.level = 0
    mean_field.conv_tol = 1.0e-11
    mean_field.kernel()
    assert mean_field.converged
    return mean_field


@pytest.fixture(scope="session")
def h2_pbe_df(h2_pbe):
    """H2/PBE with Weigend density fitting."""

    mean_field = dft.RKS(h2_pbe.mol)
    mean_field.xc = "pbe"
    mean_field.grids.level = 0
    mean_field = mean_field.density_fit(auxbasis="weigend")
    mean_field.conv_tol = 1.0e-11
    mean_field.kernel()
    assert mean_field.converged
    return mean_field


@pytest.fixture(scope="session")
def hf_pbe_df():
    """All-electron hydrogen fluoride/PBE in cc-pVDZ with density fitting."""

    molecule = gto.M(
        atom="F 0 0 0; H 0 0 1.0",
        basis="cc-pvdz",
        unit="Angstrom",
        verbose=0,
    )
    mean_field = dft.RKS(molecule)
    mean_field.xc = "pbe"
    mean_field.grids.level = 0
    mean_field = mean_field.density_fit(auxbasis="weigend")
    mean_field.conv_tol = 1.0e-11
    mean_field.kernel()
    assert mean_field.converged
    return mean_field


@pytest.fixture(scope="session")
def h2_rks_hf():
    """H2 as an RKS with exact exchange, for PySCF's dRPA."""

    molecule = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mean_field = dft.RKS(molecule)
    mean_field.xc = "hf"
    mean_field.grids.level = 0
    mean_field.conv_tol = 1.0e-12
    mean_field.kernel()
    assert mean_field.converged
    return mean_field


@pytest.fixture(scope="session")
def helium_rhf():
    """He/RHF in 6-31G: one occupied and one virtual orbital."""

    molecule = gto.M(
        atom="He 0 0 0",
        basis="6-31g",
        unit="Angstrom",
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1.0e-12
    mean_field.kernel()
    assert mean_field.converged
    return mean_field


@pytest.fixture(scope="session")
def helium_rks_hf():
    """He as an RKS with exact exchange, for PySCF's dRPA."""

    molecule = gto.M(
        atom="He 0 0 0",
        basis="6-31g",
        unit="Angstrom",
        verbose=0,
    )
    mean_field = dft.RKS(molecule)
    mean_field.xc = "hf"
    mean_field.grids.level = 0
    mean_field.conv_tol = 1.0e-12
    mean_field.kernel()
    assert mean_field.converged
    return mean_field


@pytest.fixture(scope="session")
def water_rhf():
    """Water/RHF in STO-3G: five occupied and two virtual orbitals."""

    molecule = gto.M(
        atom="""
            O  0.000000  0.000000  0.000000
            H  0.000000 -0.757000  0.587000
            H  0.000000  0.757000  0.587000
        """,
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1.0e-12
    mean_field.kernel()
    assert mean_field.converged
    return mean_field


@pytest.fixture(scope="session")
def water_pbe():
    """Water/PBE in STO-3G, whose static correction is nonzero."""

    molecule = gto.M(
        atom="""
            O  0.000000  0.000000  0.000000
            H  0.000000 -0.757000  0.587000
            H  0.000000  0.757000  0.587000
        """,
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mean_field = dft.RKS(molecule)
    mean_field.xc = "pbe"
    mean_field.grids.level = 0
    mean_field.conv_tol = 1.0e-11
    mean_field.kernel()
    assert mean_field.converged
    return mean_field


@pytest.fixture(scope="session")
def water_rks_hf():
    """Water as an RKS with exact exchange, for PySCF's dRPA."""

    molecule = gto.M(
        atom="""
            O  0.000000  0.000000  0.000000
            H  0.000000 -0.757000  0.587000
            H  0.000000  0.757000  0.587000
        """,
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mean_field = dft.RKS(molecule)
    mean_field.xc = "hf"
    mean_field.grids.level = 0
    mean_field.conv_tol = 1.0e-12
    mean_field.kernel()
    assert mean_field.converged
    return mean_field


@pytest.fixture(scope="session")
def lih_rhf():
    """LiH/RHF in STO-3G: two occupied and four virtual orbitals."""

    molecule = gto.M(
        atom="Li 0 0 0; H 0 0 1.595",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1.0e-12
    mean_field.kernel()
    assert mean_field.converged
    return mean_field


def sector_source(sector, poles, couplings):
    """Return the bare sector, poles and couplings an upfolded Hamiltonian reads."""

    return SimpleNamespace(
        sector=sector,
        poles=np.asarray(poles, dtype=np.float64),
        couplings=np.asarray(couplings, dtype=np.complex128),
    )


def _self_energy(problem, values):
    """Return the summed rational self-energy of both sectors."""

    total = 0.0
    for source in (problem.hole, problem.particle):
        denominator = values[..., None] - np.asarray(source.poles)
        total = total + np.einsum(
            "pl,...l,ql->...pq",
            source.couplings,
            1.0 / denominator,
            np.asarray(source.couplings).conj(),
            optimize=True,
        )
    return total


def dyson_greens_function(problem, frequency):
    """Return the physical Green's function by the Schur complement."""

    values = np.asarray(frequency, dtype=np.complex128)
    self_energy = _self_energy(problem, values)
    identity = np.eye(problem.nphysical, dtype=np.complex128)
    result = np.empty(values.shape + identity.shape, dtype=np.complex128)
    for index in np.ndindex(values.shape):
        inverse_green = (
            complex(values[index]) * identity
            - problem.physical_matrix
            - self_energy[index]
        )
        result[index] = np.linalg.solve(inverse_green, identity)
    return result


def _upfolded_greens_function(problem, frequency):
    """Return the physical block of the full upfolded resolvent."""

    values = np.asarray(frequency, dtype=np.complex128)
    matrix = problem.matrix
    identity = np.eye(problem.dimension, dtype=np.complex128)
    result = np.empty(
        values.shape + (problem.nphysical, problem.nphysical), dtype=np.complex128
    )
    for index in np.ndindex(values.shape):
        solution = np.linalg.solve(
            complex(values[index]) * identity - matrix,
            identity[:, : problem.nphysical],
        )
        result[index] = solution[: problem.nphysical]
    return result


def _exact_upfolded(exact) -> UpfoldedDysonHamiltonian:
    """Return the untruncated upfolded Hamiltonian of the exact poles."""

    return UpfoldedDysonHamiltonian(
        np.diag(exact.reference.mo_energy),
        exact.hole,
        exact.particle,
        static_correction=exact.static_correction,
    )


@pytest.fixture(scope="module")
def molecular_workflow(h2_rhf):
    """Run the H2 moments, upfolding and Dyson solve once per module."""

    moments = build_cayley_moments(
        h2_rhf,
        n_conserved=2,
        n_q=32,
        omega_p=1.0,
    )
    hamiltonian = build_upfolded_hamiltonian(
        moments,
        terminal_selection="scan",
        terminal_phase_count=16,
    )
    return moments, hamiltonian, diagonalize_upfolded(hamiltonian)
