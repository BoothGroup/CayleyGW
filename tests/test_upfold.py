"""The upfolded Dyson Hamiltonian: assembly, spectrum, Green's function, molecular build."""

from __future__ import annotations

import ast
import logging
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import (
    _exact_upfolded,
    _upfolded_greens_function,
    dyson_greens_function,
    sector_source,
)
from pyscf import gto, scf, tdscf
from pyscf.gw import gw_cd, gw_exact

from cayleygw import (
    ExactG0W0SelfEnergy,
    RefusalError,
    Sector,
    Tolerances,
    UpfoldedDysonHamiltonian,
    ValidationError,
    build_cayley_moments,
    build_upfolded_hamiltonian,
    diagonalize_upfolded,
    reconstruction_test,
)
from cayleygw._helpers.cayley import CayleyMap
from cayleygw._helpers.tolerances import DEFAULT_TOLERANCES
from cayleygw.realization.sector import SectorSelfEnergyRealization
from cayleygw.upfold import DysonSpectrum


def _spectral_greens_function(spectrum, frequency):
    """Return the physical Green's function from the Lehmann poles."""

    values = np.asarray(frequency, dtype=np.complex128)
    orbitals = spectrum.eigenvectors[spectrum.problem.physical_slice]
    return np.einsum(
        "pn,...n,qn->...pq",
        orbitals,
        1.0 / (values[..., None] - spectrum.energies),
        orbitals.conj(),
        optimize=True,
    )


def _greens_function_routes(spectrum, frequency):
    """Return the Lehmann, Schur-complement and upfolded Green's functions."""

    problem = spectrum.problem
    return (
        _spectral_greens_function(spectrum, frequency),
        dyson_greens_function(problem, frequency),
        _upfolded_greens_function(problem, frequency),
    )


def _assert_routes_agree(spectrum, frequency, tolerance=1.0e-9) -> None:
    """Assert the three Green's-function routes agree relative to their scale."""

    routes = _greens_function_routes(spectrum, frequency)
    scale = max(1.0, *(float(np.max(np.abs(route))) for route in routes))
    residual = max(
        float(np.max(np.abs(routes[left] - routes[right])))
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    assert residual / scale <= tolerance


def _synthetic_problem() -> UpfoldedDysonHamiltonian:
    """Return a small complex, noncommuting two-orbital Dyson problem."""

    hole = sector_source(
        Sector.HOLE,
        [-1.3, -0.55],
        [[0.25, 0.10 - 0.04j], [0.08j, 0.31]],
    )
    particle = sector_source(
        Sector.PARTICLE,
        [0.9, 1.8],
        [[0.18 + 0.03j, 0.35], [0.22j, -0.09 + 0.12j]],
    )
    return UpfoldedDysonHamiltonian(
        reference_operator=[[0.0, 0.07j], [-0.07j, 0.45]],
        static_correction=[[0.04, 0.02 - 0.01j], [0.02 + 0.01j, -0.03]],
        hole=hole,
        particle=particle,
    )


def _scalar_moments(nodes, coupling_norms, n_max):
    """Return the 1x1 moments of point masses on the unit circle."""

    nodes = np.asarray(nodes, dtype=np.complex128)
    weights = np.asarray(coupling_norms, dtype=np.float64) ** 2
    return np.asarray(
        [[[np.sum(weights * nodes**order)]] for order in range(n_max + 1)],
        dtype=np.complex128,
    )


def test_dyson_layer_remains_independent_of_all_moment_producers() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "cayleygw"
        / "_helpers"
        / "upfold"
        / "hermitian.py"
    )
    forbidden = {
        "pyscf",
        "reference_state",
        "response",
        "contour",
        "reference",
        "cayley",
        "block_cmv",
        "sector",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    assert not any(part in forbidden for imported in imports for part in imported.split("."))


def test_sector_sources_are_stored_as_given() -> None:
    """Any object exposing sector, poles and couplings is a sector source."""

    class IndependentProducer:
        """A sector source that shares no code with cayleygw."""

        sector = Sector.HOLE
        poles = np.asarray([-2.0, -0.7])
        couplings = np.asarray([[0.4 + 0.1j, 0.2], [0.05j, -0.3 + 0.2j]])

    source = IndependentProducer()
    particle = sector_source(Sector.PARTICLE, [], np.zeros((2, 0)))
    problem = UpfoldedDysonHamiltonian(np.eye(2), source, particle)

    assert problem.hole is source
    assert problem.particle is particle
    np.testing.assert_array_equal(
        problem.matrix[problem.physical_slice, problem.hole_slice],
        source.couplings,
    )


def test_upfolded_matrix_layout_hermiticity_and_diagnostics() -> None:
    problem = _synthetic_problem()
    matrix = problem.matrix

    assert problem.nphysical == 2
    assert problem.nhole == 2
    assert problem.nparticle == 2
    assert problem.dimension == 6
    np.testing.assert_allclose(
        matrix[problem.physical_slice, problem.physical_slice],
        problem.reference_operator + problem.static_correction,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        matrix[problem.hole_slice, problem.hole_slice],
        np.diag(problem.hole.poles),
    )
    np.testing.assert_array_equal(
        matrix[problem.particle_slice, problem.particle_slice],
        np.diag(problem.particle.poles),
    )
    np.testing.assert_array_equal(
        matrix[problem.physical_slice, problem.hole_slice],
        problem.hole.couplings,
    )
    np.testing.assert_array_equal(
        matrix[problem.physical_slice, problem.particle_slice],
        problem.particle.couplings,
    )
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=0.0)
    for array in (
        problem.reference_operator,
        problem.static_correction,
        problem.physical_matrix,
        problem.matrix,
    ):
        assert not array.flags.writeable


def test_roundoff_nonhermiticity_is_symmetrized() -> None:
    hole = sector_source(Sector.HOLE, [], np.zeros((2, 0)))
    particle = sector_source(Sector.PARTICLE, [], np.zeros((2, 0)))
    problem = UpfoldedDysonHamiltonian(
        reference_operator=[[0.0, 1.0 + 1.0e-12j], [1.0, 0.3]],
        hole=hole,
        particle=particle,
    )

    np.testing.assert_allclose(
        problem.reference_operator,
        problem.reference_operator.conj().T,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"reference_operator": [[0.0, 1.0], [0.0, 0.0]]}, "Hermitian"),
        ({"reference_operator": []}, "nonempty and square"),
        ({"static_correction": [[0.0]]}, "shape"),
        (
            {"hole": sector_source(Sector.PARTICLE, [1.0], [[0.2], [0.1]])},
            "Sector.HOLE",
        ),
        (
            {"particle": sector_source(Sector.HOLE, [-1.0], [[0.2], [0.1]])},
            "Sector.PARTICLE",
        ),
        (
            {"hole": sector_source(Sector.HOLE, [-1.0], [[0.2]])},
            "physical dimension",
        ),
        (
            {"hole": sector_source(Sector.HOLE, [-1.0, -0.5], [[0.2], [0.1]])},
            "number of poles",
        ),
        ({"hole": object()}, "Sector.HOLE"),
        ({"tolerances": object()}, "Tolerances"),
        ({"mo_indices": (3,)}, "each physical orbital once"),
        ({"mo_indices": (3, 3)}, "each physical orbital once"),
        ({"mo_indices": (3, -1)}, "mo_indices entry"),
    ],
)
def test_invalid_upfolded_problem_inputs_are_rejected(changes, match) -> None:
    arguments = {
        "reference_operator": np.diag([-0.4, 0.6]),
        "static_correction": np.zeros((2, 2)),
        "hole": sector_source(Sector.HOLE, [-1.0], [[0.2], [0.1]]),
        "particle": sector_source(Sector.PARTICLE, [1.2], [[0.3], [0.2]]),
    }
    arguments.update(changes)
    with pytest.raises(ValidationError, match=match):
        UpfoldedDysonHamiltonian(**arguments)


def test_three_greens_function_routes_agree_for_scalar_and_array_inputs() -> None:
    problem = _synthetic_problem()
    spectrum = diagonalize_upfolded(problem, verbose=0)
    frequencies = np.asarray([-0.8 + 0.5j, 0.2 + 0.7j, 1.4 + 0.3j])

    spectral, schur, upfolded = _greens_function_routes(spectrum, frequencies)
    np.testing.assert_allclose(spectral, schur, atol=2.0e-15)
    np.testing.assert_allclose(spectral, upfolded, atol=2.0e-15)
    np.testing.assert_allclose(schur, upfolded, atol=2.0e-15)
    for route in _greens_function_routes(spectrum, 0.1 + 0.8j):
        assert route.shape == (2, 2)
    scalar = _greens_function_routes(spectrum, 0.1 + 0.8j)
    np.testing.assert_allclose(scalar[0], scalar[1], atol=2.0e-15)
    np.testing.assert_allclose(scalar[0], scalar[2], atol=2.0e-15)


def test_spectrum_weights_satisfy_the_sum_rules() -> None:
    """The weights of a unitary eigendecomposition sum to one per orbital and per state."""

    spectrum = diagonalize_upfolded(_synthetic_problem(), verbose=0)
    problem = spectrum.problem
    vectors = spectrum.eigenvectors
    hole_weights = np.sum(np.abs(vectors[problem.hole_slice]) ** 2, axis=0)
    particle_weights = np.sum(np.abs(vectors[problem.particle_slice]) ** 2, axis=0)

    assert spectrum.nstates == 6
    assert np.all(np.diff(spectrum.energies) >= 0.0)
    assert np.all(spectrum.physical_weights >= 0.0)
    np.testing.assert_allclose(
        np.sum(spectrum.orbital_weights, axis=1),
        np.ones(2),
        atol=4.0e-15,
    )
    assert np.sum(spectrum.physical_weights) == pytest.approx(2.0, abs=4.0e-15)
    orbitals = vectors[problem.physical_slice]
    np.testing.assert_allclose(
        orbitals @ orbitals.conj().T,
        np.eye(2),
        atol=4.0e-15,
    )
    np.testing.assert_allclose(
        spectrum.physical_weights + hole_weights + particle_weights,
        np.ones(spectrum.nstates),
        atol=4.0e-15,
    )
    matrix = problem.matrix
    eigenpair = np.linalg.norm(matrix @ vectors - vectors * spectrum.energies[None, :])
    assert eigenpair / max(1.0, np.linalg.norm(matrix)) < 5.0e-15
    orthonormality = np.linalg.norm(vectors.conj().T @ vectors - np.eye(6))
    assert orthonormality / max(1.0, np.sqrt(6.0)) < 5.0e-15
    grid = np.linspace(-2.5, 2.5, 61)
    assert np.all(spectrum.spectral_function(grid, broadening=0.04) >= 0.0)
    assert np.all(
        spectrum.spectral_function(
            grid,
            broadening=0.04,
            orbital_position=1,
        )
        >= 0.0
    )
    for array in (
        spectrum.energies,
        spectrum.eigenvectors,
        spectrum.orbital_weights,
        spectrum.physical_weights,
    ):
        assert not array.flags.writeable
    high_frequency = 1.0e12j
    np.testing.assert_allclose(
        high_frequency * _spectral_greens_function(spectrum, high_frequency),
        np.eye(spectrum.nphysical),
        atol=3.0e-12,
        rtol=0.0,
    )
    retarded = _spectral_greens_function(spectrum, 0.2 + 0.6j)
    imaginary_part = (retarded - retarded.conj().T) / (2.0j)
    assert np.max(np.linalg.eigvalsh(imaginary_part)) < 0.0


def test_zero_measure_and_repeated_decoupled_poles_preserve_exact_limits() -> None:
    reference = np.asarray([[0.1, 0.08j], [-0.08j, 0.5]])
    empty_hole = sector_source(Sector.HOLE, [], np.zeros((2, 0)))
    empty_particle = sector_source(Sector.PARTICLE, [], np.zeros((2, 0)))
    empty_problem = UpfoldedDysonHamiltonian(
        reference,
        empty_hole,
        empty_particle,
    )
    empty_spectrum = diagonalize_upfolded(empty_problem, verbose=0)
    expected_energies = np.linalg.eigvalsh(reference)
    np.testing.assert_allclose(empty_spectrum.energies, expected_energies, atol=0.0)
    frequency = 0.2 + 0.6j
    np.testing.assert_allclose(
        dyson_greens_function(empty_problem, frequency),
        np.linalg.inv(frequency * np.eye(2) - reference),
        atol=3.0e-16,
    )
    np.testing.assert_allclose(empty_spectrum.physical_weights, 1.0, atol=4.0e-16)

    repeated_hole = sector_source(Sector.HOLE, [-1.0, -1.0], np.zeros((2, 2)))
    repeated_particle = sector_source(Sector.PARTICLE, [1.2, 1.2], np.zeros((2, 2)))
    repeated = diagonalize_upfolded(
        UpfoldedDysonHamiltonian(
            reference,
            repeated_hole,
            repeated_particle,
        ),
        verbose=0,
    )
    np.testing.assert_allclose(
        repeated.energies,
        np.sort(np.concatenate((expected_energies, [-1.0, -1.0, 1.2, 1.2]))),
        atol=0.0,
    )
    assert np.count_nonzero(repeated.physical_weights == 0.0) == 4


def test_invalid_spectrum_operations_are_rejected() -> None:
    problem = _synthetic_problem()
    spectrum = diagonalize_upfolded(problem, verbose=0)
    with pytest.raises(ValidationError, match="orbital_position"):
        spectrum.spectral_function([0.0], broadening=0.1, orbital_position=2)
    with pytest.raises(ValidationError, match="orbital_position"):
        spectrum.spectral_function([0.0], broadening=0.1, orbital_position=-1)
    with pytest.raises(ValidationError, match="strictly positive"):
        spectrum.spectral_function([0.0], broadening=0.0)
    with pytest.raises(ValidationError, match="real"):
        spectrum.spectral_function([0.1j], broadening=0.1)
    with pytest.raises(ValidationError, match="upfolded dimension"):
        DysonSpectrum(
            energies=np.arange(problem.dimension - 1, dtype=float),
            eigenvectors=np.eye(problem.dimension),
            problem=problem,
        )
    with pytest.raises(ValidationError, match="UpfoldedDysonHamiltonian"):
        DysonSpectrum(
            energies=spectrum.energies, eigenvectors=spectrum.eigenvectors, problem=object()
        )


@pytest.mark.pyscf
def test_exact_full_pole_h2_reproduces_the_pole_oracle(h2_rks_hf) -> None:
    exact = ExactG0W0SelfEnergy.from_mean_field(h2_rks_hf)
    problem = UpfoldedDysonHamiltonian(
        reference_operator=np.diag(exact.reference.mo_energy),
        static_correction=exact.static_correction,
        hole=exact.hole,
        particle=exact.particle,
    )
    # The oracle by hand: physical block, poles on the diagonal, couplings off it.
    couplings = np.concatenate((exact.hole.couplings, exact.particle.couplings), axis=1)
    oracle = np.block(
        [
            [np.diag(exact.reference.mo_energy) + exact.static_correction, couplings],
            [
                couplings.conj().T,
                np.diag(np.concatenate((exact.hole.poles, exact.particle.poles))),
            ],
        ]
    )

    assert problem.hole is exact.hole
    assert problem.particle is exact.particle
    np.testing.assert_allclose(problem.matrix, oracle, atol=0.0)
    spectrum = diagonalize_upfolded(problem, verbose=0)
    oracle_energies, oracle_vectors = np.linalg.eigh(oracle)
    np.testing.assert_allclose(spectrum.energies, oracle_energies, atol=0.0)
    np.testing.assert_allclose(
        spectrum.orbital_weights,
        np.abs(oracle_vectors[: problem.nphysical]) ** 2,
        atol=0.0,
    )
    frequencies = np.asarray([-1.0 + 0.4j, 0.1 + 0.7j, 1.6 + 0.5j])
    _assert_routes_agree(spectrum, frequencies)


@pytest.mark.pyscf
def test_h2_cayley_moments_complete_end_to_end_dyson_path(h2_rhf) -> None:
    exact = ExactG0W0SelfEnergy.from_mean_field(h2_rhf)
    mapping = CayleyMap(
        center=exact.reference.chemical_potential,
        scale=1.0,
    )
    hole = SectorSelfEnergyRealization.realize(
        exact.hole.cayley_moments(mapping, 3),
        Sector.HOLE,
        mapping,
    )
    particle = SectorSelfEnergyRealization.realize(
        exact.particle.cayley_moments(mapping, 3),
        Sector.PARTICLE,
        mapping,
    )
    compressed = UpfoldedDysonHamiltonian(
        reference_operator=np.diag(exact.reference.mo_energy),
        static_correction=exact.static_correction,
        hole=hole,
        particle=particle,
    )
    exact_problem = UpfoldedDysonHamiltonian(
        reference_operator=np.diag(exact.reference.mo_energy),
        static_correction=exact.static_correction,
        hole=exact.hole,
        particle=exact.particle,
    )

    frequencies = np.asarray([-1.2 + 0.4j, 0.0 + 0.8j, 1.4 + 0.5j])
    np.testing.assert_allclose(
        dyson_greens_function(compressed, frequencies),
        dyson_greens_function(exact_problem, frequencies),
        atol=3.0e-14,
    )
    compressed_spectrum = diagonalize_upfolded(compressed, verbose=0)
    exact_spectrum = diagonalize_upfolded(exact_problem, verbose=0)
    np.testing.assert_allclose(
        compressed_spectrum.energies,
        exact_spectrum.energies,
        atol=4.0e-14,
    )
    np.testing.assert_allclose(
        compressed_spectrum.physical_weights,
        exact_spectrum.physical_weights,
        atol=2.0e-14,
    )


def test_cayley_cmv_greens_function_converges_with_moment_order() -> None:
    mapping = CayleyMap(center=0.0, scale=1.0)
    particle_nodes = np.exp(1.0j * np.asarray([0.45, 1.30, 2.40]))
    hole_nodes = np.exp(-1.0j * np.asarray([0.35, 1.10, 2.20]))
    particle_couplings = np.sqrt(np.asarray([0.35, 0.22, 0.08]))
    hole_couplings = np.sqrt(np.asarray([0.28, 0.18, 0.06]))
    particle_moments = _scalar_moments(
        particle_nodes,
        particle_couplings,
        8,
    )
    hole_moments = _scalar_moments(hole_nodes, hole_couplings, 8)
    exact = UpfoldedDysonHamiltonian(
        reference_operator=[[0.1]],
        hole=sector_source(
            Sector.HOLE,
            mapping.inverse(hole_nodes),
            hole_couplings[None, :],
        ),
        particle=sector_source(
            Sector.PARTICLE,
            mapping.inverse(particle_nodes),
            particle_couplings[None, :],
        ),
    )
    frequencies = np.asarray([-0.5 + 0.4j, 0.2 + 0.7j, 1.5 + 0.3j])
    target = dyson_greens_function(exact, frequencies)
    errors = []
    dimensions = []
    for n_conserved in range(4):
        hole = SectorSelfEnergyRealization.realize(
            hole_moments,
            Sector.HOLE,
            mapping,
            n_conserved=n_conserved,
            phase_count=128,
        )
        particle = SectorSelfEnergyRealization.realize(
            particle_moments,
            Sector.PARTICLE,
            mapping,
            n_conserved=n_conserved,
            phase_count=128,
        )
        problem = UpfoldedDysonHamiltonian([[0.1]], hole, particle)
        errors.append(float(np.max(np.abs(dyson_greens_function(problem, frequencies) - target))))
        dimensions.append(problem.dimension)

    assert dimensions == [3, 5, 7, 7]
    assert np.all(np.diff(errors) < 0.0)
    assert errors[0] > 0.4
    assert errors[2] < 5.0e-3
    assert errors[3] < 2.0e-14


@pytest.mark.pyscf
def test_dominant_h2_poles_match_pyscf_exact_and_contour_deformation(
    h2_rks_hf,
) -> None:
    exact = ExactG0W0SelfEnergy.from_mean_field(h2_rks_hf)
    problem = UpfoldedDysonHamiltonian(
        np.diag(exact.reference.mo_energy),
        exact.hole,
        exact.particle,
        static_correction=exact.static_correction,
    )
    spectrum = diagonalize_upfolded(problem, verbose=0)
    dominant = spectrum.energies[np.argmax(spectrum.orbital_weights, axis=1)]

    direct_rpa = tdscf.dRPA(h2_rks_hf)
    direct_rpa.nstates = 1
    direct_rpa.conv_tol = 1.0e-12
    direct_rpa.verbose = 0
    direct_rpa.kernel()
    assert np.all(direct_rpa.converged)
    exact_gw = gw_exact.GWExact(h2_rks_hf, tdmf=direct_rpa)
    exact_gw.verbose = 0
    pyscf_exact = exact_gw.kernel()
    contour_gw = gw_cd.GWCD(h2_rks_hf)
    contour_gw.verbose = 0
    # Default grid, since not every PySCF release takes ``nw``; older ones return None.
    pyscf_contour = contour_gw.kernel()
    if pyscf_contour is None:
        pyscf_contour = contour_gw.mo_energy

    np.testing.assert_allclose(dominant, pyscf_exact, atol=2.0e-8, rtol=0.0)
    np.testing.assert_allclose(dominant, pyscf_contour, atol=2.0e-5, rtol=0.0)


@pytest.mark.pyscf
def test_cautious_h2_geometry_scan_is_continuous_and_resolves_satellites() -> None:
    bond_lengths = (0.50, 0.74, 1.00, 1.50, 2.00)
    quasiparticle_energies = []
    satellite_weights = []
    for bond_length in bond_lengths:
        molecule = gto.M(
            atom=f"H 0 0 0; H 0 0 {bond_length}",
            basis="sto-3g",
            unit="Angstrom",
            verbose=0,
        )
        mean_field = scf.RHF(molecule)
        mean_field.conv_tol = 1.0e-12
        mean_field.kernel()
        assert mean_field.converged
        exact = ExactG0W0SelfEnergy.from_mean_field(mean_field)
        problem = UpfoldedDysonHamiltonian(
            np.diag(exact.reference.mo_energy),
            exact.hole,
            exact.particle,
            static_correction=exact.static_correction,
        )
        spectrum = diagonalize_upfolded(problem, verbose=0)
        quasiparticle_states = np.argmax(spectrum.orbital_weights, axis=1)
        quasiparticle_energies.append(spectrum.energies[quasiparticle_states])
        visible = set(np.flatnonzero(spectrum.physical_weights > 1.0e-8))
        satellites = sorted(visible - set(quasiparticle_states.tolist()))
        assert len(satellites) == 2
        subspace_weights = np.stack(
            [
                spectrum.physical_weights,
                np.sum(np.abs(spectrum.eigenvectors[problem.hole_slice]) ** 2, axis=0),
                np.sum(np.abs(spectrum.eigenvectors[problem.particle_slice]) ** 2, axis=0),
            ]
        )
        assert {
            ("physical", "hole", "particle")[int(np.argmax(subspace_weights[:, index]))]
            for index in satellites
        } == {"hole", "particle"}
        satellite_weights.append(
            max(float(spectrum.physical_weights[index]) for index in satellites)
        )

    quasiparticle_energies = np.asarray(quasiparticle_energies)
    assert np.max(np.abs(np.diff(quasiparticle_energies, axis=0))) < 0.35
    assert np.all(np.diff(satellite_weights) > 0.0)
    assert satellite_weights[0] > 3.0e-3
    assert satellite_weights[-1] > 4.0e-2


def test_physical_weight_below_is_reported_and_never_gated() -> None:
    """The boundary split is measurable but is not one of the sum rules."""

    reference = np.diag([-1.0, 1.0]).astype(np.complex128)
    hole = sector_source(Sector.HOLE, [-2.0, -1.5], [[0.2, 0.1], [0.05, 0.15]])
    particle = sector_source(Sector.PARTICLE, [1.5, 2.0], [[0.1, 0.05], [0.2, 0.1]])
    hamiltonian = UpfoldedDysonHamiltonian(
        reference_operator=reference,
        hole=hole,
        particle=particle,
    )
    spectrum = diagonalize_upfolded(hamiltonian, verbose=0)

    # The total is nphysical by a sum rule; the split at a boundary is not fixed.
    below = spectrum.physical_weight_below(0.0)
    above = float(np.sum(spectrum.physical_weights)) - below
    assert below + above == pytest.approx(spectrum.nphysical, abs=1.0e-12)
    assert 0.0 < below < spectrum.nphysical

    # Monotone, and saturating outside the spectral range.
    assert spectrum.physical_weight_below(-1.0e6) == 0.0
    assert spectrum.physical_weight_below(1.0e6) == pytest.approx(spectrum.nphysical, abs=1.0e-12)
    energies = np.asarray(spectrum.energies)
    values = [spectrum.physical_weight_below(x) for x in np.sort(energies)]
    assert all(a <= b + 1.0e-15 for a, b in zip(values, values[1:]))

    # Strictly below: a boundary exactly on a pole excludes it.
    lowest = float(energies.min())
    assert spectrum.physical_weight_below(lowest) == 0.0

    for invalid in ("mid-gap", None, float("nan"), np.array([0.0, 1.0])):
        with pytest.raises(ValidationError):
            spectrum.physical_weight_below(invalid)


# build_upfolded_hamiltonian on molecular moments.


@pytest.mark.pyscf
def test_pbe_two_call_workflow_includes_static_correction_once(h2_pbe) -> None:
    """The PBE static correction enters once and the Green's function matches exact poles."""

    moments = build_cayley_moments(
        h2_pbe,
        n_conserved=3,
        n_q=128,
        omega_p=1.0,
    )
    hamiltonian = build_upfolded_hamiltonian(
        moments,
        terminal_selection="scan",
        terminal_phase_count=1,
    )
    correction = moments.adapter.build_static_self_energy_correction()
    expected_physical = np.diag(moments.adapter.reference.mo_energy) + correction

    assert np.linalg.norm(correction) > 1.0e-3
    np.testing.assert_allclose(
        hamiltonian.static_correction,
        correction,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        hamiltonian.physical_matrix,
        expected_physical,
        atol=0.0,
        rtol=0.0,
    )

    exact = ExactG0W0SelfEnergy.from_mean_field(h2_pbe)
    exact_upfolded = _exact_upfolded(exact)
    np.testing.assert_allclose(
        hamiltonian.physical_matrix,
        exact_upfolded.physical_matrix,
        atol=0.0,
        rtol=0.0,
    )
    frequencies = np.asarray([-1.0 + 0.4j, 0.1 + 0.7j, 1.5 + 0.3j])
    exact_greens_function = dyson_greens_function(exact_upfolded, frequencies)
    np.testing.assert_allclose(
        dyson_greens_function(hamiltonian, frequencies),
        exact_greens_function,
        atol=2.0e-8,
        rtol=0.0,
    )


@pytest.mark.pyscf
def test_terminal_selection_and_argument_validation(
    molecular_workflow,
) -> None:
    moments, _, _ = molecular_workflow
    scanned = build_upfolded_hamiltonian(
        moments,
        terminal_selection="scan",
        terminal_phase_count=7,
    )
    # H2/STO-3G terminates by rank, so no terminal phase is imposed.
    assert scanned.hole.selected_phase is None
    assert scanned.particle.selected_phase is None

    with pytest.raises(ValidationError, match="G0W0CayleyMoments"):
        build_upfolded_hamiltonian(object(), terminal_selection="scan")
    with pytest.raises(ValidationError, match="scan.*restricted"):
        build_upfolded_hamiltonian(moments, terminal_selection="nearest")
    with pytest.raises(ValidationError, match="terminal_phase_count"):
        build_upfolded_hamiltonian(moments, terminal_phase_count=0, terminal_selection="scan")
    for floor in (0.0, -1.0e-10, float("nan"), "1e-10", True):
        with pytest.raises(ValidationError, match="rank_floor"):
            Tolerances(rank_floor=floor)
    with pytest.raises(ValidationError, match="Tolerances"):
        build_upfolded_hamiltonian(
            moments, tolerances={"rank_floor": 1.0e-10}, terminal_selection="scan"
        )


@pytest.mark.pyscf
def test_effective_hamiltonian_reconstructs_every_conserved_moment(
    molecular_workflow,
    h2_rhf,
) -> None:
    moments, hamiltonian, _ = molecular_workflow
    report = reconstruction_test(hamiltonian)

    assert report.passed
    assert report.hole.absolute_errors.size == moments.n_max + 1
    assert report.particle.absolute_errors.size == moments.n_max + 1
    assert report.hole.conserved_order == moments.n_conserved
    assert report.particle.conserved_order == moments.n_conserved
    assert report.maximum_conserved_absolute_error < 1.0e-13
    assert report.maximum_conserved_relative_error < 1.0e-11
    np.testing.assert_allclose(
        report.hole.reconstructed_moments,
        moments.hole.moments,
        atol=1.0e-13,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        report.particle.reconstructed_moments,
        moments.particle.moments,
        atol=1.0e-13,
        rtol=0.0,
    )

    truncated = reconstruction_test(hamiltonian, n_max=1)
    assert truncated.hole.absolute_errors.size == 2
    assert truncated.particle.absolute_errors.size == 2
    with pytest.raises(ValidationError, match="exceeds"):
        reconstruction_test(hamiltonian, n_max=4)


@pytest.mark.pyscf
def test_rank_floor_reaches_both_sectors_and_the_options_table(
    molecular_workflow,
    monkeypatch,
    caplog,
) -> None:
    """``rank_floor`` reaches both sectors' tolerances, and the options line."""

    moments, default, _ = molecular_workflow
    assert default.tolerances == DEFAULT_TOLERANCES
    assert default.tolerances.rank_floor == 1.0e-10

    original = SectorSelfEnergyRealization.realize
    received = []

    def recording(*args, **kwargs):
        received.append(kwargs["tolerances"])
        return original(*args, **kwargs)

    monkeypatch.setattr(SectorSelfEnergyRealization, "realize", staticmethod(recording))
    with caplog.at_level(logging.INFO, logger="cayleygw"):
        raised = build_upfolded_hamiltonian(
            moments, tolerances=Tolerances(rank_floor=3.0e-10), terminal_selection="scan", verbose=1
        )
    assert len(received) == 2
    for tolerances in (*received, raised.tolerances):
        assert tolerances == replace(DEFAULT_TOLERANCES, rank_floor=3.0e-10)
    options = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("build_upfolded_hamiltonian: n_conserved")
    ]
    assert len(options) == 1 and "rank_floor=3e-10" in options[0]


@pytest.mark.pyscf
def test_a_sector_refusal_reaches_the_caller_unchanged(
    molecular_workflow,
    monkeypatch,
) -> None:
    """The sector's refusal already names its gate and levers; nothing is added."""

    moments, _, _ = molecular_workflow
    refusal = RefusalError(
        "hole sector refused at rank_floor=1e-10: synthetic gate. Retry with "
        "rank_floor=3e-10, then 1e-9, 2e-9, 5e-9, 1e-8.",
        kind="sector",
    )

    def refuse(*args, **kwargs):
        raise refusal

    monkeypatch.setattr(SectorSelfEnergyRealization, "realize", staticmethod(refuse))
    with pytest.raises(RefusalError) as caught:
        build_upfolded_hamiltonian(moments, n_conserved=2, terminal_selection="scan")
    assert caught.value is refusal


@pytest.mark.pyscf
def test_withheld_orders_receive_generic_reconstruction_thresholds(
    lih_rhf,
) -> None:
    """Orders above ``n_conserved`` get the generic threshold instead of a short array."""

    moments = build_cayley_moments(
        lih_rhf,
        n_conserved=3,
        n_q=512,
        omega_p=1.0,
    )
    hamiltonian = build_upfolded_hamiltonian(moments, terminal_selection="scan")
    reconstruction = reconstruction_test(hamiltonian)

    for sector in (reconstruction.hole, reconstruction.particle):
        assert sector.acceptance_thresholds.shape == (5,)
        assert sector.absolute_errors.shape == (5,)
        assert np.all(sector.acceptance_thresholds > 0.0)
        assert sector.conserved_order == 3
    assert reconstruction.passed


@pytest.mark.pyscf
@pytest.mark.parametrize("workers", [2, 8])
@pytest.mark.parametrize("phases", [4, 32])
def test_concurrent_sectors_do_not_change_the_charged_spectrum(
    h2_rhf_df,
    workers,
    phases,
) -> None:
    """Hole and particle are independent, so running them concurrently moves no root."""

    moments = build_cayley_moments(
        h2_rhf_df,
        n_conserved=3,
        n_q=32,
        omega_p=1.0,
    )
    # At 4 phases the scan has little to spread, the regime sector threads are for.
    common = {
        "terminal_selection": "scan",
        "terminal_phase_count": phases,
        "native_threads": 1,
    }
    serial = diagonalize_upfolded(build_upfolded_hamiltonian(moments, n_workers=1, **common))
    parallel = diagonalize_upfolded(
        build_upfolded_hamiltonian(moments, n_workers=workers, **common)
    )
    np.testing.assert_array_equal(parallel.energies, serial.energies)
    np.testing.assert_array_equal(
        parallel.physical_weights,
        serial.physical_weights,
    )


@pytest.mark.pyscf
@pytest.mark.parametrize("workers", [1, 4])
def test_both_sectors_log_their_stages_under_the_one_summary(
    h2_rhf_df,
    workers,
    caplog,
) -> None:
    """The sector stages of both sectors reach the main call's summary."""

    moments = build_cayley_moments(
        h2_rhf_df,
        n_conserved=3,
        n_q=32,
        omega_p=1.0,
    )
    with caplog.at_level(logging.INFO, logger="cayleygw"):
        build_upfolded_hamiltonian(
            moments,
            terminal_selection="scan",
            terminal_phase_count=32,
            n_workers=workers,
            native_threads=1,
            verbose=1,
        )
    summary = [
        [name for name, _ in record.cayleygw["rows"]]
        for record in caplog.records
        if getattr(record, "cayleygw", {}).get("event") == "summary"
        and record.cayleygw["title"] == "build_upfolded_hamiltonian"
    ]
    assert len(summary) == 1
    for sector in ("hole", "particle"):
        assert f"{sector} terminal candidates" in summary[0]
        assert f"{sector} poles and couplings" in summary[0]
    assert "sector realizations" in summary[0]


@pytest.mark.pyscf
def test_sectors_actually_run_on_separate_threads(h2_rhf_df, monkeypatch) -> None:
    """Both sectors leave the calling thread, which the equality tests cannot show."""

    moments = build_cayley_moments(
        h2_rhf_df,
        n_conserved=3,
        n_q=32,
        omega_p=1.0,
    )
    original = SectorSelfEnergyRealization.realize
    observed: list[bool] = []

    def recording(sector_moments, sector, mapping, **kwargs):
        observed.append(threading.current_thread() is threading.main_thread())
        return original(sector_moments, sector, mapping, **kwargs)

    common = {
        "terminal_selection": "scan",
        "terminal_phase_count": 4,
        "native_threads": 1,
    }
    with monkeypatch.context() as patcher:
        patcher.setattr(
            SectorSelfEnergyRealization,
            "realize",
            staticmethod(recording),
        )
        observed.clear()
        build_upfolded_hamiltonian(moments, n_workers=1, **common)
        serial = list(observed)
        observed.clear()
        build_upfolded_hamiltonian(moments, n_workers=2, **common)
        dispatched = list(observed)

    assert len(serial) == 2 and all(serial), "serial must stay on the caller"
    # Not two distinct threads: an H2 sector is so fast that one worker may run both.
    assert len(dispatched) == 2 and not any(dispatched)


@pytest.mark.pyscf
def test_a_lower_n_conserved_override_keeps_the_built_spare_count(h2_rhf) -> None:
    """Overriding ``n_conserved`` truncates the sequence: order n + 1 stays the spare."""

    built = build_cayley_moments(h2_rhf, n_conserved=3, n_q=64, omega_p=1.0)
    overridden = build_upfolded_hamiltonian(built, n_conserved=2, terminal_selection="scan")
    direct = build_upfolded_hamiltonian(
        build_cayley_moments(h2_rhf, n_conserved=2, n_q=64, omega_p=1.0),
        terminal_selection="scan",
    )
    np.testing.assert_allclose(np.asarray(overridden.matrix), np.asarray(direct.matrix))


@pytest.mark.pyscf
def test_n_conserved_override_refuses_what_the_build_did_not_conserve(h2_rhf) -> None:
    moments = build_cayley_moments(h2_rhf, n_conserved=3, n_q=64, omega_p=1.0)
    with pytest.raises(ValidationError, match="exceeds"):
        build_upfolded_hamiltonian(moments, n_conserved=9, terminal_selection="scan")


@pytest.mark.pyscf
def test_realization_reads_the_static_correction_the_build_carried(h2_pbe, monkeypatch) -> None:
    """The realization must not touch the mean field: no DF rebuild for stored moments."""

    moments = build_cayley_moments(
        h2_pbe,
        n_conserved=3,
        n_q=128,
        omega_p=1.0,
    )
    assert moments.static_correction is not None
    carried = np.array(moments.static_correction)

    def forbidden(self):
        raise AssertionError("realization asked the mean field for the static correction")

    monkeypatch.setattr(type(moments.adapter), "build_static_self_energy_correction", forbidden)
    hamiltonian = build_upfolded_hamiltonian(
        moments, terminal_selection="scan", terminal_phase_count=1
    )
    np.testing.assert_allclose(hamiltonian.static_correction, carried, rtol=0.0, atol=0.0)


@pytest.mark.pyscf
def test_a_refused_realization_still_logs_its_stages_and_summary(
    molecular_workflow,
    monkeypatch,
    caplog,
) -> None:
    """A refused build still logs its failed stage and the stage times."""

    moments, _, _ = molecular_workflow

    def refuse(*args, **kwargs):
        raise RefusalError("synthetic refusal", kind="sector")

    monkeypatch.setattr(SectorSelfEnergyRealization, "realize", staticmethod(refuse))
    with caplog.at_level(logging.INFO, logger="cayleygw"):
        with pytest.raises(RefusalError):
            build_upfolded_hamiltonian(moments, n_conserved=2, verbose=1, terminal_selection="scan")
    messages = [record.getMessage() for record in caplog.records]
    assert any(m.startswith("sector realizations: failed after") for m in messages)
    assert any(m.startswith("build_upfolded_hamiltonian: stage times") for m in messages)
