"""Tests of the sector realization: closure scan, gates, inverse map and rational self-energy.

Later sections check the closure oracle in ``closure_oracle.py`` against the
scan, and the Gram rank cut.
"""

from __future__ import annotations

import ast
import logging
import math
import re
from dataclasses import replace
from pathlib import Path

import drift
import numpy as np
import pytest
from closure_oracle import ClosureObjective, build_closure_objective

from cayleygw import ExactG0W0SelfEnergy, RefusalError, Sector, ValidationError
from cayleygw._helpers import tolerances as tolerances_module
from cayleygw._helpers.cayley import CayleyMap
from cayleygw._helpers.tolerances import DEFAULT_TOLERANCES
from cayleygw.realization import sector as sector_module
from cayleygw.realization._helpers import base as poles_module
from cayleygw.realization._helpers import sector as sector_helpers
from cayleygw.realization._helpers.base import _selection_key
from cayleygw.realization._helpers.toeplitz import gram_rank_cut
from cayleygw.realization.base import UnitaryMomentRealization
from cayleygw.realization.block_cmv import BlockCMVRealization
from cayleygw.realization.sector import SectorSelfEnergyRealization
from cayleygw.realization.toeplitz import ToeplitzRealization


def _atomic_matrix_moments(nodes, couplings, n_max):
    """Return moments of rank-one positive matrix-valued atoms."""

    nodes = np.asarray(nodes, dtype=np.complex128)
    couplings = np.asarray(couplings, dtype=np.complex128)
    return np.asarray(
        [
            np.einsum(
                "pl,l,ql->pq",
                couplings,
                nodes**order,
                couplings.conj(),
                optimize=True,
            )
            for order in range(n_max + 1)
        ]
    )


def _scalar_atomic_moments(nodes, weights, n_max):
    """Return one-dimensional matrix moments of positive scalar atoms."""

    values = [
        np.sum(np.asarray(weights) * np.asarray(nodes) ** order) for order in range(n_max + 1)
    ]
    return np.asarray(values, dtype=np.complex128)[:, None, None]


@pytest.mark.parametrize(
    "module",
    ["realization/base.py", "realization/sector.py", "realization/_helpers/sector.py"],
)
def test_sector_realization_layer_remains_independent_of_moment_producers(module) -> None:
    path = Path(__file__).resolve().parents[2] / "src" / "cayleygw" / module
    forbidden = {"pyscf", "reference_state", "response", "screening", "contour", "reference"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    assert not any(part in forbidden for imported in imports for part in imported.split("."))


def test_spectral_extraction_preserves_nodes_psd_weights_and_moments() -> None:
    nodes = np.exp(1.0j * np.asarray([0.30, 1.50, 2.70]))
    couplings = np.asarray(
        [
            [1.0, 0.4, 0.3j],
            [0.0, 0.9, 0.7],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    moments = _atomic_matrix_moments(nodes, couplings, 8)
    generic = BlockCMVRealization.realize(moments)
    spectrum = generic.spectrum()

    assert spectrum.nodes.size == 3
    assert spectrum.couplings.shape[0] == 3
    # The decomposition the extraction certified, before its radial projection.
    raw_nodes, vectors, image, offdiagonal = poles_module._unitary_eigendecomposition(
        generic.matrix, 1.0e-10 * np.sqrt(3.0)
    )
    assert np.max(np.abs(np.abs(raw_nodes) - 1.0)) < 1.0e-15
    assert offdiagonal < 2.0e-15
    projected = raw_nodes / np.abs(raw_nodes)
    assert np.linalg.norm(image - vectors * projected[None, :]) < 4.0e-15
    assert np.max(spectrum.moment_residuals) < 6.0e-15
    np.testing.assert_allclose(np.abs(spectrum.nodes), 1.0, atol=0.0)
    np.testing.assert_allclose(
        np.sort(np.mod(np.angle(spectrum.nodes), 2.0 * np.pi)),
        np.sort(np.mod(np.angle(nodes), 2.0 * np.pi)),
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        _atomic_matrix_moments(spectrum.nodes, spectrum.couplings, 8),
        moments,
        atol=5.0e-15,
    )
    residues = np.einsum("pl,ql->lpq", spectrum.couplings, spectrum.couplings.conj())
    np.testing.assert_allclose(np.sum(residues, axis=0), moments[0], atol=3.0e-15)
    for residue in residues:
        assert np.min(np.linalg.eigvalsh(residue)) > -8.0e-16


@pytest.mark.parametrize("sector", [Sector.HOLE, Sector.PARTICLE])
def test_scalar_phase_scan_preserves_guaranteed_moments_and_selects_arc(sector) -> None:
    moments = np.asarray([[[2.0 + 0.0j]]])
    scan = SectorSelfEnergyRealization.scan_closures(
        moments,
        sector,
        phase_count=32,
    )

    assert len(scan.candidates) == 32
    canonical = scan.candidates[0]
    assert not canonical.acceptable
    # The atom at u = 1 lies on the real axis, which counts as wrong-arc.
    assert canonical.wrong_arc_weight == pytest.approx(2.0)
    assert canonical.near_singular_weight == pytest.approx(2.0)
    selected = scan.require_selected()
    assert selected.acceptable
    assert float(
        np.sum(selected.spectrum.trace_weights[selected.correct_arc_mask])
    ) == pytest.approx(2.0)
    assert selected.wrong_arc_weight == pytest.approx(0.0)
    assert selected.minimum_correct_node_distance_from_one > 1.9
    for candidate in scan.candidates:
        np.testing.assert_allclose(
            candidate.realization.reconstructed_moments[:1],
            moments,
            atol=8.0e-16,
        )


def test_withheld_moment_selects_the_exact_available_scalar_phase() -> None:
    node = 1.0j
    moments = np.asarray([[[1.0 + 0.0j]], [[node]]])
    scan = SectorSelfEnergyRealization.scan_closures(
        moments,
        Sector.PARTICLE,
        n_conserved=0,
        phase_count=8,
    )
    selected = scan.require_selected()

    assert selected.phase == pytest.approx(1.5 * np.pi)
    assert selected.withheld_moment_residuals.size == 1
    assert selected.maximum_withheld_relative_residual < 2.0e-16
    np.testing.assert_allclose(selected.spectrum.nodes, np.asarray([node]), atol=3.0e-16)


def test_infeasible_later_block_schur_step_names_the_gate_and_its_lever() -> None:
    moments = np.asarray(
        [
            np.eye(2),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            1.01 * np.eye(2),
        ],
        dtype=np.complex128,
    )

    with pytest.raises(RefusalError) as caught:
        SectorSelfEnergyRealization.scan_closures(
            moments,
            Sector.PARTICLE,
            realization_algorithm="block-cmv",
        )

    message = str(caught.value)
    assert "block-cmv construction failed fitting C_0 through C_3" in message
    assert "maximum_singular_value=1.0100000000000000e+00" in message
    assert "Use realization='toeplitz'" in message
    assert caught.value.kind == "sector"
    assert isinstance(caught.value.diagnostics, RefusalError)
    assert caught.value.diagnostics.kind == "block-cmv"
    assert caught.value.diagnostics.check == "schur_contraction"
    assert caught.value.__cause__ is caught.value.diagnostics


def test_exact_sector_realization_recovers_poles_residues_and_rational_values() -> None:
    mapping = CayleyMap(center=0.4, scale=1.7)
    nodes = np.exp(1.0j * np.asarray([0.30, 1.25, 2.50]))
    couplings = np.asarray(
        [[1.0, 0.35 - 0.2j, 0.1j], [0.2, 0.8, -0.5 + 0.1j]],
        dtype=np.complex128,
    )
    moments = _atomic_matrix_moments(nodes, couplings, 8)
    result = SectorSelfEnergyRealization.realize(
        moments,
        Sector.PARTICLE,
        mapping,
    )

    expected_poles = np.asarray(mapping.inverse(nodes))
    assert result.selected_phase is None
    assert result.poles.size == 3
    assert result.couplings.shape == (2, 3)
    assert np.all(result.poles > mapping.center)
    assert np.max(np.abs(np.abs(result.nodes) - 1.0)) < 5.0e-16
    assert result.discarded_total_weight == pytest.approx(0.0)
    assert result.moment_residuals[0] < 4.0e-15
    assert result.maximum_conserved_moment_residual < 8.0e-15
    np.testing.assert_allclose(
        np.sort(result.poles),
        np.sort(expected_poles),
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        _atomic_matrix_moments(result.nodes, result.couplings, 8),
        moments,
        atol=8.0e-15,
    )
    np.testing.assert_allclose(
        _atomic_matrix_moments(result.poles, result.couplings, 2),
        _atomic_matrix_moments(expected_poles, couplings, 2),
        atol=3.0e-13,
    )
    frequencies = np.asarray([-1.0 + 0.4j, 0.7 + 0.8j, 3.0 + 1.2j])

    def rational(poles, factors):
        return np.einsum(
            "pl,zl,ql->zpq",
            factors,
            1.0 / (frequencies[:, None] - poles),
            factors.conj(),
            optimize=True,
        )

    realized = rational(result.poles, result.couplings)
    np.testing.assert_allclose(realized, rational(expected_poles, couplings), atol=2.0e-14)
    # Causal: the imaginary part is negative semidefinite in the upper half plane.
    imaginary = (realized - np.swapaxes(realized.conj(), -1, -2)) / 2.0j
    assert np.max(np.linalg.eigvalsh(imaginary)) <= 2.0e-15


def test_tiny_wrong_arc_weight_is_explicitly_discarded_under_named_tolerances(monkeypatch) -> None:
    monkeypatch.setattr(tolerances_module, "RELATIVE_TOLERANCE", 2.0e-6)
    correct = np.exp(0.7j)
    wrong = np.exp(-1.1j)
    moments = _scalar_atomic_moments(
        [correct, wrong],
        [1.0, 1.0e-6],
        n_max=5,
    )
    tolerances = replace(DEFAULT_TOLERANCES, moment_conservation=2.0e-6)
    result = SectorSelfEnergyRealization.realize(
        moments,
        Sector.PARTICLE,
        CayleyMap(center=0.0, scale=1.0),
        tolerances=tolerances,
    )

    assert result.selected_phase is None
    assert result.poles.size == 1
    assert result.discarded_wrong_arc_weight == pytest.approx(1.0e-6, rel=2.0e-8)
    assert result.discarded_total_weight == pytest.approx(1.0e-6, rel=2.0e-8)
    assert result.maximum_conserved_moment_residual < 1.1e-6
    assert np.all(result.poles > 0.0)


def test_significant_wrong_arc_weight_fails_with_attached_scan() -> None:
    moments = _scalar_atomic_moments(
        [np.exp(0.7j), np.exp(-1.1j)],
        [1.0, 0.1],
        n_max=5,
    )
    scan = SectorSelfEnergyRealization.scan_closures(
        moments,
        Sector.PARTICLE,
    )

    assert len(scan.candidates) == 1
    assert scan.selected is None
    assert scan.candidates[0].wrong_arc_weight == pytest.approx(0.1)
    with pytest.raises(RefusalError) as caught:
        scan.require_selected()
    assert caught.value.diagnostics is scan
    with pytest.raises(
        RefusalError,
        match="no sector-supported, inverse-safe realization",
    ) as caught:
        SectorSelfEnergyRealization.realize(
            moments,
            Sector.PARTICLE,
            CayleyMap(center=0.0, scale=1.0),
        )
    message = str(caught.value)
    assert message.startswith("particle sector refused at rank_floor=1e-10: ")
    assert "invalid arc weight 1.000e-01" in message
    assert "lower n_conserved, since the moments fix the terminal block" in message
    assert message.endswith("Retry with rank_floor=3e-10, then 1e-9, 2e-9, 5e-9, 1e-8.")


def _fixture_realization(name, n_conserved=None, **tolerance_overrides):
    """Realize a cached sector from ``tests/fixtures``, optionally at another ``n_conserved``."""

    stored = np.load(
        Path(__file__).resolve().parents[1] / "fixtures" / f"{name}.npz",
        allow_pickle=True,
    )
    if n_conserved is None:
        n_conserved = int(stored["n_conserved"])
    return SectorSelfEnergyRealization.realize(
        stored["values"],
        Sector.HOLE if str(stored["sector"]) == "hole" else Sector.PARTICLE,
        CayleyMap(center=float(stored["center"]), scale=float(stored["scale"])),
        **({} if n_conserved < 0 else {"n_conserved": n_conserved}),
        tolerances=replace(DEFAULT_TOLERANCES, **tolerance_overrides),
    )


def test_discarded_zeroth_moment_weight_is_redistributed_not_deleted() -> None:
    """On a real sector, a discard restores ``C[0]`` by congruence instead of losing the weight."""

    # At the default 1e-10 floor the redistributed weight exceeds the discard by one percent.
    result = _fixture_realization("magnesium-monoxide-K11-hole", rank_floor=1.0e-12)

    assert result.discarded_total_weight > 0.0
    assert result.zeroth_weight_redistributed
    assert result.redistributed_zeroth_weight > 0.0
    # The repaired order-0 breach is the discard up to cancellation between atoms.
    assert result.redistributed_zeroth_weight < result.discarded_total_weight

    # Roundoff on a 288-dimensional congruence: 1.6e-14 on CI, 4e-15 here, so pin the decade.
    drift.record("mgo_hole_K11.moment_residual_0", float(result.moment_residuals[0]))
    assert result.moment_residuals[0] < 1.0e-13
    # Eigensolver-dependent (2e-3 to 0.7 across BLAS builds), so it is reported, not gated.
    drift.record(
        "mgo_hole_K11.maximum_conservation_ratio", float(result.maximum_conservation_ratio)
    )

    # Weight moved but no node did: poles stay in the sector, residues stay rank-one PSD.
    assert np.all(result.poles < result.mapping.center)
    assert np.max(np.abs(np.abs(result.nodes) - 1.0)) < 1.0e-14
    residues = np.einsum("pl,ql->lpq", result.couplings, result.couplings.conj())
    for residue in residues[:20]:
        eigenvalues = np.linalg.eigvalsh(residue)
        assert eigenvalues[0] > -1.0e-14
        assert np.sum(eigenvalues > 1.0e-10 * eigenvalues[-1]) == 1


def test_redistribution_rescues_a_sector_that_otherwise_refuses(monkeypatch) -> None:
    """The congruence rescues a sector refused at order 0 and redistributes the named residual."""

    # An identity congruence gives the unrepaired path, whose refusal quotes the order-0 residual.
    with monkeypatch.context() as patched:
        patched.setattr(
            sector_helpers,
            "_zeroth_moment_congruence",
            lambda couplings, normalization, tolerances, diagnostics: np.eye(
                couplings.shape[0], dtype=np.complex128
            ),
        )
        # Whether it refuses is decided at the noise scale, so skip on a host where it does not.
        try:
            with pytest.raises(
                RefusalError,
                match="does not conserve a guaranteed moment: order=0",
            ) as refusal:
                _fixture_realization("magnesium-monoxide-K11-hole", n_conserved=8)
        except pytest.fail.Exception:
            drift.record("mgo_hole_K11_fit8.refuses_without_congruence", False)
            pytest.skip(
                "this fixture does not breach order 0 without the congruence on "
                "this machine, so the rescue it is here to pin has nothing to "
                "rescue; see tests/drift.py"
            )
    drift.record("mgo_hole_K11_fit8.refuses_without_congruence", True)
    refused_residual = float(re.search(r"residual=([0-9.e+-]+)", str(refusal.value)).group(1))
    drift.record("mgo_hole_K11_fit8.refused_residual", refused_residual)

    result = _skip_if_refused(
        "mgo_hole_K11_fit8",
        lambda: _fixture_realization("magnesium-monoxide-K11-hole", n_conserved=8),
    )

    assert result.zeroth_weight_redistributed
    # The discard varies by host and the two runs may drop different atoms, so pin the ratio to 2x.
    ratio = result.redistributed_zeroth_weight / refused_residual
    drift.record("mgo_hole_K11_fit8.redistributed_over_refused", float(ratio))
    assert 0.5 < ratio < 2.0
    # The refusal measured the discard, so the discard bounds the redistributed weight.
    assert result.discarded_total_weight >= result.redistributed_zeroth_weight
    # Order 0 is conserved to roundoff on the 288-dimensional congruence.
    drift.record("mgo_hole_K11_fit8.moment_residual_0", float(result.moment_residuals[0]))
    assert result.moment_residuals[0] < 1.0e-13
    # It lands inside the margin band, so it is scored against the banded gate.
    assert result.maximum_conservation_ratio <= DEFAULT_TOLERANCES.conservation_margin


def test_redistribution_is_declined_where_it_would_conserve_worse() -> None:
    """A congruence that conserves worse is declined, and the declined weight is still reported."""

    result = _fixture_realization("magnesium-monoxide-K11-particle", n_conserved=11)

    assert result.discarded_total_weight > 0.0
    assert not result.zeroth_weight_redistributed
    assert result.redistributed_zeroth_weight > 0.0
    # Declined means untouched: the order-0 residual is still the discard.
    assert result.moment_residuals[0] == pytest.approx(result.redistributed_zeroth_weight)
    assert result.maximum_conservation_ratio < 1.0


def test_empty_discard_leaves_the_realization_untouched() -> None:
    """No discard means no congruence, and nothing to record."""

    result = _fixture_realization("lithium-fluoride-K3-hole")

    assert result.discarded_total_weight == 0.0
    assert not result.zeroth_weight_redistributed
    assert result.redistributed_zeroth_weight == 0.0


def test_redistributing_a_gutted_zeroth_moment_is_refused() -> None:
    """A deficit past ``sector_redistribution`` is refused, and one inside it is corrected."""

    couplings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    nodes = np.exp(1.0j * np.asarray([0.6, 2.2]))
    moments = _atomic_matrix_moments(nodes, couplings, 3)
    normalization = UnitaryMomentRealization.normalize(moments)
    scan = SectorSelfEnergyRealization.scan_closures(
        moments,
        Sector.PARTICLE,
    )

    # Keeping one of two orthogonal unit atoms empties a direction of C[0]: deficit 1.0.
    with pytest.raises(
        RefusalError,
        match="carry too much of the zeroth moment to redistribute",
    ) as caught:
        sector_helpers._zeroth_moment_congruence(
            couplings[:, :1],
            normalization,
            DEFAULT_TOLERANCES,
            scan,
        )
    assert "deficit=1.000e+00" in str(caught.value)
    assert "threshold=0.5" in str(caught.value)
    assert caught.value.diagnostics is scan

    # A deficit inside the tolerance is corrected instead of refused.
    transform = sector_helpers._zeroth_moment_congruence(
        couplings * np.asarray([1.0, 0.8]),
        normalization,
        DEFAULT_TOLERANCES,
        scan,
    )
    repaired = transform @ (couplings * np.asarray([1.0, 0.8]))
    np.testing.assert_allclose(repaired @ repaired.conj().T, moments[0], atol=1.0e-14)


def test_near_inverse_singularity_is_rejected_without_clipping(monkeypatch) -> None:
    node = np.exp(1.0e-5j)
    moments = _scalar_atomic_moments([node], [1.0], n_max=3)
    monkeypatch.setattr(tolerances_module, "ABSOLUTE_TOLERANCE", 1.0e-4)
    scan = SectorSelfEnergyRealization.scan_closures(
        moments,
        Sector.PARTICLE,
    )

    assert scan.selected is None
    assert scan.candidates[0].near_singular_weight == pytest.approx(1.0)
    assert scan.candidates[0].spectrum.nodes[0] == pytest.approx(node)
    with pytest.raises(RefusalError) as caught:
        SectorSelfEnergyRealization.realize(
            moments,
            Sector.PARTICLE,
            CayleyMap(center=0.0, scale=1.0),
        )
    assert caught.value.diagnostics is not None


def test_scalar_terminal_family_can_fail_and_does_not_trigger_repair() -> None:
    moments = np.asarray(
        [
            np.eye(2, dtype=np.complex128),
            np.diag([0.8j, -0.8j]),
        ]
    )
    scan = SectorSelfEnergyRealization.scan_closures(
        moments,
        Sector.PARTICLE,
        phase_count=64,
    )

    assert len(scan.candidates) == 64
    assert scan.selected is None
    assert min(candidate.wrong_arc_weight for candidate in scan.candidates) > 0.8
    with pytest.raises(RefusalError, match="increase terminal_phase_count") as caught:
        SectorSelfEnergyRealization.realize(
            moments,
            Sector.PARTICLE,
            CayleyMap(center=0.0, scale=1.0),
            phase_count=64,
        )
    assert caught.value.diagnostics is not None


def _terminated_wrong_arc_measure():
    """Return a two-atom measure that both backends terminate on, with one atom on the wrong arc."""

    return _scalar_atomic_moments(
        [np.exp(0.7j), np.exp(-1.1j)],
        [1.0, 1.0],
        n_max=5,
    )


@pytest.mark.parametrize("backend", ["toeplitz", "block-cmv"])
def test_a_terminated_refusal_names_lower_n_conserved(backend) -> None:
    """With the terminal block fixed by the moments, the refusal names ``n_conserved``."""

    with pytest.raises(RefusalError) as caught:
        SectorSelfEnergyRealization.realize(
            _terminated_wrong_arc_measure(),
            Sector.PARTICLE,
            CayleyMap(center=0.0, scale=1.0),
            phase_count=8,
            realization_algorithm=backend,
        )
    message = str(caught.value)

    assert "among 1 closure:" in message
    assert "lower n_conserved, since the moments fix the terminal block" in message
    assert "terminal_phase_count" not in message


def test_lowering_n_conserved_restores_the_terminal_degree_of_freedom() -> None:
    """Lowering ``n_conserved`` frees the terminal block and withholds the surplus orders."""

    moments = _terminated_wrong_arc_measure()
    terminated = SectorSelfEnergyRealization.scan_closures(
        moments,
        Sector.PARTICLE,
        phase_count=8,
        realization_algorithm="toeplitz",
    )
    assert terminated.candidates[0].realization.terminal_from_moments
    assert len(terminated.candidates) == 1

    lowered = SectorSelfEnergyRealization.scan_closures(
        moments,
        Sector.PARTICLE,
        n_conserved=1,
        phase_count=8,
        realization_algorithm="toeplitz",
    )
    canonical = lowered.candidates[0]
    assert not canonical.realization.terminal_from_moments
    assert canonical.realization.terminal_unitary.shape[0] >= 1
    assert len(lowered.candidates) == 8
    assert canonical.withheld_moment_residuals.size == 4


def test_zero_measure_has_empty_rational_realization() -> None:
    moments = np.zeros((4, 3, 3), dtype=np.complex128)
    result = SectorSelfEnergyRealization.realize(
        moments,
        Sector.HOLE,
        CayleyMap(center=0.2, scale=1.3),
    )

    assert result.poles.size == 0
    assert result.couplings.shape == (3, 0)
    assert result.nodes.size == 0
    assert result.selected_phase is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sector": "particle"}, "Sector"),
        ({"phase_count": 0}, "positive integer"),
        ({"n_conserved": -1}, "nonnegative"),
        ({"n_conserved": 3}, "cannot exceed"),
        ({"realization_algorithm": "unknown"}, "realization_algorithm"),
        ({"realization_algorithm": "auto"}, "realization_algorithm"),
        ({"tolerances": object()}, "Tolerances"),
    ],
)
def test_invalid_sector_scan_arguments_are_rejected(kwargs, match) -> None:
    arguments = {
        "moments": np.asarray([[[1.0 + 0.0j]], [[0.2j]]]),
        "sector": Sector.PARTICLE,
    }
    arguments.update(kwargs)
    with pytest.raises(ValidationError, match=match):
        SectorSelfEnergyRealization.scan_closures(**arguments)


@pytest.mark.pyscf
@pytest.mark.parametrize("sector", [Sector.HOLE, Sector.PARTICLE])
def test_h2_exact_g0w0_sector_is_recovered_from_only_its_moments(h2_rhf, sector) -> None:
    exact = ExactG0W0SelfEnergy.from_mean_field(h2_rhf)
    reference = exact.hole if sector is Sector.HOLE else exact.particle
    mapping = CayleyMap(
        center=exact.reference.chemical_potential,
        scale=1.0,
    )
    moments = reference.cayley_moments(mapping, 3)
    result = SectorSelfEnergyRealization.realize(moments, sector, mapping)

    assert result.selected_phase is None
    np.testing.assert_allclose(result.poles, reference.poles, atol=4.0e-14)
    residues = np.einsum("pl,ql->lpq", result.couplings, result.couplings.conj())
    np.testing.assert_allclose(
        residues,
        np.einsum("pl,ql->lpq", reference.couplings, reference.couplings.conj()),
        atol=2.0e-14,
    )
    frequencies = np.asarray([-1.0 + 0.4j, 0.1 + 0.7j, 2.0 + 0.5j])
    realized = np.einsum(
        "pl,zl,ql->zpq",
        result.couplings,
        1.0 / (frequencies[:, None] - result.poles),
        result.couplings.conj(),
        optimize=True,
    )
    np.testing.assert_allclose(
        realized,
        np.einsum(
            "pl,zl,ql->zpq",
            reference.couplings,
            1.0 / (frequencies[:, None] - reference.poles),
            reference.couplings.conj(),
            optimize=True,
        ),
        atol=2.0e-14,
    )
    imaginary = (realized - np.swapaxes(realized.conj(), -1, -2)) / 2.0j
    assert np.max(np.linalg.eigvalsh(imaginary)) < 2.0e-15


def test_unitary_eigendecomposition_matches_schur_on_both_arcs() -> None:
    """The Hermitian-part route matches Schur, even on mirrored nodes with equal ``cos(theta)``."""

    from scipy.linalg import schur

    from cayleygw.realization._helpers.base import _unitary_eigendecomposition

    generator = np.random.default_rng(9021)

    def unitary_with(angles):
        basis, _ = np.linalg.qr(
            generator.normal(size=(angles.size, angles.size))
            + 1.0j * generator.normal(size=(angles.size, angles.size))
        )
        return basis @ np.diag(np.exp(1.0j * angles)) @ basis.conj().T

    lower = -np.linspace(0.2, 2.9, 12)
    cases = {
        "one arc": lower,
        "exactly mirrored": np.concatenate([lower, -lower]),
        "mirrored to 1e-13": np.concatenate([lower, -lower + 1.0e-13]),
        "degenerate": np.concatenate([lower, lower]),
    }
    for label, angles in cases.items():
        matrix = unitary_with(np.asarray(angles))
        threshold = 1.0e-10 * max(1.0, math.sqrt(matrix.shape[0]))
        nodes, vectors, image, residual = _unitary_eigendecomposition(matrix, threshold)

        assert residual <= threshold, f"{label}: residual {residual:.3e}"
        np.testing.assert_allclose(image, matrix @ vectors, atol=1.0e-12)
        np.testing.assert_allclose(
            vectors.conj().T @ vectors,
            np.eye(matrix.shape[0]),
            atol=1.0e-11,
            err_msg=f"{label}: eigenvectors are not orthonormal",
        )
        np.testing.assert_allclose(
            image,
            vectors * nodes[None, :],
            atol=1.0e-11,
            err_msg=f"{label}: not an eigendecomposition",
        )
        # Same spectrum as Schur, up to ordering.
        reference = np.diag(schur(matrix, output="complex")[0])
        key = lambda values: np.sort_complex(np.round(values, 10))
        np.testing.assert_allclose(key(nodes), key(reference), atol=1.0e-9)


def test_unitary_eigendecomposition_falls_back_when_not_normal() -> None:
    """A materially nonnormal matrix is reported through its residual, not repaired."""

    from cayleygw.realization._helpers.base import _unitary_eigendecomposition

    matrix = np.asarray([[1.0, 5.0], [0.0, -1.0]], dtype=np.complex128)
    _, vectors, image, residual = _unitary_eigendecomposition(matrix, 1.0e-10)
    assert residual > 1.0e-10
    np.testing.assert_allclose(image, matrix @ vectors, atol=1.0e-12)


def _three_atom_particle_measure(n_max=2):
    """Return a nondegenerate three-atom upper-arc scalar matrix measure."""

    nodes = np.exp(1.0j * np.asarray([0.4, 1.1, 2.2]))
    couplings = np.asarray([[1.0, 0.5, 0.25]], dtype=np.complex128)
    return _atomic_matrix_moments(nodes, couplings, n_max)


def test_an_unknown_phase_refinement_is_rejected() -> None:
    with pytest.raises(ValidationError, match="phase_refinement"):
        SectorSelfEnergyRealization.scan_closures(
            _three_atom_particle_measure(),
            Sector.PARTICLE,
            realization_algorithm="toeplitz",
            phase_refinement="bisection",
        )


def _sector_weight_ratio(result) -> float:
    """Return the used closure's invalid arc weight over its sharp threshold."""

    closure = result.selected_closure
    return closure.wrong_arc_weight / closure.weight_threshold


def test_clean_conservation_reports_no_marginal_orders() -> None:
    moments = _three_atom_particle_measure(n_max=5)
    result = SectorSelfEnergyRealization.realize(
        moments,
        Sector.PARTICLE,
        CayleyMap(center=0.0, scale=1.0),
    )

    assert not result.conservation_is_marginal
    assert 0.0 <= result.maximum_conservation_ratio <= 1.0
    assert not result.support_is_marginal
    assert 0.0 <= _sector_weight_ratio(result) <= 1.0


def test_conservation_margin_band_accepts_and_flags_instead_of_flipping() -> None:
    """A conservation residual inside the margin band passes flagged; a sharp gate refuses it."""

    # The arc gate admits the 1e-6 atom inside a widened band, so conservation alone decides.
    wide = replace(DEFAULT_TOLERANCES, arc_margin=1.0e4)
    correct = np.exp(0.7j)
    wrong = np.exp(-1.1j)
    moments = _scalar_atomic_moments(
        [correct, wrong],
        [1.0, 1.0e-6],
        n_max=5,
    )
    mapping = CayleyMap(center=0.0, scale=1.0)
    clean = SectorSelfEnergyRealization.realize(
        moments,
        Sector.PARTICLE,
        mapping,
        tolerances=replace(wide, moment_conservation=2.0e-6),
    )
    sharp = replace(wide, moment_conservation=4.0e-7, conservation_margin=1.0)
    with pytest.raises(RefusalError, match="margin=1"):
        SectorSelfEnergyRealization.realize(
            moments,
            Sector.PARTICLE,
            mapping,
            tolerances=sharp,
        )
    banded = replace(wide, moment_conservation=4.0e-7)
    flagged = SectorSelfEnergyRealization.realize(
        moments,
        Sector.PARTICLE,
        mapping,
        tolerances=banded,
    )

    assert flagged.conservation_is_marginal
    assert 1.0 < flagged.maximum_conservation_ratio <= DEFAULT_TOLERANCES.conservation_margin
    np.testing.assert_allclose(flagged.poles, clean.poles, atol=1.0e-13)


def _boundary_saturating_measure(excess: float, n_max: int = 3):
    """Return a hole measure on the edge of the contraction ball, with radius ``1 + excess``."""

    separation = 2.0e-7
    moments = _scalar_atomic_moments(
        [np.exp(-0.7j), np.exp(-(0.7 + separation) * 1j)],
        [1.0, 1.0],
        n_max=n_max,
    )
    powers = (1.0 + excess) ** np.arange(n_max + 1, dtype=float)
    return moments * powers[:, None, None]


def test_the_contraction_check_reaches_the_sector_realization(monkeypatch) -> None:
    """The sector projects an excess within the positivity tolerance and refuses one past it."""

    mapping = CayleyMap(center=0.0, scale=1.0)
    options = dict(n_conserved=1, phase_count=32, realization_algorithm="block-cmv")
    clean = SectorSelfEnergyRealization.realize(
        _boundary_saturating_measure(0.0), Sector.HOLE, mapping, **options
    )
    assert clean.selected_closure.realization.maximum_contraction_ratio <= 1.0

    within = SectorSelfEnergyRealization.realize(
        _boundary_saturating_measure(0.5 * DEFAULT_TOLERANCES.positivity),
        Sector.HOLE,
        mapping,
        **options,
    )
    assert 0.0 < within.selected_closure.realization.maximum_contraction_ratio <= 1.0
    # The projection changes no answer: the poles are those of the unperturbed measure.
    assert not within.conservation_is_marginal
    np.testing.assert_allclose(within.poles, clean.poles, atol=1.0e-9)

    with pytest.raises(RefusalError, match="contraction ball"):
        SectorSelfEnergyRealization.realize(
            _boundary_saturating_measure(5.0 * DEFAULT_TOLERANCES.positivity),
            Sector.HOLE,
            mapping,
            **options,
        )

    with pytest.raises(RefusalError, match="conserve"):
        SectorSelfEnergyRealization.realize(
            _boundary_saturating_measure(20.0e-10),
            Sector.HOLE,
            mapping,
            tolerances=replace(DEFAULT_TOLERANCES, positivity=1.0e2),
            **options,
        )


def _marginal_arc_weight_measure():
    """Return a measure whose only closure is in the arc-weight band at a 2e-10 tolerance."""

    return _scalar_atomic_moments(
        [np.exp(0.7j), np.exp(-1.1j)],
        [1.0, 1.0e-9],
        n_max=5,
    )


def test_arc_weight_margin_band_accepts_and_flags_instead_of_flipping(monkeypatch) -> None:
    """Invalid arc weight inside the band passes flagged; a sharp gate refuses it."""

    moments = _marginal_arc_weight_measure()
    mapping = CayleyMap(center=0.0, scale=1.0)
    monkeypatch.setattr(tolerances_module, "RELATIVE_TOLERANCE", 2.0e-9)
    clean = SectorSelfEnergyRealization.realize(moments, Sector.PARTICLE, mapping)
    assert not clean.support_is_marginal
    assert 0.0 <= _sector_weight_ratio(clean) <= 1.0

    monkeypatch.setattr(tolerances_module, "RELATIVE_TOLERANCE", 2.0e-10)
    with pytest.raises(RefusalError, match="no sector-supported"):
        SectorSelfEnergyRealization.realize(
            moments,
            Sector.PARTICLE,
            mapping,
            tolerances=replace(DEFAULT_TOLERANCES, arc_margin=1.0),
        )

    flagged = SectorSelfEnergyRealization.realize(moments, Sector.PARTICLE, mapping)

    assert flagged.support_is_marginal
    assert 1.0 < _sector_weight_ratio(flagged) <= DEFAULT_TOLERANCES.arc_margin
    # The conservation check is untouched and passes sharply.
    assert not flagged.conservation_is_marginal
    assert flagged.maximum_conservation_ratio <= 1.0
    np.testing.assert_allclose(flagged.poles, clean.poles, atol=1.0e-13)


def _mixed_band_scan(**overrides):
    """Return a scan holding both cleanly supported and marginal candidates."""

    moments = _scalar_atomic_moments(
        [np.exp(0.7j), np.exp(2.3j), np.exp(-1.1j)],
        [1.0, 0.5, 1.0e-9],
        n_max=2,
    )
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(tolerances_module, "RELATIVE_TOLERANCE", 2.0e-10)
        return SectorSelfEnergyRealization.scan_closures(
            moments,
            Sector.PARTICLE,
            phase_count=32,
            tolerances=replace(DEFAULT_TOLERANCES, **overrides),
        )


def test_marginal_support_is_ranked_below_every_clean_support() -> None:
    """Banded closures rank behind every clean one, and among themselves by invalid arc weight."""

    scan = _mixed_band_scan()
    ranked = scan.ranked_indices
    bands = [scan.candidates[index].support_acceptable for index in ranked]
    assert any(bands) and not all(bands), "this scan must exercise both bands"
    # Every clean candidate precedes every marginal one.
    assert bands == sorted(bands, reverse=True)
    assert scan.selected is not None
    assert scan.selected.support_acceptable
    marginal_weights = [
        scan.candidates[index].wrong_arc_weight for index, clean in zip(ranked, bands) if not clean
    ]
    assert marginal_weights == sorted(marginal_weights)


def test_unit_weight_margin_restores_the_sharp_pre_band_gate() -> None:
    """``arc_margin = 1.0`` accepts exactly the cleanly supported set."""

    banded = _mixed_band_scan()
    sharp = _mixed_band_scan(arc_margin=1.0)
    clean_indices = tuple(
        index for index, candidate in enumerate(banded.candidates) if candidate.support_acceptable
    )

    assert sharp.ranked_indices == tuple(
        index for index in banded.ranked_indices if index in clean_indices
    )
    assert not any(candidate.support_marginal for candidate in sharp.candidates)
    assert sharp.selected_index == banded.selected_index


def test_production_path_records_the_requested_policy() -> None:
    """``realize`` records the requested ``auto`` policy beside the backend that delivered."""

    mapping = CayleyMap(center=0.4, scale=1.7)
    nodes = np.exp(1.0j * np.asarray([0.30, 1.25, 2.50]))
    couplings = np.asarray(
        [[1.0, 0.35 - 0.2j, 0.1j], [0.2, 0.8, -0.5 + 0.1j]],
        dtype=np.complex128,
    )
    moments = _atomic_matrix_moments(nodes, couplings, 8)

    default = SectorSelfEnergyRealization.realize(moments, Sector.PARTICLE, mapping)
    np.testing.assert_allclose(
        np.sort(default.poles),
        np.sort(np.asarray(mapping.inverse(nodes))),
        atol=2.0e-14,
    )
    assert default.closure_scan.requested_realization_algorithm == "auto"
    assert default.closure_scan.realization_algorithm in ("block-cmv", "toeplitz")


def test_production_path_rejects_an_unknown_backend_naming_every_value() -> None:
    mapping = CayleyMap(center=0.0, scale=1.0)
    nodes = np.exp(1.0j * np.asarray([0.30, 1.25, 2.50]))
    couplings = np.asarray([[1.0, 0.5, 0.25]], dtype=np.complex128)
    moments = _atomic_matrix_moments(nodes, couplings, 6)

    with pytest.raises(ValidationError, match="toeplitz"):
        SectorSelfEnergyRealization.realize(
            moments,
            Sector.PARTICLE,
            mapping,
            realization_algorithm="unknown",
        )
    with pytest.raises(ValidationError, match="CayleyMap"):
        SectorSelfEnergyRealization.realize(moments, Sector.PARTICLE, None)
    with pytest.raises(ValidationError, match="Sector"):
        SectorSelfEnergyRealization.realize(moments, "particle", mapping)


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_scan_worker_count_does_not_change_the_candidate_sequence(
    workers,
) -> None:
    """Candidates keep their positions, so ties in the selection ``min`` resolve the same way."""

    moments = np.asarray([[[2.0 + 0.0j]]])
    common = {
        "phase_count": 64,
        "phase_refinement": "fixed",
        "native_threads": 1,
    }
    serial = SectorSelfEnergyRealization.scan_closures(
        moments,
        Sector.PARTICLE,
        n_workers=1,
        **common,
    )
    parallel = SectorSelfEnergyRealization.scan_closures(
        moments,
        Sector.PARTICLE,
        n_workers=workers,
        **common,
    )
    assert len(parallel.candidates) == len(serial.candidates)
    assert parallel.selected_index == serial.selected_index
    for left, right in zip(parallel.candidates, serial.candidates):
        assert left.phase == right.phase
        assert left.acceptable == right.acceptable
        np.testing.assert_array_equal(
            left.realization.reconstructed_moments,
            right.realization.reconstructed_moments,
        )


@pytest.mark.parametrize("value", [0, -1, 1.5, "2"])
def test_scan_rejects_invalid_worker_counts(value) -> None:
    moments = np.asarray([[[2.0 + 0.0j]]])
    for keyword in ("n_workers", "native_threads"):
        with pytest.raises(
            ValidationError,
            match="positive integer",
        ):
            SectorSelfEnergyRealization.scan_closures(
                moments,
                Sector.PARTICLE,
                phase_count=8,
                **{keyword: value},
            )


def test_scan_logs_one_stage_for_the_whole_search(caplog) -> None:
    """The scan logs one stage, since per-candidate stages would sum concurrent time."""

    moments = np.asarray([[[2.0 + 0.0j]]])
    with caplog.at_level(logging.INFO, logger="cayleygw"):
        SectorSelfEnergyRealization.scan_closures(
            moments,
            Sector.PARTICLE,
            phase_count=32,
            n_workers=4,
            native_threads=1,
            verbose=1,
        )
    messages = [record.getMessage() for record in caplog.records]
    entries = [m for m in messages if m.startswith("particle terminal candidates: phase_count=32")]
    exits = [
        m for m in messages if re.match(r"particle terminal candidates: [0-9.]+ (s|min|h)$", m)
    ]
    assert len(entries) == 1 and len(exits) == 1


@pytest.mark.parametrize(
    ("floor", "advice"),
    [
        (1.0e-12, " Retry with rank_floor=3e-10, then 1e-9, 2e-9, 5e-9, 1e-8."),
        (1.0e-10, " Retry with rank_floor=3e-10, then 1e-9, 2e-9, 5e-9, 1e-8."),
        (3.0e-10, " Retry with rank_floor=1e-9, then 2e-9, 5e-9, 1e-8."),
        (4.0e-10, " Retry with rank_floor=1e-9, then 2e-9, 5e-9, 1e-8."),
        (5.0e-9, " Retry with rank_floor=1e-8."),
        (1.0e-8, ""),
        (1.0e-6, ""),
    ],
)
def test_the_retry_advice_names_only_higher_floors_in_ladder_order(floor, advice) -> None:
    moments = _scalar_atomic_moments([np.exp(0.7j), np.exp(-1.1j)], [1.0, 0.1], n_max=5)
    tolerances = replace(DEFAULT_TOLERANCES, rank_floor=floor)

    assert sector_module._retry_advice(tolerances) == advice
    with pytest.raises(RefusalError) as caught:
        SectorSelfEnergyRealization.realize(
            moments,
            Sector.PARTICLE,
            CayleyMap(center=0.0, scale=1.0),
            tolerances=tolerances,
        )
    message = str(caught.value)
    assert message.startswith(
        f"particle sector refused at rank_floor={sector_module._floor_text(floor)}: "
    )
    if advice:
        assert message.endswith("." + advice)
    else:
        assert "Retry" not in message and message.endswith(".")


def _skip_if_refused(label: str, build):
    """Run a realization, or record the refusal and skip on a host where it sits past its gate."""

    try:
        result = build()
    except RefusalError as error:
        drift.record(f"{label}.realizes", False)
        pytest.skip(f"{label} refuses on this machine ({error}); see tests/drift.py")
    drift.record(f"{label}.realizes", True)
    return result


def _scannable_moments():
    """Return a single zeroth moment: one free terminal block whose scan selects a closure."""

    return np.asarray([[[2.0 + 0.0j]]])


def _fourier_moments():
    """Return moments that leave a free terminal block and two withheld orders at order 0."""

    return np.asarray([[[2.0 + 0.0j]], [[0.5 + 0.0j]], [[0.1 + 0.0j]]])


_LADDER_MAPPING = CayleyMap(center=0.4, scale=1.7)


def _ladder_moments():
    """Return a particle sector whose Gram spectrum spans decades."""

    return _atomic_matrix_moments(
        np.exp(1.0j * np.asarray([0.30, 1.25, 2.50, 4.10])),
        np.asarray(
            [
                [1.0, 1.0e-2, 1.0e-4, 1.0e-6],
                [0.7, 0.9e-2, 1.2e-4, 0.8e-6],
            ],
            dtype=np.complex128,
        ),
        8,
    )


def test_the_gram_eigensolve_is_thread_pinned_by_default() -> None:
    """The Gram eigensolve defaults to one thread, since its smallest eigenvalues are roundoff."""

    from cayleygw.realization.base import UnitaryMomentRealization
    from cayleygw.realization.toeplitz import build_gram_spectrum

    values = UnitaryMomentRealization.normalize(_ladder_moments()).values

    assert build_gram_spectrum(values).native_threads == 1
    np.testing.assert_array_equal(
        build_gram_spectrum(values).eigenvalues,
        build_gram_spectrum(values).eigenvalues,
    )


def test_a_scan_that_raises_still_logs_its_stage(monkeypatch, caplog) -> None:
    """A scan that fails still spent wall time, so its stage is logged."""

    def blow_up(*args, **kwargs):
        raise RefusalError("candidate evaluation failed", kind="sector")

    monkeypatch.setattr(sector_helpers, "_candidate_diagnostics", blow_up)
    with caplog.at_level(logging.INFO, logger="cayleygw"):
        with pytest.raises(RefusalError):
            SectorSelfEnergyRealization.scan_closures(
                _ladder_moments(),
                Sector.PARTICLE,
                realization_algorithm="toeplitz",
                verbose=1,
            )
    messages = [record.getMessage() for record in caplog.records]
    assert any(m.startswith("particle terminal candidates: failed after") for m in messages)


def test_a_finalization_that_raises_still_logs_its_stage(monkeypatch, caplog) -> None:
    """Every rejected backend's finalization cost is visible in the log."""

    def blow_up(*args, **kwargs):
        raise RefusalError("finalization failed", kind="sector")

    monkeypatch.setattr(sector_module, "_finalize_first_viable_closure", blow_up)
    with caplog.at_level(logging.INFO, logger="cayleygw"):
        with pytest.raises(RefusalError):
            SectorSelfEnergyRealization.realize(
                _ladder_moments(),
                Sector.PARTICLE,
                _LADDER_MAPPING,
                realization_algorithm="toeplitz",
                verbose=1,
            )
    messages = [record.getMessage() for record in caplog.records]
    assert any(m.startswith("particle poles and couplings: failed after") for m in messages)


def test_restricted_probing_is_never_worse_than_the_grid_it_restricts():
    """``restricted`` skips an interval only if its objective floor cannot beat the incumbent."""

    moments = _fourier_moments()
    shared = dict(n_conserved=0, phase_count=32)
    restricted = SectorSelfEnergyRealization.scan_closures(
        moments, Sector.PARTICLE, phase_refinement="restricted", **shared
    )
    grid = SectorSelfEnergyRealization.scan_closures(moments, Sector.PARTICLE, **shared)

    def best(scan):
        return min(_selection_key(c)[0] for c in scan.candidates if c.acceptable)

    assert best(restricted) <= best(grid)
    assert len(grid.candidates) == 32


def test_restricted_probing_bisects_toward_an_inadmissible_minimum():
    """``restricted`` bisects toward an infeasible minimiser, so a candidate lands off the grid."""

    restricted = SectorSelfEnergyRealization.scan_closures(
        _fourier_moments(),
        Sector.PARTICLE,
        n_conserved=0,
        phase_count=32,
        phase_refinement="restricted",
    )
    step = 2.0 * math.pi / 32
    off_grid = [
        c
        for c in restricted.candidates
        if c.phase is not None
        and abs(float(c.phase) / step - round(float(c.phase) / step)) > 1.0e-9
    ]
    assert off_grid, "no candidate landed between grid phases"


def test_restricted_probing_refuses_without_a_withheld_moment():
    """``restricted`` fits the withheld-residual polynomial, so it needs a withheld moment."""

    with pytest.raises(ValidationError, match="withheld"):
        SectorSelfEnergyRealization.scan_closures(
            _scannable_moments(),
            Sector.PARTICLE,
            phase_count=16,
            phase_refinement="restricted",
        )


def test_released_realization_refuses_its_moment_methods() -> None:
    """A realization that released its matrix refuses by name instead of failing on ``None``."""

    moments = np.asarray([[[2.0 + 0.0j]]])
    scan = SectorSelfEnergyRealization.scan_closures(
        moments,
        Sector.HOLE,
        phase_count=8,
    )
    realization = scan.candidates[0].realization
    for method in ("matrix_moments", "normalized_matrix_moments"):
        with pytest.raises(RefusalError, match="released its dense matri"):
            getattr(realization, method)(0)


def test_a_conservation_refusal_names_how_many_orders_breached(monkeypatch):
    """With every order breached, the refusal also reports their count and the highest."""

    def zero_thresholds(realization, *, base_tolerance=None, absolute_tolerance=None):
        return np.zeros(realization.normalization.n_max + 1)

    for backend in (UnitaryMomentRealization, ToeplitzRealization):
        monkeypatch.setattr(backend, "physical_moment_acceptance_thresholds", zero_thresholds)
    with pytest.raises(RefusalError) as refusal:
        _fixture_realization("magnesium-monoxide-K11-particle", n_conserved=9)

    message = str(refusal.value)
    assert "does not conserve" in message
    # Refusal records parse ``order=``, so it stays the lowest breached order.
    assert "order=0," in message
    breached = int(re.search(r"breached=(\d+)", message).group(1))
    highest = int(re.search(r"highest=(\d+)", message).group(1))
    assert breached == 10
    assert highest == 9


@pytest.mark.parametrize("workers", [3, 7])
def test_restricted_walks_are_the_same_scan_with_more_workers(workers):
    """More workers leave the restricted walks' candidates, order and selection unchanged."""

    moments = _fourier_moments()
    shared = dict(
        n_conserved=0,
        phase_count=32,
        phase_refinement="restricted",
        native_threads=1,
    )
    serial = SectorSelfEnergyRealization.scan_closures(
        moments, Sector.PARTICLE, n_workers=1, **shared
    )
    parallel = SectorSelfEnergyRealization.scan_closures(
        moments, Sector.PARTICLE, n_workers=workers, **shared
    )
    assert [c.phase for c in parallel.candidates] == [c.phase for c in serial.candidates]
    assert [c.acceptable for c in parallel.candidates] == [c.acceptable for c in serial.candidates]
    assert parallel.selected_index == serial.selected_index
    assert np.array_equal(
        parallel.candidates[parallel.selected_index].spectrum.nodes,
        serial.candidates[serial.selected_index].spectrum.nodes,
    )


def test_a_reclose_that_fails_its_contracts_is_a_refused_scan_not_a_crash(monkeypatch):
    """A reclose contract failure becomes a sector refusal, which the ``auto`` fallback catches."""

    moments = _fourier_moments()
    original = ToeplitzRealization.reclose

    def failing(realization, terminal, **kwargs):
        if abs(complex(np.asarray(terminal).ravel()[0]) - 1.0) > 1.0e-12:
            raise RefusalError("synthetic contract failure", kind="block-cmv")
        return original(realization, terminal, **kwargs)

    monkeypatch.setattr(ToeplitzRealization, "reclose", failing)
    with pytest.raises(RefusalError, match="failed its realization contracts"):
        SectorSelfEnergyRealization.scan_closures(
            moments,
            Sector.PARTICLE,
            n_conserved=0,
            phase_count=32,
            phase_refinement="restricted",
            realization_algorithm="toeplitz",
        )


# The closure oracle: the gates as a continuous function of the terminal unitary.


def _free_terminal_measure(atoms=14, rank=3, n_max=4, seed=5):
    """Return a measure with more atoms than a fit at ``n_conserved=2`` absorbs."""

    generator = np.random.default_rng(seed)
    nodes = np.exp(1.0j * np.linspace(-0.3, -5.9, atoms))
    couplings = generator.normal(size=(rank, atoms)) + 1.0j * generator.normal(size=(rank, atoms))
    return _atomic_matrix_moments(nodes, couplings, n_max)


def _terminated_measure():
    """Return a measure the fit absorbs exactly, leaving no closure freedom."""

    generator = np.random.default_rng(5)
    nodes = np.exp(1.0j * np.asarray([-0.4, -1.1, -2.0, -2.9, -3.7, -4.6]))
    couplings = generator.normal(size=(3, 6)) + 1.0j * generator.normal(size=(3, 6))
    return _atomic_matrix_moments(nodes, couplings, 4)


def test_closure_module_stays_independent_of_moment_producers() -> None:
    path = Path(__file__).resolve().parents[2] / "src" / "cayleygw" / "realization" / "sector.py"
    forbidden = {"pyscf", "reference_state", "response", "screening", "contour", "reference"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".")[0])
    assert forbidden.isdisjoint(imported)


@pytest.mark.parametrize("backend", ["toeplitz", "block-cmv"])
def test_objective_reproduces_the_scan_candidate_for_every_sampled_phase(
    backend,
) -> None:
    moments = _free_terminal_measure()
    objective = build_closure_objective(
        moments,
        Sector.HOLE,
        n_conserved=2,
        backend=backend,
    )
    scan = SectorSelfEnergyRealization.scan_closures(
        moments,
        Sector.HOLE,
        n_conserved=2,
        phase_count=8,
        realization_algorithm=backend,
    )

    assert not objective.terminal_is_fixed
    assert objective.terminal_dimension == 3
    for candidate in scan.candidates:
        evaluated = objective.evaluate(
            objective.scalar(candidate.phase),
            phase=candidate.phase,
        )
        assert evaluated.wrong_arc_weight == candidate.wrong_arc_weight
        assert evaluated.near_singular_weight == candidate.near_singular_weight
        assert evaluated.acceptable == candidate.acceptable
        assert evaluated.support_acceptable == candidate.support_acceptable
        assert evaluated.total_weight == pytest.approx(candidate.total_weight)


@pytest.mark.parametrize("backend", ["toeplitz", "block-cmv"])
def test_objective_reproduces_the_scan_ranking_exactly(backend) -> None:
    moments = _free_terminal_measure()
    objective = build_closure_objective(
        moments,
        Sector.HOLE,
        n_conserved=2,
        backend=backend,
    )
    scan = SectorSelfEnergyRealization.scan_closures(
        moments,
        Sector.HOLE,
        n_conserved=2,
        phase_count=8,
        realization_algorithm=backend,
    )

    for candidate in scan.candidates:
        evaluated = objective.evaluate(
            objective.scalar(candidate.phase),
            phase=candidate.phase,
        )
        assert objective.ranking_key(evaluated) == objective.ranking_key(candidate)

    from_scan = [c.phase for c in sorted(scan.candidates, key=objective.ranking_key)]
    from_objective = [
        c.phase
        for c in sorted(
            scan.candidates,
            key=lambda c: objective.ranking_key(
                objective.evaluate(objective.scalar(c.phase), phase=c.phase)
            ),
        )
    ]
    assert from_scan == from_objective


def test_the_three_families_nest() -> None:
    objective = build_closure_objective(
        _free_terminal_measure(),
        Sector.HOLE,
        n_conserved=2,
    )
    dimension = objective.terminal_dimension

    assert objective.scalar_parameter_count == 1
    assert objective.diagonal_parameter_count == dimension
    assert objective.unitary_parameter_count == dimension * dimension

    np.testing.assert_allclose(
        objective.scalar(0.7),
        objective.diagonal(np.full(dimension, 0.7)),
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        objective.generated(np.zeros((dimension, dimension), dtype=np.complex128)),
        np.eye(dimension),
        atol=1.0e-15,
    )
    # A scalar closure is also a generator, so a unitary search can start from the best phase.
    generator = 0.7j * np.eye(dimension, dtype=np.complex128)
    np.testing.assert_allclose(
        objective.generated(generator),
        objective.scalar(0.7),
        atol=1.0e-14,
    )


def test_every_family_member_is_unitary_and_evaluable() -> None:
    objective = build_closure_objective(
        _free_terminal_measure(),
        Sector.HOLE,
        n_conserved=2,
    )
    dimension = objective.terminal_dimension
    generator = np.random.default_rng(3)
    skew = generator.normal(size=(dimension, dimension)) + 1.0j * generator.normal(
        size=(dimension, dimension)
    )
    skew = skew - skew.conj().T

    for terminal in (
        objective.scalar(1.3),
        objective.diagonal(np.linspace(0.0, 2.0, dimension)),
        objective.generated(skew),
        objective.random_terminal(generator),
    ):
        np.testing.assert_allclose(
            terminal.conj().T @ terminal,
            np.eye(dimension),
            atol=1.0e-13,
        )
        assert np.isfinite(objective.evaluate(terminal).wrong_arc_weight)


def test_random_terminals_are_haar_not_qr_sign_biased() -> None:
    """A biased draw would concentrate the diagonal phases near zero."""

    objective = build_closure_objective(
        _free_terminal_measure(),
        Sector.HOLE,
        n_conserved=2,
    )
    generator = np.random.default_rng(21)
    phases = np.concatenate(
        [np.angle(np.diagonal(objective.random_terminal(generator))) for _ in range(200)]
    )
    # Haar diagonal phases are uniform; the raw LAPACK convention piles them on one side.
    assert 0.35 < np.mean(phases > 0.0) < 0.65
    assert np.mean(np.abs(phases) > np.pi / 2.0) > 0.3


def test_convenience_scalars_agree_with_full_evaluation() -> None:
    objective = build_closure_objective(
        _free_terminal_measure(),
        Sector.HOLE,
        n_conserved=2,
    )
    dimension = objective.terminal_dimension
    phases = np.linspace(0.2, 2.4, dimension)

    assert (
        objective.scalar_arc_weight(0.9)
        == objective.evaluate(objective.scalar(0.9)).wrong_arc_weight
    )
    assert (
        objective.diagonal_arc_weight(phases)
        == objective.evaluate(objective.diagonal(phases)).wrong_arc_weight
    )
    assert objective.weight_threshold > 0.0


def test_a_fixed_terminal_block_offers_no_closure_family() -> None:
    objective = build_closure_objective(
        _terminated_measure(),
        Sector.HOLE,
        n_conserved=2,
    )

    assert objective.terminal_is_fixed
    assert objective.terminal_dimension == 0
    assert objective.scalar_parameter_count == 0
    assert objective.unitary_parameter_count == 0
    # Evaluation still works; it simply has one answer.
    assert np.isfinite(objective.evaluate().wrong_arc_weight)
    for call in (
        lambda: objective.scalar(0.4),
        lambda: objective.diagonal([0.4]),
        lambda: objective.generated(np.zeros((1, 1))),
        lambda: objective.random_terminal(np.random.default_rng(0)),
    ):
        with pytest.raises(ValidationError, match="terminal block"):
            call()


def test_malformed_inputs_are_rejected() -> None:
    moments = _free_terminal_measure()

    with pytest.raises(ValidationError, match="sector must be"):
        build_closure_objective(moments, "hole")
    with pytest.raises(ValidationError, match="backend must be"):
        build_closure_objective(moments, Sector.HOLE, backend="auto")
    with pytest.raises(ValidationError, match="tolerances must be"):
        build_closure_objective(moments, Sector.HOLE, tolerances=object())
    with pytest.raises(ValidationError, match="n_conserved cannot"):
        build_closure_objective(moments, Sector.HOLE, n_conserved=99)
    with pytest.raises(ValidationError, match="nonnegative integer"):
        build_closure_objective(moments, Sector.HOLE, n_conserved=-1)
    with pytest.raises(ValidationError, match="minimum_node_distance"):
        build_closure_objective(moments, Sector.HOLE, minimum_node_distance=0.0)

    objective = build_closure_objective(moments, Sector.HOLE, n_conserved=2)
    dimension = objective.terminal_dimension
    with pytest.raises(ValidationError, match="phases must have"):
        objective.diagonal(np.zeros(dimension + 1))
    with pytest.raises(ValidationError, match="phase must be finite"):
        objective.scalar(np.inf)
    with pytest.raises(ValidationError, match="generator must have"):
        objective.generated(np.zeros((dimension + 1, dimension + 1)))
    with pytest.raises(ValidationError, match="skew-Hermitian"):
        objective.generated(np.eye(dimension, dtype=np.complex128))
    with pytest.raises(ValidationError, match="candidate must be"):
        objective.ranking_key(object())
    with pytest.raises(ValidationError, match="numpy.random"):
        objective.random_terminal(0)


def test_objective_carries_its_construction_settings() -> None:
    objective = build_closure_objective(
        _free_terminal_measure(),
        Sector.PARTICLE,
        n_conserved=3,
        backend="block-cmv",
        minimum_node_distance=1.0e-9,
    )

    assert isinstance(objective, ClosureObjective)
    assert objective.sector is Sector.PARTICLE
    assert objective.fitted_order == 3
    assert objective.backend == "block-cmv"
    assert objective.minimum_distance == pytest.approx(1.0e-9)
    assert objective.tolerances is DEFAULT_TOLERANCES
    # Withheld orders exist only when the fit stops short of the supplied data.
    assert objective.evaluate().withheld_moment_residuals.size == 1


# The Gram rank cut.


def test_the_cut_keeps_the_eigenvalues_above_the_floor():
    values = np.asarray([-1e-15, 2e-13, 5e-13, 3e-12, 8e-12, 0.2, 1.0, 3.0])
    retained_count, floor = gram_rank_cut(values, 1e-12)
    assert retained_count == int(np.count_nonzero(values > 1e-12))
    epsilon_floor = float(np.finfo(np.float64).eps) * max(
        1.0, abs(float(values[0])), abs(float(values[-1]))
    )
    assert floor == max(1e-12, epsilon_floor)
