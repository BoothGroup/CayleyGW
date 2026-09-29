"""Every example runs and prints what its docstring promises.

Each script is loaded with ``runpy`` under a name other than ``__main__``, its
controls are overridden where a test needs a smaller or pinned setting, and its
``main`` is called. The assertions read the printed output, so an example that
stops describing its result fails here.
"""

from __future__ import annotations

from pathlib import Path
import re
import runpy

import pytest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
pytestmark = pytest.mark.pyscf


def run_example(name: str, capsys, overrides: dict | None = None):
    """Load one example, override its controls and run its ``main``."""

    namespace = runpy.run_path(str(EXAMPLES / name), run_name="cayleygw_example")
    if overrides:
        namespace["main"].__globals__.update(overrides)
    namespace["main"]()
    return capsys.readouterr()


def _value_after(output: str, label: str) -> float:
    """Return the first number printed after ``label``."""

    match = re.search(re.escape(label) + r"\s*(-?\d+\.\d+)", output)
    assert match, f"{label!r} not found in output"
    return float(match.group(1))


def test_first_calculation_prints_frontier_and_log(capsys) -> None:
    captured = run_example("01_first_calculation.py", capsys)
    out = captured.out
    assert "Results: H2O/cc-pvdz G0W0@RHF with Cayley moments" in out
    assert abs(_value_after(out, "reference HOMO-LUMO gap") - 18.465) < 0.01
    assert abs(_value_after(out, "G0W0 correction to the gap") + 1.600) < 0.01
    err = captured.err
    assert "n_conserved" in err and "orbitals" in err
    assert "Moments C_0..C_4 converged at N_q = 256" in err
    assert "Upfolded Hamiltonian of dimension 216" in err
    assert re.search(r"Hole\s+block-cmv\s+96", err)
    assert abs(_value_after(err, "IP 0") - 12.166) < 0.01
    assert err.count("EA ") == 3
    assert abs(_value_after(err, "Quasiparticle gap:") - 16.865) < 0.01
    assert "HOMO and LUMO below are mean-field orbitals 4 and 5." in err
    assert err.count("total wall time") == 2
    assert "contour node solves: " not in err, "stage lines belong to the log file"


def test_log_styles_differ_only_in_the_log(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    captured = run_example("11_log_styles.py", capsys)
    out = captured.out
    assert "in the order their logs appear above: 'standard', 'bare'" in out
    assert "log file, holding every run: log_styles.log" in out
    err = captured.err
    ips = re.findall(r"IP 0\s+(\d+\.\d+)", err)
    assert len(ips) == 2 and ips[0] == ips[1]
    assert abs(float(ips[0]) - 12.166) < 0.01
    # Both styles report the selected N_q and the results ...
    assert err.count("converged at N_q = 256") == 2
    assert err.count("Upfolded Hamiltonian of dimension 216 built in") == 2
    assert err.count("Quasiparticle energies") == 2
    # ... and only the standard one the doubling, 128 -> 256, the ellipse and the stage times.
    assert err.count("error/tolerance") == 1
    assert err.count("Ellipse centre") == 1
    assert err.count("total wall time") == 2
    assert "contour node solves: " not in err, "stage lines belong to the file"
    # The file holds both runs in full, stage lines included.
    log = (tmp_path / "log_styles.log").read_text()
    assert log.count("log opened") == 2
    assert log.count("build_cayley_moments: stage times") == 2
    # The ladder solves 65 trapezoid nodes at 128, then only the 64 new pairs of 256.
    assert "contour node solves: nodes=128, solved=65" in log
    assert "automatic N_q: N_q=256" in log and "contour node solves: nodes=128, solved=64" in log
    assert log.count("IP 2: ") == 2 and "quasiparticle gap" in log


def test_the_parts_reuse_one_build_and_match_the_kernel(capsys) -> None:
    captured = run_example("02_the_parts.py", capsys)
    out = captured.out
    assert "one moment build realized at every lower order" in out
    # One build serves the seven orders of the four calls; the kernel's check builds once more.
    assert captured.err.count("Moments C_0..C_8 converged at N_q") == 2
    assert captured.err.count("Upfolded Hamiltonian of dimension") == 8
    kernel = re.search(
        r"gw\.kernel\(7\) against the four calls at n=7: IP ([+-]\d+\.\d+) meV, EA ([+-]\d+\.\d+) meV",
        out,
    )
    assert kernel and float(kernel.group(1)) == 0.0 and float(kernel.group(2)) == 0.0
    assert "  1  C_0..C_1 + C_2" in out
    assert "  7  C_0..C_7 + C_8" in out
    assert "deviation from the n=7 result (meV):" in out
    deviations = {
        int(n): float(ip)
        for n, ip in re.findall(r"n=(\d): IP\s+(\d+\.\d+)", out)
    }
    assert deviations[1] > 1.0
    assert deviations[3] < 0.05
    assert deviations[5] < 0.01


def test_omega_p_table_has_ea_first_and_a_rule_between_scales(capsys) -> None:
    out = run_example("03_omega_p.py", capsys).out
    assert "Results: CO/cc-pvdz G0W0@RHF, the Cayley scale omega_p at low conserved order" in out
    assert "  reference: EA " in out
    header = next(line for line in out.splitlines() if line.startswith("  omega_p   n   N_q"))
    assert header.split()[3:] == ["EA", "IP", "IP(4)", "IP(5)"]
    assert "    0.125   1  " in out
    assert "    2.000   2  " in out
    # A rule opens each scale's rows, the first under the header.
    assert out.count("\n  " + "-" * (len(header) - 2) + "\n") == 5


def test_contour_quadrature_reports_grids_and_ellipse(capsys) -> None:
    captured = run_example("04_contour_quadrature.py", capsys)
    out = captured.out
    assert "fixed contour grids" in out
    assert "    64  refused, increase N_q" in out
    assert "   256  " in out
    assert "  auto  " in out
    # The doubling history, the nodes solved and the refusal are in the log.
    err = captured.err
    assert "error/tolerance" in err and "converged" in err
    assert "256 (128 solved)" in err
    assert "build_upfolded_hamiltonian failed after" in err
    assert "spectral_bound='frobenius':" in out
    assert "spectral_bound='spectral':" in out
    assert "RPA excitation energies enclosed in [" in out
    assert "nearest excluded singularity:" in out
    assert "omega_p" not in out.split("Results:")[1], "omega_p is example 03"


def test_screening_and_reference_table(capsys) -> None:
    captured = run_example("05_screening_and_reference.py", capsys)
    out = captured.out
    # N2's HOMO and LUMO are mean-field orbitals 6 and 7 in every run, the frozen core included.
    assert captured.err.count("mean-field orbitals 6 and 7") == 4
    assert "mean-field orbitals 4 and 5" not in captured.err
    rows = {
        label.strip(): (int(n_mo), float(ip))
        for label, n_mo, ip in re.findall(
            r"^  (RHF, RPA screening|RHF, TDA screening|PBE, RPA screening|RHF, RPA, frozen=2)\s+(\d+)\s+\d+\s+(-?\d+\.\d+)",
            out,
            flags=re.M,
        )
    }
    assert set(rows) == {"RHF, RPA screening", "RHF, TDA screening", "PBE, RPA screening", "RHF, RPA, frozen=2"}
    assert rows["RHF, RPA screening"][0] == 28
    assert rows["RHF, RPA, frozen=2"][0] == 26
    assert abs(rows["RHF, RPA screening"][1] - rows["RHF, RPA, frozen=2"][1]) < 0.05
    assert abs(rows["RHF, RPA screening"][1] - rows["RHF, TDA screening"][1]) > 0.1
    assert abs(rows["RHF, RPA screening"][1] - rows["PBE, RPA screening"][1]) > 0.5
    assert "static correction Sigma_x^HF - v_xc" in out
    assert "RHF  Frobenius norm  0.00e+00 Ha" in out
    assert "PBE  Frobenius norm" in out


def test_spectral_function_lists_satellites_and_plots(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    out = run_example("06_spectral_function.py", capsys).out
    assert "CO/cc-pvdz G0W0@PBE" in out
    assert "HOMO-3 (orbital 3" in out
    assert "HOMO (orbital 6" in out
    assert "LUMO (orbital 7" in out
    assert out.count("quasiparticle at") == 3
    assert "satellite at" in out
    assert "spectral function on 2001 points in [-45.0, 15.0] eV" in out
    assert "plot written to" in out
    assert (tmp_path / "co_spectral_function.png").exists()


def test_closure_and_realization_compares_backends(capsys) -> None:
    captured = run_example("07_closure_and_realization.py", capsys)
    out = captured.out
    assert "terminal closure:" in out
    assert "  restricted      16" in out
    assert "  scan            64" in out
    for realization in ("block-cmv", "toeplitz", "auto"):
        assert re.search(
            rf"^  {realization}\s+(block-cmv|toeplitz)\s+(block-cmv|toeplitz)\s+24 / 24\s.*passed$",
            out,
            flags=re.M,
        )
    assert "sector diagnostics of the default realization:" in out
    assert out.count("terminal dimension 24, discarded pole weight") == 2
    err = captured.err
    assert err.count("largest relative error") == 3 and "failed" not in err
    assert "clean" in err, "the marginal flags are in the sector table"


def test_exact_reference_errors_vanish_with_order(capsys) -> None:
    captured = run_example("08_exact_reference.py", capsys)
    out = captured.out
    assert "Results: LiH/cc-pvdz G0W0@RHF against the exact reference" in out
    assert "34 RPA excitations, 68 hole and 578 particle self-energy poles" in out
    assert captured.err.count("Quasiparticle energies") == 7, "the exact one and one per order"
    assert "exact upfolded dimension 665" in out
    assert "diagonal quasiparticle equation against the full solution:" in out
    assert "finite conserved order against the exact reference" in out
    rows = {}
    for line in out.splitlines():
        match = re.match(r"^  (\d)\s+(\d+)\s+(\d+)\s+(\S+)((?:\s+-?\d+\.\d+){4})$", line)
        if match:
            rows[int(match.group(1))] = (
                float(match.group(4)),
                [abs(float(value)) for value in match.group(5).split()],
            )
    assert set(rows) == {1, 2, 3, 4, 5, 7}
    for quadrature_error, _ in rows.values():
        assert quadrature_error < 1.0e-8
    assert rows[1][1][0] > 0.1
    assert max(rows[5][1]) < 0.01
    assert max(rows[7][1]) < 0.01


def test_large_molecule_template_on_a_small_molecule(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    captured = run_example(
        "09_large_molecule.py",
        capsys,
        {
            "ATOM": "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
            "BASIS": "cc-pvdz",
            "FROZEN": 1,
            "N_CONSERVED": 2,
            "ESTIMATE_N_Q": 128,
            "NATIVE_THREADS": 1,
            "N_WORKERS": 1,
            "CONTOUR_ORBITAL_BLOCK_SIZE": 2,
        },
    )
    out = captured.out
    assert "3 atoms, 24 basis functions, 23 active orbitals," in out
    assert "memory estimate at N_q=128, block size 2:" in out
    assert "MiB  sum" in out
    assert re.search(r"^mean field: \d+\.\d s$", out, flags=re.M)
    err = captured.err
    assert "Moments C_0..C_3 converged at N_q = " in err
    assert "Upfolded Hamiltonian of dimension" in err
    assert re.search(r"IP 0\s+\d+\.\d+", err) and re.search(r"EA 0\s+-?\d+\.\d+", err)
    assert err.count("total wall time") == 2
    assert "build_cayley_moments: stage times" in (tmp_path / "large_molecule.log").read_text()


def test_large_molecule_dry_run_stops_at_the_estimate(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    out = run_example(
        "09_large_molecule.py",
        capsys,
        {
            "ATOM": "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
            "BASIS": "cc-pvdz",
            "FROZEN": 1,
            "DRY_RUN": True,
        },
    ).out
    assert "memory estimate at N_q=512, block size 4:" in out
    assert "MiB  sum" in out
    assert "contour moments:" not in out


def test_tolerances_spelled_out_reproduce_the_defaults(capsys) -> None:
    captured = run_example("12_tolerances.py", capsys)
    out = captured.out
    assert (
        "safeguards      Tolerances(rank_floor=1e-10, moment_conservation=1e-09, "
        "conservation_margin=2.0, arc_margin=100.0, positivity=1e-10)"
    ) in out
    assert "minimum_weight  0.1" in out
    assert _value_after(out, "largest IP or EA difference") == 0.0
    # The log carries the tuning settings of both calls.
    err = captured.err
    assert "n_q_tolerance" in err and err.count("terminal_phase_count") == 2


def test_pyscf_comparison_agrees_on_the_diagonal_equation(capsys) -> None:
    out = run_example("10_pyscf_comparison.py", capsys).out
    assert "H2O/cc-pvdz G0W0@PBE, 95 RPA excitations" in out
    assert "gw_exact" in out and "gw_cd" in out
    for label in ("HOMO-1", "HOMO ", "LUMO ", "LUMO+1"):
        assert label in out
    assert _value_after(out, "largest difference between gw_exact and the cayleygw diagonal equation:") < 0.01
    assert _value_after(out, "largest difference between the exact and the Cayley-moment full solution:") < 1.0
