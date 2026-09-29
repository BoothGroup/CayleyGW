"""Dispatch by index over workers, and the native-thread limit around it."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

from cayleygw.tools import parallel as parallel_module
from cayleygw.tools.parallel import (
    ambient_native_threads,
    evaluate_by_index,
    limited_native_threads,
)


def test_limit_is_installed_and_restores_the_previous_setting() -> None:
    before = ambient_native_threads()
    with limited_native_threads(1):
        assert ambient_native_threads() == 1
    assert ambient_native_threads() == before


def test_limit_is_restored_when_the_block_raises() -> None:
    """A failing node sweep must not leave the process single-threaded."""

    before = ambient_native_threads()
    with pytest.raises(RuntimeError, match="node sweep failed"):
        with limited_native_threads(1):
            raise RuntimeError("node sweep failed")
    assert ambient_native_threads() == before


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "2", None])
def test_limit_rejects_counts_that_are_not_positive_integers(count) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        with limited_native_threads(count):
            pass


def test_results_are_reproducible_at_a_fixed_native_thread_count() -> None:
    """Results repeat bit for bit at a fixed thread count; 1 and 4 threads differ by 7e-16."""

    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(256, 256)) + 1j * rng.normal(size=(256, 256))
    with limited_native_threads(1):
        repeats = [matrix @ matrix.conj().T for _ in range(4)]
    assert all(np.array_equal(repeats[0], value) for value in repeats)


def test_lowest_failing_index_is_reported_whatever_order_they_surface() -> None:
    """The blamed index must not depend on which worker got there first."""

    def failing(index: int) -> int:
        if index in (7, 3, 5):
            raise ValueError(f"index {index} rejected")
        return index * index

    for workers in (2, 4, 8):
        results, failures = evaluate_by_index(failing, range(12), n_workers=workers)
        # Work in flight still finishes, but nothing new is fed after a failure.
        assert failures[0][0] == 3
        assert [index for index, _ in failures] == sorted(index for index, _ in failures)
        assert 3 not in results
        assert len(results) + len(failures) <= 3 + 2 * workers + 1


def test_serial_evaluation_stops_at_the_first_failure() -> None:
    """One worker keeps the plain loop's fail-fast behaviour."""

    seen: list[int] = []

    def failing(index: int) -> int:
        seen.append(index)
        if index == 2:
            raise ValueError("rejected")
        return index

    results, failures = evaluate_by_index(failing, range(9), n_workers=1)
    assert seen == [0, 1, 2]
    assert [index for index, _ in failures] == [2]
    assert sorted(results) == [0, 1]


def test_results_are_keyed_by_index_not_completion_order() -> None:
    """A slow early index must still land in its own slot."""

    import time

    def uneven(index: int) -> int:
        time.sleep(0.01 if index < 3 else 0.0)
        return index * 10

    results, failures = evaluate_by_index(uneven, range(16), n_workers=8)
    assert not failures
    assert [results[index] for index in range(16)] == [index * 10 for index in range(16)]


@pytest.mark.parametrize("n_workers", [1, 2, 4])
def test_a_sink_frees_each_evaluation_before_waiting_on_the_next(
    n_workers: int,
) -> None:
    """With a sink, the pool and not the sweep sets how many evaluations are live."""

    import gc
    import weakref

    class Evaluation:
        __slots__ = ("index", "__weakref__")

        def __init__(self, index: int) -> None:
            self.index = index

    def peak_live(count: int) -> int:
        made: list[weakref.ref] = []
        seen: list[int] = []
        peak = 0

        def make(index: int) -> Evaluation:
            evaluation = Evaluation(index)
            made.append(weakref.ref(evaluation))
            return evaluation

        def sink(index: int, value: Evaluation) -> None:
            nonlocal peak
            seen.append(index)
            gc.collect()
            peak = max(peak, sum(1 for ref in made if ref() is not None))

        results, failures = evaluate_by_index(make, range(count), n_workers=n_workers, sink=sink)
        assert not failures
        assert not results, "a sink must not also retain values in the mapping"
        assert sorted(seen) == list(range(count))
        return peak

    small, large = peak_live(64), peak_live(512)
    assert large == small, (
        f"live evaluations grew from {small} to {large} when the sweep grew "
        "eightfold, so the sweep is being retained rather than consumed as it goes"
    )
    assert large <= 4 * n_workers, f"{large} evaluations live at once for {n_workers} worker(s)"


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_dispatch_rejects_invalid_native_thread_counts(value) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_by_index(lambda index: index, range(3), native_threads=value)


def test_native_threads_are_bounded_on_the_serial_path_too(monkeypatch) -> None:
    """One worker is pinned as several are, so the worker count cannot change the answer."""

    requested: list[int] = []
    real = parallel_module.limited_native_threads

    @contextmanager
    def recording(count: int):
        requested.append(count)
        with real(count) as limit:
            yield limit

    monkeypatch.setattr(parallel_module, "limited_native_threads", recording)
    for workers in (1, 2, 4):
        requested.clear()
        results, failures = evaluate_by_index(
            lambda index: index, range(4), n_workers=workers, native_threads=1
        )
        assert not failures
        assert sorted(results) == [0, 1, 2, 3]
        assert requested == [1], f"{workers} workers entered the limiter {len(requested)} times"


def test_nested_regions_at_one_count_share_a_single_installation(monkeypatch) -> None:
    """A nested region at the same count joins the outer one; a rebuild can deadlock."""

    installed: list[int] = []
    released: list[int] = []

    def fake_install(count: int):
        installed.append(count)
        return lambda: released.append(count)

    monkeypatch.setattr(parallel_module, "_install_limit", fake_install)
    with limited_native_threads(2):
        with limited_native_threads(2):
            assert installed == [2]
            assert released == []
        assert released == []
        # A different count inside installs its own and hands back.
        with limited_native_threads(3):
            assert installed == [2, 3]
        assert released == [3]
    assert released == [3, 2]


def test_a_sibling_thread_keeps_the_limit_until_the_last_region_leaves(
    monkeypatch,
) -> None:
    """Two threads pinned at one count must not restore under each other."""

    import threading

    installed: list[int] = []
    released: list[int] = []

    def fake_install(count: int):
        installed.append(count)
        return lambda: released.append(count)

    monkeypatch.setattr(parallel_module, "_install_limit", fake_install)
    first_in = threading.Event()
    second_done = threading.Event()

    def holder() -> None:
        with limited_native_threads(1):
            first_in.set()
            second_done.wait(timeout=10.0)

    thread = threading.Thread(target=holder)
    thread.start()
    assert first_in.wait(timeout=10.0)
    with limited_native_threads(1):
        assert installed == [1]
    # The holder thread is still inside its region, so nothing is released yet.
    assert released == []
    second_done.set()
    thread.join(timeout=10.0)
    assert released == [1]


def test_the_thread_controller_is_built_once_per_import_state() -> None:
    """The library walk happens once, not once per region."""

    pytest.importorskip("threadpoolctl")
    first = parallel_module._threadpool_controller()
    second = parallel_module._threadpool_controller()
    assert first is second


def test_the_dyson_solve_inherits_the_count_the_cell_was_given() -> None:
    """The Hamiltonian carries a thread count for the Dyson solve, ``None`` if not given."""

    import numpy as np
    from conftest import sector_source

    from cayleygw import Sector, UpfoldedDysonHamiltonian, diagonalize_upfolded

    reference = np.eye(2, dtype=complex)
    hole = sector_source(Sector.HOLE, [-1.0], [[0.1], [0.0]])
    particle = sector_source(Sector.PARTICLE, [1.0], [[0.0], [0.1]])
    plain = UpfoldedDysonHamiltonian(reference, hole, particle)
    assert plain.native_threads is None

    declared = UpfoldedDysonHamiltonian(reference, hole, particle, native_threads=4)
    assert declared.native_threads == 4
    # The count changes the cost, not the energies.
    np.testing.assert_allclose(
        diagonalize_upfolded(plain, verbose=0).energies,
        diagonalize_upfolded(declared, verbose=0).energies,
    )


@pytest.mark.parametrize("n_workers", [1, 4])
def test_on_complete_reports_every_index_exactly_once(n_workers) -> None:
    """A progress count needs a per-item hook, since one call runs a whole batch."""

    seen: list[int] = []
    results, failures = evaluate_by_index(
        lambda index: index * 2,
        range(8),
        n_workers=n_workers,
        on_complete=seen.append,
    )
    assert not failures
    assert results == {index: index * 2 for index in range(8)}
    assert sorted(seen) == list(range(8))


def test_on_complete_fires_alongside_a_sink() -> None:
    """The sink consumes the value; the hook only observes that it arrived."""

    stored: dict[int, int] = {}
    seen: list[int] = []
    results, failures = evaluate_by_index(
        lambda index: index,
        range(4),
        n_workers=2,
        sink=stored.__setitem__,
        on_complete=seen.append,
    )
    assert not failures
    assert results == {}
    assert stored == {index: index for index in range(4)}
    assert sorted(seen) == list(range(4))


def test_on_complete_does_not_fire_for_a_failing_index() -> None:
    def evaluate(index: int) -> int:
        if index == 2:
            raise ValueError("index two is prohibited")
        return index

    seen: list[int] = []
    _, failures = evaluate_by_index(evaluate, range(4), n_workers=1, on_complete=seen.append)
    assert [index for index, _ in failures] == [2]
    assert 2 not in seen


def test_auto_threads_follow_the_environment_then_the_cores(monkeypatch) -> None:
    from cayleygw.tools.parallel import resolve_native_threads

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(12)), raising=False)
    assert resolve_native_threads("auto") == 12
    assert resolve_native_threads("auto", 4) == 3
    assert resolve_native_threads("auto", 24) == 1
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "6")
    assert resolve_native_threads("auto", 2) == 3
    monkeypatch.setenv("OMP_NUM_THREADS", "4,2")  # nested counts are not one number
    assert resolve_native_threads("auto") == 6
    assert resolve_native_threads(5, 4) == 5


@pytest.mark.parametrize("threads, workers", [("many", 1), (0, 1), (2, 0)])
def test_thread_counts_are_validated(threads, workers) -> None:
    from cayleygw import ValidationError
    from cayleygw.tools.parallel import resolve_native_threads

    with pytest.raises(ValidationError):
        resolve_native_threads(threads, workers)
