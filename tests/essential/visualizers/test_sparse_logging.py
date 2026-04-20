import numpy as np
import pytest

# This is the underlying module where the internal global state lives (_STATE).
# We import it so we can inject a fake backend into that internal state.
import vla_foundry.visualizers.visualizer as vz_mod

# To run while in vla_foundry/visualizers directory:
# VISUALIZER=disabled uv run pytest ../../tests/essential/visualizers/test_sparse_logging.py -q
# This is the public facade module users call in real code:
from vla_foundry.visualizers import visualizer as vz
from vla_foundry.visualizers.visualizer import _pop_every_n


class CountingBackend:
    """
    Fake backend for unit tests.

    Why a fake backend?
    - In production, the visualizer forwards logs to wandb or rerun.
    - In tests, we don't want network / UI / external dependencies.
    - So we install a small object that has the same methods (log_scalar, log_images)
      and just records calls.

    What do we "count"?
    - We count which log calls got forwarded by the visualizer.
    - Sparse logging is exactly about "forward vs skip", so this directly tests it.
    """

    def __init__(self) -> None:
        # Store the calls we receive, so tests can assert on them later.
        # Each entry is (path, value) or (path, shape).
        self.scalars: list[tuple[str, float]] = []
        self.images: list[tuple[str, tuple[int, ...] | None]] = []

    def init(self, *args, **kwargs) -> None:
        # Some backends have init; our tests don't need to do anything here.
        pass

    def log_scalar(self, path: str, value, **kwargs) -> None:
        """
        This mimics the backend API: accept a scalar log.

        **kwargs means "accept extra keyword arguments".
        Visualizer forwards optional parameters to the backend, so our fake backend
        should accept them too, even if it ignores them.

        In Python, **kwargs collects any extra keyword args into a dict.
        """
        self.scalars.append((path, value))

    def log_images(self, path: str, images, **kwargs) -> None:
        """
        Same idea for images: record that we were called.

        We store the image shape (if it exists) just as a lightweight indicator
        that an image-like object was passed in.
        """
        shape = getattr(images, "shape", None)
        self.images.append((path, shape))

    def flush(self) -> None:
        # No-op for tests
        pass

    def shutdown(self) -> None:
        # No-op for tests
        pass


@pytest.fixture
def counting_backend(monkeypatch):
    """
    Pytest fixture that installs the CountingBackend as the active backend.

    A fixture is a "setup helper" that runs before tests that request it,
    and then can clean up afterward.

    The fixture uses monkeypatch:
    - monkeypatch is a built-in pytest fixture used to temporarily modify global state
      (env vars, attributes, dict entries) for the duration of a test.
    - here we use monkeypatch.setenv() to ensure VISUALIZER won't try to select wandb/rerun.
      pytest docs: monkeypatch can set environment variables safely.
    """
    # Make sure no test accidentally tries to use a real backend selected by env var.
    monkeypatch.setenv("VISUALIZER", "disabled")

    # Grab the visualizer module's internal singleton state.
    state = vz_mod._STATE

    # Snapshot state to avoid cross-test contamination.
    # This is important because _STATE is global/singleton.
    old_backend = getattr(state, "backend", None)
    old_initialized = getattr(state, "initialized", None)
    old_enabled = getattr(state, "enabled", None)
    old_rank_prefix = getattr(state, "rank_prefix", None)
    old_counters = dict(getattr(state, "counters", {}))

    backend = CountingBackend()

    # Force the visualizer into "initialized + enabled" with our fake backend.
    #
    # Why force initialized=True?
    # - In normal usage you call vz.init(...), which sets up a backend.
    # - For a unit test we skip the real init and inject our backend directly,
    #   so we mark it initialized to allow log_* calls to go through the normal path.
    state.backend = backend
    state.initialized = True
    state.enabled = True

    # Some installations add rank prefixes; for unit tests we don't want surprises.
    if hasattr(state, "rank_prefix"):
        state.rank_prefix = ""

    # IMPORTANT: clear counters, so call-counting starts fresh per test.
    # Sparse logging depends on internal counters, so we want deterministic results.
    if hasattr(state, "counters"):
        state.counters.clear()

    # "yield" in a pytest fixture means:
    # - everything before yield is setup
    # - the yielded value is what the test receives
    # - everything after yield is teardown/cleanup
    yield backend

    # Teardown: restore the previous global state so other tests aren't affected.
    state.backend = old_backend
    if old_initialized is not None:
        state.initialized = old_initialized
    if old_enabled is not None:
        state.enabled = old_enabled
    if old_rank_prefix is not None and hasattr(state, "rank_prefix"):
        state.rank_prefix = old_rank_prefix
    if hasattr(state, "counters"):
        state.counters.clear()
        state.counters.update(old_counters)


def test_every_n_logs_first_and_then_every_nth_call_scalar(counting_backend):
    """
    What this test checks:

    - We call vz.log_scalar(...) 15 times with n=5.
    - Sparse logging rule: log on call #1, then #5, #10, #15.
    - The CountingBackend records only the forwarded calls.
    - So we expect exactly 4 values: 1, 5, 10, 15.
    """
    for i in range(1, 16):
        vz.log_scalar("every_n/n5", i, n=5)

    assert [v for (p, v) in counting_backend.scalars if p == "every_n/n5"] == [1, 5, 10, 15]


def test_every_n_logs_first_and_then_every_nth_call_images(counting_backend):
    """
    Same test idea as scalars, but for images.

    We call log_images 15 times with n=5.
    We expect forwarded calls on: 1, 5, 10, 15 => total 4 forwarded logs.
    """
    for _i in range(1, 16):
        vz.log_images("every_n/img_n5", np.zeros((8, 8, 3)), n=5)

    assert len([p for (p, _shape) in counting_backend.images if p == "every_n/img_n5"]) == 4


def test_counters_are_independent_per_kind_and_path(counting_backend):
    """
    What this test checks:

    - We log to the SAME path name "same_name" using TWO different log methods:
        - log_scalar
        - log_images
    - We want call counters to be independent per (kind, path).
      That means scalar calls should not affect image sampling and vice versa.

    With n=3 and 9 calls:
      - expected forwarded calls happen on call #1, #3, #6, #9
      - so scalar should record values [1, 3, 6, 9]
      - and images should record 4 calls as well
    """
    for i in range(1, 10):
        vz.log_scalar("same_name", i, n=3)
        vz.log_images("same_name", np.zeros((8, 8, 3)), n=3)

    assert [v for (p, v) in counting_backend.scalars if p == "same_name"] == [1, 3, 6, 9]
    assert len([p for (p, _shape) in counting_backend.images if p == "same_name"]) == 4


def test_disable_blocks_logging(counting_backend):
    """
    What this test checks:

    - When vz.disable() is active, ALL log calls should be no-ops
      (backend should see zero calls).
    - When we re-enable, logging should resume.

    We use n=1 here to remove sampling from the equation:
    n=1 means "log every call" (if enabled), so any missing logs are due to disable().
    """
    vz.disable()
    for i in range(1, 10):
        vz.log_scalar("toggle/x", i, n=1)
        vz.log_images("toggle/img", np.zeros((8, 8, 3)), n=1)

    assert counting_backend.scalars == []
    assert counting_backend.images == []

    vz.enable()
    vz.log_scalar("toggle/x", 123, n=1)
    assert [v for (p, v) in counting_backend.scalars if p == "toggle/x"] == [123]


# ---------------------------------------------------------------------------
# Tests for _pop_every_n() edge cases (validation and aliases)
# ---------------------------------------------------------------------------


def test_pop_every_n_supports_every_n_and_pops_key():
    kwargs = {"every_n": 3, "other": "x"}
    assert _pop_every_n(kwargs) == 3
    assert "every_n" not in kwargs
    assert kwargs == {"other": "x"}


def test_pop_every_n_supports_n_alias_and_pops_key():
    kwargs = {"n": 4}
    assert _pop_every_n(kwargs) == 4
    assert kwargs == {}


def test_pop_every_n_none_when_missing():
    kwargs = {"other": 123}
    assert _pop_every_n(kwargs) is None
    assert kwargs == {"other": 123}


def test_pop_every_n_rejects_bool():
    with pytest.raises(TypeError):
        _pop_every_n({"n": True})


@pytest.mark.parametrize("bad_val", ["nope", object()])
def test_pop_every_n_rejects_non_coercible_values(bad_val):
    with pytest.raises(TypeError):
        _pop_every_n({"every_n": bad_val})


@pytest.mark.parametrize("bad_val", [0, -1, "-2"])
def test_pop_every_n_requires_positive_int(bad_val):
    with pytest.raises(ValueError):
        _pop_every_n({"every_n": bad_val})


def test_every_n_alias_works(counting_backend):
    """Test that `every_n=` works as an alias for `n=`."""
    for i in range(1, 11):
        vz.log_scalar("alias_test", i, every_n=5)

    # With every_n=5: log on call #1, #5, #10 => values [1, 5, 10]
    assert [v for (p, v) in counting_backend.scalars if p == "alias_test"] == [1, 5, 10]


def test_every_n_bool_rejected(counting_backend):
    """Test that n=True or n=False raises TypeError (bool is subclass of int)."""
    with pytest.raises(TypeError, match="n/every_n must be an integer >= 1"):
        vz.log_scalar("bool_test", 1.0, n=True)

    with pytest.raises(TypeError, match="n/every_n must be an integer >= 1"):
        vz.log_scalar("bool_test", 1.0, n=False)


def test_every_n_non_int_rejected(counting_backend):
    """Test that non-integer values for n raise TypeError."""
    with pytest.raises(TypeError, match="n/every_n must be an integer >= 1"):
        vz.log_scalar("non_int_test", 1.0, n="not_a_number")

    with pytest.raises(TypeError, match="n/every_n must be an integer >= 1"):
        vz.log_scalar("non_int_test", 1.0, n=[1, 2, 3])


def test_every_n_zero_or_negative_rejected(counting_backend):
    """Test that n<=0 raises ValueError."""
    with pytest.raises(ValueError, match="n/every_n must be an integer >= 1"):
        vz.log_scalar("zero_test", 1.0, n=0)

    with pytest.raises(ValueError, match="n/every_n must be an integer >= 1"):
        vz.log_scalar("negative_test", 1.0, n=-5)


def test_disable_before_init_persists(monkeypatch):
    """Test that calling disable() before init() persists through auto-init.

    This addresses the Copilot review observation that pre-init disable()
    could be overwritten when the first log call triggers auto-init.
    """
    monkeypatch.setenv("VISUALIZER", "disabled")

    # Reset state to simulate fresh import
    state = vz_mod._STATE
    old_backend = state.backend
    old_initialized = state.initialized
    old_enabled = state.enabled
    old_user_disabled = state.user_disabled

    try:
        state.backend = None
        state.initialized = False
        state.enabled = False
        state.user_disabled = False

        # Call disable() BEFORE init
        vz.disable()

        # Verify user_disabled is set
        assert state.user_disabled is True

        # Now call init() - it should respect the pre-set disabled state
        vz.init(backend="disabled")

        # enabled should still be False because user called disable() first
        assert state.enabled is False
    finally:
        # Restore state
        state.backend = old_backend
        state.initialized = old_initialized
        state.enabled = old_enabled
        state.user_disabled = old_user_disabled
