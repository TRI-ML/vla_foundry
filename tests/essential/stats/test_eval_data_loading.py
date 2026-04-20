"""Tests for vla_foundry.eval.data_loading using dashboard_fixtures/.

Each subdirectory under dashboard_fixtures/ is a self-contained rollout root
with the canonical layout: {model}/{Task}/rollouts/{timestamp}/results.json

load_episodes enforces the following rules:
- Layout: Accepts {model}/{Task}/rollouts/{timestamp}/results.json (5 parts) or
  {eval_set}/{model}/{Task}/rollouts/{timestamp}/results.json (6 parts).
  Anything else raises ValueError.
- max_sample_size_per_model: Required in every results.json — missing field
  raises ValueError.
- Conflicting budgets: Raises ValueError if different max_sample_size_per_model
  values appear across files (globally) or within the same rollouts directory.
- Overlapping episodes: When multiple timestamped runs exist under the same
  model/Task/rollouts/ with overlapping (skill_type, scenario_index) pairs,
  raises ValueError. Non-overlapping episodes across files are combined.
- Duplicate episodes: Hard error if the same (task, model, demo_id) appears
  after deduplication.
- Budget exceeded: Raises ValueError if any (task, model) pair has more
  completed episodes than max_sample_size_per_model.
"""

from pathlib import Path

import pytest

from vla_foundry.eval.data_loading import aggregate_episodes, collect_scenario_indices, load_episodes
from vla_foundry.eval.run_evaluation import check_no_overlapping_results, parse_episode_range

FIXTURES = Path(__file__).resolve().parent.parent / "test_assets" / "dashboard_fixtures"


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestMixedPendingCrashedSuccess:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.eps, self.pending, self.crashed, self.mss = load_episodes(FIXTURES / "mixed_pending_crashed_success")

    def test_completed_episodes(self):
        # 5 evaluations: 2 success, 1 failure, 1 pending, 1 crashed
        # load_episodes returns only completed (success/failure) episodes
        assert len(self.eps) == 3

    def test_pending_tracked(self):
        assert sum(v for v in self.pending.values()) > 0

    def test_crashed_tracked(self):
        assert sum(v for v in self.crashed.values()) > 0

    def test_max_sample_size(self):
        assert self.mss == 50

    def test_aggregate(self):
        stats = aggregate_episodes(self.eps, pending_by=self.pending, crashed_by=self.crashed)
        assert len(stats) == 1
        s = stats[0]
        assert s["successes"] == 2
        assert s["total"] == 3
        assert s["pending"] == 1
        assert s["crashed"] == 1


class TestCrashedEpisodes:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.eps, self.pending, self.crashed, self.mss = load_episodes(FIXTURES / "crashed_episodes")

    def test_only_completed_returned(self):
        # 3 evaluations: 2 crashed, 1 success
        assert len(self.eps) == 1

    def test_crashed_count(self):
        assert self.crashed[("TaskA", "model_a")] == 2


class TestPendingEpisodes:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.eps, self.pending, self.crashed, self.mss = load_episodes(FIXTURES / "pending_episodes")

    def test_no_completed_episodes(self):
        assert len(self.eps) == 0

    def test_pending_count(self):
        assert self.pending[("TaskA", "model_a")] == 3


class TestEmptyEvaluations:
    def test_loads_with_no_episodes(self):
        eps, pending, crashed, mss = load_episodes(FIXTURES / "empty_evaluations")
        assert len(eps) == 0
        assert mss == 50


# ---------------------------------------------------------------------------
# Nested (6-part) layout: {eval_set}/{model}/{Task}/rollouts/{timestamp}/results.json
# ---------------------------------------------------------------------------


class TestNestedEvalSets:
    """Nested 6-part layout with the same model name under two eval sets.

    Fixture layout:
      nested_eval_sets/
      ├── OSS/model_a/TaskA/rollouts/{ts}/results.json  (2 success)
      ├── OSS/model_a/TaskB/rollouts/{ts}/results.json  (1 success, 1 failure)
      └── CS/model_a/TaskA/rollouts/{ts}/results.json   (2 failure)

    The dashboard must treat OSS/model_a and CS/model_a as distinct models
    so eval-set results never collide during aggregation.
    """

    @pytest.fixture(autouse=True)
    def _load(self):
        self.eps, self.pending, self.crashed, self.mss = load_episodes(FIXTURES / "nested_eval_sets")

    def test_eval_set_prefixed_model_name(self):
        models = {ep["model"] for ep in self.eps}
        assert models == {"OSS/model_a", "CS/model_a"}

    def test_same_model_in_different_eval_sets_are_distinct(self):
        # Both eval sets contain a model literally named "model_a" running
        # the same skill task_a. They must NOT be merged.
        by_key = {(ep["task"], ep["model"]): ep["success"] for ep in self.eps}
        # OSS/model_a succeeds on task_a; CS/model_a fails on task_a.
        # If they were merged, success rates would average — instead we need
        # two separate (task, model) entries. skill_type "task_a" is normalized
        # to "TaskA" by _normalize_skill_type.
        assert ("TaskA", "OSS/model_a") in by_key
        assert ("TaskA", "CS/model_a") in by_key

    def test_per_model_episode_counts(self):
        from collections import Counter

        counts = Counter((ep["task"], ep["model"]) for ep in self.eps)
        assert counts[("TaskA", "OSS/model_a")] == 2
        assert counts[("TaskB", "OSS/model_a")] == 2
        assert counts[("TaskA", "CS/model_a")] == 2

    def test_aggregate_keeps_eval_sets_separate(self):
        stats = aggregate_episodes(self.eps, pending_by=self.pending, crashed_by=self.crashed)
        by_key = {(s["task"], s["model"]): s for s in stats}
        # OSS/model_a: TaskA is 2/2 success, TaskB is 1/2
        assert by_key[("TaskA", "OSS/model_a")]["successes"] == 2
        assert by_key[("TaskA", "OSS/model_a")]["total"] == 2
        assert by_key[("TaskB", "OSS/model_a")]["successes"] == 1
        # CS/model_a: TaskA is 0/2 — must NOT be pooled with OSS/model_a's 2/2
        assert by_key[("TaskA", "CS/model_a")]["successes"] == 0
        assert by_key[("TaskA", "CS/model_a")]["total"] == 2


# ---------------------------------------------------------------------------
# Error tests
# ---------------------------------------------------------------------------


class TestNoMaxSampleSize:
    def test_raises(self):
        with pytest.raises(ValueError, match="max_sample_size_per_model"):
            load_episodes(FIXTURES / "no_max_sample_size")


class TestConflictingMaxSampleSize:
    def test_raises(self):
        with pytest.raises(ValueError, match="Conflicting max_sample_size_per_model"):
            load_episodes(FIXTURES / "conflicting_max_sample_size_across_tasks")


class TestMismatchedMaxSampleSize:
    """Two runs for the same model+task with different max_sample_size values."""

    def test_raises(self):
        with pytest.raises(ValueError, match="Mismatched max_sample_size_per_model"):
            load_episodes(FIXTURES / "mismatched_max_sample_size")


class TestOverlappingEpisodes:
    """Two timestamped runs with overlapping (skill_type, scenario_index) pairs."""

    def test_raises(self):
        with pytest.raises(ValueError, match="Overlapping episode"):
            load_episodes(FIXTURES / "stale_overlapping_runs")


class TestExceedsMaxSampleSize:
    """More completed episodes than the declared max_sample_size_per_model budget."""

    def test_raises(self):
        with pytest.raises(ValueError, match="exceeds max_sample_size_per_model"):
            load_episodes(FIXTURES / "exceeds_max_sample_size")


class TestNoResultsFiles:
    def test_empty_dir(self, tmp_path):
        eps, pending, crashed, mss = load_episodes(tmp_path)
        assert len(eps) == 0


# ---------------------------------------------------------------------------
# collect_scenario_indices tests
# ---------------------------------------------------------------------------


class TestCollectScenarioIndices:
    def test_returns_indices(self):
        # mixed_pending_crashed_success has indices 0-4
        rollouts_dir = FIXTURES / "mixed_pending_crashed_success" / "model_a" / "TaskA" / "rollouts"
        indices = collect_scenario_indices(rollouts_dir)
        assert indices == {0, 1, 2, 3, 4}

    def test_empty_dir(self, tmp_path):
        assert collect_scenario_indices(tmp_path) == set()


# ---------------------------------------------------------------------------
# check_no_overlapping_results tests
# ---------------------------------------------------------------------------


class TestCheckNoOverlappingResults:
    """Tests for run_evaluation.check_no_overlapping_results."""

    def test_nonexistent_dir_passes(self, tmp_path):
        # Should not raise when directory doesn't exist
        check_no_overlapping_results(tmp_path / "nonexistent", {0, 1, 2}, max_sample_size=200)

    def test_no_overlap_passes(self):
        # mixed_pending_crashed_success has indices 0-4, request 5-9 — no overlap
        rollouts_dir = FIXTURES / "mixed_pending_crashed_success" / "model_a" / "TaskA" / "rollouts"
        check_no_overlapping_results(rollouts_dir, {5, 6, 7, 8, 9}, max_sample_size=50)

    def test_overlapping_indices_raises(self):
        # mixed_pending_crashed_success has indices 0-4, request 3-7 — overlap on 3,4
        rollouts_dir = FIXTURES / "mixed_pending_crashed_success" / "model_a" / "TaskA" / "rollouts"
        with pytest.raises(SystemExit, match="overlapping"):
            check_no_overlapping_results(rollouts_dir, {3, 4, 5, 6, 7}, max_sample_size=50)

    def test_exact_duplicate_raises(self):
        # Request the exact same indices that already exist
        rollouts_dir = FIXTURES / "mixed_pending_crashed_success" / "model_a" / "TaskA" / "rollouts"
        with pytest.raises(SystemExit, match="overlapping"):
            check_no_overlapping_results(rollouts_dir, {0, 1, 2, 3, 4}, max_sample_size=50)

    def test_exceeds_budget_raises(self):
        # mixed_pending_crashed_success has 5 existing indices, request 46 more with budget 50
        rollouts_dir = FIXTURES / "mixed_pending_crashed_success" / "model_a" / "TaskA" / "rollouts"
        with pytest.raises(SystemExit, match="exceeding --max_sample_size"):
            check_no_overlapping_results(rollouts_dir, set(range(5, 51)), max_sample_size=50)

    def test_within_budget_passes(self):
        # mixed_pending_crashed_success has 5 existing indices, request 5 more with budget 50
        rollouts_dir = FIXTURES / "mixed_pending_crashed_success" / "model_a" / "TaskA" / "rollouts"
        check_no_overlapping_results(rollouts_dir, {5, 6, 7, 8, 9}, max_sample_size=50)

    def test_exactly_at_budget_passes(self):
        # 5 existing + 45 new = 50 exactly at budget
        rollouts_dir = FIXTURES / "mixed_pending_crashed_success" / "model_a" / "TaskA" / "rollouts"
        check_no_overlapping_results(rollouts_dir, set(range(5, 50)), max_sample_size=50)


class TestParseEpisodeRange:
    def test_basic(self):
        assert parse_episode_range("0:5") == {0, 1, 2, 3, 4}

    def test_offset(self):
        assert parse_episode_range("100:103") == {100, 101, 102}
