"""Integration tests for aggregate balancing via dashboard fixtures.

Exercises the full pipeline: load_episodes -> build_success_arrays -> metadata.
Each fixture is a directory under dashboard_fixtures/ with the canonical layout
and contains two models evaluated on shared (and sometimes unshared) tasks.
"""

from pathlib import Path

import numpy as np
import pytest

from vla_foundry.eval.data_loading import load_episodes
from vla_foundry.eval.stats import build_success_arrays

FIXTURES = Path(__file__).resolve().parent.parent / "test_assets" / "dashboard_fixtures"


# ---------------------------------------------------------------------------
# Fixture 1: unequal rollout counts across models
# ---------------------------------------------------------------------------


class TestAggregateAndCompareUnequalRollouts:
    """model_a: 4+4, model_b: 2+4 -> per-model balancing."""

    @pytest.fixture(autouse=True)
    def _load(self):
        eps, _, _, self.mss = load_episodes(FIXTURES / "aggregate_and_compare_unequal_rollouts")
        self.arrays, self.meta = build_success_arrays(eps)

    def test_max_sample_size(self):
        assert self.mss == 10

    def test_common_tasks(self):
        assert self.meta["common_tasks"] == ["task1", "task2"]

    def test_no_excluded_tasks(self):
        assert self.meta["excluded_tasks"] == []

    def test_num_common_tasks(self):
        assert self.meta["num_common_tasks"] == 2

    def test_per_model_min_n(self):
        assert self.meta["per_model_min_n"] == {"model_a": 4, "model_b": 2}

    def test_aggregate_lengths(self):
        assert len(self.arrays["__aggregate__"]["model_a"]) == 8  # 4 * 2
        assert len(self.arrays["__aggregate__"]["model_b"]) == 4  # 2 * 2

    def test_aggregate_success_counts(self):
        # model_a: task1 [T,F,T,F] -> 2 + task2 [T,T,T,T] -> 4 = 6
        assert np.sum(self.arrays["__aggregate__"]["model_a"]) == 6
        # model_b: task1[:2] [F,F] -> 0 + task2[:2] [T,F] -> 1 = 1
        assert np.sum(self.arrays["__aggregate__"]["model_b"]) == 1

    def test_per_task_arrays_full_length(self):
        assert len(self.arrays["task1"]["model_a"]) == 4
        assert len(self.arrays["task2"]["model_a"]) == 4
        assert len(self.arrays["task1"]["model_b"]) == 2
        assert len(self.arrays["task2"]["model_b"]) == 4

    def test_original_counts(self):
        oc = self.meta["original_counts"]
        assert oc["task1"]["model_a"] == 4
        assert oc["task2"]["model_a"] == 4
        assert oc["task1"]["model_b"] == 2
        assert oc["task2"]["model_b"] == 4


# ---------------------------------------------------------------------------
# Fixture 2: one model missing a task
# ---------------------------------------------------------------------------


class TestAggregateAndCompareMissingTask:
    """model_a: 3 tasks, model_b: 2 tasks -> task3 excluded."""

    @pytest.fixture(autouse=True)
    def _load(self):
        eps, _, _, self.mss = load_episodes(FIXTURES / "aggregate_and_compare_missing_task")
        self.arrays, self.meta = build_success_arrays(eps)

    def test_common_tasks(self):
        assert self.meta["common_tasks"] == ["task1", "task2"]

    def test_excluded_tasks(self):
        assert self.meta["excluded_tasks"] == ["task3"]

    def test_num_common_tasks(self):
        assert self.meta["num_common_tasks"] == 2

    def test_per_model_min_n(self):
        # model_a: min(3, 3) = 3 across common tasks
        # model_b: min(4, 4) = 4 across common tasks
        assert self.meta["per_model_min_n"] == {"model_a": 3, "model_b": 4}

    def test_aggregate_lengths(self):
        assert len(self.arrays["__aggregate__"]["model_a"]) == 6  # 3 * 2
        assert len(self.arrays["__aggregate__"]["model_b"]) == 8  # 4 * 2

    def test_aggregate_success_counts(self):
        # model_a: task1[:3] [T,T,T] -> 3 + task2[:3] [F,F,F] -> 0 = 3
        assert np.sum(self.arrays["__aggregate__"]["model_a"]) == 3
        # model_b: task1[:4] [T,F,T,F] -> 2 + task2[:4] [T,T,T,T] -> 4 = 6
        assert np.sum(self.arrays["__aggregate__"]["model_b"]) == 6

    def test_task3_excluded_from_aggregate(self):
        # model_a agg length is 6, not 8 (would be 8 if task3 were included)
        assert len(self.arrays["__aggregate__"]["model_a"]) == 6

    def test_task3_still_in_per_task(self):
        assert "task3" in self.arrays
        assert len(self.arrays["task3"]["model_a"]) == 2
        assert "model_b" not in self.arrays["task3"]


# ---------------------------------------------------------------------------
# Fixture 3: equal rollout counts — no truncation
# ---------------------------------------------------------------------------


class TestAggregateAndCompareEqualRollouts:
    """Both models: 3 per task, 2 tasks -> no truncation."""

    @pytest.fixture(autouse=True)
    def _load(self):
        eps, _, _, self.mss = load_episodes(FIXTURES / "aggregate_and_compare_equal_rollouts")
        self.arrays, self.meta = build_success_arrays(eps)

    def test_common_tasks(self):
        assert self.meta["common_tasks"] == ["task1", "task2"]

    def test_no_excluded_tasks(self):
        assert self.meta["excluded_tasks"] == []

    def test_per_model_min_n(self):
        assert self.meta["per_model_min_n"] == {"model_a": 3, "model_b": 3}

    def test_aggregate_lengths(self):
        assert len(self.arrays["__aggregate__"]["model_a"]) == 6  # 3 * 2
        assert len(self.arrays["__aggregate__"]["model_b"]) == 6  # 3 * 2

    def test_no_truncation(self):
        for model in ["model_a", "model_b"]:
            min_n = self.meta["per_model_min_n"][model]
            expected = min_n * self.meta["num_common_tasks"]
            assert len(self.arrays["__aggregate__"][model]) == expected

    def test_aggregate_success_counts(self):
        # model_a: task1 [T,T,F] -> 2 + task2 [F,T,F] -> 1 = 3
        assert np.sum(self.arrays["__aggregate__"]["model_a"]) == 3
        # model_b: task1 [F,F,T] -> 1 + task2 [T,T,T] -> 3 = 4
        assert np.sum(self.arrays["__aggregate__"]["model_b"]) == 4

    def test_aggregate_preserves_order(self):
        # Aggregate concatenates common tasks in sorted order: task1 then task2
        agg_a = self.arrays["__aggregate__"]["model_a"]
        # First 3 from task1: T, T, F
        np.testing.assert_array_equal(agg_a[:3], [True, True, False])
        # Next 3 from task2: F, T, F
        np.testing.assert_array_equal(agg_a[3:], [False, True, False])

    def test_per_task_full_length(self):
        for task in ["task1", "task2"]:
            for model in ["model_a", "model_b"]:
                assert len(self.arrays[task][model]) == 3
