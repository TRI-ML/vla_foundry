# Statistical Comparison of Evaluation Results

This document explains how to run evaluations with statistical testing enabled, combine results across runs, and interpret the dashboard's comparison features.

## Overview

When comparing two or more models (or ablations), raw success rates can be misleading — small differences may be noise. The evaluation pipeline includes a **sequential statistical test** ([STEP](https://tri-ml.github.io/step/), Bonferroni-corrected) that determines whether observed differences are statistically significant. Results are summarized using **Compact Letter Display (CLD)**: models that do *not* share a letter are significantly different from each other.

## Workflow

### 1. Set your experimental budget

Before running any evaluations, decide the **maximum number of rollouts per model per task** you will ever collect. This is the `--max_sample_size` parameter. It configures the sequential test's stopping boundary and **must be set before seeing any results** to maintain statistical validity.

Choose this based on your compute budget. For example, if you plan to run at most 200 episodes per task per model:

```bash
--max_sample_size 200
```

### 2. Run evaluations

Run each model with the same `--max_sample_size` and the same `--output_dir`:

```bash
# Model A
uv run python vla_foundry/eval/run_evaluation.py $CHECKPOINT_A \
    --model_name model_a \
    --max_sample_size 200 \
    --num_episodes 0:200 \
    --output_dir tutorials/rollouts

# Model B
uv run python vla_foundry/eval/run_evaluation.py $CHECKPOINT_B \
    --model_name model_b \
    --max_sample_size 200 \
    --num_episodes 0:200 \
    --output_dir tutorials/rollouts
```
**Tip**: --num_episodes can be smaller than ```--max_sample_size```. For example, you might start with 100 episodes to get early results and collect more later (up to the budget) using the [incremental combining workflow](#combining-results-across-runs) described below.

### 3. View results in the dashboard

```bash
uv run --group dashboard python vla_foundry/eval/results_explorer.py tutorials/rollouts/
```

Open `http://localhost:8505`. The **Model Comparison** tab shows:
- Beta posterior violin plots for each (task, model) pair
- CLD letter annotations (computed automatically if `max_sample_size` is available)
- An aggregate comparison across all tasks

Click **Refresh** to pick up new results as evaluations run. Statistical tests are re-computed on every refresh. Note that the first time STEP is invoked with a given --max_sample_size, it may take up to a few minutes to pre-compute decision rules and save them in small cache files. These are stored in the ```sequentialized_barnard_tests``` package. Subsequent runs reuse the cache, so statistical testing should be near-instant from the second invocation onward.

## Combining Results Across Runs

You can collect results incrementally. For example, run episodes 100-199 first, then 200-299 later:

```bash
# First batch
uv run python vla_foundry/eval/run_evaluation.py $CHECKPOINT \
    --model_name model_a --max_sample_size 200 --num_episodes 100:200 --output_dir tutorials/rollouts

# Second batch (same model, same output dir, non-overlapping episodes)
uv run python vla_foundry/eval/run_evaluation.py $CHECKPOINT \
    --model_name model_a --max_sample_size 200 --num_episodes 200:300 --output_dir tutorials/rollouts
```

This produces two timestamped `results.json` files under the same rollouts directory. The dashboard **automatically combines** them if:

1. **No overlapping episode indices** — each `(skill_type, scenario_index)` pair appears in only one file
2. **Same `max_sample_size`** — both runs were started with the same `--max_sample_size`

If these conditions aren't met, loading will fail with a `ValueError`. Remove stale results files before viewing — video recordings for older runs may have been overwritten.

## Metadata

Every `results.json` must contain a `max_sample_size_per_model` field recording the evaluation budget. This field should be written before any episodes complete so the budget is locked in from the start. Every results JSON is self-contained: it carries its own budget from the very first episode, and you can share just the JSON file for others to compare against.

### Why pre-commitment matters

The sequential STEP test provides valid statistical guarantees *only* when the maximum sample size is fixed before data collection begins. Changing it after seeing A/B testing results risks [data-dredging](https://en.wikipedia.org/wiki/Data_dredging). Setting `--max_sample_size` at experiment launch time enforces this pre-commitment — the budget is decided before any results are observed.

## How Statistical Comparisons Work

We provide a brief technical overview of our statistical comparison framework. For motivation and details, see [this tech blog](https://medium.com/toyotaresearch/statistical-thinking-for-robot-policy-evaluation-from-rigorous-a-b-testing-to-effective-0ae886fbd68d).

### Per-task comparisons

For each task, the pipeline builds a boolean success/failure array per model and runs pairwise sequential tests ([`sequentialized_barnard_tests`](https://github.com/TRI-ML/sequentialized_barnard_tests)) between every pair of models. The test processes episodes sequentially and can reach a decision before consuming all data.

If two models have different sample sizes (e.g., 80 rollouts for Model A and 120 for Model B), both arrays are truncated to the shorter length for paired testing with STEP.

**Bonferroni correction** is applied across all pairwise comparisons to control the global false positive rate. With `K` models there are `K*(K-1)/2` pairs. The per-comparison significance level is:

```
alpha_per_pair = 0.05 / num_pairs
```

This ensures the probability of *any* false positive across all comparisons stays below 5%.

Each per-task test uses `n_max = max_sample_size_per_model` and processes episodes in their original order (`shuffle=False`), since outcomes within a single task are independent.

### Aggregate comparison

The aggregate comparison measures overall multi-task performance. The goal is for each model's aggregate to be an **unbiased estimate of its equally-weighted multi-task performance**, where every task carries equal weight within a model's aggregate.

To achieve this, the aggregate is **balanced per model** before statistical analysis:

1. **Selecting common tasks**: only tasks where *every* model has at least one rollout are included. Tasks missing from any model are excluded from the aggregate for all models.

2. **Per-model balancing**: for each model, the minimum rollout count across all common tasks is computed. If Model A has [50, 20, 100, 100] rollouts across 4 tasks, the minimum rollout count is 20. Then, the results are truncated to [20, 20, 20, 20] rollouts before aggregation to ensure equal task weighting within the model. If Model B has [70, 30, 80, 90] rollouts, the results are truncated to [30, 30, 30, 30] before aggregation. Then, 20 * 4 = 80 rollouts from Model A and 30 * 4 = 120 rollouts from Model B are used for plotting (see [Beta posterior visualization](#beta-posterior-visualization)).

3. **Concatenating and shuffling**: each model's balanced per-task arrays are concatenated (in sorted task order) into a single array and shuffled before running the test (`shuffle=True`). Shuffling removes ordering bias from the concatenation of results across tasks with different difficulty levels.

4. **Pairwise comparison**: the STEP test compares models pairwise, with `n_max = max_sample_size_per_model * num_tasks` (using all tasks, not just common ones). This ensures `n_max` remains stable as missing rollouts are collected later. When two models have different aggregate sizes (e.g., 80 rollouts for Model A and 120 for Model B), both results are truncated to the shorter length for that specific comparison. This is the same behavior as per-task comparisons.

The dashboard displays which tasks are included in the aggregate, rollouts per task per model, and any excluded tasks. Each violin also shows `used/budgeted` counts to show the number of rollouts used for plotting out of the evaluation budget.

### Compact Letter Display (CLD)

Test results are summarized using [Compact Letter Display](https://en.wikipedia.org/wiki/Compact_letter_display). Models are sorted by empirical success rate (highest first) and assigned letters:

- Models with entirely different letters **are** significantly different (with <=5% false positive rate)
- A model can have multiple letters (e.g., "ab") meaning it overlaps with both group "a" and group "b"
- Models that share a letter are **not** statistically separated from each other

If the test fails to decide (inconclusive — common when models have similar performance or sample sizes are small), those models will share a letter.

### Beta posterior visualization

The violin plots show **Bayesian beta posterior distributions** of each model's true success rate, computed independently of the sequential test:

- **Prior:** Beta(1, 1) (uniform)
- **Posterior:** Beta(1 + successes, 1 + failures)
- **Samples:** 2000 draws from the posterior per violin
- **Horizontal line:** posterior mean
- **Black dot:** empirical (observed) mean

These provide a visual sense of uncertainty. The CLD letters above each violin give the formal statistical conclusion.

For aggregate violins, each model's results are balanced to its minimum rollout count across all common tasks, ensuring the posterior reflects equally-weighted multi-task performance. See [Aggregate comparison](#aggregate-comparison) for details. Individual per-task violins use the full success/failure results available.

## Using Released Evaluation Results

We release per-episode breakdown JSONs for some of our evaluation results. These were collected with a computational budget of **200 episodes per model per task** (`max_sample_size = 200`). When combining these with your own runs or running statistical comparisons against them, you **must** set `--max_sample_size 200` to match.

```bash
uv run python vla_foundry/eval/run_evaluation.py $YOUR_CHECKPOINT \
    --model_name your_model \
    --max_sample_size 200 \
    --num_episodes 0:200 \
    --output_dir tutorials/rollouts
```

Place the released JSONs alongside your results in the same `tutorials/rollouts/` directory structure and launch the dashboard. The combining and statistical comparison will work automatically as long as the budget is consistent.

## What the Dashboard Shows

### Summary Tab
- Success rate, confidence intervals, and rollout counts per (task, model) pair
- Color-coded by performance

### Model Comparison Tab
- **Max sample size metadata** — displayed at the top; shows the budget or a warning if missing
- **Beta posterior violins** — Bayesian posterior distributions of the true success rate for each model, per task
  - Horizontal line = posterior mean
  - Black dot = empirical (observed) mean
- **CLD letters** — above each violin. Models sharing a letter are not significantly different (at 95% global confidence, Bonferroni-corrected)
- **Aggregate section** — rightmost column, separated by a vertical line. Combines all tasks for an overall comparison
- **Optional bar overlay** — toggle "Overlay violins on bar plots" to show empirical means as semi-transparent bars behind the violins

### Spider Tab
- Radar chart of success rates by task, one line per model

### Episode Recordings Tab
- Paginated video grid of evaluation episodes
- Filter by outcome (All / Success / Failure) and by episode ID

## File Format Reference

### results.json
```json
{
  "max_sample_size_per_model": 200,
  "num_requested_evaluations": 200,
  "num_evaluated": 150,
  "num_success": 120,
  "success_rate": 0.8,
  "evaluations": [
    {
      "skill_type": "PutMugOnSaucer",
      "scenario_index": 100,
      "is_success": true,
      "is_pending": false,
      "total_time": 12.5,
      "failure_message": null
    }
  ]
}
```

### Directory structure
```
rollouts/
  model_name/
    TaskName/
      rollouts/
        2026-04-02T03:58:51+00:00/
          results.json
          TaskName/
            demonstration_100/
              recording.html
              recording.mp4
        2026-04-02T04:26:55+00:00/   <- combined automatically if non-overlapping
          results.json
```
