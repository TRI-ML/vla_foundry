"""Statistical functions for evaluation comparison (beta posteriors, CLD, STEP test).

All functions are pure — no database or filesystem dependencies.
Requires ``scipy``, ``numpy``, and ``sequentialized-barnard-tests``
(available via ``uv sync --group dashboard``).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np


def clopper_pearson_ci(
    successes: int,
    total: int,
    confidence: float = 0.9,
) -> tuple[float, float]:
    """Clopper-Pearson confidence interval for a binomial proportion.

    Args:
        successes: Number of successes.
        total: Total number of trials.
        confidence: Confidence level (0 to 1). Default 0.9 matches the
            internal leaderboard.

    Returns:
        ``(lower_bound, upper_bound)`` of the confidence interval.
    """
    from scipy.stats import binomtest

    if total == 0:
        return 0.0, 1.0

    r = binomtest(successes, total).proportion_ci(confidence)
    return r.low, r.high


def build_success_arrays(
    episodes: list[dict],
) -> tuple[dict, dict]:
    """Convert raw episodes to per-(task, model) boolean arrays.

    Returns ``(arrays_by_task, metadata)``.

    ``arrays_by_task`` has the shape
    ``{task: {model: array}, "__aggregate__": {model: balanced_array}}``.

    The ``"__aggregate__"`` entry is built so that each model's aggregate
    is an unbiased estimate of its equally-weighted multi-task performance.
    Only tasks common to *all* models are included.  For each model, the
    per-task contribution is balanced to that model's own minimum count
    across common tasks (``per_model_min_n``), so every task has equal
    weight *within* that model.  Different models may therefore have
    different aggregate array lengths; the pairwise STEP test handles
    this via ``min(len(A), len(B))`` truncation.

    Per-task arrays are kept at their full length.

    ``metadata`` contains information about the balancing:

    * ``common_tasks`` – sorted list of tasks used in the aggregate.
    * ``excluded_tasks`` – sorted list of tasks excluded (not all models
      have rollouts).
    * ``per_model_min_n`` – ``{model: int}`` rollouts per common task
      used in that model's aggregate.
    * ``num_common_tasks`` – ``len(common_tasks)``.
    * ``original_counts`` – ``{task: {model: int}}`` with original array
      lengths before any truncation.
    """
    data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for ep in episodes:
        data[ep["task"]][ep["model"]].append(bool(ep["success"]))

    result: dict[str, dict[str, np.ndarray]] = {}
    original_counts: dict[str, dict[str, int]] = {}

    for task, model_dict in data.items():
        result[task] = {}
        original_counts[task] = {}
        for model, bools in model_dict.items():
            result[task][model] = np.array(bools, dtype=bool)
            original_counts[task][model] = len(bools)

    # Determine common tasks (tasks where every model has ≥1 rollout).
    all_models = sorted({m for task_dict in data.values() for m in task_dict})
    all_tasks = sorted(data.keys())
    common_tasks = sorted(
        t for t in all_tasks if all(t in data and m in data[t] and len(data[t][m]) > 0 for m in all_models)
    )
    excluded_tasks = sorted(set(all_tasks) - set(common_tasks))

    # Build balanced aggregate — each model balanced to its own minimum.
    per_model_min_n: dict[str, int] = {}
    if common_tasks and all_models:
        for m in all_models:
            per_model_min_n[m] = min(len(result[t][m]) for t in common_tasks)
        agg: dict[str, list] = defaultdict(list)
        for t in common_tasks:
            for m in all_models:
                agg[m].extend(result[t][m][: per_model_min_n[m]].tolist())
        result["__aggregate__"] = {m: np.array(v, dtype=bool) for m, v in agg.items()}
    else:
        result["__aggregate__"] = {}

    metadata = {
        "common_tasks": common_tasks,
        "excluded_tasks": excluded_tasks,
        "per_model_min_n": per_model_min_n,
        "num_common_tasks": len(common_tasks),
        "original_counts": original_counts,
    }
    return result, metadata


def compute_cld_step(
    success_arrays_by_task: dict,
    max_sample_size_per_model: int,
    aggregate_metadata: dict,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[dict[str, dict[str, str]], str]:
    """Compute per-task and aggregate CLD using the STEP sequential test.

    Calls ``compare_success_and_get_cld_auto`` with Bonferroni correction
    across all pairwise comparisons.

    Per-task comparisons use ``shuffle=False``; the aggregate uses ``shuffle=True``
    (episodes from different tasks are concatenated, so order should be shuffled).

    Args:
        success_arrays_by_task: First element of ``build_success_arrays`` output.
        max_sample_size_per_model: Maximum number of rollouts per model. Must be
            set based on the experimental budget *before* collecting data.
            ``load_episodes`` enforces that all results files share the same
            value, so this is always the recorded pre-commitment budget.
        aggregate_metadata: Second element of ``build_success_arrays`` output.
            Used to set ``n_max`` for the aggregate test and to generate
            balancing warning messages.
        confidence_level: Global confidence level (default 0.95).
        seed: RNG seed for the aggregate shuffle.

    Returns:
        A tuple ``(cld_by_task, warning_msg)``.  ``cld_by_task`` maps each task
        key (including ``"__aggregate__"``) to ``{model: cld_letter}``.
        ``warning_msg`` is non-empty if any array length exceeds
        ``max_sample_size_per_model`` or if aggregate balancing is applied.
    """
    # Check for sample size violations (exclude aggregate since it's derived).
    violations: list[tuple[str, str, int]] = []
    for task, model_dict in success_arrays_by_task.items():
        if task == "__aggregate__":
            continue
        for model, arr in model_dict.items():
            if len(arr) > max_sample_size_per_model:
                violations.append((task, model, len(arr)))

    warning_parts: list[str] = []
    if violations:
        ex_task, ex_model, ex_len = violations[0]
        warning_parts.append(
            f"Some rollout counts exceed the pre-committed `max_sample_size` "
            f'(e.g. task "{ex_task}", model "{ex_model}": {ex_len} > '
            f"{max_sample_size_per_model}). "
            f"Only the first **{max_sample_size_per_model}** observations "
            f"per model are used for statistical testing. "
            f"Extra observations are ignored to preserve the validity of "
            f"the sequential test's error guarantees."
        )

    # Aggregate balancing info.
    num_common_tasks = aggregate_metadata["num_common_tasks"]
    per_model_min_n: dict[str, int] = aggregate_metadata["per_model_min_n"]
    excluded_tasks = aggregate_metadata["excluded_tasks"]
    common_tasks = aggregate_metadata["common_tasks"]

    if excluded_tasks:
        warning_parts.append(
            f"**Aggregate excludes {len(excluded_tasks)} task(s)** not shared by all models: "
            f"{', '.join(excluded_tasks)}. "
            f"Only tasks common to all models are included in the aggregate comparison."
        )
    if common_tasks:
        per_model_summary = ", ".join(f"{m}: {n}" for m, n in sorted(per_model_min_n.items()))
        warning_parts.append(
            f"**Aggregate balanced for equal task weighting** across "
            f"{num_common_tasks} task(s) ({', '.join(common_tasks)}). "
            f"Rollouts per task per model: {per_model_summary}."
        )
    elif not common_tasks and success_arrays_by_task.get("__aggregate__") == {}:
        warning_parts.append("**No tasks are common to all models** — aggregate comparison is not available.")

    warning_msg = "\n\n".join(warning_parts)

    cld_by_task: dict[str, dict[str, str]] = {}
    rng = np.random.default_rng(seed)
    num_all_tasks = sum(1 for k in success_arrays_by_task if k != "__aggregate__")

    for task, model_dict in success_arrays_by_task.items():
        models = sorted(model_dict.keys())
        if not models:
            cld_by_task[task] = {}
            continue
        if len(models) == 1:
            cld_by_task[task] = {models[0]: "a"}
            continue

        is_agg = task == "__aggregate__"
        # For the aggregate test the maximum sequence length is
        # max_sample_size_per_model * num_all_tasks (one budget per task).
        # We use *all* tasks (not just common ones) so that n_max remains
        # stable as missing rollouts are collected later, preserving
        # pre-commitment.
        n_max = max_sample_size_per_model * num_all_tasks if is_agg else max_sample_size_per_model
        # Copy arrays — compare_success_and_get_cld shuffles them in-place.
        arrays = [model_dict[m].copy() for m in models]

        try:
            cld = compare_success_and_get_cld_auto(
                models,
                arrays,
                confidence_level,
                n_max,
                shuffle=is_agg,
                rng=rng if is_agg else None,
                verbose=False,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "CLD computation failed for task %r; marking as '?'",
                task,
                exc_info=True,
            )
            cld = {m: "?" for m in models}

        cld_by_task[task] = cld

    return cld_by_task, warning_msg


def compare_success_and_get_cld_auto(
    model_name_list: list[str],  # [model_0, ...]
    success_array_list: list[np.ndarray],  # [success_array_for_model_0, ...]
    global_confidence_level: float,
    n_max: int,
    shuffle: bool,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
) -> dict[str, str]:
    """Compares multiple success arrays and returns their Compact Letter Display (CLD)
    representation based on pairwise sequential tests (STEP or Lai, auto-selected).

    Args:
        model_name_list: A list of model names.
        success_array_list: A list of binary arrays indicating success/failure
            for each model.
        global_confidence_level: The desired global confidence level for the
            multiple comparisons.
        n_max: The maximum sequence length for the sequential test. For
            per-task comparisons this equals ``max_sample_size_per_model``; for
            aggregate comparisons it equals
            ``max_sample_size_per_model * num_tasks``.
        shuffle: Whether to shuffle the True/False ordering of each success array
            before comparison. Set it to False if each True/False outcome is
            independent within each array. Set to True if, for example, each array is a
            concatenation of results from multiple tasks and you want to measure the
            aggregate multi-task performance.
        rng: Optional random number generator instance for shuffling. Only used if
            shuffle is True.
        verbose: Whether to print detailed output. Defaults to True.
    Returns:
        A dictionary mapping model names to their CLD letters.
    """
    from sequentialized_barnard_tests import Decision, Hypothesis, MirroredLaiTest, MirroredStepTest
    from sequentialized_barnard_tests.auto import get_mirrored_test
    from sequentialized_barnard_tests.tools.plotting import compact_letter_display

    if shuffle and rng is None:
        raise ValueError("rng must be provided when shuffle is True.")
    num_models = len(model_name_list)
    # Set up the sequential statistical test.
    global_alpha = 1 - global_confidence_level
    num_comparisons = num_models * (num_models - 1) // 2
    individual_alpha = global_alpha / num_comparisons
    individual_confidence_level = 1 - individual_alpha
    test = get_mirrored_test(
        alternative=Hypothesis.P0LessThanP1,
        alpha=individual_alpha,
        n_max=n_max,
    )
    if verbose:
        if isinstance(test, MirroredStepTest):
            method_name = "STEP"
        elif isinstance(test, MirroredLaiTest):
            method_name = "Lai"
        else:
            method_name = type(test).__name__
        print("Statistical Test Specs:")
        print(f"  Method: {method_name}")
        print(f"  Global Confidence: {round(global_confidence_level, 5)}")
        print(f"    ({round(individual_confidence_level, 5)} per comparison)")
        print(f"  Maximum Sequence Length (n_max): {n_max}\n")

    # No explicit reset needed — run_on_sequence() calls reset() internally.

    # Prepare success array per model.
    success_array_dict = dict()  # model_name -> success_array
    for idx in np.arange(num_models):
        model = model_name_list[idx]
        success_array = success_array_list[idx]
        if shuffle:
            rng.shuffle(success_array)
        success_array_dict[model] = success_array

    # Run pairwise comparisons.
    comparisons_dict = dict()  # (model_name_a, model_name_b) -> Decision
    for idx_a in np.arange(num_models):
        for idx_b in np.arange(idx_a + 1, num_models):
            model_a = model_name_list[idx_a]
            model_b = model_name_list[idx_b]
            array_a = success_array_dict[model_a]
            array_b = success_array_dict[model_b]
            len_common = min(len(array_a), len(array_b))
            array_a = array_a[:len_common]
            array_b = array_b[:len_common]
            # Run the test.
            test_result = test.run_on_sequence(array_a, array_b)
            comparisons_dict[(model_a, model_b)] = test_result.decision

    # Compact Letter Display algorithm to summarize results
    input_list_to_cld = list()
    for key, val in comparisons_dict.items():
        if val != Decision.FailToDecide:
            input_list_to_cld.append(key)
    models_sorted_by_success_rates = [
        model
        for model, _ in sorted(
            success_array_dict.items(),
            key=lambda kv_pair: np.mean(kv_pair[1]) if len(kv_pair[1]) else 0.0,
            reverse=True,
        )
    ]
    letters_list = compact_letter_display(input_list_to_cld, models_sorted_by_success_rates)
    if verbose:
        print("Statistical Test Results (Compact Letter Display):")
    str_padding = max([len(model) for model in models_sorted_by_success_rates])
    return_dict = dict()
    for letters, model in zip(letters_list, models_sorted_by_success_rates, strict=True):
        return_dict[model] = letters
        num_successes = np.sum(success_array_dict[model])
        num_trials = len(success_array_dict[model])
        empirical_success_rate = 0.0 if num_trials == 0 else np.mean(success_array_dict[model])
        if verbose:
            print(
                f"  CLD for {model:<{str_padding}}: {letters}\n"
                f"    Success Rate {num_successes} / {num_trials} = "
                f"{round(empirical_success_rate, 3)}",
            )

    # Ranks are determined if each policy has a unique single letter.
    all_order_determined = all([len(letters) == 1 for letters in letters_list]) and len(set(letters_list)) == len(
        model_name_list
    )
    if verbose:
        if all_order_determined:
            print(f"All models separated with global confidence of {round(global_confidence_level, 5)}.")
        else:
            print(
                "Not all models were separated with global confidence of "
                f"{round(global_confidence_level, 5)}. Models that share "
                "a same letter are not separated from each other with "
                "statistical significance. For more information on how to "
                "interpret the letters, see: "
                "https://en.wikipedia.org/wiki/Compact_letter_display.\n"
            )
    return return_dict


# ---------------------------------------------------------------------------
# Plot builders (Plotly figures from raw episodes)
# ---------------------------------------------------------------------------

COLORS = [
    "#4DBBD5",
    "#E64B35",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#DC0000",
    "#7E6148",
    "#B09C85",
]


def _with_alpha(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def model_comparison_chart(
    episodes: list[dict],
    max_sample_size_per_model: int | None = None,
    overlay_on_bars: bool = False,
    confidence_level: float = 0.95,
    n_posterior_samples: int = 2000,
    seed: int = 42,
    task_order_hint: list[str] | None = None,
    model_order_hint: list[str] | None = None,
) -> tuple:
    """Plotly adaptation of ``plot_model_comparison`` from sequentialized_barnard_tests.

    Draws beta posterior violin plots per (task, model) with:
    - Horizontal mean lines through each violin (``meanline_visible=True``).
    - Empirical mean dots (black circle markers).
    - CLD letter annotations above each violin's posterior mean (STEP test,
      Bonferroni-corrected). Only shown when ``max_sample_size_per_model`` is set.
    - Optional semi-transparent bar overlay showing the empirical mean height.
    - A vertical separator before the aggregate section.

    Args:
        episodes: Raw episode dicts with ``task``, ``model``, and ``success`` keys.
        max_sample_size_per_model: The ``max_sample_size_per_model`` value from
            the results files (returned by ``load_episodes``). Used as the STEP
            test budget. When ``None``, violins are shown without CLD annotations.
        overlay_on_bars: If ``True``, draw semi-transparent bars behind each
            violin showing the empirical mean.
        confidence_level: Global confidence level for CLD (default 0.95).
        n_posterior_samples: Number of samples drawn from the beta posterior for
            each violin (default 2000).
        seed: RNG seed for posterior sampling and aggregate shuffling.

    Returns:
        ``(figure, warning_msg)``.  ``warning_msg`` is non-empty when any
        rollout count exceeds ``max_sample_size_per_model``.
    """
    import plotly.graph_objects as go
    from sequentialized_barnard_tests.tools.plotting import draw_samples_from_beta_posterior

    if not episodes:
        return go.Figure().update_layout(title="No data"), ""

    rng = np.random.default_rng(seed)
    arrays_by_task, agg_metadata = build_success_arrays(episodes)

    pure_tasks = sorted(k for k in arrays_by_task if k != "__aggregate__")
    discovered_models = sorted({m for task_dict in arrays_by_task.values() for m in task_dict})
    if model_order_hint:
        hint_set = set(model_order_hint)
        all_models = [m for m in model_order_hint if m in discovered_models]
        all_models += sorted(m for m in discovered_models if m not in hint_set)
    else:
        all_models = discovered_models

    cld_by_task: dict[str, dict[str, str]] = {}
    warning_msg = ""
    if max_sample_size_per_model is not None:
        cld_by_task, warning_msg = compute_cld_step(
            arrays_by_task, max_sample_size_per_model, agg_metadata, confidence_level, seed + 1
        )

    if task_order_hint:
        # Preserve the caller's order; append any tasks not in the hint at the end.
        hint_set = set(task_order_hint)
        task_order = [t for t in task_order_hint if t in pure_tasks]
        task_order += sorted(t for t in pure_tasks if t not in hint_set)
    else:
        task_order = sorted(
            pure_tasks,
            key=lambda t: float(np.mean([np.mean(arr) for arr in arrays_by_task[t].values()])),
            reverse=True,
        )

    n = len(all_models)
    cmap = {m: COLORS[i % len(COLORS)] for i, m in enumerate(all_models)}
    gw = 0.8
    slot = gw / max(n, 1)
    vw = min(0.35, slot * 0.8)

    fig = go.Figure()
    shown: set[str] = set()
    annotations: list[dict] = []

    def _add_violin(
        xp: float,
        model: str,
        arr: np.ndarray,
        cld_key: str,
        width: float,
        used: int | None = None,
        budgeted: int | None = None,
        successes: int | None = None,
    ) -> None:
        samples = draw_samples_from_beta_posterior(arr, rng, n_posterior_samples)
        posterior_mean = float(np.mean(samples))
        empirical_mean = float(np.mean(arr))
        color = cmap[model]

        if overlay_on_bars:
            fig.add_trace(
                go.Bar(
                    x=[xp],
                    y=[empirical_mean],
                    width=width * 1.2,
                    marker_color=_with_alpha(color, 0.3),
                    marker_line_width=0,
                    showlegend=False,
                    legendgroup=model,
                    hoverinfo="skip",
                )
            )

        fig.add_trace(
            go.Violin(
                y=samples,
                x=np.full(len(samples), xp),
                name=model,
                legendgroup=model,
                showlegend=model not in shown,
                scalegroup="all",
                points=False,
                box_visible=False,
                meanline=dict(visible=True, color="black", width=1.5),
                line_color=color,
                fillcolor=color,
                opacity=0.7,
                width=width * 1.25,
                hoverinfo="skip",
            )
        )
        shown.add(model)

        # Empirical mean dot with hover tooltip.
        hover_lines = [f"<b>{model}</b>"]
        hover_lines.append(f"Empirical: {empirical_mean:.1%}")
        hover_lines.append(f"Posterior mean: {posterior_mean:.1%}")
        if successes is not None and used is not None:
            hover_lines.append(f"Successes: {successes}/{used}")
        if budgeted is not None:
            hover_lines.append(f"Budget: {budgeted}")
        fig.add_trace(
            go.Scatter(
                x=[xp],
                y=[empirical_mean],
                mode="markers",
                marker=dict(color="black", size=6, symbol="circle"),
                legendgroup=model,
                showlegend=False,
                hoverinfo="text",
                hovertext="<br>".join(hover_lines),
            )
        )

        cld_letter = cld_by_task.get(cld_key, {}).get(model, "")
        if cld_letter and n > 1:
            violin_bottom = float(np.percentile(samples, 1))
            annotations.append(
                dict(
                    x=xp,
                    y=max(violin_bottom - 0.1, -0.05),
                    text=f"<b>{cld_letter}</b>",
                    showarrow=False,
                    font=dict(family="CMU Serif, Latin Modern Roman, Computer Modern, serif", size=18, weight=400),
                )
            )

        # "successes / used (budgeted)" annotation below the violin.
        if successes is not None and used is not None and budgeted is not None:
            annotations.append(
                dict(
                    x=xp,
                    y=-0.06,
                    text=f"{successes}/{used} ({budgeted})",
                    showarrow=False,
                    textangle=-45,
                    font=dict(
                        family="CMU Serif, Latin Modern Roman, Computer Modern, serif",
                        size=16,
                        color="gray",
                        weight=400,
                    ),
                )
            )

    agg_per_model_min_n = agg_metadata["per_model_min_n"]
    agg_num_common = agg_metadata["num_common_tasks"]
    num_all_tasks = len(pure_tasks)

    for ti, task in enumerate(task_order):
        for ai, model in enumerate(all_models):
            if model not in arrays_by_task.get(task, {}):
                continue
            xp = ti + (ai - (n - 1) / 2) * (gw / max(n, 1))
            arr = arrays_by_task[task][model]
            per_task_used = len(arr)
            per_task_successes = int(np.sum(arr))
            per_task_budgeted = max_sample_size_per_model if max_sample_size_per_model is not None else None
            _add_violin(
                xp,
                model,
                arr,
                task,
                vw,
                used=per_task_used,
                budgeted=per_task_budgeted,
                successes=per_task_successes,
            )

    agg_x = len(task_order) + 1
    agg = arrays_by_task.get("__aggregate__", {})
    agg_budgeted = (
        max_sample_size_per_model * num_all_tasks if max_sample_size_per_model is not None and num_all_tasks else None
    )
    for ai, model in enumerate(all_models):
        if model not in agg:
            continue
        xp = agg_x + (ai - (n - 1) / 2) * (gw * 1.5 / max(n, 1))
        model_min = agg_per_model_min_n.get(model, 0)
        agg_used = model_min * agg_num_common if agg_num_common else None
        agg_successes = int(np.sum(agg[model])) if len(agg[model]) else None
        _add_violin(
            xp,
            model,
            agg[model],
            "__aggregate__",
            vw * 1.5,
            used=agg_used,
            budgeted=agg_budgeted,
            successes=agg_successes,
        )

    ticks = list(range(len(task_order))) + [agg_x]
    tick_text = [(t[:25] + "\u2026" if len(t) > 25 else t) for t in task_order] + ["Aggregate"]

    fig.update_layout(
        # title="Success Rate Distribution (Beta Posterior)",
        font=dict(family="CMU Serif, Latin Modern Roman, Computer Modern, serif", size=20, weight=400),
        yaxis=dict(title="Success Rate", range=[-0.05, 1.15]),
        xaxis=dict(
            tickangle=-45,
            tickvals=ticks,
            ticktext=tick_text,
            range=[-0.7, agg_x + 1.0],
        ),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", yanchor="bottom"),
        annotations=annotations,
        shapes=[
            dict(
                type="line",
                x0=len(task_order) - 0.15,
                x1=len(task_order) - 0.15,
                y0=-0.02,
                y1=1.1,
                line=dict(color="gray", width=2),
            )
        ]
        if task_order
        else [],
        margin=dict(b=150, t=100, l=160, r=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        violingap=0,
        violinmode="overlay",
        barmode="overlay",
    )

    return fig, warning_msg


def spider_chart(data: list[dict]):
    """Radar chart of success rates by task and model."""
    import plotly.graph_objects as go

    if not data:
        return go.Figure().update_layout(title="No data")
    models = sorted({s["model"] for s in data})
    tasks = sorted({s["task"] for s in data})
    lookup = {(s["task"], s["model"]): s for s in data}
    short = [t[:22] + "\u2026" if len(t) > 22 else t for t in tasks]
    fig = go.Figure()
    for i, model in enumerate(models):
        rates = [lookup.get((t, model), {}).get("pct", 0) for t in tasks]
        fig.add_trace(
            go.Scatterpolar(
                r=rates + [rates[0]],
                theta=short + [short[0]],
                fill="toself",
                name=model,
                line_color=COLORS[i % len(COLORS)],
                fillcolor=_with_alpha(COLORS[i % len(COLORS)], 0.2),
            )
        )
    fig.update_layout(
        title="Success Rate by Task",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%"), angularaxis=dict(direction="clockwise")
        ),
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
        margin=dict(t=60, b=60, l=80, r=80),
        paper_bgcolor="white",
    )
    return fig
