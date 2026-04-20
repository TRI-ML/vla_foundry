"""Generate PDF violin plots from a YAML config file.

Batch equivalent of the "Download Plot as PDF" button in ``results_explorer.py``:
loads episodes from one or more rollout roots, filters tasks/models, and
writes a beta-posterior violin plot per entry.

Usage::

    uv run --group dashboard python vla_foundry/eval/generate_violin_plots.py plots.yaml

Config format::

    # Optional shared defaults (any plot key can override).
    defaults:
      width: 1800
      height: 600
      font_scale: 1.5
      bar_overlay: true
      output_dir: ./plots       # relative to the config file

    plots:
      main_16:
        paths:
          - vla_foundry/eval/eval_results
        # tasks: "seen" -> "16 Main Tasks", "unseen" -> "3 Unseen Tasks",
        #        "all" -> every task found, a task-group name from rename.yaml,
        #        or an explicit list of (display) task names.
        tasks: seen
        # Order in the list = color order left-to-right in the plot.
        models: [CS/LBM-MT, CS/LBM-ST, CS/LBM-FT]

      unseen_3:
        paths:
          - vla_foundry/eval/eval_results
        tasks: unseen
        width: 1200
        height: 600

Paths are interpreted as rollout roots (the directory passed to
``results_explorer.py``).  Episodes from every path are concatenated before
plotting, so you can combine results from multiple experiment directories.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

from vla_foundry.eval.data_loading import load_episodes, load_task_groups
from vla_foundry.eval.stats import build_success_arrays, model_comparison_chart

SEEN_GROUP = "16 Main Tasks"
UNSEEN_GROUP = "3 Unseen Tasks"


def _resolve_tasks(
    tasks_cfg,
    task_groups: dict[str, list[str]],
    available_tasks: set[str],
) -> list[str]:
    if tasks_cfg is None or tasks_cfg == "all":
        return sorted(available_tasks)
    if isinstance(tasks_cfg, str):
        key = {"seen": SEEN_GROUP, "unseen": UNSEEN_GROUP}.get(tasks_cfg, tasks_cfg)
        if key in task_groups:
            return list(task_groups[key])
        known = sorted(task_groups.keys()) + ["all", "seen", "unseen"]
        raise ValueError(f"Unknown task group {tasks_cfg!r}. Known: {known}")
    if isinstance(tasks_cfg, (list, tuple)):
        return [str(t) for t in tasks_cfg]
    raise ValueError(f"Unsupported 'tasks' value: {tasks_cfg!r}")


def _load_combined(
    paths: list[Path],
) -> tuple[list[dict], int | None, dict[str, list[str]]]:
    all_eps: list[dict] = []
    max_sizes: set[int] = set()
    task_groups: dict[str, list[str]] = {}
    for p in paths:
        eps, _pending, _crashed, mss = load_episodes(p)
        all_eps.extend(eps)
        if mss is not None:
            max_sizes.add(mss)
        groups = load_task_groups(p)
        for k, v in groups.items():
            task_groups.setdefault(k, list(v))
    if len(max_sizes) > 1:
        print(
            f"WARNING: paths report different max_sample_size_per_model values {max_sizes}; using min for CLD.",
            file=sys.stderr,
        )
    mss_combined = min(max_sizes) if max_sizes else None
    return all_eps, mss_combined, task_groups


def _save_table(
    name: str,
    episodes: list[dict],
    task_order: list[str],
    models: list[str] | None,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write ``{name}.csv`` (long format) and ``{name}.md`` (wide, readable).

    Rows: one per (task, model) plus ``Aggregate (balanced)`` — the
    per-task-balanced aggregate used by the violin plot — and
    ``Aggregate (unweighted)`` — raw sum of successes / sum of totals.
    """
    import pandas as pd

    arrays_by_task, _meta = build_success_arrays(episodes)
    discovered = sorted({m for task_dict in arrays_by_task.values() for m in task_dict})
    if models:
        model_order = [m for m in models if m in discovered]
        model_order += [m for m in discovered if m not in set(models)]
    else:
        model_order = discovered

    rows: list[dict] = []

    def _row(task_label: str, model: str, succ: int, total: int) -> dict:
        return {
            "task": task_label,
            "model": model,
            "successes": succ,
            "total": total,
            "success_rate": (succ / total) if total else 0.0,
        }

    for task in task_order:
        for model in model_order:
            arr = arrays_by_task.get(task, {}).get(model)
            if arr is None:
                continue
            rows.append(_row(task, model, int(arr.sum()), int(len(arr))))

    agg = arrays_by_task.get("__aggregate__", {})
    for model in model_order:
        arr = agg.get(model)
        if arr is None or len(arr) == 0:
            continue
        rows.append(_row("Aggregate (balanced)", model, int(arr.sum()), int(len(arr))))

    for model in model_order:
        total_succ = 0
        total_n = 0
        for task in task_order:
            arr = arrays_by_task.get(task, {}).get(model)
            if arr is None:
                continue
            total_succ += int(arr.sum())
            total_n += int(len(arr))
        if total_n == 0:
            continue
        rows.append(_row("Aggregate (unweighted)", model, total_succ, total_n))

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)

    # Wide, readable markdown: rows = task, columns = one "succ/total (pct)" per model.
    if df.empty:
        md_path = output_dir / f"{name}.md"
        md_path.write_text("_(no data)_\n")
        return csv_path, md_path

    def _cell(row: pd.Series) -> str:
        return f"{int(row['successes'])}/{int(row['total'])} ({100 * row['success_rate']:.1f}%)"

    df["cell"] = df.apply(_cell, axis=1)
    row_order = list(task_order) + ["Aggregate (balanced)", "Aggregate (unweighted)"]
    wide = df.pivot(index="task", columns="model", values="cell")
    wide = wide.reindex(index=[r for r in row_order if r in wide.index])
    wide = wide.reindex(columns=[m for m in model_order if m in wide.columns])
    wide = wide.fillna("")

    headers = ["task", *wide.columns.tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for task, row in wide.iterrows():
        lines.append("| " + " | ".join([str(task), *(str(v) for v in row.tolist())]) + " |")
    md_path = output_dir / f"{name}.md"
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def _scale_fonts(fig, scale: float) -> None:
    def _apply(font_obj):
        if font_obj and font_obj.size:
            font_obj.size = round(font_obj.size * scale)

    _apply(fig.layout.font)
    for ax_attr in ("xaxis", "yaxis"):
        ax = getattr(fig.layout, ax_attr, None)
        if ax:
            _apply(ax.tickfont)
            if ax.title:
                _apply(ax.title.font)
    if fig.layout.legend:
        _apply(fig.layout.legend.font)
    for ann in fig.layout.annotations:
        _apply(ann.font)


def _render_plot(
    name: str,
    plot_cfg: dict,
    defaults: dict,
    output_dir: Path,
    config_dir: Path,
) -> Path:
    cfg = {**defaults, **plot_cfg}
    raw_paths = cfg.get("paths") or []
    if not raw_paths:
        raise ValueError(f"Plot '{name}' has no 'paths'.")
    paths: list[Path] = []
    for p in raw_paths:
        pp = Path(p)
        if not pp.is_absolute():
            pp = (config_dir / pp).resolve()
        if not pp.exists():
            raise FileNotFoundError(f"Plot '{name}': path does not exist: {pp}")
        paths.append(pp)

    episodes, mss, task_groups = _load_combined(paths)
    if not episodes:
        raise ValueError(f"Plot '{name}': no episodes loaded from {paths}.")

    available_tasks = {ep["task"] for ep in episodes}
    task_order = _resolve_tasks(cfg.get("tasks", "all"), task_groups, available_tasks)
    task_set = set(task_order)
    episodes = [ep for ep in episodes if ep["task"] in task_set]

    models = cfg.get("models")
    if models:
        model_set = set(models)
        episodes = [ep for ep in episodes if ep["model"] in model_set]

    if not episodes:
        raise ValueError(
            f"Plot '{name}': no episodes remain after filtering (tasks={cfg.get('tasks')!r}, models={models!r})."
        )

    fig, warning = model_comparison_chart(
        episodes,
        mss,
        bool(cfg.get("bar_overlay", True)),
        task_order_hint=task_order,
        model_order_hint=list(models) if models else None,
    )
    if warning:
        print(f"[{name}] {warning}", file=sys.stderr)

    font_scale = float(cfg.get("font_scale", 1.5))
    width = int(cfg.get("width", 1800))
    height = int(cfg.get("height", 600))
    fig_export = copy.deepcopy(fig)
    _scale_fonts(fig_export, font_scale)
    fig_export.update_layout(width=width, height=height)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.pdf"
    fig_export.write_image(str(out_path))

    _save_table(name, episodes, task_order, list(models) if models else None, output_dir)
    return out_path


def _split_top_level(cfg: dict) -> tuple[dict, dict]:
    """Accept either ``{defaults: ..., plots: {...}}`` or a flat mapping."""
    if "plots" in cfg:
        return cfg.get("defaults", {}) or {}, cfg["plots"] or {}
    defaults = cfg.get("defaults", {}) or {}
    plots = {k: v for k, v in cfg.items() if k != "defaults"}
    return defaults, plots


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PDF violin plots from a YAML config.")
    parser.add_argument("config", type=Path, help="YAML config file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write PDFs (overrides defaults.output_dir).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Only render these plot names.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text()) or {}
    defaults, plots = _split_top_level(cfg)
    if not plots:
        print("No plots defined in config.", file=sys.stderr)
        return 1

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        out_cfg = defaults.get("output_dir", "plots")
        out_p = Path(out_cfg)
        output_dir = (out_p if out_p.is_absolute() else config_path.parent / out_p).resolve()

    selected = set(args.only) if args.only else None
    errors: list[tuple[str, Exception]] = []
    for name, plot_cfg in plots.items():
        if selected is not None and name not in selected:
            continue
        try:
            out = _render_plot(name, plot_cfg, defaults, output_dir, config_path.parent)
            print(f"Wrote {out}")
        except Exception as exc:  # noqa: BLE001
            errors.append((name, exc))
            print(f"[{name}] FAILED: {exc}", file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} plot(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
