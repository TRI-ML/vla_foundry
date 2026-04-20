"""Print a summary table of evaluation results from a rollouts directory."""

import sys
from pathlib import Path

from vla_foundry.eval.data_loading import aggregate_episodes, load_episodes
from vla_foundry.eval.stats import clopper_pearson_ci


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m vla_foundry.eval.print_results <rollouts_dir>", file=sys.stderr)
        sys.exit(1)

    root = Path(sys.argv[1])
    episodes, pending_by, crashed_by, _max_sample_size = load_episodes(root)
    if not episodes:
        print("No episodes found.")
        return

    stats = aggregate_episodes(episodes, ci_fn=clopper_pearson_ci, pending_by=pending_by, crashed_by=crashed_by)

    print(f"\n{'Task':<30} {'Model':<15} {'Success Rate':>13} {'N':>5} {'90% CI':>20}")
    print("-" * 88)
    for s in sorted(stats, key=lambda x: x["pct"], reverse=True):
        ci = f"[{s['ci_low']:.1%}, {s['ci_high']:.1%}]"
        print(f"{s['task']:<30} {s['model']:<15} {s['pct']:>11.1f}% {s['total']:>5} {ci:>20}")


if __name__ == "__main__":
    main()
