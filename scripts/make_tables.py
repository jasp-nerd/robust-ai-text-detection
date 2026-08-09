"""Regenerate all results tables from committed run artifacts.

Usage: uv run python scripts/make_tables.py [--slice attack|generator|domain] [--eval raid/eval]

Every table in the README/write-up comes from this script — no hand-typed numbers.
"""

import argparse
import json
from pathlib import Path


def load_runs(runs_dir: Path, eval_name: str) -> dict[str, dict]:
    out = {}
    fname = eval_name.replace("/", "_") + ".json"
    for d in sorted(runs_dir.iterdir()):
        if d.name.endswith("-smoke") or not (d / fname).exists():
            continue
        out[d.name] = json.loads((d / fname).read_text())
    return out


def overall_table(runs: dict[str, dict]) -> str:
    lines = [
        "| run | AUROC | TPR@5%FPR | TPR@1%FPR | n |",
        "|---|---|---|---|---|",
    ]
    ranked = sorted(runs.items(), key=lambda kv: -kv[1]["metrics"]["overall"]["auroc"])
    for name, r in ranked:
        o = r["metrics"]["overall"]
        lines.append(
            f"| {name} | {o['auroc']:.3f} | {o['tpr_at_fpr_0.05']:.3f} | "
            f"{o['tpr_at_fpr_0.01']:.3f} | {r['n_eval']:,} |"
        )
    return "\n".join(lines)


def slice_table(runs: dict[str, dict], slice_col: str, metric: str = "tpr_at_fpr_0.05") -> str:
    all_values: list[str] = sorted(
        {v for r in runs.values() for v in r["metrics"]["slices"].get(slice_col, {})}
    )
    header = "| " + slice_col + " | " + " | ".join(runs) + " |"
    sep = "|---" * (len(runs) + 1) + "|"
    lines = [header, sep]
    for value in all_values:
        cells = []
        for r in runs.values():
            rep = r["metrics"]["slices"].get(slice_col, {}).get(value)
            cells.append(f"{rep[metric]:.3f}" if rep else "—")
        lines.append(f"| {value} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results/runs"))
    parser.add_argument("--eval", default="raid/eval")
    parser.add_argument("--slice", default=None, choices=["attack", "generator", "domain"])
    parser.add_argument("--metric", default="tpr_at_fpr_0.05")
    args = parser.parse_args()

    runs = load_runs(args.runs, args.eval)
    print(f"## Overall — {args.eval}\n")
    print(overall_table(runs))
    if args.slice:
        print(f"\n## {args.metric} by {args.slice} — {args.eval}\n")
        print(slice_table(runs, args.slice, args.metric))
