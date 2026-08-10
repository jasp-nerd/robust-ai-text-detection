"""Phase 3d: ensemble analysis from dumped per-sample scores.

Combines the supervised (ModernBERT) and zero-shot (Fast-DetectGPT) score files on the
RAID eval set. Their failure modes are complementary (RESEARCH_LOG 2026-08-09), so a
rank-based combination should beat both. Rank-averaging avoids scale mismatch between
probability outputs and unbounded curvature scores; no learned weights (nothing to
overfit, nothing to tune).

Usage:
  uv run python scripts/analyze_ensemble.py \
      results/runs/modernbert-defense/raid_eval_scores.parquet \
      results/runs/fdg-neo-defense/raid_eval_scores.parquet

Writes results/runs/ensemble-rankavg/<basename>.json in the standard artifact format.
"""

import argparse
import json
from pathlib import Path

import polars as pl
from scipy.stats import rankdata

from detector.evaluation.harness import evaluate_slices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score_files", nargs=2, type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/runs/ensemble-rankavg"))
    args = parser.parse_args()

    a = pl.read_parquet(args.score_files[0])
    b = pl.read_parquet(args.score_files[1])
    if len(a) != len(b):
        raise SystemExit(f"row count mismatch: {len(a)} vs {len(b)}")
    if "id" in a.columns and "id" in b.columns:
        b = b.rename({"score": "score_b"}).select("id", "score_b")
        merged = a.join(b, on="id", how="inner")
        assert len(merged) == len(a), "id join lost rows"
    else:  # positional alignment (same eval parquet + same seed => same order)
        merged = a.with_columns(pl.Series("score_b", b["score"]))

    ranks = (
        rankdata(merged["score"].to_numpy()) + rankdata(merged["score_b"].to_numpy())
    ) / (2 * len(merged))
    scored = merged.drop("score", "score_b").with_columns(pl.Series("score", ranks))

    result = {
        "run": "ensemble-rankavg",
        "components": [str(p) for p in args.score_files],
        "n_eval": len(scored),
        "metrics": evaluate_slices(scored),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / (args.score_files[0].stem.replace("_scores", "") + ".json")
    out_path.write_text(json.dumps(result, indent=2))
    o = result["metrics"]["overall"]
    print(
        f"ensemble-rankavg: AUROC {o['auroc']:.4f} | TPR@5% {o['tpr_at_fpr_0.05']:.4f} | "
        f"TPR@1% {o['tpr_at_fpr_0.01']:.4f}  -> {out_path}"
    )


if __name__ == "__main__":
    main()
