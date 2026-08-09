"""Train a detector from a YAML config, evaluate on the configured eval sets, and write
one JSON artifact per eval set under results/runs/<name>/.

Usage: uv run python scripts/run_experiment.py configs/tfidf_logreg_mage.yaml [--limit 2000]

--limit subsamples both train and eval for a fast smoke run (artifacts are then written
to results/runs/<name>-smoke/ so real results are never overwritten).
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from detector.data.audits import ARTIFACT_PATTERNS
from detector.evaluation.harness import calibration_transfer, evaluate_slices
from detector.models.baselines import StylometricGBM, TfidfLogReg

MODEL_TYPES = {
    "tfidf_logreg": lambda cfg, seed: TfidfLogReg(
        max_features=cfg.get("max_features", 100_000), seed=seed
    ),
    "stylometric_gbm": lambda cfg, seed: StylometricGBM(
        feature_subset=cfg.get("feature_subset"), seed=seed
    ),
}


def load_split(data_dir: Path, dataset: str, split: str) -> pl.DataFrame:
    return pl.read_parquet(data_dir / dataset / f"{split}.parquet")


def drop_artifact_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Remove machine rows carrying chat-assistant boilerplate (trivial shortcuts)."""
    pattern = "|".join(f"(?:{p})" for p in ARTIFACT_PATTERNS.values())
    return df.filter(~((pl.col("label") == 1) & pl.col("text").str.contains(pattern)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("results/runs"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed = cfg.get("seed", 0)
    name = cfg["name"] + ("-smoke" if args.limit else "")
    run_dir = args.out / name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg["train"]
    train = load_split(args.data, train_cfg["dataset"], train_cfg["split"])
    if train_cfg.get("filter_artifacts", True):
        before = len(train)
        train = drop_artifact_rows(train)
        print(f"artifact filter: dropped {before - len(train):,} machine rows")
    if args.limit:
        train = train.sample(min(args.limit, len(train)), seed=seed)

    model = MODEL_TYPES[cfg["model"]["type"]](cfg["model"], seed)
    t0 = time.time()
    model.fit(train["text"].to_list(), train["label"].to_numpy())
    train_seconds = time.time() - t0
    print(f"trained {cfg['model']['type']} on {len(train):,} rows in {train_seconds:.0f}s")

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    train_generators = set(train["generator"].unique().to_list())
    for ev in cfg["eval"]:
        df = load_split(args.data, ev["dataset"], ev["split"])
        if args.limit:
            df = df.sample(min(args.limit * 2, len(df)), seed=seed)
        scores = model.predict_scores(df["text"].to_list())
        scored = df.with_columns(pl.Series("score", np.asarray(scores)))
        scored = scored.with_columns(
            (
                pl.col("generator").is_in(sorted(train_generators))
                | (pl.col("label") == 0)
            ).alias("generator_seen")
        )
        result = {
            "run": name,
            "config": cfg,
            "git_commit": commit,
            "train_rows": len(train),
            "train_seconds": round(train_seconds, 1),
            "eval": f"{ev['dataset']}/{ev['split']}",
            "n_eval": len(scored),
            "metrics": evaluate_slices(scored),
            "seen_vs_unseen": {
                label: evaluate_slices(sub, slice_cols=())["overall"]
                for label, sub in [
                    ("seen", scored.filter(pl.col("generator_seen"))),
                    ("unseen", scored.filter(~pl.col("generator_seen"))),
                ]
                if sub.filter(pl.col("label") == 1).height > 0
            },
            "calibration_transfer": calibration_transfer(scored).to_dicts(),
        }
        out_path = run_dir / f"{ev['dataset']}_{ev['split']}.json"
        out_path.write_text(json.dumps(result, indent=2))
        o = result["metrics"]["overall"]
        print(
            f"{name} on {ev['dataset']}/{ev['split']}: "
            f"AUROC {o['auroc']:.4f} | TPR@5% {o['tpr_at_fpr_0.05']:.4f} | "
            f"TPR@1% {o['tpr_at_fpr_0.01']:.4f}  -> {out_path}"
        )


if __name__ == "__main__":
    main()
