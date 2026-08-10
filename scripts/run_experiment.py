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


def _fast_detect_gpt(cfg: dict, seed: int):
    from detector.models.fast_detect_gpt import FastDetectGPT  # lazy: pulls in torch

    return FastDetectGPT(
        scorer=cfg.get("scorer", "EleutherAI/gpt-neo-2.7B"),
        sampler=cfg.get("sampler"),
        max_tokens=cfg.get("max_tokens", 512),
    )


def _encoder(cfg: dict, seed: int):
    from detector.models.encoder import EncoderDetector  # lazy: pulls in torch

    return EncoderDetector(
        model_name=cfg.get("model_name", "roberta-base"),
        max_tokens=cfg.get("max_tokens", 512),
        batch_size=cfg.get("batch_size", 8),
        grad_accum=cfg.get("grad_accum", 4),
        epochs=cfg.get("epochs", 1),
        lr=float(cfg.get("lr", 2e-5)),
        seed=seed,
    )


def _binoculars(cfg: dict, seed: int):
    from detector.models.binoculars import Binoculars  # lazy: pulls in torch

    return Binoculars(
        observer=cfg.get("observer", "Qwen/Qwen2.5-1.5B"),
        performer=cfg.get("performer", "Qwen/Qwen2.5-1.5B-Instruct"),
        max_tokens=cfg.get("max_tokens", 512),
        load_in_8bit=cfg.get("load_in_8bit", False),
    )


MODEL_TYPES = {
    "tfidf_logreg": lambda cfg, seed: TfidfLogReg(
        max_features=cfg.get("max_features", 100_000), seed=seed
    ),
    "stylometric_gbm": lambda cfg, seed: StylometricGBM(
        feature_subset=cfg.get("feature_subset"), seed=seed
    ),
    "fast_detect_gpt": _fast_detect_gpt,
    "binoculars": _binoculars,
    "encoder": _encoder,
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
    parser.add_argument(
        "--seed", type=int, default=None, help="override config seed (variance runs)"
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    cfg["seed"] = seed
    name = (
        cfg["name"]
        + (f"-seed{seed}" if args.seed is not None else "")
        + ("-smoke" if args.limit else "")
    )
    run_dir = args.out / name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = MODEL_TYPES[cfg["model"]["type"]](cfg["model"], seed)
    train_cfg = cfg.get("train")
    train_generators: set[str] = set()
    train_rows, train_seconds = 0, 0.0
    if train_cfg is not None:
        train = load_split(args.data, train_cfg["dataset"], train_cfg["split"])
        if train_cfg.get("filter_artifacts", True):
            before = len(train)
            train = drop_artifact_rows(train)
            print(f"artifact filter: dropped {before - len(train):,} machine rows")
        if args.limit:
            train = train.sample(min(args.limit, len(train)), seed=seed)
        t0 = time.time()
        model.fit(train["text"].to_list(), train["label"].to_numpy())
        train_seconds = time.time() - t0
        train_generators = set(train["generator"].unique().to_list())
        train_rows = len(train)
        print(f"trained {cfg['model']['type']} on {train_rows:,} rows in {train_seconds:.0f}s")
        if cfg.get("save_model") and hasattr(model, "save"):
            model.save(str(run_dir / "model"))
            print(f"saved model to {run_dir / 'model'}")

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    for ev in cfg["eval"]:
        df = load_split(args.data, ev["dataset"], ev["split"])
        if ev.get("sample"):
            df = df.sample(min(ev["sample"], len(df)), seed=seed)
        if args.limit:
            df = df.sample(min(args.limit * 2, len(df)), seed=seed)
        texts = df["text"].to_list()
        variant = ""
        if ev.get("normalize_unicode"):
            from detector.data.normalize import normalize_text

            texts = [normalize_text(t) for t in texts]
            variant = "_norm"
        scores = model.predict_scores(texts)
        scored = df.with_columns(pl.Series("score", np.asarray(scores)))
        if ev.get("save_scores"):
            scored.drop("text").write_parquet(
                run_dir / f"{ev['dataset']}_{ev['split']}{variant}_scores.parquet"
            )
        scored = scored.with_columns(
            (pl.col("generator").is_in(sorted(train_generators)) | (pl.col("label") == 0)).alias(
                "generator_seen"
            )
        )
        result = {
            "run": name,
            "config": cfg,
            "git_commit": commit,
            "train_rows": train_rows,
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
        out_path = run_dir / f"{ev['dataset']}_{ev['split']}{variant}.json"
        out_path.write_text(json.dumps(result, indent=2))
        o = result["metrics"]["overall"]
        print(
            f"{name} on {ev['dataset']}/{ev['split']}{variant}: "
            f"AUROC {o['auroc']:.4f} | TPR@5% {o['tpr_at_fpr_0.05']:.4f} | "
            f"TPR@1% {o['tpr_at_fpr_0.01']:.4f}  -> {out_path}"
        )


if __name__ == "__main__":
    main()
