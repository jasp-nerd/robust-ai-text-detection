"""Dataset download + normalization to the unified schema.

Each ``prepare_*`` function downloads a dataset from the Hugging Face Hub and writes
normalized parquet files under ``data/processed/<name>/<split>.parquet`` with exactly the
unified columns (see `detector.data.schema.Sample`). Raw HF caches live wherever
``datasets`` puts them; everything downstream reads only the normalized parquet.

Label convention everywhere: 1 = machine-generated, 0 = human-written.
(Careful: MAGE uses the opposite convention upstream — 1 = human. We flip it here.)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from datasets import load_dataset

UNIFIED_COLUMNS = [
    "text",
    "label",
    "generator",
    "domain",
    "attack",
    "decoding",
    "source_dataset",
]


def _write(
    df: pl.DataFrame, out_dir: Path, name: str, split: str, extra_cols: list[str] | None = None
) -> Path:
    df = df.select(UNIFIED_COLUMNS + (extra_cols or []))
    path = out_dir / name / f"{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def parse_mage_src(src: str) -> tuple[str, str]:
    """Split a MAGE ``src`` field into (domain, generator).

    Human rows look like ``cmv_human``; machine rows like
    ``cmv_machine_specified_gpt-3.5-turbo`` or ``xsum_machine_continuation_opt-13b``.
    """
    if src.endswith("_human"):
        return src.removesuffix("_human"), "human"
    if "_machine_" in src:
        domain, rest = src.split("_machine_", 1)
        # rest = "<prompt_type>_<model>"; prompt types are specified/continuation/topical
        parts = rest.split("_", 1)
        generator = parts[1] if len(parts) == 2 else rest
        return domain, generator
    return src, "unknown"


def prepare_mage(out_dir: Path) -> list[Path]:
    ds = load_dataset("yaful/MAGE")
    paths = []
    for split in ds:
        df = pl.from_arrow(ds[split].data.table)
        parsed = [parse_mage_src(s) for s in df["src"]]
        df = df.with_columns(
            pl.Series("domain", [p[0] for p in parsed]),
            pl.Series("generator", [p[1] for p in parsed]),
            # MAGE: label 1 = human. Ours: 1 = machine.
            (1 - pl.col("label")).alias("label"),
            pl.lit("none").alias("attack"),
            pl.lit("unknown").alias("decoding"),
            pl.lit("mage").alias("source_dataset"),
        )
        paths.append(_write(df, out_dir, "mage", split))
    return paths


def prepare_raid(out_dir: Path) -> list[Path]:
    """RAID train split (the test split is unlabeled; eval goes via the leaderboard)."""
    ds = load_dataset("liamdugan/raid", "raid")
    df = pl.from_arrow(ds["train"].data.table)
    df = df.rename({"generation": "text", "model": "generator"}).with_columns(
        (pl.col("generator") != "human").cast(pl.Int64).alias("label"),
        pl.col("attack").fill_null("none"),
        pl.col("decoding").fill_null("unknown").replace({"None": "unknown"}),
        pl.lit("raid").alias("source_dataset"),
    )
    # id linkage kept for leakage-safe splitting (see scripts/make_raid_splits.py)
    return [_write(df, out_dir, "raid", "train", extra_cols=["id", "source_id", "adv_source_id"])]


def prepare_hc3(out_dir: Path) -> list[Path]:
    """HC3 English. The original repo uses a deprecated loading script, so we read the
    auto-converted parquet branch. Each row has one question with lists of human and
    ChatGPT answers; we explode to one text per row."""
    ds = load_dataset(
        "parquet",
        data_files="hf://datasets/Hello-SimpleAI/HC3@refs/convert/parquet/all/train/0000.parquet",
    )
    rows: dict[str, list] = {c: [] for c in UNIFIED_COLUMNS}
    for row in ds["train"]:
        for answer, label, generator in [
            *[(a, 0, "human") for a in row["human_answers"] or []],
            *[(a, 1, "chatgpt") for a in row["chatgpt_answers"] or []],
        ]:
            if not answer or not answer.strip():
                continue
            rows["text"].append(answer)
            rows["label"].append(label)
            rows["generator"].append(generator)
            rows["domain"].append(row["source"])
            rows["attack"].append("none")
            rows["decoding"].append("unknown")
            rows["source_dataset"].append("hc3")
    return [_write(pl.DataFrame(rows), out_dir, "hc3", "test")]


PREPARERS = {
    "mage": prepare_mage,
    "raid": prepare_raid,
    "hc3": prepare_hc3,
}
