"""Build the curated training mixture (Phase 3d, following MELD's data-first finding).

MAGE train (generator diversity) + a stratified sample of RAID train_pool
(decoding/attack diversity), exact-deduped, artifact-filterable downstream.
Writes data/processed/mix1/train.parquet.

Honesty note: once RAID train_pool enters training, our RAID eval is no longer
out-of-distribution (same generators/domains, disjoint source documents). The
mixture run therefore tests "does attack/decoding exposure help under attack",
not cross-dataset generalization — HC3 remains the only fully-OOD eval.
"""

import pathlib

import polars as pl

from detector.data.loaders import UNIFIED_COLUMNS

RAID_PER_CELL = 40  # per (generator, domain, attack) cell
SEED = 0

mage = pl.read_parquet("data/processed/mage/train.parquet")
pool = pl.read_parquet("data/processed/raid/train_pool.parquet")

raid_machine = (
    pool.filter(pl.col("label") == 1)
    .with_columns(pl.int_range(pl.len()).shuffle(seed=SEED).alias("_r"))
    .sort("_r")
    .group_by("generator", "domain", "attack", maintain_order=True)
    .head(RAID_PER_CELL)
    .drop("_r")
    .select(UNIFIED_COLUMNS)
)
raid_human = (
    pool.filter(pl.col("label") == 0)
    .with_columns(pl.int_range(pl.len()).shuffle(seed=SEED).alias("_r"))
    .sort("_r")
    .group_by("domain", "attack", maintain_order=True)
    .head(RAID_PER_CELL * 3)
    .drop("_r")
    .select(UNIFIED_COLUMNS)
)
mix = pl.concat([mage.select(UNIFIED_COLUMNS), raid_machine, raid_human])
n0 = len(mix)
mix = (
    mix.with_columns(
        pl.col("text")
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9]+", " ")
        .str.strip_chars()
        .hash()
        .alias("_h")
    )
    .unique(subset="_h", keep="first", maintain_order=True)
    .drop("_h")
)
out = "data/processed/mix1/train.parquet"
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
mix.write_parquet(out)
print(
    f"mix1: {len(mix):,} rows ({n0 - len(mix):,} dups dropped) | machine {mix['label'].mean():.2%}"
)
