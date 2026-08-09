"""Carve leakage-safe project splits from RAID's labeled train set.

RAID's official test split is unlabeled (hidden leaderboard), so all local work uses the
labeled train split. Rows are linked: every adversarial row derives from a clean
generation (adv_source_id) which derives from a human source document (source_id).
Splitting naively would put a clean generation in train and its attacked variant in
eval. We therefore split by source_id hash:

- eval pool (5% of source_ids), then stratified-sample to a fixed-size eval set:
  up to 25 machine rows per (generator, domain, attack) cell and up to 75 human rows
  per (domain, attack) cell.
- train pool: everything from the remaining 95% of source_ids.

Outputs: data/processed/raid/eval.parquet, data/processed/raid/train_pool.parquet.
"""

import polars as pl

EVAL_FRACTION = 0.05
MACHINE_PER_CELL = 25
HUMAN_PER_CELL = 75
SEED = 0


def main() -> None:
    df = pl.read_parquet("data/processed/raid/train.parquet")
    in_eval = pl.col("source_id").hash(seed=SEED) % 100 < int(EVAL_FRACTION * 100)
    eval_pool = df.filter(in_eval)
    train_pool = df.filter(~in_eval)

    machine = (
        eval_pool.filter(pl.col("label") == 1)
        .with_columns(pl.int_range(pl.len()).shuffle(seed=SEED).alias("_r"))
        .sort("_r")
        .group_by("generator", "domain", "attack", maintain_order=True)
        .head(MACHINE_PER_CELL)
        .drop("_r")
    )
    human = (
        eval_pool.filter(pl.col("label") == 0)
        .with_columns(pl.int_range(pl.len()).shuffle(seed=SEED).alias("_r"))
        .sort("_r")
        .group_by("domain", "attack", maintain_order=True)
        .head(HUMAN_PER_CELL)
        .drop("_r")
    )
    # group_by moves key columns first; restore a common column order before concat
    eval_set = pl.concat([machine.select(df.columns), human.select(df.columns)])

    eval_set.write_parquet("data/processed/raid/eval.parquet")
    train_pool.write_parquet("data/processed/raid/train_pool.parquet")
    print(
        f"eval: {len(eval_set):,} rows ({len(machine):,} machine / {len(human):,} human), "
        f"train_pool: {len(train_pool):,} rows"
    )
    overlap = (
        eval_set.select("source_id")
        .unique()
        .join(train_pool.select("source_id").unique(), on="source_id", how="semi")
    )
    assert overlap.height == 0, "source_id leakage between eval and train_pool"
    print("source_id leakage check: OK")


if __name__ == "__main__":
    main()
