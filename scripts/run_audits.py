"""Run data-hygiene audits on normalized datasets; write JSON artifacts to results/audits/.

Usage: uv run python scripts/run_audits.py mage hc3
"""

import argparse
import json
from pathlib import Path

import polars as pl

from detector.data.audits import (
    artifact_report,
    cross_split_leakage,
    dedup_exact,
    length_shortcut_audit,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("results/audits"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        splits = {p.stem: pl.read_parquet(p) for p in sorted((args.data / name).glob("*.parquet"))}
        report: dict = {"dataset": name, "splits": {}}
        for split, df in splits.items():
            _, n_dup = dedup_exact(df)
            worst = artifact_report(df).head(5)
            report["splits"][split] = {
                "rows": len(df),
                "machine_fraction": float(df["label"].mean()),
                "exact_duplicates": n_dup,
                "length_audit": length_shortcut_audit(df),
                "worst_artifact_offenders": worst.to_dicts(),
            }
        if "train" in splits and "test" in splits:
            report["train_test_leakage_rows"] = cross_split_leakage(splits["train"], splits["test"])
        out_path = args.out / f"{name}.json"
        out_path.write_text(json.dumps(report, indent=2))
        print(f"wrote {out_path}")
        for split, r in report["splits"].items():
            la = r["length_audit"]
            print(
                f"  {split}: {r['rows']:,} rows, dups={r['exact_duplicates']:,}, "
                f"len-AUROC={la['auroc_length_only']:.3f} "
                f"(median words M/H: {la['median_words_machine']:.0f}/{la['median_words_human']:.0f})"
            )
        if "train_test_leakage_rows" in report:
            print(f"  train->test leakage: {report['train_test_leakage_rows']} rows")
