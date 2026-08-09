"""Evaluation harness: every detector is scored once per dataset, then this module
slices the scored frame into the project's standard reporting conditions.

Input contract: a polars DataFrame with the unified schema columns plus a ``score``
column where higher = more likely machine-generated.

Standard conditions:
- overall
- per generator, per domain, per attack (clean vs each attack)
- calibration transfer: threshold calibrated on one domain's human text, applied to
  every other domain — the deployment failure mode identified by Shen et al. (2026)
  and MAGE's own threshold analysis.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from detector.evaluation.metrics import detection_report, threshold_at_fpr


def _report(df: pl.DataFrame, fprs: tuple[float, ...]) -> dict[str, float] | None:
    machine = df.filter(pl.col("label") == 1)["score"].to_numpy()
    human = df.filter(pl.col("label") == 0)["score"].to_numpy()
    if machine.size == 0 or human.size == 0:
        return None
    return detection_report(machine, human, fprs)


def evaluate_slices(
    scored: pl.DataFrame,
    fprs: tuple[float, ...] = (0.05, 0.01),
    slice_cols: tuple[str, ...] = ("generator", "domain", "attack"),
) -> dict:
    """Overall metrics plus per-slice metrics.

    Per-slice machine scores are always compared against ALL human scores in the frame
    (matching RAID's protocol: the human negatives are shared across generator/attack
    slices; domain slices use the domain's own humans when available).
    """
    out: dict = {"overall": _report(scored, fprs), "slices": {}}
    human_all = scored.filter(pl.col("label") == 0)
    for col in slice_cols:
        out["slices"][col] = {}
        for (value,), group in sorted(
            scored.filter(pl.col("label") == 1).group_by(col), key=lambda kv: str(kv[0][0])
        ):
            humans = (
                human_all.filter(pl.col(col) == value)
                if col == "domain" and human_all.filter(pl.col(col) == value).height > 0
                else human_all
            )
            rep = _report(pl.concat([group, humans]), fprs)
            if rep is not None:
                out["slices"][col][str(value)] = rep
    return out


def calibration_transfer(scored: pl.DataFrame, target_fpr: float = 0.05) -> pl.DataFrame:
    """Calibrate the threshold on each domain's human text, then measure the achieved
    FPR and TPR on every other domain. Returns a tidy frame with one row per
    (calibration_domain, eval_domain)."""
    domains = sorted(scored["domain"].unique().to_list())
    rows = []
    for cal in domains:
        cal_human = scored.filter((pl.col("domain") == cal) & (pl.col("label") == 0))
        if cal_human.height < 50:
            continue
        t = threshold_at_fpr(cal_human["score"].to_numpy(), target_fpr)
        for ev in domains:
            sub = scored.filter(pl.col("domain") == ev)
            human = sub.filter(pl.col("label") == 0)["score"].to_numpy()
            machine = sub.filter(pl.col("label") == 1)["score"].to_numpy()
            if human.size < 50 or machine.size < 50:
                continue
            rows.append(
                {
                    "calibration_domain": cal,
                    "eval_domain": ev,
                    "target_fpr": target_fpr,
                    "achieved_fpr": float(np.mean(human >= t)),
                    "tpr": float(np.mean(machine >= t)),
                }
            )
    return pl.DataFrame(rows)
