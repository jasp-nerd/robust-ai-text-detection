"""Detection metrics.

The project's primary metric is TPR @ k% FPR: the threshold is chosen on human-written
(negative) scores so that at most k% of human texts are flagged, and we report the fraction
of machine-generated (positive) texts caught at that threshold. This follows RAID
(Dugan et al., 2024): AUROC alone hides what happens in the low-FPR regime where
real deployments must operate.

Convention everywhere in this package: higher score = more likely machine-generated;
label 1 = machine-generated, label 0 = human-written.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score


def threshold_at_fpr(human_scores: ArrayLike, target_fpr: float) -> float:
    """Smallest threshold t such that P(human_score >= t) <= target_fpr.

    Computed as an empirical quantile of the human (negative) score distribution.
    """
    scores = np.asarray(human_scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("need at least one human score to calibrate a threshold")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
    # At most floor(target_fpr * n) human scores may sit at or above the threshold.
    # Descending order; the threshold is nudged just above the (k+1)-th largest score so
    # the >= comparison downstream can never exceed the budget, ties included.
    desc = np.sort(scores)[::-1]
    k = int(np.floor(target_fpr * scores.size))
    return float(np.nextafter(desc[k], np.inf))


def tpr_at_fpr(
    machine_scores: ArrayLike,
    human_scores: ArrayLike,
    target_fpr: float = 0.01,
) -> float:
    """Fraction of machine texts scoring >= the threshold calibrated at target_fpr."""
    machine = np.asarray(machine_scores, dtype=np.float64)
    if machine.size == 0:
        raise ValueError("need at least one machine score")
    t = threshold_at_fpr(human_scores, target_fpr)
    return float(np.mean(machine >= t))


def auroc(machine_scores: ArrayLike, human_scores: ArrayLike) -> float:
    machine = np.asarray(machine_scores, dtype=np.float64)
    human = np.asarray(human_scores, dtype=np.float64)
    y = np.concatenate([np.ones_like(machine), np.zeros_like(human)])
    s = np.concatenate([machine, human])
    return float(roc_auc_score(y, s))


def detection_report(
    machine_scores: ArrayLike,
    human_scores: ArrayLike,
    fprs: tuple[float, ...] = (0.05, 0.01),
) -> dict[str, float]:
    """The standard metric bundle reported for every detector and condition."""
    report = {"auroc": auroc(machine_scores, human_scores)}
    for fpr in fprs:
        report[f"tpr_at_fpr_{fpr:g}"] = tpr_at_fpr(machine_scores, human_scores, fpr)
    report["n_machine"] = float(np.asarray(machine_scores).size)
    report["n_human"] = float(np.asarray(human_scores).size)
    return report
