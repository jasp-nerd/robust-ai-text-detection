import numpy as np
import pytest
from sklearn.metrics import roc_curve

from detector.evaluation import auroc, detection_report, threshold_at_fpr, tpr_at_fpr


def test_perfect_separation() -> None:
    machine = np.array([0.9, 0.8, 0.95])
    human = np.array([0.1, 0.2, 0.05])
    assert auroc(machine, human) == 1.0
    assert tpr_at_fpr(machine, human, 0.05) == 1.0


def test_no_separation() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=2000)
    assert abs(auroc(scores[:1000], scores[1000:]) - 0.5) < 0.05


def test_threshold_respects_fpr_budget() -> None:
    rng = np.random.default_rng(1)
    human = rng.normal(size=10_000)
    for target in (0.05, 0.01):
        t = threshold_at_fpr(human, target)
        achieved = float(np.mean(human >= t))
        assert achieved <= target


def test_tpr_at_fpr_matches_sklearn_roc() -> None:
    """Cross-check against sklearn's ROC curve on a non-trivial overlap case."""
    rng = np.random.default_rng(2)
    human = rng.normal(0.0, 1.0, size=5000)
    machine = rng.normal(1.5, 1.0, size=5000)
    y = np.concatenate([np.zeros(5000), np.ones(5000)])
    s = np.concatenate([human, machine])
    fpr, tpr, _ = roc_curve(y, s)
    ours = tpr_at_fpr(machine, human, 0.01)
    # sklearn interpolation point at fpr<=0.01
    sk = tpr[fpr <= 0.01].max()
    assert abs(ours - sk) < 0.02


def test_report_keys() -> None:
    r = detection_report([0.9], [0.1, 0.2])
    assert set(r) == {"auroc", "tpr_at_fpr_0.05", "tpr_at_fpr_0.01", "n_machine", "n_human"}


def test_empty_inputs_raise() -> None:
    with pytest.raises(ValueError):
        threshold_at_fpr([], 0.05)
    with pytest.raises(ValueError):
        tpr_at_fpr([], [0.1], 0.05)
