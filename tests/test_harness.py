import numpy as np
import polars as pl

from detector.evaluation.harness import calibration_transfer, evaluate_slices


def _scored(n: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    label = np.repeat([0, 1], n // 2)
    score = np.where(label == 1, rng.normal(2, 1, n), rng.normal(0, 1, n))
    return pl.DataFrame(
        {
            "text": ["x"] * n,
            "label": label,
            "score": score,
            "generator": np.where(label == 1, rng.choice(["gpt", "llama"], n), "human"),
            "domain": rng.choice(["news", "reddit"], n),
            "attack": np.where(label == 1, rng.choice(["none", "paraphrase"], n), "none"),
            "decoding": ["unknown"] * n,
            "source_dataset": ["synthetic"] * n,
        }
    )


def test_evaluate_slices_shapes() -> None:
    out = evaluate_slices(_scored())
    assert out["overall"]["auroc"] > 0.8
    assert set(out["slices"]) == {"generator", "domain", "attack"}
    assert "gpt" in out["slices"]["generator"]
    assert "human" not in out["slices"]["generator"]


def test_calibration_transfer_tidy() -> None:
    df = calibration_transfer(_scored(2000))
    assert set(df.columns) == {
        "calibration_domain",
        "eval_domain",
        "target_fpr",
        "achieved_fpr",
        "tpr",
    }
    # same-domain calibration must respect the FPR budget on its own domain
    same = df.filter(pl.col("calibration_domain") == pl.col("eval_domain"))
    assert (same["achieved_fpr"] <= same["target_fpr"] + 1e-9).all()
