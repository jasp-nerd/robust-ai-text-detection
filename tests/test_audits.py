import polars as pl

from detector.data.audits import (
    artifact_report,
    cross_split_leakage,
    dedup_exact,
    length_shortcut_audit,
)


def _df(rows: list[dict]) -> pl.DataFrame:
    base = {"generator": "g", "label": 1, "domain": "d"}
    return pl.DataFrame([{**base, **r} for r in rows])


def test_artifact_report_catches_boilerplate() -> None:
    df = _df(
        [
            {"text": "Sure! Here is the essay you asked for."},
            {"text": "As an AI language model, I cannot feel."},
            {"text": "A perfectly organic sentence about frogs."},
        ]
    )
    rep = artifact_report(df)
    assert rep["worst_pattern_rate"][0] > 0


def test_dedup_exact_normalizes() -> None:
    df = _df(
        [
            {"text": "Hello,   World!"},
            {"text": "hello world"},
            {"text": "something else"},
        ]
    )
    out, dropped = dedup_exact(df)
    assert dropped == 1
    assert len(out) == 2


def test_cross_split_leakage() -> None:
    train = _df([{"text": "shared sample"}, {"text": "train only"}])
    test = _df([{"text": "Shared   sample"}, {"text": "test only"}])
    assert cross_split_leakage(train, test) == 1


def test_length_shortcut_audit_flags_confounded_split() -> None:
    rows = [{"text": "long " * 200, "label": 1} for _ in range(50)]
    rows += [{"text": "short text", "label": 0} for _ in range(50)]
    stats = length_shortcut_audit(_df(rows))
    assert stats["auroc_length_only"] > 0.95  # deliberately confounded

    balanced = [{"text": "same length here", "label": i % 2} for i in range(100)]
    stats2 = length_shortcut_audit(_df(balanced))
    assert abs(stats2["auroc_length_only"] - 0.5) < 0.05
