"""Data-hygiene audits, run on every dataset before it is used for anything.

Motivated by documented failures in this literature:
- 98.5% of DetectRL's Claude split contains generation artifacts like
  "Sure! Here is..." (Dingfelder & Riess, 2025) — detectors learn the artifact.
- A 4M-parameter BERT-tiny reaches 0.967 AUROC on RAID (Thorat, 2026) — benchmark
  splits contain shortcuts.
- Length is a classic shortcut feature: if a length-only "detector" beats chance by a
  wide margin, the human/machine split is confounded.
"""

from __future__ import annotations

import polars as pl

from detector.evaluation.metrics import auroc

# Telltale strings that mark chat-assistant boilerplate or template leakage rather than
# organic text. Case-sensitive where casing is part of the artifact.
ARTIFACT_PATTERNS: dict[str, str] = {
    "sure_here": r"^\s*(Sure|Certainly|Of course)[!,.]? [Hh]ere('s| is| are)",
    "as_an_ai": r"[Aa]s an AI( language)? (model|assistant)",
    "i_cannot_assist": r"I (cannot|can't|am unable to) (assist|help|provide|fulfill)",
    "chatml_tokens": r"<\|im_(start|end)\|>|<\|endoftext\|>",
    "inst_tokens": r"\[/?INST\]|<<SYS>>",
    "knowledge_cutoff": r"[Kk]nowledge cutoff|[Aa]s of my (last|knowledge)",
    "heres_a": r"^\s*Here('s| is) (a|an|the|your)\b",
}


def artifact_report(df: pl.DataFrame, text_col: str = "text") -> pl.DataFrame:
    """Fraction of rows matching each artifact pattern, split by label and generator."""
    flags = [
        pl.col(text_col).str.contains(pattern).alias(name)
        for name, pattern in ARTIFACT_PATTERNS.items()
    ]
    out = (
        df.with_columns(flags)
        .group_by("generator", "label")
        .agg(
            pl.len().alias("n"),
            *[pl.col(name).mean().alias(name) for name in ARTIFACT_PATTERNS],
        )
        .with_columns(
            pl.max_horizontal(*ARTIFACT_PATTERNS).alias("worst_pattern_rate"),
        )
        .sort("worst_pattern_rate", descending=True)
    )
    return out


def _normalized(text_col: str = "text") -> pl.Expr:
    return pl.col(text_col).str.to_lowercase().str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()


def dedup_exact(df: pl.DataFrame, text_col: str = "text") -> tuple[pl.DataFrame, int]:
    """Drop rows whose whitespace/case/punctuation-normalized text already occurred.

    First occurrence wins. Returns (deduped_df, n_dropped). This catches exact and
    trivially-reformatted duplicates; near-duplicate (MinHash) detection is a separate,
    heavier pass used for cross-split leakage checks.
    """
    n0 = len(df)
    out = (
        df.with_columns(_normalized(text_col).hash().alias("_h"))
        .unique(subset="_h", keep="first", maintain_order=True)
        .drop("_h")
    )
    return out, n0 - len(out)


def cross_split_leakage(train: pl.DataFrame, test: pl.DataFrame, text_col: str = "text") -> int:
    """Number of test rows whose normalized text also appears in train."""
    train_h = train.select(_normalized(text_col).hash().alias("_h"))
    test_h = test.select(_normalized(text_col).hash().alias("_h"))
    return test_h.join(train_h.unique(), on="_h", how="semi").height


def length_shortcut_audit(df: pl.DataFrame, text_col: str = "text") -> dict[str, float]:
    """How much of a 'detector' is text length alone?

    Returns word-count stats per class and the AUROC of length as the score.
    AUROC near 0.5 = fine; substantially above/below = the split is length-confounded
    and length must be controlled (matching or stratification) before training.
    """
    words = df.with_columns(pl.col(text_col).str.split(" ").list.len().alias("_words"))
    machine = words.filter(pl.col("label") == 1)["_words"].to_numpy()
    human = words.filter(pl.col("label") == 0)["_words"].to_numpy()
    return {
        "auroc_length_only": auroc(machine, human),
        "median_words_machine": float(pl.Series(machine).median()),
        "median_words_human": float(pl.Series(human).median()),
    }
