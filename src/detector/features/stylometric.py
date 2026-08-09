"""Interpretable stylometric features.

The feature set is deliberately small and centered on the only feature group shown to be
robust across 27 generators and 10 domains (El Attar et al., 2026): lexical richness —
type-token ratio, hapax proportion, and lexical density. We approximate lexical density
(content-word fraction) with a stopword-based proxy to avoid a POS-tagger dependency;
the approximation is documented in the write-up.

Secondary features (surface statistics, punctuation, character entropy) are included so
their contribution can be ablated — the same study found several feature groups *hurt*
out-of-distribution generalization, and we want to reproduce that.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[.!?]+")

LEXICAL_RICHNESS = ["type_token_ratio", "hapax_proportion", "lexical_density"]
SECONDARY = [
    "mean_word_length",
    "mean_sentence_words",
    "std_sentence_words",
    "comma_rate",
    "punct_rate",
    "uppercase_rate",
    "char_entropy",
    "stopword_rate",
]
FEATURE_NAMES = LEXICAL_RICHNESS + SECONDARY


def extract_features(text: str) -> np.ndarray:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    n_words = len(words)
    if n_words == 0:
        return np.zeros(len(FEATURE_NAMES))
    counts = Counter(words)
    sentences = [s for s in _SENT_RE.split(text) if s.strip()]
    sent_lens = [len(_WORD_RE.findall(s)) for s in sentences] or [n_words]
    n_chars = len(text)
    char_counts = Counter(text)
    char_entropy = -sum(
        (c / n_chars) * math.log2(c / n_chars) for c in char_counts.values()
    )
    n_stop = sum(1 for w in words if w in ENGLISH_STOP_WORDS)
    values = {
        "type_token_ratio": len(counts) / n_words,
        "hapax_proportion": sum(1 for c in counts.values() if c == 1) / n_words,
        # proxy: content words ≈ non-stopwords
        "lexical_density": 1.0 - n_stop / n_words,
        "mean_word_length": float(np.mean([len(w) for w in words])),
        "mean_sentence_words": float(np.mean(sent_lens)),
        "std_sentence_words": float(np.std(sent_lens)),
        "comma_rate": text.count(",") / n_words,
        "punct_rate": sum(1 for ch in text if ch in ".,;:!?—–-()\"'") / n_chars,
        "uppercase_rate": sum(1 for ch in text if ch.isupper()) / n_chars,
        "char_entropy": char_entropy,
        "stopword_rate": n_stop / n_words,
    }
    return np.array([values[name] for name in FEATURE_NAMES])


def extract_matrix(texts: list[str]) -> np.ndarray:
    return np.stack([extract_features(t) for t in texts])
