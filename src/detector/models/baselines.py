"""Interpretable baseline detectors.

Both expose the same minimal interface: ``fit(texts, labels)`` and
``predict_scores(texts) -> np.ndarray`` where higher = more likely machine-generated.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from detector.features.stylometric import FEATURE_NAMES, extract_matrix


class TfidfLogReg:
    """TF-IDF (word 1-2 grams) + logistic regression. The classic strong-simple baseline."""

    def __init__(self, max_features: int = 100_000, seed: int = 0):
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=(1, 2),
                        min_df=5,
                        sublinear_tf=True,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
            ]
        )

    def fit(self, texts: list[str], labels: np.ndarray) -> "TfidfLogReg":
        self.pipeline.fit(texts, labels)
        return self

    def predict_scores(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)[:, 1]


class StylometricGBM:
    """Gradient boosting over interpretable stylometric features.

    ``feature_subset`` allows ablations, e.g. lexical-richness-only.
    """

    def __init__(self, feature_subset: list[str] | None = None, seed: int = 0):
        self.feature_names = feature_subset or FEATURE_NAMES
        self._idx = [FEATURE_NAMES.index(f) for f in self.feature_names]
        self.clf = HistGradientBoostingClassifier(random_state=seed)

    def fit(self, texts: list[str], labels: np.ndarray) -> "StylometricGBM":
        x = extract_matrix(texts)[:, self._idx]
        self.clf.fit(x, labels)
        return self

    def predict_scores(self, texts: list[str]) -> np.ndarray:
        x = extract_matrix(texts)[:, self._idx]
        return self.clf.predict_proba(x)[:, 1]
