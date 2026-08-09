import numpy as np

from detector.features.stylometric import FEATURE_NAMES, extract_features
from detector.models.baselines import StylometricGBM, TfidfLogReg


def test_feature_vector_shape_and_ranges() -> None:
    v = extract_features("The quick brown fox jumps over the lazy dog. It was quick.")
    assert v.shape == (len(FEATURE_NAMES),)
    ttr = v[FEATURE_NAMES.index("type_token_ratio")]
    assert 0 < ttr <= 1


def test_empty_text_is_zero_vector() -> None:
    assert extract_features("").sum() == 0


def _toy_corpus(n: int = 200) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(0)
    human = ["short human note " + " ".join(rng.choice(list("abcdef"), 5)) for _ in range(n // 2)]
    machine = [
        "Furthermore, it is important to note that the aforementioned considerations "
        "demonstrate significant implications. " * 3
        for _ in range(n // 2)
    ]
    texts = human + machine
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    return texts, labels


def test_baselines_learn_separable_toy_data() -> None:
    texts, labels = _toy_corpus()
    for model in [TfidfLogReg(max_features=1000), StylometricGBM()]:
        model.fit(texts, labels)
        scores = model.predict_scores(texts)
        assert scores[labels == 1].mean() > scores[labels == 0].mean()
