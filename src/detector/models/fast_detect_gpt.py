"""Fast-DetectGPT (Bao et al., ICLR 2024) — conditional probability curvature, analytic form.

Score: d(x) = (sum_j ll_j - sum_j mu_j) / sqrt(sum_j var_j), where ll_j is the scoring
model's log-prob of the observed token and mu_j/var_j are the mean/variance of the
log-prob under alternative tokens drawn from the sampling model's distribution at the
same position (App. B closed form — no sampling needed when expectations are computed
over the full vocabulary).

Higher score = more likely machine-generated (machine text sits at high curvature).

Configs used in this project:
- dev (Mac/MPS): sampler == scorer == EleutherAI/gpt-neo-2.7B (the paper's "sampling
  model = scoring model" variant, Table 9 of the paper).
- faithful black-box (GPU): sampler EleutherAI/gpt-j-6B, scorer EleutherAI/gpt-neo-2.7B
  (vocab-aligned by truncation to the shorter vocabulary).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.functional import log_softmax
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FastDetectGPT:
    def __init__(
        self,
        scorer: str = "EleutherAI/gpt-neo-2.7B",
        sampler: str | None = None,
        max_tokens: int = 512,
        device: str | None = None,
    ):
        self.device = device or best_device()
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(scorer)
        self.score_model = AutoModelForCausalLM.from_pretrained(scorer, torch_dtype=dtype)
        self.score_model.to(self.device).eval()
        self.sample_model = None
        if sampler and sampler != scorer:
            self.sample_model = AutoModelForCausalLM.from_pretrained(sampler, torch_dtype=dtype)
            self.sample_model.to(self.device).eval()

    def fit(self, texts: list[str], labels: np.ndarray) -> "FastDetectGPT":
        return self  # zero-shot: nothing to fit

    @torch.no_grad()
    def _score_one(self, text: str) -> float:
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_tokens
        ).to(self.device)
        ids = enc.input_ids
        if ids.shape[1] < 2:
            return 0.0
        score_logits = self.score_model(**enc).logits[0, :-1].float()
        if self.sample_model is not None:
            sample_logits = self.sample_model(**enc).logits[0, :-1].float()
            v = min(score_logits.shape[-1], sample_logits.shape[-1])
            score_logits, sample_logits = score_logits[:, :v], sample_logits[:, :v]
        else:
            sample_logits = score_logits
        log_p_score = log_softmax(score_logits, dim=-1)
        p_sample = torch.softmax(sample_logits, dim=-1)
        targets = ids[0, 1:].clamp(max=log_p_score.shape[-1] - 1)
        ll = log_p_score.gather(1, targets.unsqueeze(1)).squeeze(1)
        mu = (p_sample * log_p_score).sum(-1)
        var = (p_sample * log_p_score.square()).sum(-1) - mu.square()
        denom = var.sum().clamp(min=1e-8).sqrt()
        return float((ll.sum() - mu.sum()) / denom)

    def predict_scores(self, texts: list[str]) -> np.ndarray:
        return np.array([self._score_one(t) for t in tqdm(texts, desc="fast-detect-gpt")])
