"""Binoculars (Hans et al., ICML 2024) — perplexity / cross-perplexity ratio.

B(s) = log-PPL_performer(s) / X-PPL(observer, performer, s), where X-PPL is the average
per-token cross-entropy between the observer's next-token distribution and the
performer's, on the same tokenization. Lower B = more likely machine-generated; we
return the NEGATED score so that the project convention (higher = machine) holds.

The observer/performer pair must be a closely matched base/instruct pair sharing a
tokenizer. Paper pair: tiiuae/falcon-7b + tiiuae/falcon-7b-instruct (needs ~29GB bf16 —
GPU server, or 8-bit). Mac dev pair: Qwen/Qwen2.5-1.5B + -Instruct (~6GB fp16).
Published thresholds are pair-specific; we always recalibrate on human text via the
project harness, so no fixed threshold is used.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.functional import cross_entropy, log_softmax, softmax
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from detector.models.fast_detect_gpt import best_device


class Binoculars:
    def __init__(
        self,
        observer: str = "Qwen/Qwen2.5-1.5B",
        performer: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_tokens: int = 512,
        device: str | None = None,
    ):
        self.device = device or best_device()
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(observer)
        perf_tok = AutoTokenizer.from_pretrained(performer)
        if self.tokenizer.get_vocab() != perf_tok.get_vocab():
            raise ValueError(
                "observer and performer must share a tokenizer (Binoculars requirement)"
            )
        self.observer = AutoModelForCausalLM.from_pretrained(observer, torch_dtype=dtype)
        self.performer = AutoModelForCausalLM.from_pretrained(performer, torch_dtype=dtype)
        self.observer.to(self.device).eval()
        self.performer.to(self.device).eval()

    def fit(self, texts: list[str], labels: np.ndarray) -> Binoculars:
        return self  # zero-shot

    @torch.no_grad()
    def _score_one(self, text: str) -> float:
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_tokens
        ).to(self.device)
        ids = enc.input_ids
        if ids.shape[1] < 2:
            return 0.0
        obs_logits = self.observer(**enc).logits[0, :-1].float()
        perf_logits = self.performer(**enc).logits[0, :-1].float()
        targets = ids[0, 1:]
        # log perplexity of the text under the performer
        log_ppl = cross_entropy(perf_logits, targets)
        # cross-perplexity: observer's distribution scored against performer's log-probs
        x_ppl = -(softmax(obs_logits, dim=-1) * log_softmax(perf_logits, dim=-1)).sum(-1).mean()
        b = (log_ppl / x_ppl).item()
        return -b  # negate: project convention is higher = machine

    def predict_scores(self, texts: list[str]) -> np.ndarray:
        return np.array([self._score_one(t) for t in tqdm(texts, desc="binoculars")])
