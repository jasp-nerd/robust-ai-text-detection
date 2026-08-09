"""Fine-tuned encoder detector (RoBERTa / ModernBERT / DeBERTa families).

Deliberately a plain, readable training loop rather than a Trainer abstraction:
AdamW + linear warmup, mixed precision on CUDA, gradient accumulation. Shen et al.
(2026) showed this vanilla recipe matches specialized detector architectures —
the interesting variables are the data mixture and the objective, not the trainer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from detector.models.fast_detect_gpt import best_device


class EncoderDetector:
    def __init__(
        self,
        model_name: str = "roberta-base",
        max_tokens: int = 512,
        batch_size: int = 8,
        grad_accum: int = 4,
        epochs: int = 1,
        lr: float = 2e-5,
        warmup_fraction: float = 0.06,
        seed: int = 0,
        device: str | None = None,
    ):
        self.device = device or best_device()
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.epochs = epochs
        self.lr = lr
        self.warmup_fraction = warmup_fraction
        self.seed = seed
        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.to(self.device)

    def _batches(self, texts: list[str], labels: np.ndarray | None, shuffle: bool):
        idx = np.arange(len(texts))
        if shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(idx)
        loader = DataLoader(idx.tolist(), batch_size=self.batch_size, shuffle=False)
        for batch_idx in loader:
            ii = [int(i) for i in batch_idx]
            enc = self.tokenizer(
                [texts[i] for i in ii],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_tokens,
                padding=True,
            ).to(self.device)
            y = None
            if labels is not None:
                y = torch.tensor([int(labels[i]) for i in ii], device=self.device)
            yield enc, y

    def fit(self, texts: list[str], labels: np.ndarray) -> EncoderDetector:
        self.model.train()
        steps_per_epoch = (len(texts) + self.batch_size - 1) // self.batch_size
        total_updates = max(1, steps_per_epoch * self.epochs // self.grad_accum)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.warmup_fraction * total_updates), total_updates
        )
        use_amp = self.device == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)
        for epoch in range(self.epochs):
            bar = tqdm(
                self._batches(texts, labels, shuffle=True),
                total=steps_per_epoch,
                desc=f"epoch {epoch + 1}/{self.epochs}",
            )
            for step, (enc, y) in enumerate(bar):
                with torch.autocast(self.device, dtype=torch.float16, enabled=use_amp):
                    loss = self.model(**enc, labels=y).loss / self.grad_accum
                scaler.scale(loss).backward()
                if (step + 1) % self.grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                if step % 50 == 0:
                    bar.set_postfix(loss=f"{loss.item() * self.grad_accum:.4f}")
        return self

    @torch.no_grad()
    def predict_scores(self, texts: list[str]) -> np.ndarray:
        self.model.eval()
        out = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for enc, _ in tqdm(
            self._batches(texts, None, shuffle=False), total=n_batches, desc="score"
        ):
            logits = self.model(**enc).logits
            out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
