# Robust AI-Generated Text Detection — an open research project

**TL;DR.** We benchmarked five families of AI-text detectors under distribution shift and
adversarial attack, reproduced the field's core failure modes, falsified one published
robustness claim, and found that a zero-training pipeline — Unicode input normalization
followed by a rank-average ensemble of a fine-tuned encoder and a zero-shot statistical
detector — raises detection at 1% false-positive rate on the adversarial RAID grid from
12% (best simple baseline) to **58%**, and AUROC to **0.947**.

> This repository is a research write-up as much as a codebase. Every experiment is
> logged with hypothesis → setup → result → decision in
> [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) — including the failures. The
> literature review that grounds it is [`docs/literature-review.md`](docs/literature-review.md)
> (~60 sources, [`docs/references.bib`](docs/references.bib)). Every number below
> regenerates from committed artifacts via `scripts/make_tables.py`.

## 1. Motivation and research question

Detectors of machine-generated text report near-perfect in-distribution scores and then
collapse in the conditions of real use: unseen generators, unseen domains, adversarial
edits, and the low-false-positive operating points that deployment demands
[shen2026rethinking; dugan2024raid; dycke2026aitdna]. The consequences of getting this
wrong are not abstract: commercial detectors averaged 61% false positives on non-native
English essays [liang2023gpt], OpenAI retired its own classifier over accuracy
[openai2023classifier], and universities — including in the Netherlands — have ruled
detector scores inadmissible as evidence of fraud [vu2024genai].

**Research question:** how far can an open, reproducible detector get on
out-of-distribution and adversarial robustness, and which method families actually
generalize?

## 2. Setup

**Data.** MAGE [li2024mage] (447K texts, 27 generators — supervised training and
in-distribution-ish eval), HC3 [guo2023hc3] (cross-dataset transfer check; we show it is
length-confounded — length alone scores 0.73 AUROC — so it serves only as an easy floor),
and RAID [dugan2024raid] as the robustness centrepiece: we carve a leakage-safe
33,396-text eval grid from its labeled train split, stratified over 11 generators ×
8 domains × 12 attack conditions, split by source document so no attacked variant of a
training text can appear in eval.

**Protocol.** Primary metric TPR at 5%/1% FPR with thresholds calibrated on human text
(RAID's protocol); AUROC secondary. Data hygiene enforced before any training:
generation-artifact filtering (2.2% of MAGE's ChatGPT rows say "As an AI…"),
normalized-text dedup, length-shortcut audits, and per-slice reporting by generator,
domain, and attack. Fixed seeds; seed-variance measured where affordable.

**Methods benchmarked** (each behind the same eval harness):
- *Interpretable baselines:* TF-IDF + logistic regression; gradient boosting over
  stylometric features, with a lexical-richness-only ablation [elattar2026linguistic].
- *Zero-shot statistical:* Fast-DetectGPT [bao2024fastdetectgpt] in the paper-faithful
  GPT-J-6B config and the cheaper self-sampling GPT-Neo-2.7B config; Binoculars
  [hans2024binoculars] with matched Qwen2.5 pairs at 0.5B and 3B.
- *Fine-tuned encoders:* RoBERTa-base (the "strong vanilla baseline" of
  [shen2026rethinking]) and ModernBERT-base [thorat2026dactyl; drayson2025collapse].
- *Robustness interventions:* Unicode input normalization; rank-average ensembling;
  curated MAGE+RAID training mixture (in progress).

All compute ran on VU Amsterdam's shared L4 GPUs plus a consumer laptop for development.

## 3. Main results

RAID eval grid (33,396 texts, all generators/domains/attacks), sorted by AUROC —
regenerate with `uv run python scripts/make_tables.py`:

| detector | AUROC | TPR@5%FPR | TPR@1%FPR |
|---|---|---|---|
| **Normalize → ensemble (ModernBERT + FDG)** | **0.947** | **0.761** | **0.582** |
| Normalize → ModernBERT-MAGE | 0.927 | 0.637 | 0.448 |
| Ensemble, clean inputs | 0.902 | 0.544 | 0.428 |
| ModernBERT-base (MAGE) | 0.875 | 0.512 | 0.310 |
| Normalize → Fast-DetectGPT (Neo) | 0.829 | 0.672 | 0.528 |
| RoBERTa-base (MAGE) | 0.825 | 0.414 | 0.258 |
| Fast-DetectGPT (GPT-J, paper config) | 0.809 | 0.488 | 0.319 |
| Fast-DetectGPT (Neo-2.7B, self-sampling) | 0.798 | 0.556 | 0.416 |
| TF-IDF + logreg | 0.750 | 0.268 | 0.124 |
| Binoculars (Qwen 0.5B pair) | 0.739 | 0.356 | 0.196 |
| Binoculars (Qwen 3B pair) | 0.726 | 0.395 | 0.216 |
| Stylometric GBM | 0.675 | 0.064 | 0.021 |

**What we learned (details and per-attack tables in the [research log](docs/RESEARCH_LOG.md)):**

1. **The OOD collapse is real and AUROC hides it.** Every supervised model near-ceiling
   in-distribution (ModernBERT 0.98 AUROC on MAGE) drops hard on RAID — and TPR@1%FPR
   drops far harder than AUROC suggests. Two configs of the *same* zero-shot method even
   swap rank between AUROC and TPR@1%.
2. **Supervised and zero-shot detectors fail in opposite places.** Encoders are killed
   by zero-width-space insertion (TPR 0.05) but survive synonym swaps (0.52); statistical
   scorers are the mirror image (0.55 / 0.19). Homoglyph substitution kills both.
   This complementarity is why the plain rank-average ensemble works.
3. **Unicode normalization is a free lunch nobody serves.** NFKC + zero-width stripping
   + a 40-entry confusables map restores both character-level attacks to clean-input
   detection levels (homoglyph 0.08→0.77) at zero cost elsewhere — yet no benchmark or
   leaderboard we reviewed mandates it as preprocessing.
4. **A published robustness claim did not replicate across datasets:** the
   lexical-richness feature trio, reported as the most shift-robust interpretable
   signal [elattar2026linguistic], transfers *below chance* (0.44 AUROC) from MAGE to
   HC3 in our setup. (Their evaluation shifted within one corpus; ours crosses corpora.)
5. **Binoculars did not improve from 0.5B to 3B observers** — pair matching, not raw
   scale, seems to carry the method; and the cheap self-sampling Fast-DetectGPT config
   beat the paper-faithful two-model config at low FPR.
6. **Low-FPR metrics are noisy.** Across seeds/retrains, AUROC moves ±0.007 but TPR@1%
   for weak detectors varies 2.5×, and an identical retrain of ModernBERT moved TPR@1%
   by 7 points. Single-run low-FPR claims near the noise floor should not be trusted —
   ours included.
7. *Pending:* curated MAGE+RAID mixture training (the literature's top-ranked
   intervention [li2026meld]) — results land here when the run completes.

## 4. What didn't work (kept on purpose)

- Lexical-richness-only stylometrics (below chance cross-dataset — see finding 4).
- Binoculars at accessible scale (both pairs below Fast-DetectGPT everywhere).
- Falcon-7B 8-bit replication: blocked by a transformers caching-allocator bug that
  pre-allocates fp16-sized memory for quantized models on 24GB cards (backlogged).
- Assorted infrastructure lessons (silent git-pull failures, zombie GPU jobs, nightly
  SSH-key wipes on shared university hardware) are logged for anyone reproducing this
  on similar academic infra.

## 5. Limitations and ethics

This project measures English-only, document-level detection of fully machine-generated
text; realistic human-AI co-writing is harder and detectors serving other "notions" of
AI text do not transfer [dycke2026aitdna]. RAID's generators are 2023-era; no public
benchmark covers 2025–26 frontier models (a gap we document, not fill). Our subgroup
false-positive behavior (non-native writers) is untested. An adaptive adversary with an
RL-optimized paraphraser defeats all published detectors [ranganath2026stealthrl] —
including, we must assume, ours.

Following RAID's ethics position: **detector output is probabilistic evidence, never
proof, and should not be the basis of disciplinary action.** At 1% FPR, an institution
processing 75,000 papers a year would wrongly flag ~750 of them. Dutch universities'
policy — a detector score may at most prompt a conversation, with fraud determined only
by an examination board [vu2024genai] — is the deployment model this work assumes.

## 6. Reproducing

```bash
git clone https://github.com/jasp-nerd/robust-ai-text-detection
cd robust-ai-text-detection && uv sync
uv run pytest                                        # 23 tests
uv run python scripts/prepare_data.py mage hc3 raid  # ~3GB download
uv run python scripts/make_raid_splits.py
uv run python scripts/run_experiment.py configs/tfidf_logreg_mage.yaml   # minutes, CPU
uv run python scripts/run_experiment.py configs/fast_detect_gpt_neo.yaml # needs a GPU
uv run python scripts/make_tables.py --slice attack
```

Every YAML in `configs/` is one experiment; every JSON in `results/runs/` carries its
config and git commit. Citations resolve in [`docs/references.bib`](docs/references.bib).

## License

MIT. If you use this work, please cite the underlying papers credited throughout —
they did the heavy lifting.
