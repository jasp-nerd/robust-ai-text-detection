# Literature Review: Robust Detection of AI-Generated Text

*Compiled 2026-08-09. Citation keys refer to [`references.bib`](references.bib). Method of review:
four parallel deep-reading passes (benchmarks/datasets; zero-shot statistical methods; supervised
methods and robustness recipes; evaluation methodology and ethics) over arXiv full texts, the
ACL Anthology, Consensus (peer-reviewed indexing), the RAID public leaderboard, and shared-task
pages. Every source was assessed on venue, lab track record, citations relative to age, recency,
and code/data availability. Where a claim rests on an unreviewed 2025–2026 preprint, we say so.*

---

## 1. The problem, honestly stated

Detectors of machine-generated text (MGT) routinely report near-perfect in-distribution scores and
then fail in exactly the conditions of real use. Three independent 2026 papers converge on the same
diagnosis:

- A plain fine-tuned RoBERTa **matches or exceeds the specialized detectors** of the very
  benchmarks those detectors were built for — most reported architectural gains are actually
  in-distribution data access. But transfer TPR@1%FPR collapses across benchmarks (e.g.
  MAGE→FAID 0.038), and **60.4% of human texts from unseen domains receive p(machine) ≥ 0.95**
  [shen2026rethinking].
- Detection performance is largely benchmark-relative: once the task definition is aligned,
  **no detector is consistently best across datasets**, and detectors only work for narrow
  "notions" of AI text (en-bloc generation), not for realistic human–AI co-writing
  [dycke2026aitdna].
- Detection datasets are riddled with shortcuts: a 4M-parameter BERT-tiny reaches 0.967 AUROC
  on RAID [thorat2026dactyl]; 98.5% of DetectRL's Claude split carries literal generation
  artifacts ("Sure! Here is…") [dingfelder2025contamination].

There is also a theoretical ceiling: for any detector, AUROC ≤ ½ + TV − TV²/2 where TV is the
total-variation distance between machine and human text distributions, which shrinks as models
improve [sadasivan2023can] — contested in scope (retrieval-based detection escapes it
[krishna2023paraphrasing]), but a useful reminder that the interesting question is not "can we
hit 99%" but *at what FPR, under what shift, against what attacker*.

**Metric consensus.** AUROC alone is misleading: it is uncorrelated with TPR at FPR below 1%
[hans2024binoculars], averages over irrelevant operating points [krishna2023paraphrasing], and
hides catastrophic threshold miscalibration — GLTR has 99.3% FPR at its naive threshold
[dugan2024raid]. The field's better practice, standardized by RAID: calibrate the threshold on
human text at a target FPR (5%, 1%) and report the TPR achieved. We adopt exactly this.

## 2. Benchmarks and datasets

| Dataset | Venue | Scale | Generators | Strengths | Weaknesses |
|---|---|---|---|---|---|
| RAID [dugan2024raid] | ACL'24 | 6.2M texts | 11 (2023-era) | 8 domains × 4 decodings × 11 attacks; FPR-calibrated protocol; hidden test + leaderboard | generators frozen in 2023; English-centric |
| MAGE [li2024mage] | ACL'24 | 447K | 27 (2023-era) | best generator-diversity-per-byte for training; 8 "wildness" testbeds | one attack type; AUROC/AvgRec only |
| M4GT-Bench [wang2024m4gt] | ACL'24 | ~200K | 9 | multilingual (9 langs); boundary-detection task | not adversarial; GitHub-only |
| MGTBench [he2024mgtbench] | CCS'24 | ~21K | 6 | harness design; transferability analyses | small, 2023-era |
| HC3 [guo2023hc3] | preprint | 27K | ChatGPT | historical; trivial-floor OOD probe | single generator, known artifacts |
| DetectRL [wu2024detectrl] | NeurIPS'24 D&B | ~100K | 4 | realistic scenarios + heuristic attacks | 98.5% of Claude split contaminated — use the cleansed release [dingfelder2025contamination] |
| EvoBench [yu2025evobench] | ACL'25 Findings | 22.5K | 7 families, ~30 *versions* | the only benchmark testing generalization across model **updates** (Fast-DetectGPT loses up to 25 AUROC across versions alone) | GitHub-only, moderate size |
| Humanize-16K [bao2026triospect] | preprint | 16K | 6 (2024-era) | commercial humanizer attacks; reports TPR@1%FPR | released inside Triospect repo |
| SHIELD [ayoobi2025shield] | preprint | — | — | hardness-controllable humanification attack | early, low citations |
| AITDNA [dycke2026aitdna] | preprint (UKP) | 362 texts | 5 (incl. GPT-5.2, Gemini-3-flash) | real co-writing with full edit/prompt genesis; the hardest realistic eval | small; eval-only |
| PAN'25/'26 Voight-Kampff [bevendorff2026pan] | CLEF | ~10⁵ | surprise sets | deliberately unseen generators + obfuscations; calibration scored | Zenodo-gated, no redistribution |

**Key gap found:** as of August 2026, **no public benchmark systematically covers 2025–2026
frontier generators** (GPT-5.x, Claude 4/5, Gemini 2/3, DeepSeek V3.x, Llama 4, Qwen3). The
closest are EvoBench (late-2024 snapshots) and MELD's private MELD-eval. RAID's prompt templates
and generation code are MIT-licensed and designed for extension — generating a small
frontier-generator test set is a genuine, publishable contribution this project can make.

## 3. Method families

### 3.1 Zero-shot statistical methods

Score texts with pretrained LMs' probability signals; no detection training data.

- **Binoculars** [hans2024binoculars]: ratio of perplexity to cross-perplexity between a matched
  base/instruct pair (Falcon-7B/-Instruct). >90% TPR at 0.01% FPR on ChatGPT text zero-shot,
  and — uniquely well-documented — **robust on non-native English essays**. Fails on memorized
  text; requires a shared tokenizer; 512-token window; published thresholds are pair-specific.
- **Fast-DetectGPT** [bao2024fastdetectgpt]: conditional probability curvature via per-position
  alternative-token sampling; one forward pass (340× faster than DetectGPT [mitchell2023detectgpt]).
  Black-box config GPT-J-6B sampler + GPT-Neo-2.7B scorer; the repo now recommends
  Llama-3-8B/-Instruct pairs. 87% TPR@1%FPR on ChatGPT; degrades gracefully under T5 paraphrase
  (0.964→0.872 AUROC).
- **2026 wave (preprints, weight accordingly):** Triospect wraps any statistical score with
  content/expression-transformed variants — on adversarial RAID it lifts Binoculars from
  0.807→0.901 AUROC (TPR@1%: 46→59%) across 17 attacks, but gains nothing against synonym
  substitution [bao2026triospect]. Luminol-AIDetect's perplexity-under-shuffling neutralizes
  homoglyph/zero-width attacks (FNR 0.006 where Binoculars and Fast-DetectGPT both hit 0.979)
  with a single 2.7B model [lacava2026luminol]. Restricting any statistical score to
  **low-probability tokens** is a near-free upgrade (+4.2 AUROC on Fast-DetectGPT)
  [guo2026uncertainty].

Shared failure modes: paraphrase attacks (universal — an RL-optimized paraphraser drives all
tested detectors to ~0.02 TPR@1%FPR [ranganath2026stealthrl]), short texts, memorized text,
character-level attacks absent Unicode normalization, and multilingual text under English-centric
scoring models.

### 3.2 Supervised fine-tuned detectors

Near-perfect in-distribution; the research question is entirely about what survives shift.

- **Current state of the art (open): MELD** [li2026meld] — Ettin-400M (ModernBERT family) with
  multi-task auxiliary heads (generator/attack/domain), uncertainty-weighted losses, EMA
  clean/attacked distillation, and a hard-negative ranking loss, trained on a curated 6.6M-row
  mixture. 99.24 TPR@1%FPR on RAID *including attacks*; 99.9 TPR@1%FPR zero-shot on 2025-era
  generators. **Their own control experiment shows most of the transfer gain comes from the data
  mixture, not the architecture** — retraining a 2023 RoBERTa on MELD's mix lifts it from 38.6
  to 99.9 AUROC on modern generators. The objective's contribution concentrates at low FPR
  (ablation: removing the ranking loss drops HC3 TPR@1% from 98.5 to 44.6).
- **PAN 2026** [thorat2026dactyl; bevendorff2026pan]: rank-2 system = Bayesian data *filtering*
  (BERT-tiny+BNN votes to drop ~15% suspected shortcut/mislabeled rows) + ModernBERT-large with
  partial-AUROC objective + multicalibration. Shared-task pattern 2024→2026: **data curation and
  calibration beat architectural novelty**.
- **Backbones:** ModernBERT-family is the 2026 default (long context, speed, powers MELD and the
  best transfer baseline [drayson2025collapse]); DeBERTa-v3-large shows the best raw-AUROC
  robustness on surprise sets but calibrates poorly [thorat2026dactyl]; RoBERTa remains a
  legitimate cheap baseline [shen2026rethinking]. Long context is *not* a robustness lever
  (512 tokens ≈ 4096 [shen2026rethinking]).
- **Feature-based:** Ghostbuster's weak-LM feature search beat its own RoBERTa baseline OOD by
  13.8 F1 [verma2024ghostbuster]. Of 284 interpretable linguistic features, **only lexical
  richness (type–token ratio, hapax legomena, lexical density) survives across 27 generators and
  10 domains**; several feature groups actively hurt OOD [elattar2026linguistic].

### 3.3 Ranked evidence: what improves OOD/adversarial robustness

1. **Curated multi-source data mixing** (incl. diverse human-only corpora) — strong, replicated
   [li2026meld; thorat2026dactyl]; but *naive* pooling does not close the OOD gap
   [shen2026rethinking] and can add shortcuts [thorat2026dactyl]. Curation > volume.
2. **Hard-negative mining / ranking / partial-AUROC objectives** shaping the low-FPR tail —
   strong ablations [li2026meld; thorat2026dactyl].
3. **Attack augmentation + invariance training** (EMA distillation, adversarial paraphraser) —
   strong for the trained attack channel [hu2023radar; li2026meld]; does not confer
   generator-shift robustness (RADAR is near-random on 2026 generators).
4. **Multi-task auxiliary supervision + uncertainty weighting** — moderate-strong, one lab
   [li2026meld].
5. **Post-hoc calibration** — helps scored metrics, can hurt on surprise distributions
   [thorat2026dactyl].
6. **K-shot target-domain adaptation (FOMAML+LoRA) + confidence ensembling** — moderate
   [shen2026rethinking].
7. Statistical+supervised **ensembling / side channels** (lexical richness, low-prob tokens) —
   promising, less standardized [basani2025diveye; guo2026uncertainty; elattar2026linguistic].

## 4. Evaluation pitfalls this project guards against

Distilled from [dycke2026aitdna; dugan2024raid; krishna2023paraphrasing; sadasivan2023can;
dingfelder2025contamination; doughman2024exploring; liang2023gpt; geirhos2020shortcut]:

1. **Task definition:** state the detection notion (document-level, en-bloc machine generation),
   the AI-token threshold τ it implies, and the attacker model; report co-written/hybrid text
   behavior separately where data allows.
2. **Metrics:** TPR@5%/1%FPR with thresholds calibrated on held-out human text; report *achieved*
   FPR; report **calibration transfer** (threshold set on domain A, applied to domain B) — this
   is where deployments actually fail.
3. **Data hygiene:** grep all machine text for generation artifacts; MinHash near-dedup within
   and across splits; topic-aware splits; length-distribution matching plus a **length-only
   baseline audit**; a **BERT-tiny shortcut canary** — if a 4M-param model scores high, the
   split is measuring artifacts.
4. **Shift coverage:** always report seen vs unseen generator families separately; include
   cross-dataset transfer; include the RAID attack suite and paraphrase at multiple strengths.
5. **Trivial-baseline floor:** majority class, length, perplexity-only, AI-vocabulary
   bag-of-words — headline numbers only mean something above these.
6. **Subgroup FPR** where testable (non-native English, easy-to-read text, short texts) — the
   documented false-positive magnets [liang2023gpt; doughman2024exploring].

## 5. Ethics and deployment reality (summary; full discussion in the write-up)

Commercial detectors averaged **61.2% FPR on non-native TOEFL essays** [liang2023gpt]; OpenAI
retired its own classifier for "low rate of accuracy" [openai2023classifier]; Vanderbilt and at
least a dozen universities disabled Turnitin's detector — 1% FPR at Vanderbilt's volume means
~750 wrongly flagged papers a year [vanderbilt2023turnitin]. Dutch institutions (SURF advice;
VU Amsterdam policy) treat detector output as at most a signal for a *conversation*, never as
fraud evidence [vu2024genai]. Spoofing attacks (making human text look machine-generated) exist
alongside evasion [sadasivan2023can]. Expert humans who use LLMs frequently outperform most
detectors [russell2025people]. This project follows RAID's ethics position: we oppose the use of
detectors in disciplinary or punitive contexts; output is probabilistic evidence, never proof.

## 6. Decisions for this project (the experiment matrix)

**Datasets.**
- *Supervised training pool:* MAGE (Apache-2.0, generator diversity) + a stratified RAID-train
  subsample (decoding/penalty/attack diversity) — hash-deduped against all eval pools.
- *In-distribution eval:* MAGE test; held-out RAID-train slice.
- *Cross-dataset OOD eval:* M4GT-Bench (English splits), HC3 (floor check), AITDNA (realistic
  co-writing; hardest), EvoBench if time allows.
- *Adversarial eval:* RAID attack partitions; DIPPER-style paraphrase at multiple strengths.
- *Stretch:* self-generated frontier-generator test set via RAID's MIT pipeline; RAID leaderboard
  submission (declaring RAID-train usage).

**Methods, in order.**
1. Interpretable baselines: TF-IDF + logistic regression; gradient boosting over stylometric
   features with the lexical-richness trio isolated.
2. Zero-shot: Fast-DetectGPT (GPT-Neo-2.7B config on Mac for dev; GPT-J-6B+Neo / Llama-3-8B pair
   on the L4), then Binoculars (Falcon pair 8-bit on L4; small matched pair for Mac dev,
   recalibrated). Unicode normalization studied as a cheap input defense.
3. Supervised: ModernBERT-base → large with curated mixture; DeBERTa-v3-large comparison;
   RoBERTa-base as the honest baseline; published HF detectors (Desklib, ModernBERT-Detect,
   e5-small-lora, vanguard/gradient) evaluated under our identical harness as external anchors.
4. Robustness interventions, in ranked-evidence order: data-mixture curation first, then a
   hard-negative ranking / partial-AUROC term, then attack augmentation; ablate each.

**Protocol:** §4 checklist implemented in code (artifact grep, dedup, length audit, BERT-tiny
canary, calibration transfer, seen/unseen split reporting) before any model training.
