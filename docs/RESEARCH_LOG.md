# Research Log

Chronological record of every experiment and decision. Each entry follows:
**Hypothesis → Setup → Result → Decision.** Negative results stay in the log forever;
their code is removed from `src/` once the decision is made.

---

## 2026-08-09 — Project start

**Context.** Goal: an open, reproducible detector of AI-generated text, evaluated primarily on
out-of-distribution and adversarial robustness rather than in-distribution accuracy. A scouting
pass over the 2024–2026 literature shows the field's consistent finding: supervised detectors
saturate in-distribution but degrade sharply under generator/domain shift and paraphrase
attacks, while zero-shot statistical methods generalize better but are more fragile to attacks.

**Decisions made at t=0 (before deep review, revisable):**
- Primary metric: TPR @ 1% FPR (following RAID), with AUROC as secondary.
- Evaluation always reported across four conditions: in-distribution, unseen generator,
  unseen domain / cross-dataset, and under adversarial attack.
- Every reported number must be regenerable from a committed run artifact.

---

## 2026-08-09 — Literature review completed; experiment matrix decided

**Setup.** Four parallel deep-reading passes over ~60 sources (full-text arXiv reads for ~20 core
papers), plus the live RAID leaderboard and PAN 2026 task pages. Synthesis in
[literature-review.md](literature-review.md); all citations in [references.bib](references.bib).

**What the review changed relative to the initial plan:**
1. *The central research question sharpened.* The field's own 2026 self-critique
   (Shen et al.; MELD's Appendix E control; DACTYL's BERT-tiny probe) says the decisive factor
   for robustness is **training-data curation, not architecture**. Our Phase 3d interventions are
   re-ranked accordingly: data-mixture curation first, low-FPR-shaping losses second, attack
   augmentation third. Architectural novelty is explicitly out of scope.
2. *Dataset lineup revised.* MAGE promoted to primary supervised-training corpus
   (generator-diversity per byte, Apache-2.0); RAID train subsampled for decoding/attack
   diversity; AITDNA added as a realistic co-writing eval set (it did not exist in the initial
   plan); DetectRL only in its cleansed form (98.5% of its Claude data is contaminated);
   HC3 demoted to a floor-check.
3. *Protocol hardened before any modeling.* New mandatory audits from the pitfalls literature:
   generation-artifact grep, MinHash near-dedup, topic-aware splits, length-only baseline,
   BERT-tiny shortcut canary, calibration-transfer reporting, seen/unseen generator separation.
4. *Gap identified.* No public benchmark covers 2025–2026 frontier generators. Stretch goal:
   generate a small frontier test set using RAID's MIT-licensed pipeline.
5. *Zero-shot compute plan fixed.* Fast-DetectGPT (Neo-2.7B config) runs on the MacBook;
   the faithful GPT-J-6B+Neo config and Binoculars (Falcon pair, 8-bit) need the L4 server.

**Decision:** proceed to Phase 2 (data pipeline + audits) with the matrix in
literature-review.md §6.

---

## 2026-08-09 — Data audits: MAGE clean, HC3 length-confounded

**Hypothesis.** Public detection datasets contain shortcut features; audits must run
before any training (protocol §4).

**Setup.** `scripts/run_audits.py` over normalized MAGE and HC3: artifact-pattern rates,
exact-duplicate counts, train→test leakage, and a length-only "detector" AUROC.

**Result.** MAGE: length-only AUROC 0.499 (train) — clean; 24 leaked train→test rows
(dropped downstream); 2.2% of its ChatGPT rows carry "As an AI…" boilerplate (machine
rows matching any artifact pattern are filtered at training time: 894 rows).
HC3: **length-only AUROC 0.731** — ChatGPT answers are 2× longer than human answers
(median 173 vs 83 words). HC3 numbers therefore overstate any real detector.

**Decision.** HC3 stays eval-only as a floor check, never for training; its results are
always reported with the length-confound caveat. Artifacts committed in `results/audits/`.

---

## 2026-08-09 — Phase 3a: interpretable baselines. OOD collapse reproduced; a published robustness claim does not transfer

**Hypothesis.** (a) Simple supervised baselines trained on MAGE will look reasonable
in-dataset and degrade sharply on RAID (unseen generators/domains/attacks).
(b) Per El Attar et al. (2026), the lexical-richness feature trio should be the most
robust stylometric signal under shift.

**Setup.** Train on MAGE train (318K rows after artifact filter, seed 0). Eval on MAGE
test (60.7K, includes MAGE's own OOD testbeds), HC3 (85K, floor check), RAID eval (33.4K,
full generator×domain×attack grid). Configs in `configs/`, artifacts in `results/runs/`.

| model | MAGE test AUROC / TPR@1% | HC3 AUROC / TPR@1% | RAID eval AUROC / TPR@1% |
|---|---|---|---|
| TF-IDF + logreg | 0.804 / 0.194 | 0.868 / 0.237 | 0.750 / 0.124 |
| Stylometric GBM (11 feats) | 0.812 / 0.251 | 0.717 / 0.115 | 0.675 / 0.021 |
| Lexical-richness only (3 feats) | 0.707 / 0.128 | **0.437** / 0.064 | 0.603 / 0.029 |

**Findings.**
1. Hypothesis (a) confirmed: everything degrades OOD, and TPR@1%FPR collapses far harder
   than AUROC suggests (stylometric GBM: 0.68 AUROC but **2%** TPR@1% on RAID) —
   the AUROC-hides-deployment-failure pattern from the literature, reproduced.
2. Hypothesis (b) **rejected in our setting**: lexical-richness-only transfers *below
   chance* to HC3 (0.437). The El Attar result was measured across testbeds *within*
   MAGE; it does not survive a real cross-dataset shift to QA-style text, where the
   direction of the human/machine lexical-density gap apparently flips. Caveats: our
   lexical density is a stopword-proxy (theirs is POS-based), and their SVM ≠ our GBM.
3. TF-IDF remains the best *transferring* simple baseline — consistent with PAN 2026's
   own baseline table, where TF-IDF SVM (0.978) beat Binoculars in-domain.

**Decision.** Keep TF-IDF+logreg as the running reference baseline in all future tables.
Keep the full stylometric GBM for interpretability analyses only; drop the
lexical-only variant from the headline table (kept in `results/` as a negative result).

---

## 2026-08-09 — Phases 3b/3c (first wave): zero-shot vs supervised, and the attack asymmetry

*(All runs on the VU compute servers: L4 24GB, shared. Tables regenerable via
`scripts/make_tables.py`.)*

**Hypotheses.** (a) Fine-tuned encoders dominate in-distribution but drop hard on RAID;
zero-shot statistical detectors degrade less across datasets. (b) ModernBERT > RoBERTa
as detector backbone (2026 consensus). (c) Attacks hit method families differently.

**Setup.** Supervised: RoBERTa-base and ModernBERT-base, vanilla fine-tune on MAGE train
(318K, artifact-filtered, seed 0). Zero-shot: Fast-DetectGPT (paper-faithful black-box:
GPT-J-6B sampler + GPT-Neo-2.7B scorer, analytic form) and Binoculars with a small
Qwen2.5-0.5B pair (the Falcon-7B pair OOMed beside a 17GB neighbor process; 8-bit rerun
in progress). Evals: MAGE test / HC3 / RAID eval grid (33.4K, 11 generators × 8 domains
× 12 attack conditions).

| detector | MAGE AUROC / TPR@1% | HC3 | RAID eval |
|---|---|---|---|
| ModernBERT-base (MAGE) | 0.981 / 0.845 | 0.994 / 0.941 | **0.875** / 0.310 |
| RoBERTa-base (MAGE) | 0.976 / 0.708 | 0.985 / 0.900 | 0.825 / 0.258 |
| Fast-DetectGPT (GPT-J) | 0.628 / 0.289 | 0.993 / 0.957 | 0.809 / **0.319** |
| Binoculars (0.5B pair) | 0.595 / 0.112 | 0.954 / 0.486 | 0.739 / 0.196 |
| TF-IDF + logreg (ref) | 0.804 / 0.195 | 0.868 / 0.237 | 0.750 / 0.124 |

**Findings.**
1. (a) confirmed with a twist: ModernBERT wins RAID *AUROC*, but Fast-DetectGPT wins
   RAID *TPR@1%FPR* — the supervised advantage disappears exactly in the operating
   regime that matters. Fast-DetectGPT's MAGE weakness (0.628) traces to MAGE's many
   base-model generators (OPT/T5/GPT-J family text looks "human" to a GPT-J sampler —
   scoring-model relatedness cuts both ways).
2. (b) confirmed: ModernBERT > RoBERTa on every eval at every metric (+0.05 AUROC /
   +5pp TPR@1% on RAID).
3. (c) confirmed, and the asymmetry is sharp (TPR@5% by attack, RAID):
   - **homoglyph** breaks everyone (ModernBERT 0.71→0.03; FDG 0.66→0.08);
   - **zero_width_space** breaks only the encoders (ModernBERT 0.05, RoBERTa 0.06)
     while LLM scorers barely notice (FDG 0.55, Binoculars 0.57);
   - **synonym swap** breaks the statistical methods (FDG 0.19, Binoculars 0.13) but
     not ModernBERT (0.52);
   - **paraphrase** *helps* RoBERTa (0.61 vs 0.57 clean) — RAID's
     attack-toward-training-distribution effect, reproduced.
4. Binoculars at 0.5B scale is far below its paper numbers (7B pair) — observer scale
   matters; the 8-bit Falcon run will quantify the gap.
5. Seed variance (3 seeds, TF-IDF + stylometric): AUROC stable (±0.007), but RAID
   TPR@5% for the weak stylometric model swings 0.05–0.13 across seeds. **Low-FPR
   metrics of weak detectors are seed-unstable; single-seed low-FPR claims near the
   noise floor are not meaningful.**

**Decisions.**
- Phase 3d intervention order: (1) Unicode/NFKC normalization as input defense — the
  attack table says it should recover both character-level attacks almost for free;
  (2) supervised+statistical ensemble (their failure modes are complementary on every
  axis measured); (3) attack-augmented training for the encoder; (4) RAID-train mixture.
- ModernBERT-base is the supervised backbone going forward; RoBERTa retired to
  reference-row status.

---

## 2026-08-10 — Phase 3b complete: two zero-shot surprises

**Setup.** Final two zero-shot runs: Fast-DetectGPT with sampler = scorer = GPT-Neo-2.7B
(the paper's cheaper Table-9 variant), and Binoculars with a Qwen2.5-3B pair (the
Falcon-7B 8-bit replication is backlogged: transformers' caching-allocator warmup
pre-allocates fp16-sized memory for quantized models and OOMs the 24GB L4).

| config | RAID AUROC | RAID TPR@5% | RAID TPR@1% | HC3 AUROC | MAGE AUROC |
|---|---|---|---|---|---|
| FDG, GPT-J-6B sampler + Neo scorer | **0.809** | 0.488 | 0.319 | 0.993 | 0.628 |
| FDG, Neo-2.7B sampler = scorer | 0.798 | **0.556** | **0.416** | 0.993 | 0.604 |
| Binoculars 0.5B pair | 0.739 | 0.356 | 0.196 | 0.954 | 0.595 |
| Binoculars 3B pair | 0.726 | 0.395 | 0.216 | 0.926 | 0.612 |

**Surprise 1 — the cheap config wins where it matters.** The single-model Neo-2.7B
variant has slightly *lower* RAID AUROC than the paper-faithful GPT-J config but far
*higher* TPR at low FPR (0.416 vs 0.319 at 1% — the best of ALL eleven runs so far,
supervised included). AUROC ranking and low-FPR ranking disagree between two variants
of the *same method* — the strongest instance yet of the field's warning that AUROC is
the wrong lens. Hypothesis: self-sampling produces a tighter human score distribution
(thinner right tail), which is what low-FPR thresholds reward.

**Surprise 2 — Binoculars does not simply scale.** 0.5B → 3B changed RAID AUROC from
0.739 to 0.726 (down), TPR@1% from 0.196 to 0.216 (up a little), HC3 down. The paper's
Falcon-7B strength evidently isn't raw observer scale — pair matching (how close
base and instruct siblings are) plausibly matters more. Worth a controlled follow-up;
for now Binoculars-at-accessible-scale is simply not competitive with Fast-DetectGPT.

**Decision.** Fast-DetectGPT (Neo-2.7B) is the zero-shot representative for Phase 3d
ensembling. Phase 3b closed; Falcon replication stays on the backlog.

---

## 2026-08-10 — Phase 3d, interventions 1+2: normalization defense and the ensemble

**Hypotheses.** (1) Unicode/NFKC normalization + zero-width stripping + a small
confusables map recovers the character-level attacks (homoglyph, zero-width) at no cost
elsewhere. (2) Rank-averaging the supervised (ModernBERT-MAGE) and zero-shot
(Fast-DetectGPT Neo) scores beats both — their failure modes are complementary.

**Setup.** Both detectors re-scored the RAID eval grid twice (clean inputs vs
normalized inputs), dumping per-sample scores; ensemble = mean of within-run score
ranks, no learned weights (`scripts/analyze_ensemble.py`). Clean re-scores reproduced
the previous runs' numbers (FDG exactly; ModernBERT retrained: AUROC matched to 0.001,
TPR@1% differed 0.31→0.38 across retrains — training nondeterminism on identical data;
noted as a caveat for all single-run TPR@1% claims).

**Results (RAID eval, all conditions incl. attacks):**

| detector | AUROC | TPR@5% | TPR@1% |
|---|---|---|---|
| ModernBERT clean | 0.874 | 0.468 | 0.378 |
| ModernBERT normalized | 0.927 | 0.637 | 0.448 |
| FDG-neo clean | 0.798 | 0.556 | 0.416 |
| FDG-neo normalized | 0.829 | 0.672 | 0.528 |
| Ensemble clean | 0.902 | 0.544 | 0.428 |
| **Ensemble normalized** | **0.947** | **0.761** | **0.582** |

Per-attack (FDG, TPR@5%): homoglyph 0.078→**0.773**, zero-width 0.199→**0.761** —
both restored to clean level (0.757); every other attack unchanged or slightly up.
Synonym substitution remains the open weakness (0.29→0.31).

**Findings.** Both hypotheses confirmed. The stacked pipeline (normalize → both
detectors → rank-average) more than **quadruples** the best simple baseline's TPR@1%
under attack (0.124 → 0.582) with zero additional training. Normalization is a pure
Pareto improvement and should be a default preprocessing step for any detector —
notable that no leaderboard we reviewed mandates it.

**Decision.** One experiment left: the curated MAGE+RAID training mixture
(ModernBERT), testing the literature's top-ranked intervention. Caveat pre-registered
in `scripts/make_mixture.py`: with RAID in training, RAID eval becomes
semi-in-distribution; HC3 stays the only fully-OOD eval for that model.

---

## 2026-08-10 — Phase 3d complete: the data mixture, and the program's close

**Hypothesis.** Per MELD's control experiment, curated training-data mixing is the
strongest single robustness intervention; a MAGE+RAID mixture should lift attack
robustness without costing in-distribution accuracy.

**Setup.** ModernBERT-base, identical recipe to the MAGE-only run, trained on mix1:
MAGE train + a stratified RAID train-pool sample covering every generator × domain ×
attack cell (371,317 rows after dedup). Pre-registered caveat: RAID eval shares
generators/domains/attacks with this training set (source documents remain disjoint),
so its numbers measure attack/decoding exposure, not cross-dataset generalization.
HC3 is the only fully-OOD eval for this model.

| eval | MAGE-only ModernBERT | mix1 ModernBERT |
|---|---|---|
| RAID eval (semi-ID for mix1), TPR@1% | 0.310 | **0.905** (0.954 normalized) |
| MAGE test, TPR@1% | 0.845 | 0.848 |
| HC3 (fully OOD), TPR@1% | 0.941 | **0.953** |

**Findings.** Hypothesis confirmed on all three axes: attack exposure is worth ~60
points of TPR@1% on the attacked grid, in-distribution accuracy is unchanged, and the
fully-OOD floor check improves rather than degrades. The result mirrors MELD's
Appendix E at 1/400 of the training scale. The honest framing matters: mix1's RAID
number is not comparable to the OOD rows in the main table, and we present it
separately.

---

## 2026-08-11 — Season 2: the M4GT reversal

**Hypothesis.** On a genuinely unseen corpus (M4GT English, via the source-filtered
COLING-2025 mirror; generators include llama3-8b and gpt4, postdating MAGE), the mix1
model's RAID gains either transfer (data mixing is a general robustness win) or they
don't (the mixture overfits RAID's attack styles).

**Result — they don't, and it's worse than neutral.**

| detector | M4GT AUROC | M4GT TPR@5% | M4GT TPR@1% |
|---|---|---|---|
| ModernBERT (MAGE only) | **0.920** | **0.783** | **0.674** |
| Fast-DetectGPT (Neo, zero-shot) | 0.845 | 0.712 | 0.636 |
| ModernBERT (mix1) | 0.855 | 0.219 | **0.000** |
| TF-IDF + logreg (MAGE) | 0.793 | 0.376 | 0.145 |
| Binoculars (0.5B) | 0.735 | 0.532 | 0.379 |

The RAID-attack champion scores literally zero at 1% FPR on M4GT: its human-score
distribution has a heavy machine-side tail on unseen-domain human text — the exact
high-confidence-wrong failure mode of Shen et al. (2026), reproduced by our own best
model. Attack-exposure training specialized the decision boundary to RAID's style of
text and broke calibration everywhere else. HC3's earlier "improvement" was a mirage
of an easy, single-generator eval.

**Decision.** The MAGE-only checkpoint is the honest general-purpose release; mix1 is
published only as a RAID-specialist with an explicit warning. Both are on the Hub:
[general](https://huggingface.co/jaspai/modernbert-ai-text-detector),
[raid-mix](https://huggingface.co/jaspai/modernbert-ai-text-detector-raid-mix).
Lesson recorded: in-benchmark robustness gains must be re-earned on a disjoint corpus
before being believed — our own pre-registered caveat turned out to be the headline.

---

## 2026-08-11 — Season 2 close: the frontier probe completes the triangle

**Setup.** 465 texts generated by Qwen3-4B-Instruct-2507 (a 2025 generator absent from
every training set), continuation-style from RAID-domain human openings, paired with
the matched human documents. Declared confound: continuation-from-opening is RAID's
own generation format, so mix1 is in-format here while the MAGE-only model is doubly
out (format and generator); the zero-shot detector carries no such asymmetry.

| detector | frontier AUROC | frontier TPR@1% | (recall M4GT TPR@1%) |
|---|---|---|---|
| ModernBERT mix1 | 0.999 | **0.985** | 0.000 |
| Fast-DetectGPT (Neo, zero-shot) | 0.989 | 0.854 | 0.636 |
| ModernBERT MAGE-only | 0.881 | **0.126** | 0.674 |

**Finding.** The two supervised checkpoints fail in perfectly complementary places:
each is near-ceiling exactly where the other collapses, and which one "generalizes"
depends entirely on which eval you pick. The zero-shot method never ranks first and
never collapses (worst TPR@1% across RAID/M4GT/frontier: 0.42). Detection robustness
is not a scalar; any single-number leaderboard claim hides a collapse axis. This is
the strongest argument in the project for the ensemble result of Phase 3d and for
reporting multi-corpus results as a profile, never an average.

**Falcon-7B replication: closed as blocked.** Three attempts across two transformers
major versions. Root causes found each time (fp32 fallback from a dropped dtype
kwarg; a caching-allocator warmup that pre-allocates unquantized-size memory; and in
transformers 5.x, BitsAndBytesConfig 8-bit quantization silently not applied at all —
weights load fp16 and the 29GB pair cannot fit 24GB). The science this replication
would add is already covered by our 0.5B/3B matched-pair scale curve and the paper's
published Falcon numbers; further debugging is not worth the compute. Recorded here
so the next person hits the wall with a map.

**Program close.** Fifteen detector configurations evaluated under one harness;
two zero-training interventions (normalization, ensembling) taking adversarial-grid
TPR@1% from 0.124 to 0.582 without RAID exposure; one training intervention (mix1)
reaching 0.905/0.954 with it. Remaining backlog, in order of value: a third fully-OOD
eval dataset (M4GT) to give mixture models a fair cross-dataset test; Falcon-7B
Binoculars once the transformers warmup bug is fixed; a frontier-generator test set
via RAID's pipeline; publishing a trained checkpoint to Hugging Face (requires a
save_model retrain and an HF token).
