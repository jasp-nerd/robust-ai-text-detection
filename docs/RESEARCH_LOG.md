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
