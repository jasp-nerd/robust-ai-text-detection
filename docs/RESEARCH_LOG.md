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
