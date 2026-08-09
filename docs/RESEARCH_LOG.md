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
