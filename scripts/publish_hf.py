"""Publish saved detector checkpoints to the Hugging Face Hub with model cards.

Usage (HF_TOKEN in env):  uv run python scripts/publish_hf.py
Uploads both checkpoints if present. Cards state metrics, lineage, and limitations.
"""

import json
import os
from pathlib import Path

from huggingface_hub import HfApi

REPO_URL = "https://github.com/jasp-nerd/robust-ai-text-detection"

MODELS = [
    {
        "dir": "results/runs/encoder-modernbert-base-mage/model",
        "repo": "jaspai/modernbert-ai-text-detector",
        "headline": "Recommended general-purpose checkpoint: best cross-dataset transfer.",
        "metrics_runs": [
            ("MAGE test (near in-distribution)", "results/runs/encoder-modernbert-base-mage/mage_test.json"),
            ("HC3 (cross-dataset)", "results/runs/encoder-modernbert-base-mage/hc3_test.json"),
            ("RAID eval grid, incl. attacks (OOD)", "results/runs/encoder-modernbert-base-mage/raid_eval.json"),
            ("M4GT (cross-dataset)", "results/runs/saved-mage-ood/m4gt_test.json"),
        ],
        "training": "MAGE train (318K rows after artifact filtering), 1 epoch, lr 3e-5, 512 tokens.",
        "warning": "",
    },
    {
        "dir": "results/runs/modernbert-mix1/model",
        "repo": "jaspai/modernbert-ai-text-detector-raid-mix",
        "headline": "RAID-specialist variant: trained with attack exposure (MAGE + stratified RAID mixture).",
        "metrics_runs": [
            ("MAGE test", "results/runs/modernbert-mix1/mage_test.json"),
            ("HC3 (cross-dataset)", "results/runs/modernbert-mix1/hc3_test.json"),
            ("RAID eval grid, incl. attacks (semi-in-distribution)", "results/runs/modernbert-mix1/raid_eval.json"),
            ("M4GT (cross-dataset)", "results/runs/saved-mix1-ood/m4gt_test.json"),
        ],
        "training": "MAGE + stratified RAID train-pool mixture (371K rows, every generator/domain/attack cell), 1 epoch, lr 3e-5, 512 tokens.",
        "warning": (
            "\n> **Warning.** This variant is heavily specialized to RAID-style attacked text. "
            "On the truly out-of-distribution M4GT set its TPR at 1% FPR is **0.00** — it assigns "
            "high machine-confidence to unseen-domain human text. Prefer the "
            "[general checkpoint](https://huggingface.co/jaspai/modernbert-ai-text-detector) "
            "unless your inputs resemble RAID.\n"
        ),
    },
]

CARD = """---
license: mit
language: en
base_model: answerdotai/ModernBERT-base
pipeline_tag: text-classification
tags: [ai-text-detection, machine-generated-text]
---

# {repo_name}

{headline}

Fine-tuned ModernBERT-base classifier for detecting machine-generated English text
(label 1 = machine). Built in the open research project
[robust-ai-text-detection]({repo_url}), where every number regenerates from committed
artifacts and the full research log, literature review, and negative results live.
{warning}
## Metrics

Thresholds must be calibrated on human text from your own distribution; the scores
below use per-dataset calibration. AUROC alone is misleading for this task — use the
low-FPR columns.

| eval | AUROC | TPR@5%FPR | TPR@1%FPR |
|---|---|---|---|
{metric_rows}

## Training

{training}

## Intended use and limitations

Research use. Output is probabilistic evidence, never proof of authorship: at 1% FPR,
an institution processing 75,000 documents a year would wrongly flag ~750. Do not use
as the basis of disciplinary action. Untested on non-native-writer false-positive
rates; English only; document-level only; defeated by adaptive paraphrase attacks.
Pair with Unicode/NFKC input normalization (see the repository) — it neutralizes
homoglyph and zero-width attacks for free.

## Citation

Please cite the datasets and methods this builds on (MAGE, RAID, ModernBERT — full
BibTeX in the [repository]({repo_url})).
"""


def metric_rows(runs: list) -> str:
    rows = []
    for label, path in runs:
        o = json.loads(Path(path).read_text())["metrics"]["overall"]
        rows.append(
            f"| {label} | {o['auroc']:.3f} | {o['tpr_at_fpr_0.05']:.3f} | {o['tpr_at_fpr_0.01']:.3f} |"
        )
    return "\n".join(rows)


def main() -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    for m in MODELS:
        if not Path(m["dir"]).exists():
            print(f"skip (no checkpoint): {m['dir']}")
            continue
        card = CARD.format(
            repo_name=m["repo"].split("/")[1],
            headline=m["headline"],
            repo_url=REPO_URL,
            warning=m["warning"],
            metric_rows=metric_rows(m["metrics_runs"]),
            training=m["training"],
        )
        Path(m["dir"], "README.md").write_text(card)
        api.create_repo(m["repo"], exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=m["dir"], repo_id=m["repo"], repo_type="model")
        print(f"published https://huggingface.co/{m['repo']}")


if __name__ == "__main__":
    main()
