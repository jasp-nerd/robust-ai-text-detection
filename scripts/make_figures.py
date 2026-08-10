"""Generate the paper figures from committed run artifacts.

Usage: uv run python scripts/make_figures.py
Writes SVG (embedded in README) and PNG (inspection) to results/figures/.

Design choices follow colorblind-safe practice: Okabe-Ito hues for categorical
series (validated for CVD separation), viridis for the heatmap, direct labels,
minimal chrome. One hue per job; emphasis reserved for the headline pipeline.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("results/runs")
OUT = Path("results/figures")
BLUE, VERMILLION, GRAY = "#0072B2", "#D55E00", "#999999"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "xtick.color": "#666666",
        "ytick.color": "#333333",
        "axes.labelcolor": "#333333",
        "svg.fonttype": "none",
    }
)


def load(run: str, eval_file: str) -> dict:
    return json.loads((RUNS / run / eval_file).read_text())


def fig_main_results() -> None:
    rows = [
        ("Normalize → ensemble", "ensemble-rankavg-norm", "raid_eval_norm.json", True),
        ("Ensemble (clean inputs)", "ensemble-rankavg", "raid_eval.json", False),
        ("Normalize → ModernBERT", "modernbert-defense", "raid_eval_norm.json", False),
        ("Normalize → Fast-DetectGPT", "fdg-neo-defense", "raid_eval_norm.json", False),
        ("ModernBERT-base (MAGE)", "encoder-modernbert-base-mage", "raid_eval.json", False),
        ("RoBERTa-base (MAGE)", "encoder-roberta-base-mage", "raid_eval.json", False),
        ("Fast-DetectGPT (Neo-2.7B)", "fast-detect-gpt-neo2.7b", "raid_eval.json", False),
        ("Fast-DetectGPT (GPT-J)", "fast-detect-gpt-gptj", "raid_eval.json", False),
        ("TF-IDF + logistic reg.", "tfidf-logreg-mage", "raid_eval.json", False),
        ("Binoculars (0.5B pair)", "binoculars-qwen2.5-0.5b", "raid_eval.json", False),
        ("Binoculars (3B pair)", "binoculars-qwen2.5-3b", "raid_eval.json", False),
        ("Stylometric GBM", "stylometric-gbm-mage", "raid_eval.json", False),
    ]
    data = []
    for label, run, f, emph in rows:
        o = load(run, f)["metrics"]["overall"]
        data.append((label, o["auroc"], o["tpr_at_fpr_0.01"], emph))
    data.sort(key=lambda r: r[2])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
    y = np.arange(len(data))
    for ax, idx, title, xlim in [
        (axes[0], 2, "TPR @ 1% FPR", (0, 0.72)),
        (axes[1], 1, "AUROC", (0.5, 1.0)),
    ]:
        vals = [r[idx] for r in data]
        colors = [VERMILLION if r[3] else BLUE for r in data]
        ax.barh(y, vals, height=0.62, color=colors, zorder=3)
        for yi, v in zip(y, vals, strict=True):
            ax.text(v + (xlim[1] - xlim[0]) * 0.012, yi, f"{v:.3f}",
                    va="center", fontsize=8.5, color="#333333")
        ax.set_title(title, fontsize=11, loc="left", color="#333333")
        ax.set_xlim(*xlim)
        ax.grid(axis="x", color="#eeeeee", zorder=0)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [r[0] for r in data])
    fig.suptitle(
        "Detection on the adversarial RAID grid (33,396 texts: 11 generators \u00d7 8 domains \u00d7 12 attack conditions)",
        fontsize=10.5, x=0.01, ha="left", color="#333333",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "main_results.svg", bbox_inches="tight")
    fig.savefig(OUT / "main_results.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


ATTACK_ORDER = [
    "none", "paraphrase", "synonym", "homoglyph", "zero_width_space", "whitespace",
    "upper_lower", "article_deletion", "insert_paragraphs", "perplexity_misspelling",
    "alternative_spelling", "number",
]
ATTACK_LABELS = [a.replace("_", " ") for a in ATTACK_ORDER]


def fig_attack_heatmap() -> None:
    methods = [
        ("ModernBERT", "encoder-modernbert-base-mage", "raid_eval.json"),
        ("RoBERTa", "encoder-roberta-base-mage", "raid_eval.json"),
        ("Fast-DetectGPT", "fast-detect-gpt-neo2.7b", "raid_eval.json"),
        ("Binoculars 0.5B", "binoculars-qwen2.5-0.5b", "raid_eval.json"),
        ("TF-IDF", "tfidf-logreg-mage", "raid_eval.json"),
    ]
    grid = np.zeros((len(methods), len(ATTACK_ORDER)))
    for i, (_, run, f) in enumerate(methods):
        slices = load(run, f)["metrics"]["slices"]["attack"]
        for j, att in enumerate(ATTACK_ORDER):
            grid[i, j] = slices[att]["tpr_at_fpr_0.05"]

    fig, ax = plt.subplots(figsize=(9.5, 3.1))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(ATTACK_ORDER)), ATTACK_LABELS, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(methods)), [m[0] for m in methods], fontsize=9)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                    fontsize=7.5, color="white" if v < 0.45 else "black")
    ax.set_title(
        "TPR @ 5% FPR by attack — encoders and statistical scorers break in opposite places",
        fontsize=10.5, loc="left", color="#333333",
    )
    fig.colorbar(im, ax=ax, shrink=0.85, label="TPR @ 5% FPR")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "attack_heatmap.svg", bbox_inches="tight")
    fig.savefig(OUT / "attack_heatmap.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_defense() -> None:
    panels = [
        ("Fast-DetectGPT (Neo-2.7B)", "fdg-neo-defense"),
        ("ModernBERT-base (MAGE)", "modernbert-defense"),
    ]
    # one shared row order (mean clean TPR across both panels) — with sharey, a
    # per-panel order would let the second panel's labels overwrite the first's
    cleans = {
        run: load(run, "raid_eval.json")["metrics"]["slices"]["attack"] for _, run in panels
    }
    order = sorted(
        ATTACK_ORDER,
        key=lambda a: np.mean([cleans[r][a]["tpr_at_fpr_0.05"] for _, r in panels]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, (title, run) in zip(axes, panels, strict=True):
        clean = cleans[run]
        norm = load(run, "raid_eval_norm.json")["metrics"]["slices"]["attack"]
        y = np.arange(len(order))
        c = np.array([clean[a]["tpr_at_fpr_0.05"] for a in order])
        n = np.array([norm[a]["tpr_at_fpr_0.05"] for a in order])
        ax.hlines(y, c, n, color="#cccccc", lw=1.5, zorder=2)
        ax.scatter(c, y, s=42, color=BLUE, zorder=3, label="clean input")
        ax.scatter(n, y, s=42, color=VERMILLION, zorder=3, label="normalized input")
        ax.set_yticks(y, [a.replace("_", " ") for a in order], fontsize=8.5)
        ax.set_title(title, fontsize=10.5, loc="left", color="#333333")
        ax.set_xlim(0, 0.85)
        ax.grid(axis="x", color="#eeeeee", zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("TPR @ 5% FPR", fontsize=9)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(
        "Unicode input normalization restores the character-level attacks and costs nothing elsewhere",
        fontsize=10.5, x=0.01, ha="left", color="#333333",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "defense_dumbbell.svg", bbox_inches="tight")
    fig.savefig(OUT / "defense_dumbbell.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_main_results()
    fig_attack_heatmap()
    fig_defense()
    print(f"wrote figures to {OUT}/")
