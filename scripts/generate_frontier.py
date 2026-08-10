"""Generate a small frontier-generator test set (Phase S2).

No public benchmark covers post-2024 generators, so we make a probe set ourselves:
take human documents from the RAID eval split (clean, one per title), and have a
2025-era open-weights model (Qwen3-4B-Instruct) write a text for the same title and
domain. Detectors then face a generator none of them saw in any training set.

Writes data/processed/frontier/test.parquet (human rows + generated rows).
Runs on a 24GB GPU; ~60 generations per domain by default.
"""

import argparse

import polars as pl
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-4B-Instruct-2507"

PROMPTS = {
    "abstracts": "Write a plausible scientific paper abstract titled: {title}",
    "books": "Write the opening passage of a book titled: {title}",
    "news": "Write a news article with the headline: {title}",
    "poetry": "Write a poem titled: {title}",
    "recipes": "Write a recipe for: {title}",
    "reddit": "Write a reddit post titled: {title}",
    "reviews": "Write a review with the title: {title}",
    "wiki": "Write an encyclopedia article about: {title}",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-domain", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    args = parser.parse_args()

    raid = pl.read_parquet("data/processed/raid/eval.parquet")
    humans = (
        raid.filter((pl.col("label") == 0) & (pl.col("attack") == "none"))
        .filter(pl.col("domain").is_in(list(PROMPTS)))
        .unique(subset=["title"], keep="first")
        .group_by("domain", maintain_order=True)
        .head(args.per_domain)
    )
    print(f"seed titles: {len(humans)} across {humans['domain'].n_unique()} domains")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    model.to("cuda").eval()

    rows = []
    for r in tqdm(humans.iter_rows(named=True), total=len(humans), desc="generate"):
        prompt = PROMPTS[r["domain"]].format(title=r["title"])
        messages = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()
        if len(text.split()) < 30:
            continue
        rows.append(
            {
                "text": text,
                "label": 1,
                "generator": "qwen3-4b-2507",
                "domain": r["domain"],
                "attack": "none",
                "decoding": "sampling",
                "source_dataset": "frontier",
            }
        )
        rows.append(
            {
                "text": r["text"],
                "label": 0,
                "generator": "human",
                "domain": r["domain"],
                "attack": "none",
                "decoding": "unknown",
                "source_dataset": "frontier",
            }
        )
    out_df = pl.DataFrame(rows)
    out_path = "data/processed/frontier/test.parquet"
    import pathlib

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out_path)
    print(f"wrote {len(out_df)} rows ({out_df['label'].mean():.1%} machine) to {out_path}")


if __name__ == "__main__":
    main()
