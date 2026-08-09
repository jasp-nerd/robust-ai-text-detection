# Robust AI-Generated Text Detection — an open research project

> **Status: work in progress.** This repository is run as an open research project: every
> experiment is logged in [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md), the literature that
> informs it is reviewed in [`docs/literature-review.md`](docs/literature-review.md), and this
> README will grow into the final write-up — including the approaches that *didn't* work.

## Research question

Detectors of machine-generated text routinely report near-perfect scores in-distribution, then
collapse when the generator, domain, or writing style shifts — precisely the conditions of real
use. **How far can an open, reproducible detector get on out-of-distribution and adversarial
robustness, and which method families actually generalize?**

## Planned structure

```
src/detector/        # installable package: data, features, models, eval
configs/             # one YAML per experiment
scripts/             # thin CLI entry points
notebooks/           # exploration & figures only — never the source of truth
docs/                # literature review, research log, references.bib
results/             # committed metric artifacts (JSON) + figures
```

## Reproducing

```bash
uv sync
uv run pytest
```

More to come as the project develops.

## License

MIT
