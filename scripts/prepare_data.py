"""Download and normalize datasets to the unified schema.

Usage:
    uv run python scripts/prepare_data.py mage hc3
    uv run python scripts/prepare_data.py raid          # ~2.4GB download
"""

import argparse
from pathlib import Path

from detector.data.loaders import PREPARERS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", choices=sorted(PREPARERS))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    for name in args.datasets:
        print(f"preparing {name} ...")
        for path in PREPARERS[name](args.out):
            print(f"  wrote {path}")
