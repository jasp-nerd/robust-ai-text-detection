"""Unified sample schema.

Every dataset ingested by this project is normalized to these fields so that all
detectors face identical inputs and the eval harness can slice by generator, domain,
and attack without dataset-specific code.
"""

from __future__ import annotations

from dataclasses import dataclass

HUMAN = 0
MACHINE = 1


@dataclass(frozen=True, slots=True)
class Sample:
    text: str
    label: int  # HUMAN (0) or MACHINE (1)
    generator: str  # e.g. "gpt-4o", or "human"
    domain: str  # e.g. "news", "reddit", "abstracts"
    attack: str  # adversarial attack applied, "none" if clean
    decoding: str  # decoding strategy if known, else "unknown"
    source_dataset: str  # e.g. "raid", "mage"

    def __post_init__(self) -> None:
        if self.label not in (HUMAN, MACHINE):
            raise ValueError(f"label must be {HUMAN} or {MACHINE}, got {self.label}")
