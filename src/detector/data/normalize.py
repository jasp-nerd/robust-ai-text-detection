"""Unicode input normalization — a cheap defense against character-level attacks.

Motivated directly by our RAID attack table (RESEARCH_LOG 2026-08-09): homoglyph
substitution collapses every detector (ModernBERT 0.71→0.03 TPR@5%) and zero-width
insertion collapses the encoders. Both attacks only work because the tokenizer sees
different bytes than the reader sees; normalizing the input should undo them.

Three steps: NFKC normalization, removal of invisible/zero-width code points, and a
small confusables map for the Cyrillic/Greek lookalikes that NFKC leaves untouched.
"""

from __future__ import annotations

import unicodedata

# Zero-width and invisible formatting characters (the RAID zero_width_space attack
# inserts U+200B; the rest are close cousins worth stripping).
_INVISIBLE = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0xFEFF, 0x00AD]
)

# Common Cyrillic/Greek homoglyphs of Latin letters (the high-frequency subset used by
# homoglyph attacks; intentionally small and auditable rather than exhaustive).
_CONFUSABLES = str.maketrans(
    {
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
        "і": "i", "ѕ": "s", "ј": "j", "ԁ": "d", "һ": "h", "ո": "n", "ν": "v",
        "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
        "Р": "P", "С": "C", "Т": "T", "Х": "X", "Ѕ": "S", "І": "I", "Ј": "J",
        "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
        "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
        "ο": "o", "α": "a",
    }
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_INVISIBLE)
    return text.translate(_CONFUSABLES)
