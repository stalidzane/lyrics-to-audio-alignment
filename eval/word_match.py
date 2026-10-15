"""
Match a system's predicted words to the reference lyrics.

Systems that transcribe (plain Whisper) produce their own word sequence, which
differs from the reference lyrics (misheard, missing, extra words). We link the
two sequences with difflib and give each reference word the start time of the
predicted word it matched, or NaN if nothing matched. Forced aligners that
return exactly one time per reference word don't need this step.

Normalisation is Unicode-aware so es/de/fr words keep their letters
("über" stays "über", not "ber").
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Sequence

import numpy as np

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})


def normalize(word: str) -> str:
    """Lowercase, unify apostrophes, keep only letters/digits/apostrophes."""
    w = unicodedata.normalize("NFC", word).translate(_APOSTROPHES).lower()
    return re.sub(r"[^\w']|_", "", w)


def align_predictions(ref_words: Sequence[str], pred_words: Sequence[dict]) -> np.ndarray:
    """One predicted onset per reference word (NaN where unmatched).

    pred_words: list of {"text": str, "start": float, ...} in time order.
    """
    ref_norm = [normalize(w) for w in ref_words]
    pred_norm = [normalize(w["text"]) for w in pred_words]
    pred_onsets = np.full(len(ref_words), np.nan, dtype=float)

    sm = SequenceMatcher(a=ref_norm, b=pred_norm, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                if ref_norm[i1 + k]:  # never count two empty tokens as a match
                    pred_onsets[i1 + k] = pred_words[j1 + k]["start"]
    return pred_onsets
