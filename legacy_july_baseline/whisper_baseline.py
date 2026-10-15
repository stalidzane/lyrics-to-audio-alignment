"""
Whisper baseline for word-level lyrics alignment on JamendoLyrics.

Pipeline per song:
    1. Transcribe the audio with Whisper (word_timestamps=True) -> Whisper's own
       words, each with a start time.
    2. Align Whisper's word sequence to the *reference* lyrics words. Whisper
       mishears lyrics, so the two sequences don't line up 1:1; we match the
       parts that agree and leave the rest unpredicted (NaN).
    3. For each reference word, take the matched Whisper start time as the
       predicted onset -> feed to metrics.py.

Two preprocessing conditions are provided so you can produce the "raw vs.
band-pass" comparison directly:
    - raw:      Whisper on the original mix.
    - bandpass: scipy Butterworth band-pass (~200 Hz-4 kHz) to emphasise vocals,
                then Whisper.

Word matching note
------------------
We use difflib.SequenceMatcher over normalised tokens. It only links tokens that
match exactly, so mistranscribed words become gaps (coverage < 1). That is the
honest behaviour of a plain-Whisper baseline; better aligners (WhisperX, forced
alignment) come later in the plan and will raise coverage.

Dependencies: numpy, scipy, openai-whisper (`pip install openai-whisper`),
plus data/jamendo_loader.py and eval/metrics.py.

Run from the repo root:
    python -m experiments.whisper_baseline
"""

from __future__ import annotations

import re
import csv
import os
from difflib import SequenceMatcher
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt

from data.jamendo_loader import load_songs
from eval.word_match import align_predictions  # moved + fixed for non-English letters
from eval.metrics import evaluate_song, aggregate, per_song_table, summary_table, DEFAULT_TOLERANCES


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def bandpass(audio: np.ndarray, sr: int, low: float = 200.0,
             high: float = 4000.0, order: int = 4) -> np.ndarray:
    """Butterworth band-pass to emphasise the vocal range.

    Zero-phase (sosfiltfilt) so it doesn't shift onsets in time -- important,
    since we're measuring onset timing. Tune `low`/`high` to taste.
    """
    nyq = sr / 2.0
    high = min(high, nyq - 1.0)          # guard against >= Nyquist
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


# --------------------------------------------------------------------------- #
# Whisper transcription + word matching
# --------------------------------------------------------------------------- #
def transcribe_words(model, audio: np.ndarray,
                     language: Optional[str] = None) -> list[dict]:
    """Run Whisper and flatten to a list of {text, start, end} word dicts."""
    result = model.transcribe(
        audio, word_timestamps=True, language=language, fp16=False,
    )
    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append({
                "text": w["word"].strip(),
                "start": float(w["start"]),
                "end": float(w["end"]),
            })
    return words


# --------------------------------------------------------------------------- #
# End-to-end runner
# --------------------------------------------------------------------------- #
def run_baseline(
    ds,
    model,
    n: Optional[int] = 5,
    language: Optional[str] = None,
    preprocess: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    method: str = "whisper_raw",
    verbose: bool = True,
) -> list:
    """Transcribe, align, and score each song; return list[SongResult]."""
    results = []
    for song in load_songs(ds, n=n, language=language):
        audio = song["audio"]
        if preprocess is not None:
            audio = preprocess(audio, song["sr"])
        pred_words = transcribe_words(model, audio, language=song["language"])
        pred_onsets = align_predictions(song["words"], pred_words)
        r = evaluate_song(
            song["name"], song["language"],
            pred_onsets, song["onsets"], method=method,
        )
        results.append(r)
        if verbose:
            m = r.metrics
            print(f"  {song['name']:<35} cov={m['coverage']:.2f} "
                  f"AAE={m['aae']:.3f} PCO@0.2={m['pco'][0.2]:.3f}")
    return results




def save_results_csv(results, path, tolerances=DEFAULT_TOLERANCES):
    """Append/write SongResults to a CSV, one row per song/method."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tolerances = tuple(float(t) for t in tolerances)
    fieldnames = (
        ["song_id", "language", "method", "n_words", "n_matched", "coverage",
         "aae", "median_ae"]
        + [f"pco_{t}" for t in tolerances]
        + [f"pco_matched_{t}" for t in tolerances]
    )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            m = r.metrics
            row = {
                "song_id": r.song_id,
                "language": r.language,
                "method": r.method,
                "n_words": m["n_words"],
                "n_matched": m["n_matched"],
                "coverage": m["coverage"],
                "aae": m["aae"],
                "median_ae": m["median_ae"],
            }
            for t in tolerances:
                row[f"pco_{t}"] = m["pco"][t]
                row[f"pco_matched_{t}"] = m["pco_matched"][t]
            writer.writerow(row)

if __name__ == "__main__":
    import whisper
    from datasets import load_dataset

    MODEL_NAME = "small"   # "base" = faster/worse, "medium"/"large-v3" = better/slower
    N_SONGS = 5

    print(f"Loading Whisper '{MODEL_NAME}' ...")
    model = whisper.load_model(MODEL_NAME)

    ds = load_dataset("jamendolyrics/jamendolyrics", "en", split="test")

    print("\n[raw] Whisper on original mix:")
    res_raw = run_baseline(ds, model, n=N_SONGS, method="whisper_raw")

    print("\n[bandpass] Whisper on 200Hz-4kHz band-passed audio:")
    res_bp = run_baseline(ds, model, n=N_SONGS, preprocess=bandpass,
                          method="whisper_bandpass")

    all_res = res_raw + res_bp
    save_results_csv(all_res, "results/baseline_en.csv")

    print("\n=== Per-song ===")
    print(per_song_table(all_res))
    print("\n=== raw ===")
    print(summary_table(aggregate(res_raw)))
    print("\n=== bandpass ===")
    print(summary_table(aggregate(res_bp)))
