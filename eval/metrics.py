"""
Evaluation metrics for word-level lyrics-to-audio alignment.

Given, for each reference word, a *predicted* onset time and the *ground-truth*
onset time, this module computes the standard alignment metrics used in the
lyrics-alignment literature (JamendoLyrics benchmark and follow-ups):

    - AAE       : Average Absolute Error (seconds)
    - median AE : Median Absolute Error (seconds)
    - PCO@t     : Percentage of Correct Onsets within tolerance t seconds
                  (a.k.a. "percentage of correct segments"), reported at
                  t = 0.03, 0.05, 0.1, 0.2, 0.3 s.
    - coverage  : fraction of reference words that actually received a
                  prediction (Whisper / an aligner may drop words).

Design notes
------------
* The unit of evaluation is a single reference word. You are expected to have
  already established the correspondence "predicted onset for reference word i"
  <-> "ground-truth onset for reference word i" (e.g. via text alignment of
  Whisper's transcript to the reference lyrics). This module does NOT do that
  matching -- it only scores an already-aligned pair of onset sequences.

* Missing predictions are encoded as NaN in the `pred` array. They are:
    - excluded from AAE / median AE (you can't measure an error you didn't make),
    - counted as INCORRECT in the primary PCO ("pco", denominator = all words),
      because for a karaoke product a missing word is a failure.
  A secondary PCO over matched words only ("pco_matched") is also reported so
  you can separate "aligner accuracy" from "aligner coverage".

* Two ways to aggregate across songs are provided:
    - macro: mean of per-song metrics (each song weighted equally). This is the
      convention most alignment papers report, so it is the default shown in the
      summary table.
    - micro: pool all words together (each word weighted equally). Useful when
      songs differ a lot in length.

Dependencies: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

DEFAULT_TOLERANCES: tuple[float, ...] = (0.03, 0.05, 0.1, 0.2, 0.3)


# --------------------------------------------------------------------------- #
# Core per-song computation
# --------------------------------------------------------------------------- #
def absolute_errors(pred: Sequence[float], gt: Sequence[float]) -> np.ndarray:
    """Element-wise |pred - gt|.

    `pred` may contain NaN for reference words that received no prediction;
    the corresponding error is NaN. Raises if shapes differ.
    """
    pred_arr = np.asarray(pred, dtype=float)
    gt_arr = np.asarray(gt, dtype=float)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(
            f"pred and gt must have the same shape, got "
            f"{pred_arr.shape} vs {gt_arr.shape}"
        )
    if np.isnan(gt_arr).any():
        raise ValueError("ground-truth onsets must not contain NaN")
    return np.abs(pred_arr - gt_arr)


def compute_metrics(
    pred: Sequence[float],
    gt: Sequence[float],
    tolerances: Iterable[float] = DEFAULT_TOLERANCES,
) -> dict:
    """Compute alignment metrics for one song (or any pooled set of words).

    Returns a dict with:
        n_words     : total reference words
        n_matched   : reference words that received a (non-NaN) prediction
        coverage    : n_matched / n_words
        aae         : mean absolute error over matched words (seconds)
        median_ae   : median absolute error over matched words (seconds)
        pco         : {t: fraction correct within t, unmatched counted wrong}
        pco_matched : {t: fraction correct within t, over matched words only}
    """
    tolerances = tuple(float(t) for t in tolerances)
    ae = absolute_errors(pred, gt)
    n_words = int(ae.size)
    matched_mask = ~np.isnan(ae)
    n_matched = int(matched_mask.sum())
    matched_ae = ae[matched_mask]

    if n_words == 0:
        aae = float("nan")
        median_ae = float("nan")
        pco = {t: float("nan") for t in tolerances}
        pco_matched = {t: float("nan") for t in tolerances}
    else:
        aae = float(matched_ae.mean()) if n_matched else float("nan")
        median_ae = float(np.median(matched_ae)) if n_matched else float("nan")
        pco = {
            t: float((matched_ae <= t).sum()) / n_words for t in tolerances
        }
        if n_matched:
            pco_matched = {
                t: float((matched_ae <= t).sum()) / n_matched
                for t in tolerances
            }
        else:
            pco_matched = {t: float("nan") for t in tolerances}

    return {
        "n_words": n_words,
        "n_matched": n_matched,
        "coverage": (n_matched / n_words) if n_words else float("nan"),
        "aae": aae,
        "median_ae": median_ae,
        "pco": pco,
        "pco_matched": pco_matched,
    }


# --------------------------------------------------------------------------- #
# Per-song container + dataset-level aggregation
# --------------------------------------------------------------------------- #
@dataclass
class SongResult:
    """Metrics for a single song, tagged with an id, language, and method."""

    song_id: str
    language: str
    metrics: dict
    method: str = "baseline"
    abs_errors: np.ndarray = field(default_factory=lambda: np.empty(0))


def evaluate_song(
    song_id: str,
    language: str,
    pred: Sequence[float],
    gt: Sequence[float],
    method: str = "baseline",
    tolerances: Iterable[float] = DEFAULT_TOLERANCES,
) -> SongResult:
    """Score one song and return a SongResult (keeps the raw errors around)."""
    ae = absolute_errors(pred, gt)
    metrics = compute_metrics(pred, gt, tolerances)
    return SongResult(
        song_id=song_id,
        language=language,
        metrics=metrics,
        method=method,
        abs_errors=ae,
    )


def _aggregate_macro(results: Sequence[SongResult], tolerances) -> dict:
    """Mean of per-song metrics (each song weighted equally)."""
    aae = np.nanmean([r.metrics["aae"] for r in results])
    median_ae = np.nanmean([r.metrics["median_ae"] for r in results])
    coverage = np.nanmean([r.metrics["coverage"] for r in results])
    pco = {
        t: np.nanmean([r.metrics["pco"][t] for r in results])
        for t in tolerances
    }
    pco_matched = {
        t: np.nanmean([r.metrics["pco_matched"][t] for r in results])
        for t in tolerances
    }
    return {
        "n_songs": len(results),
        "aae": float(aae),
        "median_ae": float(median_ae),
        "coverage": float(coverage),
        "pco": {t: float(v) for t, v in pco.items()},
        "pco_matched": {t: float(v) for t, v in pco_matched.items()},
    }


def _aggregate_micro(results: Sequence[SongResult], tolerances) -> dict:
    """Pool all words across songs (each word weighted equally)."""
    all_ae = np.concatenate([r.abs_errors for r in results]) if results else np.empty(0)
    matched = all_ae[~np.isnan(all_ae)]
    n_words = int(all_ae.size)
    n_matched = int(matched.size)
    aae = float(matched.mean()) if n_matched else float("nan")
    median_ae = float(np.median(matched)) if n_matched else float("nan")
    pco = {
        t: (float((matched <= t).sum()) / n_words if n_words else float("nan"))
        for t in tolerances
    }
    pco_matched = {
        t: (float((matched <= t).sum()) / n_matched if n_matched else float("nan"))
        for t in tolerances
    }
    return {
        "n_words": n_words,
        "n_matched": n_matched,
        "coverage": (n_matched / n_words) if n_words else float("nan"),
        "aae": aae,
        "median_ae": median_ae,
        "pco": pco,
        "pco_matched": pco_matched,
    }


def aggregate(
    results: Sequence[SongResult],
    tolerances: Iterable[float] = DEFAULT_TOLERANCES,
) -> dict:
    """Aggregate per-song results overall and grouped by language.

    Returns:
        {
          "overall": {"macro": {...}, "micro": {...}},
          "by_language": {lang: {"macro": {...}, "micro": {...}}, ...},
        }
    """
    tolerances = tuple(float(t) for t in tolerances)
    if not results:
        raise ValueError("no results to aggregate")

    out = {
        "overall": {
            "macro": _aggregate_macro(results, tolerances),
            "micro": _aggregate_micro(results, tolerances),
        },
        "by_language": {},
    }
    languages = sorted({r.language for r in results})
    for lang in languages:
        subset = [r for r in results if r.language == lang]
        out["by_language"][lang] = {
            "macro": _aggregate_macro(subset, tolerances),
            "micro": _aggregate_micro(subset, tolerances),
        }
    return out


# --------------------------------------------------------------------------- #
# Pretty printing (no pandas dependency)
# --------------------------------------------------------------------------- #
def _fmt(x: float) -> str:
    return "  nan" if x != x else f"{x:5.3f}"  # x!=x is True only for NaN


def per_song_table(
    results: Sequence[SongResult],
    tolerances: Iterable[float] = DEFAULT_TOLERANCES,
) -> str:
    tolerances = tuple(float(t) for t in tolerances)
    header = (
        f"{'song':<20}{'lang':<6}{'method':<12}{'n':>5}{'cov':>7}"
        f"{'AAE':>8}{'medAE':>8}"
        + "".join(f"{'PCO@'+str(t):>9}" for t in tolerances)
    )
    lines = [header, "-" * len(header)]
    for r in results:
        m = r.metrics
        row = (
            f"{r.song_id:<20}{r.language:<6}{r.method:<12}"
            f"{m['n_words']:>5}{_fmt(m['coverage']):>7}"
            f"{_fmt(m['aae']):>8}{_fmt(m['median_ae']):>8}"
            + "".join(f"{_fmt(m['pco'][t]):>9}" for t in tolerances)
        )
        lines.append(row)
    return "\n".join(lines)


def summary_table(
    agg: dict,
    tolerances: Iterable[float] = DEFAULT_TOLERANCES,
    averaging: str = "macro",
) -> str:
    """Render the by-language + overall summary (default: macro averaging)."""
    tolerances = tuple(float(t) for t in tolerances)
    header = (
        f"{'group':<16}{'avg':<7}{'cov':>7}{'AAE':>8}{'medAE':>8}"
        + "".join(f"{'PCO@'+str(t):>9}" for t in tolerances)
    )
    lines = [header, "-" * len(header)]

    def add_row(name: str, block: dict):
        lines.append(
            f"{name:<16}{averaging:<7}{_fmt(block['coverage']):>7}"
            f"{_fmt(block['aae']):>8}{_fmt(block['median_ae']):>8}"
            + "".join(f"{_fmt(block['pco'][t]):>9}" for t in tolerances)
        )

    for lang, blocks in agg["by_language"].items():
        add_row(lang, blocks[averaging])
    add_row("OVERALL", agg["overall"][averaging])
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Demo on dummy predictions
# --------------------------------------------------------------------------- #
def _demo() -> None:
    """Fabricate a few songs and print the metric tables end-to-end."""
    rng = np.random.default_rng(0)
    songs = [
        ("song_en_1", "en", 60),
        ("song_en_2", "en", 45),
        ("song_de_1", "de", 50),
        ("song_es_1", "es", 40),
    ]
    results = []
    for song_id, lang, n in songs:
        gt = np.cumsum(rng.uniform(0.2, 0.8, size=n))  # increasing onset times
        pred = gt + rng.normal(0.0, 0.15, size=n)      # noisy predictions
        # simulate ~10% dropped words (Whisper misses some)
        drop = rng.random(n) < 0.10
        pred[drop] = np.nan
        results.append(evaluate_song(song_id, lang, pred, gt, method="whisper_raw"))

    print("=== Per-song ===")
    print(per_song_table(results))
    print("\n=== Summary (macro = mean over songs) ===")
    print(summary_table(aggregate(results), averaging="macro"))
    print("\n=== Summary (micro = pooled words) ===")
    print(summary_table(aggregate(results), averaging="micro"))


if __name__ == "__main__":
    _demo()
