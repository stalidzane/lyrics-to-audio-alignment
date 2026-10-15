"""
Score saved predictions against the Jamendo references.

Reads predictions/<dataset>/<system>/<song>.json in either format:
    {"words":  [{text,start,...}, ...]}   a transcript (plain Whisper):
                                          matched to the reference with eval/word_match.py
    {"onsets": [t0, t1, ...]}             one time per reference word (forced aligners;
                                          null for words the aligner skipped)

Writes:
    results/<dataset>/<system>.csv        one row per song (same columns as the July CSV)
    results/<dataset>/words/<system>.csv  one row per reference word (for plots/error analysis)
    results/<dataset>/summary.csv         all systems x language, macro averages
and prints a summary table.

Run from the repo root:
    python3 -m eval.score                     # every folder in predictions/
    python3 -m eval.score whisper-large-v3-mix --dataset jamendo
    python3 -m eval.score whisper-large-v3-mix --dataset jamendopp
    python3 -m eval.score --dataset company
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from eval.metrics import DEFAULT_TOLERANCES, aggregate, evaluate_song
from eval.word_match import align_predictions

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# set from --dataset in main()
PRED_DIR = os.path.join(REPO_ROOT, "predictions", "jamendo")
REF_DIR = os.path.join(REPO_ROOT, "data", "jamendo", "refs")
RES_DIR = os.path.join(REPO_ROOT, "results", "jamendo")
SUMMARY_TOLS = (0.05, 0.1, 0.3)


def load_ref(name: str) -> dict:
    with open(os.path.join(REF_DIR, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def predicted_onsets(pred: dict, ref_words: list[str]) -> np.ndarray:
    if "onsets" in pred:
        on = np.array([np.nan if t is None else float(t) for t in pred["onsets"]])
        if len(on) != len(ref_words):
            raise ValueError(f"{len(on)} onsets for {len(ref_words)} reference words")
        return on
    return align_predictions(ref_words, pred["words"])


def score_system(system: str):
    folder = os.path.join(PRED_DIR, system)
    results, word_rows = [], []
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(folder, fn), encoding="utf-8") as f:
            pred = json.load(f)
        name = fn[:-5]
        ref = load_ref(name)
        ref_words = [w["text"] for w in ref["words"]]
        gt = np.array([w["start"] for w in ref["words"]], dtype=float)
        on = predicted_onsets(pred, ref_words)
        r = evaluate_song(name, ref["language"], on, gt, method=system)
        results.append(r)
        for i, (w, g, p) in enumerate(zip(ref_words, gt, on)):
            word_rows.append({"song_id": name, "language": ref["language"], "idx": i,
                              "word": w, "gt": g, "pred": p,
                              "error": (p - g) if not np.isnan(p) else np.nan})
    return results, word_rows


def write_song_csv(results, path):
    tols = tuple(float(t) for t in DEFAULT_TOLERANCES)
    fields = (["song_id", "language", "method", "n_words", "n_matched", "coverage",
               "aae", "median_ae"] + [f"pco_{t}" for t in tols]
              + [f"pco_matched_{t}" for t in tols])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            m = r.metrics
            row = {"song_id": r.song_id, "language": r.language, "method": r.method,
                   **{k: m[k] for k in ("n_words", "n_matched", "coverage", "aae", "median_ae")}}
            row.update({f"pco_{t}": m["pco"][t] for t in tols})
            row.update({f"pco_matched_{t}": m["pco_matched"][t] for t in tols})
            w.writerow(row)


def write_word_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["song_id", "language", "idx", "word", "gt", "pred", "error"])
        w.writeheader()
        w.writerows(rows)


def summary_rows(system, results):
    agg = aggregate(results)
    blocks = [("all", agg["overall"]["macro"])] + [
        (lang, b["macro"]) for lang, b in sorted(agg["by_language"].items())]
    out = []
    for lang, m in blocks:
        n = m["n_songs"]
        row = {"system": system, "language": lang, "n_songs": n,
               "coverage": m["coverage"], "median_ae": m["median_ae"], "aae": m["aae"]}
        row.update({f"pco_{t}": m["pco"][t] for t in SUMMARY_TOLS})
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="*", help="prediction folders (default: all)")
    ap.add_argument("--dataset", default="jamendo", help="jamendo, jamendopp or company")
    args = ap.parse_args()
    global PRED_DIR, REF_DIR, RES_DIR
    PRED_DIR = os.path.join(REPO_ROOT, "predictions", args.dataset)
    REF_DIR = os.path.join(REPO_ROOT, "data", args.dataset, "refs")
    RES_DIR = os.path.join(REPO_ROOT, "results", args.dataset)
    systems = args.systems or sorted(
        d for d in os.listdir(PRED_DIR) if os.path.isdir(os.path.join(PRED_DIR, d)))

    os.makedirs(os.path.join(RES_DIR, "words"), exist_ok=True)
    summary_path = os.path.join(RES_DIR, "summary.csv")
    existing = []
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            existing = [r for r in csv.DictReader(f) if r["system"] not in systems]

    new = []
    for s in systems:
        results, words = score_system(s)
        if not results:
            print(f"{s}: no predictions yet")
            continue
        write_song_csv(results, os.path.join(RES_DIR, f"{s}.csv"))
        write_word_csv(words, os.path.join(RES_DIR, "words", f"{s}.csv"))
        new += summary_rows(s, results)

    rows = existing + new
    fields = (["system", "language", "n_songs", "coverage", "median_ae", "aae"]
              + [f"pco_{t}" for t in SUMMARY_TOLS])
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"{'system':<28} {'lang':<4} {'songs':>5} {'cover':>6} {'medAE':>6} "
          + " ".join(f"{'PCO@'+str(t):>8}" for t in SUMMARY_TOLS))
    for r in rows:
        f_ = lambda x: f"{float(x):.3f}"
        print(f"{r['system']:<28} {r['language']:<4} {int(r['n_songs']):>5} {f_(r['coverage']):>6} "
              f"{f_(r['median_ae']):>6} " + " ".join(f"{f_(r[f'pco_{t}']):>8}" for t in SUMMARY_TOLS))
    print(f"\nmacro averages (each song weighted equally) -> {summary_path}")


if __name__ == "__main__":
    main()
