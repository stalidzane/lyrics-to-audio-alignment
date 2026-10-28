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

Columns in the summary:
    medAE   median over songs of each song's median absolute error (robust to the few
            songs where Whisper skipped or repeated a chorus)
    offset  median signed error (pred - ref) of matched words within 1 s; negative = early
    PCO@t   share of reference words with |error| <= t; unmatched words count as wrong
    backend Whisper backend/model read from the prediction files' meta (mixed = folder
            contains more than one; fix that before reporting)
    corr@t  the same after subtracting ONE constant offset per system and dataset
            (median signed error of the words within 1 s of their reference, so a few
            completely misaligned songs cannot drag it), i.e. what a game could get with a
            single fixed latency calibration. Same matching, same denominator.

Run from the repo root:
    python3 -m eval.score                     # every folder in predictions/
    python3 -m eval.score whisper-large-v3-mix --dataset jamendo
    python3 -m eval.score whisper-large-v3-mix --dataset jamendopp
    python3 -m eval.score --dataset company

Folders starting with "_" (e.g. predictions/jamendo/_archive_openai/) are ignored.
Scoring all systems rebuilds summary.csv from scratch, so it never keeps rows for
systems that no longer exist in predictions/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import sys
import warnings

import numpy as np

# groups where a system failed on every song give empty means: expected, shown as nan
warnings.filterwarnings("ignore", category=RuntimeWarning)

# allow `python3 eval/score.py` as well as `python3 -m eval.score`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.metrics import DEFAULT_TOLERANCES, aggregate, evaluate_song
from eval.word_match import align_predictions

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# set from --dataset in main()
PRED_DIR = os.path.join(REPO_ROOT, "predictions", "jamendo")
REF_DIR = os.path.join(REPO_ROOT, "data", "jamendo", "refs")
RES_DIR = os.path.join(REPO_ROOT, "results", "jamendo")
SUMMARY_TOLS = (0.05, 0.1, 0.3)
OFFSET_WINDOW = 1.0  # s; only words within this error are used to estimate the constant offset


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
    results, word_rows, songs = [], [], []
    backends = set()
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(folder, fn), encoding="utf-8") as f:
            pred = json.load(f)
        name = fn[:-5]
        meta = pred.get("meta", {})
        backends.add(meta.get("backend", "?") if "backend" in meta else meta.get("system", "?"))
        ref = load_ref(name)
        ref_words = [w["text"] for w in ref["words"]]
        gt = np.array([w["start"] for w in ref["words"]], dtype=float)
        on = predicted_onsets(pred, ref_words)
        r = evaluate_song(name, ref["language"], on, gt, method=system)
        results.append(r)
        songs.append((name, ref["language"], on, gt))
        for i, (w, g, p) in enumerate(zip(ref_words, gt, on)):
            word_rows.append({"song_id": name, "language": ref["language"], "idx": i,
                              "word": w, "gt": g, "pred": p,
                              "error": (p - g) if not np.isnan(p) else np.nan})
    backend = backends.pop() if len(backends) == 1 else "MIXED:" + "+".join(sorted(map(str, backends)))
    if not results:
        return results, word_rows, [], float("nan"), backend
    # one constant offset per system: median signed error over all matched words
    errs = np.concatenate([on - gt for _, _, on, gt in songs])
    errs = errs[~np.isnan(errs)]
    near = errs[np.abs(errs) < OFFSET_WINDOW]   # ignore words aligned to the wrong place
    offset = float(np.median(near)) if near.size else (float(np.median(errs)) if errs.size else 0.0)
    corrected = [evaluate_song(n, lang, on - offset, gt, method=system + "+offset")
                 for n, lang, on, gt in songs]
    return results, word_rows, corrected, offset, backend


def write_song_csv(results, path, corrected=None):
    tols = tuple(float(t) for t in DEFAULT_TOLERANCES)
    fields = (["song_id", "language", "method", "failed", "n_words", "n_matched", "coverage",
               "aae", "median_ae"] + [f"pco_{t}" for t in tols]
              + [f"pco_matched_{t}" for t in tols] + [f"corr_{t}" for t in SUMMARY_TOLS])
    corr = {r.song_id: r for r in (corrected or [])}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            m = r.metrics
            row = {"song_id": r.song_id, "language": r.language, "method": r.method,
                   "failed": int(m["coverage"] == 0),
                   **{k: m[k] for k in ("n_words", "n_matched", "coverage", "aae", "median_ae")}}
            if r.song_id in corr:
                row.update({f"corr_{t}": corr[r.song_id].metrics["pco"][t] for t in SUMMARY_TOLS})
            row.update({f"pco_{t}": m["pco"][t] for t in tols})
            row.update({f"pco_matched_{t}": m["pco_matched"][t] for t in tols})
            w.writerow(row)


def write_word_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["song_id", "language", "idx", "word", "gt", "pred", "error"])
        w.writeheader()
        w.writerows(rows)


def summary_rows(system, results, corrected, words, backend):
    agg = aggregate(results)
    blocks = [("all", agg["overall"]["macro"])] + [
        (lang, b["macro"]) for lang, b in sorted(agg["by_language"].items())]
    out = []
    for lang, m in blocks:
        keep = lambda r: lang == "all" or r.language == lang
        res = [r for r in results if keep(r)]
        cor = [r for r in corrected if keep(r)]
        errs = [w["error"] for w in words
                if (lang == "all" or w["language"] == lang) and not np.isnan(w["error"])]
        row = {"system": system, "backend": backend, "language": lang, "n_songs": m["n_songs"],
               "n_failed": sum(r.metrics["coverage"] == 0 for r in res),
               "coverage": m["coverage"],
               "median_ae": float(np.nanmedian([r.metrics["median_ae"] for r in res])),
               "aae": m["aae"],
               "offset": (float(np.median([e for e in errs if abs(e) < OFFSET_WINDOW]))
                          if any(abs(e) < OFFSET_WINDOW for e in errs) else float("nan"))}
        row.update({f"pco_{t}": m["pco"][t] for t in SUMMARY_TOLS})
        row.update({f"corr_{t}": float(np.nanmean([r.metrics["pco"][t] for r in cor]))
                    for t in SUMMARY_TOLS})
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
        d for d in os.listdir(PRED_DIR)
        if os.path.isdir(os.path.join(PRED_DIR, d)) and not d.startswith(("_", ".")))

    os.makedirs(os.path.join(RES_DIR, "words"), exist_ok=True)
    summary_path = os.path.join(RES_DIR, "summary.csv")
    ok_path = os.path.join(RES_DIR, "summary_ok.csv")
    existing, existing_ok = [], []
    if args.systems and os.path.exists(summary_path):  # partial rescore: keep the other systems' rows
        with open(summary_path) as f:
            existing = [r for r in csv.DictReader(f)
                        if r["system"] not in systems and "n_failed" in r]  # drop old-format rows
        if os.path.exists(ok_path):
            with open(ok_path) as f:
                existing_ok = [r for r in csv.DictReader(f) if r["system"] not in systems]

    new, new_ok = [], []
    for s in systems:
        results, words, corrected, offset, backend = score_system(s)
        if not results:
            print(f"{s}: no predictions yet")
            continue
        write_song_csv(results, os.path.join(RES_DIR, f"{s}.csv"), corrected)
        write_word_csv(words, os.path.join(RES_DIR, "words", f"{s}.csv"))
        if backend.startswith("MIXED"):
            print(f"WARNING {s}: predictions come from different backends ({backend})")
        new += summary_rows(s, results, corrected, words, backend)
        # the same, only over the songs the system produced an alignment for
        ok = {r.song_id for r in results if r.metrics["coverage"] > 0}
        if ok:
            new_ok += summary_rows(s, [r for r in results if r.song_id in ok],
                                   [r for r in corrected if r.song_id in ok],
                                   [w for w in words if w["song_id"] in ok], backend)

    rows = existing + new
    fields = (["system", "backend", "language", "n_songs", "n_failed", "coverage", "median_ae", "aae", "offset"]
              + [f"pco_{t}" for t in SUMMARY_TOLS] + [f"corr_{t}" for t in SUMMARY_TOLS])
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(ok_path, "w", newline="") as f:  # failed songs left out (n_failed = 0 there)
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(existing_ok + new_ok)

    print(f"dataset: {args.dataset}   systems scored: {', '.join(systems)}")
    print(f"{'system':<28} {'backend':<13} {'lang':<4} {'songs':>5} {'fail':>4} {'cover':>6} {'medAE':>6} {'offset':>7} "
          + " ".join(f"{'PCO@'+str(t):>8}" for t in SUMMARY_TOLS) + " |"
          + " ".join(f"{'corr@'+str(t):>9}" for t in SUMMARY_TOLS))
    for r in rows:
        f_ = lambda x: f"{float(x):.3f}"
        print(f"{r['system']:<28} {r['backend']:<13} {r['language']:<4} {int(r['n_songs']):>5} {int(r['n_failed']):>4} {f_(r['coverage']):>6} "
              f"{f_(r['median_ae']):>6} {float(r['offset']):>+7.3f} "
              + " ".join(f"{f_(r[f'pco_{t}']):>8}" for t in SUMMARY_TOLS) + " |"
              + " ".join(f"{f_(r[f'corr_{t}']):>9}" for t in SUMMARY_TOLS))
    print(f"\nmacro averages (each song weighted equally) -> {summary_path}")
    print(f"same without the songs a system failed on (fail > 0 above) -> {ok_path}")


if __name__ == "__main__":
    main()
