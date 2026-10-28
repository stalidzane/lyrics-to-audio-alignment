"""
Write Audacity label tracks to compare the reference with a system for one song.

    python3 eval/labels.py --dataset jamendo --song Avercage_-_Embers --system mms-vocals

Writes to outputs/labels/<dataset>/<song>/:
    reference.txt          reference word onsets
    <system>.txt           the system's onsets for the same words (label = word, error in s)
    <system>_errors.csv    word, reference, predicted, error (s) -- to scan in a spreadsheet
In Audacity: open the song's wav (data/<dataset>/audio/<song>.wav), then
File > Import > Labels... for each .txt. Words more than 1 s off are marked with "!!".
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.score import predicted_onsets  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--song", required=True)
    ap.add_argument("--system", required=True, help="e.g. mms-vocals, whisper-small-mix")
    args = ap.parse_args()

    with open(os.path.join(REPO_ROOT, "data", args.dataset, "refs", f"{args.song}.json"), encoding="utf-8") as f:
        ref = json.load(f)
    with open(os.path.join(REPO_ROOT, "predictions", args.dataset, args.system, f"{args.song}.json"), encoding="utf-8") as f:
        pred = json.load(f)
    words = [w["text"] for w in ref["words"]]
    gt = [float(w["start"]) for w in ref["words"]]
    on = predicted_onsets(pred, words)

    out = os.path.join(REPO_ROOT, "outputs", "labels", args.dataset, args.song)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "reference.txt"), "w", encoding="utf-8") as f:
        for w, t in zip(words, gt):
            f.write(f"{t:.3f}\t{t:.3f}\t{w}\n")
    n_bad = 0
    with open(os.path.join(out, f"{args.system}.txt"), "w", encoding="utf-8") as f, \
         open(os.path.join(out, f"{args.system}_errors.csv"), "w", newline="", encoding="utf-8") as g:
        wr = csv.writer(g)
        wr.writerow(["idx", "word", "reference_s", "predicted_s", "error_s"])
        for i, (w, t, p) in enumerate(zip(words, gt, on)):
            if p != p:  # NaN: word not found by the system
                wr.writerow([i, w, f"{t:.3f}", "", ""])
                continue
            err = p - t
            mark = "!! " if abs(err) > 1 else ""
            n_bad += abs(err) > 1
            f.write(f"{p:.3f}\t{p:.3f}\t{mark}{w} ({err:+.2f})\n")
            wr.writerow([i, w, f"{t:.3f}", f"{p:.3f}", f"{err:+.3f}"])
    print(f"{len(words)} words, {n_bad} more than 1 s off -> {out}")
    first = next((i for i, (t, p) in enumerate(zip(gt, on)) if p == p and abs(p - t) <= 1), None)
    if first is not None:
        print(f"first word within 1 s: #{first} '{words[first]}' at {gt[first]:.1f} s (reference)")


if __name__ == "__main__":
    main()
