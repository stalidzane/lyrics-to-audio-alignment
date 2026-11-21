"""
Robustness of the constant offset correction: is an offset estimated on OTHER
songs as good as the in-sample one used in the thesis tables?

For every system, input and dataset:
  in-sample   one offset from all songs, applied to all songs (= corr@t in eval/score.py)
  leave-one-out  for each song, the offset is estimated from all OTHER songs of the dataset
              and applied to the held-out song (the realistic case: calibrate on songs that
              are already timed, apply to a new song)
  k songs     the offset is estimated from k randomly drawn songs and applied to all the
              remaining songs ("calibrate once on a few timed songs"); repeated N_DRAWS
              times: mean and 5-95 % range of the macro PCO over draws, and the 90th
              percentile of |k-song offset - in-sample offset|

Offset estimate as in score.py: median signed error of the words within 1 s, pooled over the
calibration songs. PCO: share of reference words within the tolerance, unmatched words count
as wrong, averaged over songs (failed songs count as 0).

    python3 eval/offset_loo.py                       # all three datasets
    python3 eval/offset_loo.py --dataset company
Writes results/offset_robustness.csv and prints a table.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.word_match import align_predictions  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOLS = (0.05, 0.1, 0.3)
K_SONGS = (1, 3, 5, 10)
K_TOLS = (0.05, 0.1)
N_DRAWS = 500
WINDOW = 1.0
SEED = 0


def load_errors(dataset: str, system: str) -> dict[str, np.ndarray]:
    """per song: predicted - reference onset for every reference word (NaN = no time)"""
    pred_dir = os.path.join(REPO, "predictions", dataset, system)
    ref_dir = os.path.join(REPO, "data", dataset, "refs")
    errs = {}
    for fn in sorted(os.listdir(pred_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(pred_dir, fn), encoding="utf-8") as f:
            pred = json.load(f)
        with open(os.path.join(ref_dir, fn), encoding="utf-8") as f:
            ref = json.load(f)
        gt = np.array([w["start"] for w in ref["words"]], float)
        if "onsets" in pred:
            on = np.array([np.nan if t is None else float(t) for t in pred["onsets"]], float)
        else:
            on = align_predictions([w["text"] for w in ref["words"]], pred["words"])
        errs[fn[:-5]] = on - gt
    return errs


def estimate(err_list) -> float:
    e = np.concatenate(err_list) if err_list else np.array([])
    e = e[~np.isnan(e)]
    near = e[np.abs(e) < WINDOW]
    return float(np.median(near)) if near.size else (float(np.median(e)) if e.size else 0.0)


def pco(err: np.ndarray, tol: float) -> float:
    return float(np.mean(np.abs(np.nan_to_num(err, nan=np.inf)) <= tol))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=["jamendo", "jamendopp", "company"])
    datasets = ap.parse_args().dataset
    rng = np.random.default_rng(SEED)
    rows = []
    for ds in datasets:
        pred_root = os.path.join(REPO, "predictions", ds)
        systems = sorted(d for d in os.listdir(pred_root)
                         if os.path.isdir(os.path.join(pred_root, d))
                         and not d.startswith(("_", ".")) and not d.endswith("nolang"))
        for system in systems:
            errs = load_errors(ds, system)
            names = list(errs)
            if len(names) < 3:
                continue
            row = {"dataset": ds, "system": system, "n_songs": len(names)}
            o_all = estimate([errs[n] for n in names])
            row["offset_in_sample"] = round(o_all, 4)
            loo = {n: estimate([errs[m] for m in names if m != n]) for n in names}
            row["offset_loo_min"] = round(min(loo.values()), 4)
            row["offset_loo_max"] = round(max(loo.values()), 4)
            for t in TOLS:
                row[f"raw_{t}"] = round(np.mean([pco(errs[n], t) for n in names]), 4)
                row[f"in_sample_{t}"] = round(np.mean([pco(errs[n] - o_all, t) for n in names]), 4)
                row[f"loo_{t}"] = round(np.mean([pco(errs[n] - loo[n], t) for n in names]), 4)
            for k in K_SONGS:
                if k >= len(names):
                    continue
                vals = {t: [] for t in K_TOLS}
                dev = []
                for _ in range(N_DRAWS):
                    cal = set(rng.choice(len(names), size=k, replace=False))
                    o = estimate([errs[names[i]] for i in cal])
                    dev.append(abs(o - o_all))
                    rest = [errs[names[i]] for i in range(len(names)) if i not in cal]
                    for t in K_TOLS:
                        vals[t].append(np.mean([pco(e - o, t) for e in rest]))
                row[f"k{k}_offset_dev_p90_ms"] = round(1000 * float(np.percentile(dev, 90)), 1)
                for t in K_TOLS:
                    row[f"k{k}_{t}_mean"] = round(float(np.mean(vals[t])), 4)
                    row[f"k{k}_{t}_p05"] = round(float(np.percentile(vals[t], 5)), 4)
                    row[f"k{k}_{t}_p95"] = round(float(np.percentile(vals[t], 95)), 4)
            rows.append(row)
            print(f"{ds:<10} {system:<24} offset {o_all:+.3f} (LOO {row['offset_loo_min']:+.3f}..{row['offset_loo_max']:+.3f})"
                  f" | corr 0.1 in-sample {row['in_sample_0.1']:.1%}  LOO {row['loo_0.1']:.1%}"
                  + "".join(f" | k={k}: {row[f'k{k}_0.1_mean']:.1%} [{row[f'k{k}_0.1_p05']:.1%}-{row[f'k{k}_0.1_p95']:.1%}]"
                            for k in K_SONGS if f"k{k}_0.1_mean" in row))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    out = os.path.join(REPO, "results", "offset_robustness.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
