"""
Human agreement: compare different annotators' timings of the same company song.

    lyrics2audio/bin/python experiments/agreement.py
    lyrics2audio/bin/python experiments/agreement.py --exclude stalidzanelinda

Reads every fully timed LRC version in ~/Downloads/company-s3/users/<user>/<slug>/<slug>.lrc
(same parsing as data/export_company.py), and for every pair of annotators who both timed
a song, matches their words by text and compares the onsets (b - a).

Writes results/agreement/pairs.csv (one row per song x annotator pair) and prints a summary:
matched share, median |difference|, signed median offset, share of words within
0.05 / 0.1 / 0.3 s, and the same after removing the pair's constant offset
(median of differences within 1 s, as in eval/score.py).

Most versions in the bucket are copies of another user's timing (mostly identical onsets);
those pairs are marked "copy" and left out of the summary. Two versions may also have been
timed against different audio uploads; a large constant offset is flagged "offset>0.5s".
"""
import argparse
import csv
import itertools
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.export_company import parse_lrc, section_labels  # noqa: E402
from eval.word_match import align_predictions  # noqa: E402

TOLS = (0.05, 0.1, 0.3)


def timed_words(lrc: Path):
    lines = parse_lrc(lrc, section_labels(lrc.with_suffix(".txt")))
    words = [w for line in lines for w in line]
    if words and not any(t == 0.0 for i, (_, t) in enumerate(words) if i > 0):
        return words
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "Downloads" / "company-s3"))
    ap.add_argument("--exclude", nargs="*", default=[], help="annotator user names to leave out")
    ap.add_argument("--max_identical", type=float, default=0.5,
                    help="pairs where more than this share of onsets are identical are copies, not "
                         "independent timings (default 0.5)")
    args = ap.parse_args()
    src = Path(args.src).expanduser()

    versions = {}  # slug -> {user: words}
    for lrc in sorted((src / "users").glob("*/*/*.lrc")):
        user, slug = lrc.parent.parent.name, lrc.parent.name
        if user in args.exclude or lrc.stem != slug:
            continue
        words = timed_words(lrc)
        if words:
            versions.setdefault(slug, {})[user] = words

    out_rows = []
    for slug, by_user in sorted(versions.items()):
        for a, b in itertools.combinations(sorted(by_user), 2):
            wa, wb = by_user[a], by_user[b]
            ref = [w for w, _ in wa]
            ta = np.array([t for _, t in wa])
            tb = align_predictions(ref, [{"text": w, "start": t} for w, t in wb])
            d = tb - ta
            d = d[~np.isnan(d)]
            if d.size < 10:
                continue
            identical = float(np.mean(np.abs(d) < 0.001))
            near = d[np.abs(d) < 1.0]
            off = float(np.median(near)) if near.size else float(np.median(d))
            row = {"song": slug, "a": a, "b": b, "n_a": len(wa), "n_b": len(wb),
                   "matched": round(d.size / len(wa), 3),
                   "median_abs": round(float(np.median(np.abs(d))), 3),
                   "identical": round(identical, 3),
                   "offset": round(off, 3),
                   "flag": ("copy" if identical > args.max_identical
                            else "offset>0.5s" if abs(off) > 0.5 else "")}
            for t in TOLS:  # over matched words
                row[f"within_{t}"] = round(float(np.mean(np.abs(d) <= t)), 3)
                row[f"corr_{t}"] = round(float(np.mean(np.abs(d - off) <= t)), 3)
            out_rows.append(row)

    out = ROOT / "results" / "agreement"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    indep = [r for r in out_rows if r["flag"] != "copy"]
    ok = [r for r in indep if not r["flag"]]
    print(f"{len(versions)} songs with timed versions; {len(out_rows)} annotator pairs on "
          f"{len({r['song'] for r in out_rows})} songs")
    print(f"{len(out_rows) - len(indep)} pairs are copies (> {args.max_identical:.0%} identical onsets) and left out; "
          f"{len(indep)} independent pairs on {len({r['song'] for r in indep})} songs, "
          f"{len(indep) - len(ok)} of them with offset > 0.5 s (different audio upload?)")
    for name, rows in (("independent pairs", indep), ("independent, offset < 0.5 s", ok)):
        if not rows:
            continue
        m = lambda k: st.mean(r[k] for r in rows)
        print(f"\n{name} (n={len(rows)}), means over pairs:")
        print(f"  matched {m('matched'):.3f}  median|diff| {st.median(r['median_abs'] for r in rows):.3f}s"
              f"  |offset| median {st.median(abs(r['offset']) for r in rows):.3f}s")
        print("  within " + "  ".join(f"{t}s: {m(f'within_{t}'):.3f}" for t in TOLS)
              + "   | after offset " + "  ".join(f"{t}s: {m(f'corr_{t}'):.3f}" for t in TOLS))
    pairs = {}
    for r in indep:
        pairs[(r["a"], r["b"])] = pairs.get((r["a"], r["b"]), 0) + 1
    print("\nindependent annotator pairs (songs):", ", ".join(f"{a}-{b}: {n}" for (a, b), n in
                                             sorted(pairs.items(), key=lambda x: -x[1])))
    print(f"-> {out / 'pairs.csv'}")


if __name__ == "__main__":
    main()
