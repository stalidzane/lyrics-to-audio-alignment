"""
Human agreement on the songs timed again for the thesis.

For each song, every pair of timings is compared word by word. All timings were made on the same
audio file. Words are matched by text, as for Whisper, because the lyrics were slightly corrected
in the new timings (e.g. "streetlamp" -> "street lamp"); unmatched words count as wrong.

Timings compared (rhythm-game songs):
    reference   the timing used as reference in the evaluation (data/company/refs/<slug>.json)
    annotator   the new timing by the second annotator in the bucket (--annotator = their user name)
    linda       the author's own new timing (data/company/<annotator slug>.lrc)
    + Mr. Brightside: the timing delivered with the stems vs the reference. The mix delivered with
      the stems is a different file, shifted against the reference's track, so this one pair is
      first moved onto the same audio by cross-correlation.

    python3 experiments/agreement_check.py --annotator <user name of the second annotator>
Writes results/agreement/agreement_check.csv (one row per pair) and prints a table.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.export_company import parse_lrc        # noqa: E402
from data.export_stems import decode, lag_seconds  # noqa: E402
from eval.word_match import normalize            # noqa: E402

REPO = Path(__file__).resolve().parent.parent
S3 = Path(os.path.expanduser("~/Downloads/company-s3"))
if not S3.exists():                                # the same folder as seen from the Linux VM
    S3 = Path(os.path.expanduser("~/mnt/company-s3"))
TOLS = (0.05, 0.1, 0.3)
# (title, reference slug, annotator slug)
SONGS = [("The Sound of Silence", "xIImn_1oJ", "txJTzNtwU"),
         ("Dancing Queen", "EEiitGqQ8H", "SQuK69v0xC"),
         ("Guerrilla Radio", "R1LGghIk4O", "Zz7MUQRbsp"),
         ("Pipars", "kw2gggCCBe", "oM_tBMQCJ")]


def lrc_words(path: Path) -> list[tuple[str, float]]:
    return [(w, t) for line in parse_lrc(path, set()) for w, t in line]


def ref_words(slug: str) -> tuple[list[tuple[str, float]], Path]:
    ref = json.loads((REPO / "data" / "company" / "refs" / f"{slug}.json").read_text(encoding="utf-8"))
    return [(w["text"], float(w["start"])) for w in ref["words"]], S3 / ref["source_audio"]


def compare(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> dict:
    """b relative to a; both already on the same audio file"""
    sm = SequenceMatcher(a=[normalize(w) for w, _ in a], b=[normalize(w) for w, _ in b], autojunk=False)
    pairs = [(a[i][1], b[j][1]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag == "equal"
             for i, j in zip(range(i1, i2), range(j1, j2))]
    d = np.array([y - x for x, y in pairs])
    near = d[np.abs(d) < 1]
    off = float(np.median(near)) if near.size else float("nan")
    row = {"words": len(a), "matched": len(pairs), "offset_s": round(off, 3),
           "identical_pct": round(100 * float(np.mean(np.abs(d) < 0.001)), 1)}
    for t in TOLS:
        row[f"pco_{t}"] = round(100 * len(d[np.abs(d) <= t]) / len(a), 1)   # unmatched words count as wrong
        row[f"corr_{t}"] = round(100 * len(d[np.abs(d - off) <= t]) / len(a), 1)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", required=True, help="bucket user name of the second annotator")
    annotator = ap.parse_args().annotator
    rows = []
    for title, ref_slug, ann_slug in SONGS:
        ref, _ = ref_words(ref_slug)
        ann = lrc_words(S3 / "users" / annotator / ann_slug / f"{ann_slug}.lrc")
        mine = REPO / "data" / "company" / f"{ann_slug}.lrc"
        timings = {"reference": ref, "annotator": ann}
        if mine.exists():
            timings["linda"] = lrc_words(mine)
        names = list(timings)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = compare(timings[names[i]], timings[names[j]])
                rows.append({"song": title, "pair": f"{names[i]} vs {names[j]}", "file_shift_ms": 0.0, **r})
    # Mr. Brightside: timing delivered with the stems vs the reference
    ref, ref_track = ref_words("rZbPhqYcA")
    stem_dir = next((REPO / "data" / "company" / "blaster tracks [ai thesis]").glob("*mr brightside*"))
    new = lrc_words(next(stem_dir.glob("rZbPhqYcA*.lrc")))
    mix = next(p for p in stem_dir.glob("*.mp3") if "vox" not in p.name.lower())
    shift = lag_seconds(decode(ref_track, 8000), decode(mix, 8000), 8000)
    rows.append({"song": "Mr. Brightside", "pair": "reference vs stems delivery",
                 "file_shift_ms": round(shift * 1000, 1), **compare([(w, t + shift) for w, t in ref], new)})

    out = REPO / "results" / "agreement"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "agreement_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'song':<21} {'pair':<28} {'shift':>6} {'ident':>6} {'offset':>7}  "
          + "  ".join(f"raw{t:<5} corr{t:<4}" for t in TOLS))
    for r in rows:
        print(f"{r['song']:<21} {r['pair']:<28} {r['file_shift_ms']:>+6.1f} {r['identical_pct']:>5.1f}% "
              f"{r['offset_s']:>+7.3f}  " + "  ".join(f"{r[f'pco_{t}']:>7.1f} {r[f'corr_{t}']:>8.1f}" for t in TOLS))
    print("-> results/agreement/agreement_check.csv")


if __name__ == "__main__":
    main()
