"""
System 4: Montreal Forced Aligner (Kaldi HMM-GMM), pretrained models, no training.

Run inside the `mfa` conda env, from the thesis repo root:
    conda activate mfa
    python3 -u experiments/run_mfa.py --dataset company --songs xIImn_1oJ R1LGghIk4O   # test
    python3 -u experiments/run_mfa.py --dataset jamendo --input vocals

Per language it builds a temporary MFA corpus (<song>.wav linked + <song>.lab with the
reference lyrics on one line), runs `mfa align` once, reads the "words" tier of each
TextGrid and writes predictions/<dataset>/mfa-<input>/<song>.json.
Songs whose language has no model (Latvian) are skipped and listed at the end.
Words MFA does not know (not in the dictionary) are still aligned, as <unk>/spn.
Songs MFA cannot align at all are written with all onsets null, so they count as
failures (coverage 0) instead of silently disappearing from the averages.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# language -> (acoustic model, dictionary); names as downloaded with `mfa model download`
MODELS = {
    "en": ("english_mfa", "english_us_mfa"),
    "es": ("spanish_mfa", "spanish_mfa"),
    "de": ("german_mfa", "german_mfa"),
    "fr": ("french_mfa", "french_mfa"),
    "it": ("italian_cv", "italian_cv"),
    "pt": ("portuguese_mfa", "portuguese_mfa"),
    "sv": ("swedish_mfa", "swedish_mfa"),
    "ro": ("romanian_cv", "romanian_cv"),
}


def input_path(data_dir: str, name: str, which: str) -> str:
    if which == "vocals":
        return os.path.join(data_dir, "separated", "htdemucs", name, "vocals.wav")
    return os.path.join(data_dir, "audio", f"{name}.wav")


def read_word_tier(path: str) -> list[dict]:
    """Minimal TextGrid reader: intervals of the tier named 'words', silences dropped."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    tiers = re.split(r'\n\s*item \[\d+\]:', text)
    for tier in tiers:
        if re.search(r'name = "words"', tier):
            out = []
            for xmin, xmax, label in re.findall(
                    r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "(.*)"', tier):
                if label.strip():
                    out.append({"text": label, "start": float(xmin), "end": float(xmax)})
            return out
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="jamendo")
    ap.add_argument("--input", choices=["mix", "vocals"], default="vocals")
    ap.add_argument("--languages", nargs="+")
    ap.add_argument("--songs", nargs="+")
    ap.add_argument("--beam", type=int, default=100)
    ap.add_argument("--retry_beam", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--system", help="output folder name (default mfa-<input>)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep_tmp", action="store_true")
    args = ap.parse_args()

    data_dir = os.path.join(REPO_ROOT, "data", args.dataset)
    system = args.system or f"mfa-{args.input}"
    out_dir = os.path.join(REPO_ROOT, "predictions", args.dataset, system)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(data_dir, "index.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if (not args.languages or r["language"] in args.languages)
                and (not args.songs or r["name"] in args.songs)]
    todo = [r for r in rows if args.overwrite
            or not os.path.exists(os.path.join(out_dir, f"{r['name']}.json"))]
    no_model = sorted({r["language"] for r in todo if r["language"] not in MODELS})
    print(f"{system}: {len(rows)} songs, {len(todo)} to align"
          + (f"; no MFA model for: {', '.join(no_model)}" if no_model else ""), flush=True)

    t_all, done = time.time(), 0
    for lang in sorted({r["language"] for r in todo if r["language"] in MODELS}):
        am, dic = MODELS[lang]
        songs = [r["name"] for r in todo if r["language"] == lang]
        tmp = tempfile.mkdtemp(prefix=f"mfa_{args.dataset}_{lang}_")
        corpus, aligned = os.path.join(tmp, "corpus"), os.path.join(tmp, "aligned")
        os.makedirs(corpus)
        words_by_song = {}
        for name in songs:
            src = input_path(data_dir, name, args.input)
            if not os.path.exists(src):
                print(f"  MISSING input {src} -- skipped", flush=True)
                continue
            with open(os.path.join(data_dir, "refs", f"{name}.json"), encoding="utf-8") as f:
                words = [w["text"] for w in json.load(f)["words"]]
            words_by_song[name] = words
            os.symlink(src, os.path.join(corpus, f"{name}.wav"))
            with open(os.path.join(corpus, f"{name}.lab"), "w", encoding="utf-8") as f:
                f.write(" ".join(words) + "\n")
        print(f"  [{lang}] {len(words_by_song)} songs with {am} / {dic}", flush=True)
        t0 = time.time()
        cmd = ["mfa", "align", corpus, dic, am, aligned, "--clean", "--overwrite",
               "--beam", str(args.beam), "--retry_beam", str(args.retry_beam), "-j", str(args.jobs)]
        res = subprocess.run(cmd)
        secs = time.time() - t0
        if res.returncode != 0:
            # usually NoAlignmentsError: no song of this language could be aligned
            print(f"  [{lang}] mfa align failed (exit {res.returncode}); all {len(words_by_song)} "
                  f"song(s) of this language are scored as failures", flush=True)
            os.makedirs(aligned, exist_ok=True)  # empty: every song below counts as failed
        for name, words in words_by_song.items():
            tg = os.path.join(aligned, f"{name}.TextGrid")
            failed = not os.path.exists(tg)
            pred_words = [] if failed else read_word_tier(tg)
            out = {"meta": {"system": system, "dataset": args.dataset, "song": name, "language": lang,
                            "input": args.input, "backend": "mfa", "acoustic_model": am,
                            "dictionary": dic, "beam": args.beam, "retry_beam": args.retry_beam,
                            "n_ref_words": len(words), "n_aligned_words": len(pred_words),
                            "batch_runtime_s": round(secs, 1), "machine": platform.platform(),
                            "date": time.strftime("%Y-%m-%d %H:%M")},
                   "words": pred_words}
            out["meta"]["failed"] = failed
            if failed:  # count it in the scores as a song with no aligned words
                out["onsets"] = [None] * len(words)
                note = "  FAILED: MFA could not align this song (scored as 0 words)"
            elif len(pred_words) == len(words):
                out["onsets"] = [w["start"] for w in pred_words]
                note = ""
            else:
                note = f" ({len(pred_words)} of {len(words)} words; matched by text)"
            with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=1)
            done += 1
            print(f"  {name:<40} {lang} {len(pred_words):>4}/{len(words)} words{note}", flush=True)
        if not args.keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"done: {done} songs in {(time.time() - t_all) / 60:.1f} min -> {out_dir}")
    if no_model:
        print(f"not aligned (no MFA model): {', '.join(no_model)}")


if __name__ == "__main__":
    main()
