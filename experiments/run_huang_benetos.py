"""
System 3: Huang & Benetos multilingual lyrics aligner (CRNN + CTC, phoneme posteriors,
dynamic-programming alignment), using the authors' released code and checkpoint:
    https://github.com/jhuang448/LyricsAlignment-Multilingual

Run inside the `hb` conda env (Python 3.9), from the thesis repo root:
    export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib
    python3 -u experiments/run_huang_benetos.py --hb_dir ~/uni/LyricsAlignment-Multilingual \
        --dataset jamendo --input vocals
    ... --dataset company --songs xIImn_1oJ kw2gggCCBe      # test on a few songs

What it does, per dataset/input:
  1. phonemizes each reference word with espeak exactly like their phonemize.py
     (punctuation removed, lower case, phones separated by ';', [start, end] phone indices
     with one gap between words) and writes <song>.csv to a temp annot dir;
  2. links the audio (Demucs vocals by default, as the model expects) as <song>.wav;
  3. runs their eval.py once for all songs (model loaded once; CPU unless --cuda);
  4. reads <song>_align.csv (start, end, ... per word) and writes
     predictions/<dataset>/hb-<input>/<song>.json with one onset per reference word.

Words that espeak turns into no phonemes (e.g. only punctuation) are not aligned (null).
Languages outside the model's training set (lv, pt, sv) still run; unknown phonemes
map to the model's <unk> class -- report this.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

from phonemizer.backend import EspeakBackend
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESPEAK = {"en": "en-us", "fr": "fr-fr", "de": "de", "it": "it", "es": "es",
          "pt": "pt", "sv": "sv", "lv": "lv"}
SEPARATOR = Separator(phone=";", word=None)
PUNCT = Punctuation(';:,.!"?()-')


def input_path(data_dir: str, name: str, which: str) -> str:
    if which == "vocals":
        return os.path.join(data_dir, "separated", "htdemucs", name, "vocals.wav")
    return os.path.join(data_dir, "audio", f"{name}.wav")


def phonemize_song(words: list[str], backend: EspeakBackend):
    """Rows in their format + the reference-word index of each row."""
    rows, kept = [], []
    st = ed = 0
    for i, word in enumerate(words):
        cleaned = PUNCT.remove(word).lower().strip()
        if not cleaned:
            continue
        phone = backend.phonemize([cleaned], separator=SEPARATOR, strip=True)[0].strip(";").strip()
        if not phone:
            continue
        ed += len(phone.split(";"))
        rows.append([word, phone, [st, ed]])
        kept.append(i)
        st = ed + 1
        ed = st
    return rows, kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hb_dir", required=True, help="path to the cloned LyricsAlignment-Multilingual repo")
    ap.add_argument("--checkpoint", default="checkpoints/checkpoint_Baseline", help="relative to --hb_dir")
    ap.add_argument("--dataset", default="jamendo", help="jamendo, jamendopp or company")
    ap.add_argument("--input", choices=["mix", "vocals"], default="vocals")
    ap.add_argument("--languages", nargs="+")
    ap.add_argument("--songs", nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--system", help="output folder name (default hb-<input>)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep_tmp", action="store_true", help="keep the temp folder (debugging)")
    args = ap.parse_args()

    hb_dir = os.path.abspath(os.path.expanduser(args.hb_dir))
    data_dir = os.path.join(REPO_ROOT, "data", args.dataset)
    system = args.system or f"hb-{args.input}"
    out_dir = os.path.join(REPO_ROOT, "predictions", args.dataset, system)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(data_dir, "index.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if (not args.languages or r["language"] in args.languages)
                and (not args.songs or r["name"] in args.songs)]
    todo = [r for r in rows if args.overwrite
            or not os.path.exists(os.path.join(out_dir, f"{r['name']}.json"))]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{system}: {len(rows)} songs, {len(todo)} to align", flush=True)
    if not todo:
        return

    tmp = tempfile.mkdtemp(prefix=f"hb_{args.dataset}_")
    annot, audio, pred = (os.path.join(tmp, d) for d in ("annot", "audio", "pred"))
    for d in (annot, audio, pred):
        os.makedirs(d)

    backends, meta = {}, {}
    for r in todo:
        name, lang = r["name"], r["language"]
        src = input_path(data_dir, name, args.input)
        if not os.path.exists(src):
            print(f"  MISSING input {src} -- skipped", flush=True)
            continue
        with open(os.path.join(data_dir, "refs", f"{name}.json"), encoding="utf-8") as f:
            words = [w["text"] for w in json.load(f)["words"]]
        if lang not in backends:
            backends[lang] = EspeakBackend(ESPEAK.get(lang, "en-us"))
        song_rows, kept = phonemize_song(words, backends[lang])
        with open(os.path.join(annot, f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["word", "phonemizer", "phone_idx"])
            w.writerows(song_rows)
        os.symlink(src, os.path.join(audio, f"{name}.wav"))
        meta[name] = (lang, words, kept)
    print(f"  phonemized {len(meta)} songs -> {tmp}", flush=True)

    cmd = [sys.executable, "eval.py", f"--sepa_dir={audio}/", f"--annot_dir={annot}/",
           f"--load_model={os.path.join(hb_dir, args.checkpoint)}", f"--pred_dir={pred}/",
           "--model=baseline", "--ext", ".wav"] + (["--cuda"] if args.cuda else [])
    t0 = time.time()
    subprocess.run(cmd, cwd=hb_dir, check=True)
    secs = time.time() - t0

    done = 0
    for name, (lang, words, kept) in meta.items():
        out_csv = os.path.join(pred, f"{name}_align.csv")
        if not os.path.exists(out_csv):
            print(f"  NO OUTPUT for {name}", flush=True)
            continue
        with open(out_csv, encoding="utf-8") as f:
            aligned = [line.rstrip("\n").split(",", 2) for line in f if line.strip()]
        onsets = [None] * len(words)
        out_words = []
        for k, a in zip(kept, aligned):
            start, end = float(a[0]), float(a[1])
            onsets[k] = start
            out_words.append({"text": words[k], "start": start, "end": end})
        note = "" if len(aligned) == len(kept) else f" (expected {len(kept)} rows, got {len(aligned)})"
        out = {"meta": {"system": system, "dataset": args.dataset, "song": name, "language": lang,
                        "input": args.input, "backend": "huang-benetos",
                        "model": args.checkpoint, "hb_repo": hb_dir,
                        "n_ref_words": len(words), "n_aligned_words": len(out_words),
                        "batch_runtime_s": round(secs, 1), "machine": platform.platform(),
                        "date": time.strftime("%Y-%m-%d %H:%M")},
               "onsets": onsets, "words": out_words}
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        done += 1
        print(f"  {name:<40} {lang} {len(out_words):>4}/{len(words)} words{note}", flush=True)

    if not args.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"done: {done} songs in {secs / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
