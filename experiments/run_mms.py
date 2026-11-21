"""
Forced alignment with the MMS aligner (wav2vec2 + CTC + Viterbi), system 2.

Uses the ctc-forced-aligner package by MahmoudAshraf97 and its default model
MahmoudAshraf/mms-300m-1130-forced-aligner (MMS, 1,130 languages; CC-BY-NC 4.0).
Installed in the `audio` conda env (envs/audio.yml), from GitHub, NOT the PyPI package of the
same name:
    conda activate audio
    pip uninstall -y ctc-forced-aligner
    pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git

Unlike Whisper, the aligner gets the reference lyrics and only places them in time,
so every reference word gets a time (coverage ~100 %).

Writes predictions/<dataset>/mms-<input>/<name>.json:
    {"meta": {...}, "onsets": [t per reference word, null if not aligned],
     "words": [{text, start, end, score}]}
score.py uses "onsets" (one per reference word).

Run from the repo root inside the audio env:
    python3 -m experiments.run_mms --dataset company --songs xIImn_1oJ R1LGghIk4O kw2gggCCBe
    python3 -m experiments.run_mms --dataset jamendo --limit 2
    python3 -m experiments.run_mms --dataset jamendo --input vocals
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
import warnings

import torch
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    load_audio,
    postprocess_results,
    preprocess_text,
)

# harmless: the package wraps ffmpeg's read-only audio buffer in a tensor without copying
warnings.filterwarnings("ignore", message="The given buffer is not writable")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISO3 = {"en": "eng", "es": "spa", "de": "deu", "fr": "fra", "it": "ita",
        "pt": "por", "sv": "swe", "lv": "lav", "lt": "lit", "et": "est", "ru": "rus"}


def input_path(data_dir: str, name: str, which: str) -> str:
    if which == "vocals":
        return os.path.join(data_dir, "separated", "htdemucs", name, "vocals.wav")
    return os.path.join(data_dir, "audio", f"{name}.wav")


def word_field(w: dict, *keys):
    for k in keys:
        if k in w:
            return w[k]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="jamendo", help="jamendo, jamendopp or company")
    ap.add_argument("--input", choices=["mix", "vocals"], default="mix")
    ap.add_argument("--languages", nargs="+", help="only these languages (default: all)")
    ap.add_argument("--songs", nargs="+", help="only these song names")
    ap.add_argument("--limit", type=int, help="stop after N new songs (test run)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu",
                    help="mps (Apple GPU) or cpu; use cpu if mps errors out")
    ap.add_argument("--batch", type=int, default=8, help="windows per forward pass")
    ap.add_argument("--system", help="output folder name (default mms-<input>)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    data_dir = os.path.join(REPO_ROOT, "data", args.dataset)
    system = args.system or f"mms-{args.input}"
    out_dir = os.path.join(REPO_ROOT, "predictions", args.dataset, system)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(data_dir, "index.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if (not args.languages or r["language"] in args.languages)
                and (not args.songs or r["name"] in args.songs)]

    print(f"{system}: {len(rows)} songs on {args.device}; loading the aligner model ...", flush=True)
    model, tokenizer = load_alignment_model(args.device, dtype=torch.float32)

    done, t_all = 0, time.time()
    for i, r in enumerate(rows, 1):
        name, lang = r["name"], r["language"]
        out_path = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(out_path) and not args.overwrite:
            continue
        path = input_path(data_dir, name, args.input)
        if not os.path.exists(path):
            print(f"  [{i}/{len(rows)}] MISSING input {path} -- skipped", flush=True)
            continue
        with open(os.path.join(data_dir, "refs", f"{name}.json"), encoding="utf-8") as f:
            ref = json.load(f)
        ref_words = [w["text"] for w in ref["words"]]

        t0 = time.time()
        audio = load_audio(path, model.dtype, model.device)
        emissions, stride = generate_emissions(model, audio, batch_size=args.batch)
        tokens_starred, text_starred = preprocess_text(
            " ".join(ref_words), romanize=True, language=ISO3.get(lang, "eng"))
        segments, scores, blank = get_alignments(emissions, tokens_starred, tokenizer)
        spans = get_spans(tokens_starred, segments, blank)
        result = postprocess_results(text_starred, spans, stride, scores)
        secs = time.time() - t0

        words = [{"text": word_field(w, "text", "word"),
                  "start": float(word_field(w, "start")),
                  "end": float(word_field(w, "end")),
                  "score": (float(word_field(w, "score")) if word_field(w, "score") is not None else None)}
                 for w in result]
        out = {"meta": {
                   "system": system, "dataset": args.dataset, "song": name, "language": lang,
                   "input": args.input, "backend": "mms-fa",
                   "model": "MahmoudAshraf/mms-300m-1130-forced-aligner",
                   "device": args.device, "runtime_s": round(secs, 1),
                   "n_ref_words": len(ref_words), "n_aligned_words": len(words),
                   "machine": platform.platform(), "date": time.strftime("%Y-%m-%d %H:%M")},
               "words": words}
        if len(words) == len(ref_words):
            out["onsets"] = [w["start"] for w in words]
            note = ""
        else:  # some words vanished in text normalisation: score.py matches by text instead
            note = f"  ({len(words)} of {len(ref_words)} words returned; matched by text)"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        done += 1
        print(f"  [{i}/{len(rows)}] {name:<40} {lang} {len(words):>4} words {secs:6.1f}s{note}", flush=True)
        if args.limit and done >= args.limit:
            break
    print(f"done: {done} new songs in {(time.time() - t_all) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
