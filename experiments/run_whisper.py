"""
Run plain Whisper (free transcription, word timestamps) on songs.

Saves Whisper's raw words per song; scoring is a separate step (eval/score.py),
so you can re-score without re-running Whisper. Songs already done are skipped,
so an interrupted run just continues where it stopped.

Inputs (from data/export_jamendo.py and experiments/separate_demucs.py):
    --input mix     data/jamendo/audio/<name>.wav
    --input vocals  data/jamendo/separated/htdemucs/<name>/vocals.wav

Output:
    predictions/<dataset>/<system>/<name>.json   {"meta": {...}, "words": [{text,start,end,probability}]}
    with <system> = whisper-<model>-<input> unless --system is given.

Examples (repo root):
    python -m experiments.run_whisper --model large-v3 --languages en --limit 1   # smoke test
    python -m experiments.run_whisper --model large-v3                            # all 79 songs
    python -m experiments.run_whisper --model large-v3 --input vocals             # after Demucs
    python -m experiments.run_whisper --model large-v3 --dataset company          # company songs

Backends:
    mlx     (default) mlx-whisper, Apple-silicon GPU, same weights as OpenAI's.
            pip install mlx-whisper
    openai  openai-whisper. Mostly CPU on a Mac -> slow.

The song's language is passed to Whisper (it is known for the benchmark), so we
measure timing, not language identification. Mention this in the thesis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
import librosa

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO_ROOT, "data", "jamendo")   # set from --dataset in main()

MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def input_path(name: str, which: str) -> str:
    if which == "mix":
        return os.path.join(DATA, "audio", f"{name}.wav")
    if which == "vocals":
        return os.path.join(DATA, "separated", "htdemucs", name, "vocals.wav")
    raise ValueError(which)


def load_index(languages, names=None):
    with open(os.path.join(DATA, "index.csv")) as f:
        rows = [r for r in csv.DictReader(f) if not languages or r["language"] in languages]
    if names:
        rows = [r for r in rows if r["name"] in names]
    return rows


def make_transcriber(backend: str, model: str):
    """Return fn(audio_16k, language) -> Whisper result dict."""
    if backend == "mlx":
        import mlx_whisper
        repo = MLX_REPOS.get(model, model)

        def run(audio, language):
            return mlx_whisper.transcribe(
                audio, path_or_hf_repo=repo, word_timestamps=True,
                language=language, verbose=None,
            )
        return run, repo
    if backend == "openai":
        import whisper
        m = whisper.load_model(model)

        def run(audio, language):
            return m.transcribe(audio, word_timestamps=True, language=language, fp16=False)
        return run, model
    raise ValueError(backend)


def flatten_words(result: dict) -> list[dict]:
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []) or []:
            words.append({
                "text": w["word"].strip(),
                "start": float(w["start"]),
                "end": float(w["end"]),
                "probability": float(w.get("probability", float("nan"))),
            })
    return words


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--backend", choices=["mlx", "openai"], default="mlx")
    ap.add_argument("--input", choices=["mix", "vocals"], default="mix")
    ap.add_argument("--dataset", default="jamendo", help="folder under data/: jamendo, jamendopp or company")
    ap.add_argument("--languages", nargs="+", help="only these languages (default: all)")
    ap.add_argument("--songs", nargs="+", help="only these song names")
    ap.add_argument("--limit", type=int, help="stop after N new songs (smoke test)")
    ap.add_argument("--system", help="output folder name (default whisper-<model>-<input>)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-language", action="store_true",
                    help="let Whisper detect the language itself (system name gets -nolang)")
    args = ap.parse_args()

    global DATA
    DATA = os.path.join(REPO_ROOT, "data", args.dataset)
    system = args.system or f"whisper-{args.model}-{args.input}" + ("-nolang" if args.no_language else "")
    out_dir = os.path.join(REPO_ROOT, "predictions", args.dataset, system)
    os.makedirs(out_dir, exist_ok=True)
    transcribe, model_id = make_transcriber(args.backend, args.model)

    rows = load_index(args.languages, args.songs)
    print(f"{system}: {len(rows)} songs ({args.backend}, {model_id})")
    done_new = 0
    t_all = time.time()
    for i, r in enumerate(rows, 1):
        name, lang = r["name"], r["language"]
        out_path = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(out_path) and not args.overwrite:
            continue
        path = input_path(name, args.input)
        if not os.path.exists(path):
            print(f"  [{i}/{len(rows)}] MISSING input {path} -- skipped")
            continue
        audio, _ = librosa.load(path, sr=16000, mono=True)
        t0 = time.time()
        result = transcribe(audio, None if args.no_language else lang)
        secs = time.time() - t0
        words = flatten_words(result)
        meta = {
            "system": system, "dataset": args.dataset, "song": name, "language": lang, "input": args.input,
            "backend": args.backend, "model": model_id, "word_timestamps": True,
            "language_given": not args.no_language, "detected_language": result.get("language"), "runtime_s": round(secs, 1),
            "audio_s": round(len(audio) / 16000, 1), "machine": platform.platform(),
            "date": time.strftime("%Y-%m-%d %H:%M"),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "text": result.get("text", ""), "words": words},
                      f, ensure_ascii=False, indent=1)
        done_new += 1
        print(f"  [{i}/{len(rows)}] {name:<40} {lang}->{result.get('language')} {len(words):>4} words "
              f"{secs:6.1f}s for {meta['audio_s']:.0f}s audio")
        if args.limit and done_new >= args.limit:
            break
    print(f"done: {done_new} new songs in {(time.time() - t_all) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
