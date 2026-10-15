"""
Export JamendoLyrics MultiLang (HuggingFace, 79 songs: en/es/de/fr) to disk once.

Every system (Whisper, Demucs, MMS, Huang & Benetos) then reads the same
files, so nothing depends on HF decoding again and runs can be resumed.

Writes (gitignored, under data/jamendo/):
    audio/<name>.wav    original sample rate, original channels (Demucs wants this)
    refs/<name>.json    {"name", "language", "words": [{text,start,end,line_end}],
                         "lines": [...]}  -- the ground truth
    index.csv           name,language,n_words,duration_s

Run from the repo root (needs internet the first time):
    python -m data.export_jamendo
    python -m data.export_jamendo --languages en      # English only
Already-exported songs are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import soundfile as sf
from datasets import load_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "data", "jamendo")
LANGUAGES = ("en", "es", "de", "fr")


def _to_jsonable(x):
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _decode_native(audio_obj) -> tuple[np.ndarray, int]:
    """Decode without resampling or downmixing. Returns (channels x samples, sr)."""
    if hasattr(audio_obj, "get_all_samples"):          # torchcodec (datasets >= 4)
        s = audio_obj.get_all_samples()
        return s.data.detach().cpu().numpy().astype(np.float32), int(s.sample_rate)
    if isinstance(audio_obj, dict) and "array" in audio_obj:  # older datasets
        wav = np.asarray(audio_obj["array"], dtype=np.float32)
        return (wav[None, :] if wav.ndim == 1 else wav), int(audio_obj["sampling_rate"])
    raise TypeError(f"Unrecognized audio object: {type(audio_obj)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    args = ap.parse_args()

    os.makedirs(os.path.join(OUT_DIR, "audio"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "refs"), exist_ok=True)
    index_path = os.path.join(OUT_DIR, "index.csv")
    rows = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            rows = {r["name"]: r for r in csv.DictReader(f)}

    for lang in args.languages:
        ds = load_dataset("jamendolyrics/jamendolyrics", lang, split="test")
        print(f"[{lang}] {len(ds)} songs")
        for ex in ds:
            name = ex["name"]
            wav_path = os.path.join(OUT_DIR, "audio", f"{name}.wav")
            ref_path = os.path.join(OUT_DIR, "refs", f"{name}.json")
            if os.path.exists(wav_path) and os.path.exists(ref_path) and name in rows:
                continue
            wav, sr = _decode_native(ex["audio"])
            sf.write(wav_path, wav.T, sr, subtype="PCM_16")
            ref = {k: ex[k] for k in ("name", "language", "words", "lines")}
            for k in ("artist", "title", "genre", "polyphonic", "non_lexical", "lyric_overlap"):
                if k in ex:
                    ref[k] = ex[k]
            with open(ref_path, "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(ref), f, ensure_ascii=False, indent=1)
            rows[name] = {"name": name, "language": ex["language"],
                          "n_words": len(ex["words"]),
                          "duration_s": f"{wav.shape[1] / sr:.1f}"}
            print(f"  saved {name} ({rows[name]['duration_s']} s, {sr} Hz, {wav.shape[0]} ch)")

    with open(index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "language", "n_words", "duration_s"])
        w.writeheader()
        for r in sorted(rows.values(), key=lambda r: (r["language"], r["name"])):
            w.writerow(r)
    print(f"index: {len(rows)} songs -> {index_path}")


if __name__ == "__main__":
    main()
