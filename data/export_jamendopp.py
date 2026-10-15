"""
Export JamendoLyrics++ (Kick et al. 2025; 80 extra English songs) into the same
layout as data/jamendo/, so every system and eval/score.py work on it unchanged.

Source repo (same file structure as the original JamendoLyrics GitHub repo):
    https://github.com/tikick/jamendolyricspp
        JamendoLyrics++.csv              URL,Filepath,Artist,Title,Genre
        mp3/<name>.mp3                   audio
        lyrics/<name>.txt                lyrics, one line per lyric line
        annotations/words/<name>.csv     word_start,word_end (one row per word)

Get it once (about 80 mp3 files, use Wi-Fi):
    git clone --depth 1 https://github.com/tikick/jamendolyricspp.git data/raw/jamendolyricspp

Then (repo root):
    python -m data.export_jamendopp

Writes data/jamendopp/{audio/<name>.wav, refs/<name>.json, index.csv, report.txt}.
All songs are English. Words are the whitespace-separated tokens of the lyrics
file, matched 1:1 to the rows of the word annotation file; songs where the
counts differ are skipped and listed in report.txt.
Run the systems with --dataset jamendopp (same as --dataset company).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "data", "raw", "jamendolyricspp")
OUT = os.path.join(REPO_ROOT, "data", "jamendopp")


def read_words(lyrics_path: str) -> tuple[list[str], list[bool], list[dict]]:
    """Whitespace tokens, a line_end flag per token, and the lyric lines."""
    words, line_end, lines = [], [], []
    with open(lyrics_path, encoding="utf-8") as f:
        for raw in f:
            toks = raw.split()
            if not toks:
                continue
            lines.append({"text": " ".join(toks), "first_word": len(words)})
            words.extend(toks)
            line_end.extend([False] * (len(toks) - 1) + [True])
    return words, line_end, lines


def read_times(csv_path: str) -> list[tuple[float, float]]:
    with open(csv_path, newline="") as f:
        return [(float(r["word_start"]), float(r["word_end"])) for r in csv.DictReader(f)]


def to_wav(src: str, dst: str) -> float:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vn", "-c:a", "pcm_s16le", dst], check=True)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", dst], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC, help="path of the cloned jamendolyricspp repo")
    args = ap.parse_args()

    meta_path = os.path.join(args.src, "JamendoLyrics++.csv")
    if not os.path.exists(meta_path):
        raise SystemExit(f"{meta_path} not found. Clone the repo first (see the top of this file).")
    for d in ("audio", "refs"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)

    rows, report = [], []
    with open(meta_path, newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    for m in meta:
        name = os.path.splitext(m["Filepath"])[0]
        mp3 = os.path.join(args.src, "mp3", m["Filepath"])
        lyr = os.path.join(args.src, "lyrics", f"{name}.txt")
        ann = os.path.join(args.src, "annotations", "words", f"{name}.csv")
        missing = [p for p in (mp3, lyr, ann) if not os.path.exists(p)]
        if missing:
            report.append(f"SKIP {name}: missing {', '.join(os.path.relpath(p, args.src) for p in missing)}")
            continue
        words, line_end, lines = read_words(lyr)
        times = read_times(ann)
        if len(words) != len(times):
            report.append(f"SKIP {name}: {len(words)} words in lyrics but {len(times)} annotated")
            continue
        for ln in lines:
            ln["start"] = times[ln.pop("first_word")][0]
        wav = os.path.join(OUT, "audio", f"{name}.wav")
        dur = to_wav(mp3, wav) if not os.path.exists(wav) else None
        ref = {"name": name, "language": "en", "artist": m.get("Artist"), "title": m.get("Title"),
               "genre": m.get("Genre"), "url": m.get("URL"),
               "words": [{"text": w, "start": s, "end": e, "line_end": le}
                         for w, (s, e), le in zip(words, times, line_end)],
               "lines": lines}
        with open(os.path.join(OUT, "refs", f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(ref, f, ensure_ascii=False, indent=1)
        rows.append({"name": name, "language": "en", "n_words": len(words),
                     "duration_s": f"{dur:.1f}" if dur else "existing"})
        report.append(f"OK {name}: {len(words)} words")

    with open(os.path.join(OUT, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "language", "n_words", "duration_s"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"{len(rows)} of {len(meta)} songs exported -> {OUT} (see report.txt for skipped songs)")


if __name__ == "__main__":
    main()
