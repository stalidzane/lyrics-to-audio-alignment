"""
Turn downloaded company files into the same layout as data/jamendo/, so every
system and eval/score.py work on them unchanged.

Input : company_data/raw/   (anything: .lrc files and audio, any folder structure)
Output: data/company/       (gitignored)
    audio/<song>.wav          converted once with ffmpeg (44.1 kHz, original channels)
    refs/<song>.json          {"name","language","words":[{text,start,end,line_end}],"lines":[...]}
    lrc/<file>.lrc            every LRC version, untouched (for the annotator-agreement analysis)
    index.csv                 name,language,n_words,duration_s,versions,audio_source
    report.txt                what was matched, skipped and why

Matching: an audio file belongs to a song when its file name (without extension)
equals the song id, i.e. the LRC file name without " (1)" etc. or its
[[blaster_id:...]] tag. Several LRC versions of one song: the version without a
" (N)" suffix is the reference; the others are only copied to lrc/.

Language: LRC has no language tag. It is GUESSED from the lyrics text with
`langdetect` (pip install langdetect) and marked in report.txt. Correct it by creating company_data/languages.csv with
lines "song_id,language" before running; that file always wins.

Usage (repo root):  python -m data.prepare_company
Needs ffmpeg (brew install ffmpeg).
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess

from data.lrc_loader import is_unannotated, parse_words, slug_of

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO_ROOT, "company_data", "raw")
LANG_CSV = os.path.join(REPO_ROOT, "company_data", "languages.csv")
OUT = os.path.join(REPO_ROOT, "data", "company")
AUDIO_EXT = (".mp3", ".wav", ".m4a", ".mp4", ".aac", ".flac", ".ogg", ".opus", ".webm")
LATVIAN = set("āēīūšžčķļņģĀĒĪŪŠŽČĶĻŅĢ")

LINE_RE = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)$")
WORD_TAG_RE = re.compile(r"<\d+:\d+(?:\.\d+)?>")
BLASTER_RE = re.compile(r"\[\[blaster_id:(.*?)\]\]")


def song_id_of(path: str) -> str:
    """Prefer the [[blaster_id:..]] tag, else the file name without ' (N)'."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = BLASTER_RE.match(line.strip())
            if m:
                return m.group(1).strip()
            if not line.startswith("[["):
                break
    return slug_of(path)


def parse_lines(path: str) -> list[dict]:
    """Lyric lines with their start time (section-only lines skipped)."""
    lines = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            m = LINE_RE.match(raw.rstrip("\n"))
            if not m:
                continue
            text = WORD_TAG_RE.sub("", m.group(3)).strip()
            if not text or (text.startswith("[") and text.endswith("]")):
                continue
            lines.append({"start": int(m.group(1)) * 60 + float(m.group(2)), "text": " ".join(text.split())})
    return lines


def words_with_line_ends(path: str) -> list[dict]:
    pairs = parse_words(path, keep_case=True)
    line_starts = {round(l["start"], 3) for l in parse_lines(path)}
    words = []
    for i, (w, t) in enumerate(pairs):
        nxt = pairs[i + 1][1] if i + 1 < len(pairs) else None
        words.append({"text": w, "start": float(t), "end": None,
                      "line_end": nxt is None or round(nxt, 3) in line_starts})
    return words


def guess_language(words: list[dict]) -> tuple[str, str]:
    """Guess the lyrics language. Returns (iso 639-1 code, method).

    Uses `langdetect` (pip install langdetect; small, offline) on the whole
    lyrics text; falls back to a Latvian-letter check if it isn't installed.
    Always verify the result in company_data/languages.csv.
    """
    text = " ".join(w["text"] for w in words)
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0          # deterministic results
        return detect(text), "langdetect"
    except ImportError:
        return ("lv" if any(ch in LATVIAN for ch in text) else "en"), "latvian-letters fallback"
    except Exception:                     # e.g. no letters at all
        return "unknown", "langdetect failed"


def convert(src: str, dst: str) -> float:
    """ffmpeg -> 44.1 kHz WAV (audio only). Returns duration in seconds."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vn", "-ar", "44100",
                    "-c:a", "pcm_s16le", dst], check=True)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", dst], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def main() -> None:
    for d in ("audio", "refs", "lrc"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)

    lrc_files, audio_files = [], {}
    for root, _, files in os.walk(RAW):
        for fn in files:
            p = os.path.join(root, fn)
            stem, ext = os.path.splitext(fn)
            if ext.lower() == ".lrc":
                lrc_files.append(p)
            elif ext.lower() in AUDIO_EXT:
                audio_files.setdefault(slug_of(stem + ".lrc"), p)

    overrides = {}
    if os.path.exists(LANG_CSV):
        with open(LANG_CSV) as f:
            overrides = {r[0].strip(): r[1].strip() for r in csv.reader(f) if len(r) >= 2}

    songs: dict[str, list[str]] = {}
    for p in sorted(lrc_files):
        shutil.copy2(p, os.path.join(OUT, "lrc", os.path.basename(p)))
        songs.setdefault(song_id_of(p), []).append(p)

    report, rows = [], []
    for sid, versions in sorted(songs.items()):
        annotated = [v for v in versions if parse_words(v) and not is_unannotated(parse_words(v))]
        if not annotated:
            report.append(f"SKIP {sid}: no timed words in any version")
            continue
        plain = [v for v in annotated if not re.search(r" \(\d+\)\.lrc$", v, re.I)]
        ref_path = (plain or annotated)[0]
        audio = audio_files.get(sid) or audio_files.get(slug_of(ref_path))
        if audio is None:
            report.append(f"NO AUDIO {sid}: {len(versions)} LRC version(s); reference kept, no audio")
        words = words_with_line_ends(ref_path)
        if sid in overrides:
            lang, lang_note = overrides[sid], ""
        else:
            lang, how = guess_language(words)
            lang_note = f" (language guessed: {how})"

        duration = ""
        if audio:
            wav = os.path.join(OUT, "audio", f"{sid}.wav")
            if not os.path.exists(wav):
                duration = f"{convert(audio, wav):.1f}"
            else:
                duration = "existing"
        ref = {"name": sid, "language": lang, "source_lrc": os.path.basename(ref_path),
               "words": words, "lines": parse_lines(ref_path)}
        with open(os.path.join(OUT, "refs", f"{sid}.json"), "w", encoding="utf-8") as f:
            json.dump(ref, f, ensure_ascii=False, indent=1)
        rows.append({"name": sid, "language": lang, "n_words": len(words), "duration_s": duration,
                     "versions": len(versions), "audio_source": os.path.basename(audio) if audio else ""})
        report.append(f"OK {sid}: {len(words)} words, {lang}{lang_note}, {len(versions)} version(s), "
                      f"audio={'yes' if audio else 'NO'}")

    # index.csv lists only songs WITH audio: that is what the systems can run on
    with open(os.path.join(OUT, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "language", "n_words", "duration_s", "versions", "audio_source"])
        w.writeheader()
        w.writerows(r for r in rows if r["audio_source"])
    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    with_audio = sum(1 for r in rows if r["audio_source"])
    langs = {}
    for r in rows:
        langs[r["language"]] = langs.get(r["language"], 0) + 1
    print(f"{len(songs)} songs found in LRC files; {len(rows)} with timed words; {with_audio} also have audio")
    print("languages: " + ", ".join(f"{k} {v}" for k, v in sorted(langs.items())) + "  (check report.txt)")
    print(f"multi-version songs: {sum(1 for v in songs.values() if len(v) > 1)}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
