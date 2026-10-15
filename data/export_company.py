"""
Convert the company songs downloaded with s3.mjs into the same layout as data/jamendo/,
so run_whisper.py, the other systems and eval/score.py work with --dataset company.

Input (default ~/Downloads/company-s3, as downloaded from S3):
    users/<user>/<slug>/<slug>.lrc|.txt|.json      lyrics timing per user, section labels, title/artist
    tracks/<y>/<m>/<d>/<user>-<slug>.<mp3|m4a|...>  audio; ONE track per slug (newest file used)

A song = one slug. Several users can have timed the same slug; they share one track.

Writes (gitignored, under data/company/):
    audio/<slug>.wav          44.1 kHz 16-bit WAV, original channels (ffmpeg)
    refs/<slug>.json          {"name","language","title","artist","reference_user","versions",
                               "words":[{text,start,end,line_end}],"lines":[...]}
    lrc/<slug>/<user>.lrc     every user's LRC, untouched (for the annotator-agreement analysis)
    index.csv                 name,language,n_words,duration_s,title,artist,versions  (usable songs)
    report.txt                every song: OK or SKIP and why

Reference timing (refs/) when several users timed a song: the fully timed version of the user
who uploaded the track; otherwise the fully timed version with the most words.
Skipped: songs without audio, songs with no fully timed version (words left at 00:00.000).
Section labels ("Verse", "[Chorus]", "Intrumental", "---") are dropped; sung lines without
word tags (e.g. "Ah-ah-ah") are kept as words.

Language: "slug,language" lines in data/company_languages.csv win; otherwise langdetect
(installed in the lyrics2audio env) guesses it, marked "guessed" in report.txt.

Run from the repo root:
    python -m data.export_company
    python -m data.export_company --src /other/folder
Already-converted audio is not converted again. Needs ffmpeg.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "data" / "company"
LANG_CSV = REPO_ROOT / "data" / "company_languages.csv"
AUDIO_EXT = {".mp3", ".m4a", ".mp4", ".wav", ".aac", ".ogg", ".flac", ".opus", ".webm"}
LATVIAN = set("āēīūšžčķļņģĀĒĪŪŠŽČĶĻŅĢ")

LINE_RE = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)$")
TAG_RE = re.compile(r"<(\d+):(\d+(?:\.\d+)?)>")
SECTION_RE = re.compile(
    r"^\[?\s*(intro|verse|pre-?chorus|chorus|post-?chorus|bridge|hook|refrain|outro|"
    r"instrumental|intrumental|interlude|break|solo|coda)\b[^\]]*\]?$|^\[.*\]$|^-+$",
    re.IGNORECASE,
)


def seconds(mm: str, ss: str) -> float:
    return int(mm) * 60 + float(ss)


def section_labels(txt_path: Path) -> set[str]:
    """Labels written as [Label] in the plain-text lyrics, e.g. {'chorus', 'intrumental'}."""
    if not txt_path.exists():
        return set()
    labels = set()
    for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.fullmatch(r"\[([^\[\]]*)\]", line.strip())
        if m:
            labels.add(m.group(1).strip().lower())
    return labels


def parse_lrc(lrc_path: Path, labels: set[str]) -> list[list[tuple[str, float]]]:
    """Lyric lines as lists of (word, start). Metadata and section-label lines are dropped."""
    lines = []
    for raw in lrc_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.match(raw.strip())
        if not m:
            continue  # [[ti:..]], [[blaster_id:..]], empty lines
        line_start, rest = seconds(m.group(1), m.group(2)), m.group(3)
        plain = TAG_RE.sub("", rest).strip()
        if not plain:
            continue
        if "<" not in rest and (plain.strip("[]").strip().lower() in labels or SECTION_RE.match(plain)):
            continue  # section label, not sung
        # first word gets the line time, every following word its own <mm:ss.xxx>
        parts = TAG_RE.split(rest)  # [text, mm, ss, text, mm, ss, text, ...]
        words = [(w, line_start) for w in parts[0].split()]
        for i in range(1, len(parts), 3):
            t = seconds(parts[i], parts[i + 1])
            words += [(w, t) for w in parts[i + 2].split()]
        if words:
            lines.append(words)
    return lines


def guess_language(text: str) -> str:
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        return detect(text)
    except ImportError:
        return "lv" if any(ch in LATVIAN for ch in text) else "en"


def convert(src: Path, dst: Path) -> float:
    """Audio -> 44.1 kHz 16-bit WAV. Returns duration in seconds."""
    if not dst.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ar", "44100",
                        "-c:a", "pcm_s16le", str(dst)], check=True)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(dst)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "Downloads" / "company-s3"))
    src = Path(ap.parse_args().src).expanduser()

    for d in ("audio", "refs", "lrc"):
        (OUT / d).mkdir(parents=True, exist_ok=True)

    languages = {}
    if LANG_CSV.exists():
        with open(LANG_CSV, encoding="utf-8") as f:
            languages = {r[0].strip(): r[1].strip() for r in csv.reader(f) if len(r) >= 2}

    # lyrics versions per slug: slug -> {user: lrc path}
    songs: dict[str, dict[str, Path]] = {}
    for lrc in sorted((src / "users").glob("*/*/*.lrc")):
        songs.setdefault(lrc.parent.name, {})[lrc.parent.parent.name] = lrc

    # one track per slug (newest); file name is <user>-<slug>, user names may contain "-"
    tracks: dict[str, tuple[Path, str]] = {}
    for p in (src / "tracks").rglob("*"):
        if p.suffix.lower() not in AUDIO_EXT:
            continue
        pieces = p.stem.split("-")
        for i in range(1, len(pieces)):
            slug, owner = "-".join(pieces[i:]), "-".join(pieces[:i])
            if slug in songs:
                if slug not in tracks or p.stat().st_mtime > tracks[slug][0].stat().st_mtime:
                    tracks[slug] = (p, owner)
                break

    rows, report = [], []
    for slug, versions in sorted(songs.items()):
        # keep every user's LRC for the agreement analysis
        (OUT / "lrc" / slug).mkdir(exist_ok=True)
        for user, lrc in versions.items():
            shutil.copy2(lrc, OUT / "lrc" / slug / f"{user}.lrc")

        meta = {}
        for lrc in versions.values():
            if lrc.with_suffix(".json").exists():
                meta = json.loads(lrc.with_suffix(".json").read_text(encoding="utf-8"))
                break
        title, artist = meta.get("title", ""), meta.get("artist", "")
        label = f"{slug} ({artist} - {title})" if title else slug

        if slug not in tracks:
            report.append(f"SKIP {label}: no audio ({len(versions)} lyrics version(s))")
            continue
        audio, owner = tracks[slug]

        # parse every version; keep the fully timed ones
        timed = {}
        for user, lrc in versions.items():
            lines = parse_lrc(lrc, section_labels(lrc.with_suffix(".txt")))
            words = [w for line in lines for w in line]
            if words and not any(t == 0.0 for i, (_, t) in enumerate(words) if i > 0):
                timed[user] = lines
        if not timed:
            report.append(f"SKIP {label}: no fully timed version ({len(versions)} version(s), words at 00:00.000)")
            continue
        ref_user = owner if owner in timed else max(timed, key=lambda u: sum(len(l) for l in timed[u]))
        lines = timed[ref_user]

        text = " ".join(w for line in lines for w, _ in line)
        lang = languages.get(slug) or guess_language(text)
        how = "" if slug in languages else ", language guessed"

        ref_words, ref_lines = [], []
        for line in lines:
            for j, (w, t) in enumerate(line):
                ref_words.append({"text": w, "start": t, "end": None, "line_end": j == len(line) - 1})
            ref_lines.append({"start": line[0][1], "text": " ".join(w for w, _ in line)})
        ref = {"name": slug, "language": lang, "title": title, "artist": artist,
               "reference_user": ref_user, "versions": sorted(versions), "timed_versions": sorted(timed),
               "source_audio": str(audio.relative_to(src)), "words": ref_words, "lines": ref_lines}
        (OUT / "refs" / f"{slug}.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")

        duration = convert(audio, OUT / "audio" / f"{slug}.wav")
        rows.append({"name": slug, "language": lang, "n_words": len(ref_words), "duration_s": f"{duration:.1f}",
                     "title": title, "artist": artist, "versions": len(timed)})
        report.append(f"OK   {label}: {len(ref_words)} words, {lang}{how}, {duration:.0f} s, "
                      f"reference by {ref_user}, timed versions: {', '.join(sorted(timed))}")
        print(f"  {label}: {len(ref_words)} words, {lang}, {len(timed)} timed version(s)")

    with open(OUT / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "language", "n_words", "duration_s", "title", "artist", "versions"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["language"], r["name"])))
    (OUT / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"{len(rows)} songs ready, {len(songs) - len(rows)} skipped (see {OUT / 'report.txt'})")


if __name__ == "__main__":
    main()
