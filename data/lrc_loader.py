"""
LRC loader for the real game-song data.

Mirrors jamendo_loader's contract so metrics.py and the aligner-benchmark code
can be reused unchanged. Parses enhanced-LRC files (per-word <mm:ss.xxx> tags)
into the same shape the rest of the pipeline expects.

Two entry points:

    load_song(lrc_path)            -> dict(song_id, title, artist, words, onsets, ...)
    iter_songs(lrc_dir)            -> yields one dict per slug (first version found)

The returned dict matches whisper_baseline's `song` usage:
    song["words"]   : list[str]      reference word tokens (original casing)
    song["onsets"]  : np.ndarray     onset time (s) per word, same length
    song["language"]: str | None     (LRC has no language tag; defaults to None)
    song["song_id"] : str            the slug

Audio is NOT loaded here (LRC carries no audio). When you wire in the real
stems, add an `audio_dir` and load `<slug>.wav` at 16 kHz mono alongside, same
as jamendo_loader does.
"""
import os
import re
import numpy as np

WORD_RE = re.compile(r"<(\d+):(\d+(?:\.\d+)?)>\s*([^<\[\]]+)")
LINE_TS_RE = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\]")


def _ts(mm, ss):
    return int(mm) * 60 + float(ss)


def slug_of(filename):
    base = os.path.basename(filename)
    if base.lower().endswith(".lrc"):
        base = base[:-4]
    return re.sub(r" \(\d+\)$", "", base)


def parse_words(path, keep_case=True):
    """Return list of (word, onset_sec) for every timed word in an LRC file.

    The leading [mm:ss.xxx] on a lyric line is treated as the onset of that
    line's first word; subsequent words carry their own <mm:ss.xxx>. Metadata
    ([[..]]) and bare section tags ([Chorus]) are skipped.
    """
    words = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("[["):
                continue
            m_line = LINE_TS_RE.match(line)
            rest = line[m_line.end():] if m_line else line
            stripped = rest.strip()
            if stripped.startswith("[") and stripped.endswith("]") and "<" not in stripped:
                continue  # section-only line like [00:44.0][Verse]
            if m_line:
                line_onset = _ts(m_line.group(1), m_line.group(2))
                first_chunk = re.split(r"<", rest, maxsplit=1)[0]
                fw = first_chunk.strip()
                if fw:
                    words.append((fw if keep_case else fw.lower(), line_onset))
            for mm, ss, wtext in WORD_RE.findall(rest):
                w = wtext.strip()
                if w:
                    words.append((w if keep_case else w.lower(), _ts(mm, ss)))
    return words


def _meta(path):
    ti = ar = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.match(r"\[\[ti:(.*?)\]\]", line)
            if m:
                ti = m.group(1).strip()
            m = re.match(r"\[\[ar:(.*?)\]\]", line)
            if m:
                ar = m.group(1).strip()
            if not line.startswith("[["):
                break
    return ti, ar


def is_unannotated(word_onset_pairs):
    """True if every onset is 0 -> lyrics typed but timing not done."""
    return bool(word_onset_pairs) and max(o for _, o in word_onset_pairs) == 0.0


def load_song(lrc_path, language=None):
    """Load one LRC file into the pipeline's `song` dict shape."""
    pairs = parse_words(lrc_path, keep_case=True)
    words = [w for w, _ in pairs]
    onsets = np.array([o for _, o in pairs], dtype=float)
    ti, ar = _meta(lrc_path)
    return {
        "song_id": slug_of(lrc_path),
        "title": ti,
        "artist": ar,
        "language": language,
        "words": words,
        "onsets": onsets,
        "path": lrc_path,
    }


def iter_songs(lrc_dir, skip_unannotated=True):
    """Yield one song dict per slug (first non-empty version found).

    For benchmarking an aligner you want one reference per song. Where a slug
    has multiple annotator versions, this takes the first annotated one; pick a
    canonical version explicitly if that matters for your experiment.
    """
    files = sorted(f for f in os.listdir(lrc_dir) if f.lower().endswith(".lrc"))
    seen = set()
    for f in files:
        slug = slug_of(f)
        if slug in seen:
            continue
        path = os.path.join(lrc_dir, f)
        if skip_unannotated and is_unannotated(parse_words(path)):
            continue
        seen.add(slug)
        yield load_song(path)


if __name__ == "__main__":
    import sys
    # Run from repo root: python -m data.lrc_loader [lrc_dir]
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lrc_files")
    n = 0
    for song in iter_songs(d):
        n += 1
        print(f"{song['song_id']:<14} {len(song['words']):>4} words | "
              f"{(song['title'] or '?')[:30]:<30} | "
              f"onset range {song['onsets'].min():.1f}..{song['onsets'].max():.1f}s")
    print(f"\n{n} songs loaded.")
