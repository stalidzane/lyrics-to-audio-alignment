"""
Export the 5 songs with isolated vocal stems into two datasets the pipeline understands:

    data/stems/       audio/<slug>.wav = the full mix, refs/<slug>.json from the LRC, index.csv
                      (Demucs vocals are added by experiments/separate_demucs.py as usual)
    data/stems_iso/   the same mix and refs, but separated/htdemucs/<slug>/vocals.wav = the studio
                      stem, shifted to line up with the mix, so "--input vocals" means the stem

The stems are offset from the mixes by a few tens of ms (MP3 padding and/or mix processing).
The offset is measured by cross-correlation (20-90 s of each song, 8 kHz) and removed by shifting
the stem at 44.1 kHz; the residual offset after the shift is printed as a check.

    python3 data/export_stems.py                   # default source folder below
    python3 data/export_stems.py --src "<folder with one sub-folder per song>"
Then on the Mac:
    bash experiments/run_all.sh stems && bash experiments/run_all.sh stems_iso
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.export_company import parse_lrc  # same LRC parsing as the rhythm-game songs

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SRC = REPO / "data" / "company" / "blaster tracks [ai thesis]"
SR = 44100


def decode(path: Path, sr: int) -> np.ndarray:
    """mono float32 samples"""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768


def lag_seconds(mix: np.ndarray, stem: np.ndarray, sr: int, max_lag: float = 0.5) -> float:
    """How many seconds LATER the stem's content appears than in the mix
    (positive: the stem must be moved earlier). Cross-correlation over 20-90 s, via FFT."""
    a = mix[20 * sr:90 * sr]
    b = stem[20 * sr:90 * sr]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    size = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.conj(np.fft.rfft(a, size)) * np.fft.rfft(b, size), size)  # c[k] = sum a[t] b[t+k]
    m = int(max_lag * sr)
    lags = np.r_[np.arange(0, m), np.arange(-m, 0)]
    vals = np.r_[c[:m], c[-m:]]
    return float(lags[np.argmax(vals)]) / sr


def shift(x: np.ndarray, seconds_later: float, sr: int, length: int) -> np.ndarray:
    """move content earlier by `seconds_later` (later if negative), keep `length` samples"""
    k = int(round(seconds_later * sr))
    y = x[k:] if k >= 0 else np.r_[np.zeros(-k, dtype=x.dtype), x]
    return np.r_[y, np.zeros(max(0, length - len(y)), dtype=x.dtype)][:length]


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "-",
                    "-c:a", "pcm_s16le", str(path)], input=pcm, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    args = ap.parse_args()
    src = Path(args.src)
    out_mix, out_iso = REPO / "data" / "stems", REPO / "data" / "stems_iso"
    rows = []
    for folder in sorted(p for p in src.iterdir() if p.is_dir()):
        lrcs = list(folder.glob("*.lrc"))
        mp3s = list(folder.glob("*.mp3"))
        stems = [p for p in mp3s if "vox" in p.name.lower()]
        mixes = [p for p in mp3s if p not in stems]
        if len(lrcs) != 1 or len(stems) != 1 or len(mixes) != 1:
            print(f"skipped {folder.name}: expected one .lrc, one mix and one ISO VOX mp3")
            continue
        slug = lrcs[0].stem.replace(" copy", "").strip()
        artist, _, title = folder.name.replace(" FOLDER", "").partition("�")
        lines = parse_lrc(lrcs[0], set())
        words = [{"text": w, "start": t, "end": None, "line_end": i == len(line) - 1}
                 for line in lines for i, (w, t) in enumerate(line)]
        ref = {"name": slug, "language": "en", "title": title.strip(), "artist": artist.strip(),
               "source": f"stems delivery: {folder.name}", "words": words,
               "lines": [{"start": line[0][1], "text": " ".join(w for w, _ in line)} for line in lines]}

        mix = decode(mixes[0], SR)
        stem = decode(stems[0], SR)
        lag = lag_seconds(decode(mixes[0], 8000), decode(stems[0], 8000), 8000)
        aligned = shift(stem, lag, SR, len(mix))
        residual = lag_seconds(mix[::1], aligned, SR, max_lag=0.05)

        for out in (out_mix, out_iso):
            write_wav(out / "audio" / f"{slug}.wav", mix, SR)
            (out / "refs").mkdir(parents=True, exist_ok=True)
            (out / "refs" / f"{slug}.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
        write_wav(out_iso / "separated" / "htdemucs" / slug / "vocals.wav", aligned, SR)
        rows.append({"name": slug, "language": "en", "n_words": len(words),
                     "duration_s": round(len(mix) / SR, 1), "title": ref["title"], "artist": ref["artist"],
                     "stem_shift_ms": round(lag * 1000, 1)})
        print(f"{slug:<11} {ref['title'][:28]:<28} {len(words):>4} words  stem shifted by "
              f"{lag * 1000:+.1f} ms, residual {residual * 1000:+.2f} ms")
    for out in (out_mix, out_iso):
        with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} songs -> {out_mix} and {out_iso}")


if __name__ == "__main__":
    main()
