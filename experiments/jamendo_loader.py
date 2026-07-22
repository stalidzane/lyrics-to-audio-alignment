"""
Loader for the JamendoLyrics dataset (HuggingFace: jamendolyrics/jamendolyrics).

Turns each dataset row into a plain dict:

    {
      "name":     str,                # track id, e.g. "HILA_-_Give_Me_the_Same"
      "language": str,                # "en", "es", ...
      "audio":    np.ndarray,         # mono float32 waveform at `sr`
      "sr":       int,                # sample rate (default 16000, Whisper-ready)
      "words":    list[str],          # word tokens, in order
      "onsets":   np.ndarray,         # ground-truth word START times (seconds)
      "offsets":  np.ndarray,         # ground-truth word END times (seconds)
      "word_entries": list[dict],     # raw word dicts (start/end/text/line_end)
    }

`onsets` is exactly what metrics.py expects as ground truth: one start time per
reference word, in order.

Why the audio handling looks the way it does
--------------------------------------------
Recent `datasets` (4.x) decodes Audio features via torchcodec, so row["audio"]
is a torchcodec `AudioDecoder`, not the old {"array", "sampling_rate"} dict.
We pull samples with .get_all_samples(), downmix to mono, and resample to 16 kHz
(Whisper's native rate) with librosa.

Dependencies: numpy, librosa (torch/torchcodec come in via `datasets`).
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np


def decode_audio(audio_obj, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Decode a torchcodec AudioDecoder (or old-style dict) to mono float32.

    Returns (waveform, sample_rate) with waveform resampled to `target_sr`.
    """
    # New path: torchcodec AudioDecoder
    if hasattr(audio_obj, "get_all_samples"):
        samples = audio_obj.get_all_samples()
        wav = samples.data                       # torch tensor [channels, n]
        sr = int(samples.sample_rate)
        wav = wav.mean(dim=0)                     # downmix to mono
        wav = wav.detach().cpu().numpy().astype(np.float32)
    # Fallback: old-style dict {"array", "sampling_rate"}
    elif isinstance(audio_obj, dict) and "array" in audio_obj:
        wav = np.asarray(audio_obj["array"], dtype=np.float32)
        sr = int(audio_obj["sampling_rate"])
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
    else:
        raise TypeError(f"Unrecognized audio object type: {type(audio_obj)}")

    if sr != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return wav, sr


def load_song(ex: dict, target_sr: int = 16000, decode: bool = True) -> dict:
    """Convert one dataset row into the standard song dict (see module docstring).

    Set decode=False to skip audio decoding (fast) when you only need the
    ground-truth words/onsets, e.g. to test the eval pipeline.
    """
    word_entries = ex["words"]
    words = [w["text"] for w in word_entries]
    onsets = np.array([w["start"] for w in word_entries], dtype=float)
    offsets = np.array([w["end"] for w in word_entries], dtype=float)

    song = {
        "name": ex["name"],
        "language": ex["language"],
        "words": words,
        "onsets": onsets,
        "offsets": offsets,
        "word_entries": word_entries,
        "audio": None,
        "sr": target_sr,
    }
    if decode:
        song["audio"], song["sr"] = decode_audio(ex["audio"], target_sr)
    return song


def load_songs(
    ds,
    n: Optional[int] = None,
    language: Optional[str] = None,
    target_sr: int = 16000,
    decode: bool = True,
) -> Iterator[dict]:
    """Yield song dicts from a HF dataset split.

    Args:
        ds:        a datasets.Dataset (e.g. load_dataset(..., split="test")).
        n:         max number of songs to yield (None = all).
        language:  if set, only yield songs whose "language" matches.
        target_sr: resample audio to this rate (16000 for Whisper).
        decode:    decode audio (True) or skip it (False, words/onsets only).

    Example:
        ds = load_dataset("jamendolyrics/jamendolyrics", "en", split="test")
        for song in load_songs(ds, n=5):
            print(song["name"], len(song["words"]), song["audio"].shape)
    """
    yielded = 0
    for ex in ds:
        if language is not None and ex["language"] != language:
            continue
        yield load_song(ex, target_sr=target_sr, decode=decode)
        yielded += 1
        if n is not None and yielded >= n:
            break


if __name__ == "__main__":
    # Quick self-check against the real dataset (needs internet + deps).
    from datasets import load_dataset

    ds = load_dataset("jamendolyrics/jamendolyrics", "en", split="test")
    for song in load_songs(ds, n=3):
        print(
            f"{song['name']:<35} lang={song['language']} "
            f"words={len(song['words']):>4} "
            f"audio={song['audio'].shape} sr={song['sr']} "
            f"first_onset={song['onsets'][0]:.2f}s"
        )
