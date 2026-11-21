"""
Separate vocals for every song with Demucs (htdemucs), once.

Output (gitignored):
    data/<dataset>/separated/htdemucs/<name>/vocals.wav
    data/<dataset>/separated/htdemucs/<name>/no_vocals.wav

Used by every system run with --input vocals. Songs already separated are skipped.

Demucs pins older torch/torchaudio versions, so it is installed in the `audio` conda env
(envs/audio.yml), not in the main lyrics2audio venv. This script finds the demucs command
in that env by itself; --demucs points it elsewhere:
    python3 -m experiments.separate_demucs --dataset company
    python3 -m experiments.separate_demucs --demucs /path/to/envs/audio/bin/demucs --dataset company

Default device is "mps" (Apple GPU); use --device cpu if mps errors out.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_demucs() -> str:
    """The demucs executable inside the 'audio' conda env (or the old venv), so no conda activate is needed."""
    home = os.path.expanduser("~")
    conda_exe = os.environ.get("CONDA_EXE", "")
    candidates = [os.path.join(conda_exe.rsplit("/bin/", 1)[0], "envs/audio/bin/demucs")] if conda_exe else []
    for base in ("miniconda3", "miniforge3", "anaconda3", "mambaforge", "opt/miniconda3", "opt/anaconda3"):
        candidates.append(os.path.join(home, base, "envs/audio/bin/demucs"))
    candidates += ["/opt/homebrew/Caskroom/miniconda/base/envs/audio/bin/demucs",
                   "/opt/homebrew/Caskroom/miniforge/base/envs/audio/bin/demucs",
                   os.path.join(home, "venvs/demucs/bin/demucs")]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return "demucs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demucs", help="demucs command; default: the demucs in the 'audio' conda env")
    ap.add_argument("--model", default="htdemucs")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dataset", default="jamendo", help="folder under data/: jamendo, jamendopp or company")
    ap.add_argument("--languages", nargs="+", help="only these languages (default: all)")
    ap.add_argument("--batch", type=int, default=8, help="songs per demucs call")
    args = ap.parse_args()

    demucs = shlex.split(args.demucs) if args.demucs else [find_demucs()]
    demucs[0] = os.path.expanduser(demucs[0])
    if not (os.path.exists(demucs[0]) or shutil.which(demucs[0])):
        sys.exit(f"demucs not found: {demucs[0]}\n"
                 "find it with:  conda activate audio && which demucs   then pass --demucs <that path>")
    print("demucs:", " ".join(demucs))

    DATASET = os.path.join(REPO_ROOT, "data", args.dataset)  # dataset folder
    out_root = os.path.join(DATASET, "separated")
    with open(os.path.join(DATASET, "index.csv")) as f:
        rows = [r for r in csv.DictReader(f) if not args.languages or r["language"] in args.languages]
    todo = [r["name"] for r in rows
            if not os.path.exists(os.path.join(out_root, args.model, r["name"], "vocals.wav"))]
    print(f"{len(rows)} songs, {len(todo)} to separate with {args.model} on {args.device}")

    t0 = time.time()
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        files = [os.path.join(DATASET, "audio", f"{n}.wav") for n in chunk]
        cmd = demucs + ["-n", args.model, "--two-stems", "vocals",
               "-d", args.device, "-o", out_root] + files
        print(f"  [{i + len(chunk)}/{len(todo)}] {', '.join(chunk)}")
        subprocess.run(cmd, check=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {os.path.join(out_root, args.model)}")


if __name__ == "__main__":
    main()
