"""
Plot PCO against the tolerance, one panel per dataset, one line per system, with the
game's windows marked.

    python3 visualisation/plot_pco_tolerance.py                 # raw PCO, Demucs vocals
    python3 visualisation/plot_pco_tolerance.py --corrected     # after the constant offset correction
    python3 visualisation/plot_pco_tolerance.py --input mix

Reads results/<dataset>/words/<system>.csv (one row per reference word, written by
eval/score.py) and the offsets from results/<dataset>/summary.csv.
PCO is computed exactly as in eval/score.py: per song (words without a time count as wrong),
then averaged over songs; songs a system failed on count as 0.
Writes visualisation/figures/pco_tolerance_<input>[_corrected].png
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# (system prefix, label, colour, line style, marker); fixed order = fixed colour per system
SYSTEMS = [("whisper-small", "Whisper small", "#2a78d6", (0, (5, 2)), "o"),
           ("whisper-large-v3", "Whisper large-v3", "#eb6834", (0, (1, 1.5)), "s"),
           ("mms", "MMS", "#1baf7a", "-", "^"),
           ("hb", "Huang & Benetos", "#eda100", "-", "D"),
           ("mfa", "MFA", "#e87ba4", (0, (5, 1.5, 1, 1.5)), "v")]
# (tolerance, label above the plot, alignment so neighbouring labels do not collide)
WINDOWS = [(0.05, "0.05 s ", "right"), (0.1, " perfect", "left"), (0.3, "good", "center"), (0.6, "okay", "center")]
DATASET_NAMES = {"jamendo": "JamendoLyrics MultiLang", "jamendopp": "JamendoLyrics++",
                 "company": "Rhythm-game songs"}


def read_errors(path):
    """{song: array of signed errors, NaN for words without a time}"""
    songs = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            songs[r["song_id"]].append(float(r["error"]) if r["error"] not in ("", "nan") else np.nan)
    return [np.array(v) for v in songs.values()]


def macro_pco(songs, taus, offset=0.0):
    """mean over songs of the share of words with |error - offset| <= tau"""
    per_song = []
    for e in songs:
        ae = np.abs(e - offset)                      # NaN stays NaN -> counts as wrong
        per_song.append([(ae <= t).sum() / len(e) for t in taus])
    return 100 * np.mean(per_song, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="vocals", choices=["vocals", "mix"])
    ap.add_argument("--corrected", action="store_true",
                    help="plot PCO after the constant offset correction instead of the raw PCO")
    ap.add_argument("--max_tol", type=float, default=0.65)
    args = ap.parse_args()
    taus = np.round(np.arange(0, args.max_tol + 1e-9, 0.005), 3)

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, len(DATASET_NAMES), figsize=(7.0, 3.0), sharey=True)
    for ax, (ds, ds_name) in zip(axes, DATASET_NAMES.items()):
        res = os.path.join(ROOT, "results", ds)
        with open(os.path.join(res, "summary.csv"), encoding="utf-8") as f:
            offsets = {r["system"]: float(r["offset"]) for r in csv.DictReader(f) if r["language"] == "all"}
        for w, _, _ in WINDOWS:
            ax.axvline(w, color="#b5b3ad", lw=0.8, ls="--", zorder=0)
        for prefix, label, colour, ls, marker in SYSTEMS:
            name = f"{prefix}-{args.input}"
            path = os.path.join(res, "words", f"{name}.csv")
            if not os.path.exists(path):
                continue
            y = macro_pco(read_errors(path), taus, offsets[name] if args.corrected else 0.0)
            ax.plot(taus, y, color=colour, ls=ls, lw=1.8, label=label,
                    marker=marker, markevery=[int(round(w / 0.005)) for w, _, _ in WINDOWS if w <= args.max_tol],
                    markersize=4.5, markeredgecolor="white", markeredgewidth=0.7)
            if prefix == "hb":   # sanity check against the tables
                print(f"{ds} {name}{' corrected' if args.corrected else ''}: "
                      f"PCO@0.1 {y[20]:.1f}, PCO@0.3 {y[60]:.1f}")
        ax.set_xlim(0, args.max_tol)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ax.set_xticklabels(["0", ".1", ".2", ".3", ".4", ".5", ".6"])
        ax.set_xlabel("Tolerance (s)")
        ax.set_title(ds_name, fontsize=9, pad=14)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        for w, wl, ha in WINDOWS:
            ax.text(w, 101, wl, ha=ha, va="bottom", fontsize=6.5, color="#52514e")
    axes[0].set_ylabel("Corrected PCO (%)" if args.corrected else "PCO (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))

    out_dir = os.path.join(ROOT, "visualisation", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"pco_tolerance_{args.input}{'_corrected' if args.corrected else ''}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
