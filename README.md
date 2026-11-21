# Word-Level Lyrics-to-Audio Alignment for a Rhythm Game

Code for the bachelor's thesis *Word-Level Lyrics-to-Audio Alignment for a Rhythm Game: A Comparative
Evaluation of Publicly Available Tools*. Four publicly available systems are used as released, without
any training, to time the word onsets of song lyrics, and are compared with human reference timings:

| System | Type | Script |
|---|---|---|
| Whisper (small, large-v3) | speech transcription | `experiments/run_whisper.py` |
| MMS forced aligner | speech-trained forced aligner (wav2vec2 + CTC) | `experiments/run_mms.py` |
| Huang & Benetos | singing-trained forced aligner (CRNN + CTC, phonemes) | `experiments/run_huang_benetos.py` |
| Montreal Forced Aligner (MFA) | HMM-GMM forced aligner | `experiments/run_mfa.py` |

Each system is run on the full mix and on vocals separated with Demucs (htdemucs).

## Data

| Dataset | Songs | Languages | Source |
|---|---|---|---|
| `jamendo` | 79 | en, es, de, fr | JamendoLyrics MultiLang (Hugging Face) |
| `jamendopp` | 80 | en | [JamendoLyrics++](https://github.com/tikick/jamendolyricspp) |
| `company` | 80 | en, lv, it, de, es, fr, pt, ro, sv | proprietary rhythm-game songs with tapped references |

The rhythm-game songs and their timings are licensed and **not included** in this repository; the
scripts for them are included so the processing is documented. All data, predictions and results are
gitignored and are created by the scripts below.

Every dataset is exported to the same layout, so all systems and the evaluation work on any of them:

```
data/<dataset>/audio/<song>.wav                         full mix
data/<dataset>/separated/htdemucs/<song>/vocals.wav     Demucs vocals
data/<dataset>/refs/<song>.json                         reference words with onsets
data/<dataset>/index.csv                                one row per song (language, words, duration)
```

## Environments

The systems need incompatible package versions, so they run in separate environments. The exact
package lists are in `envs/`:

| Environment | Used for | File |
|---|---|---|
| `lyrics2audio` (venv in the repo root) | Whisper (mlx-whisper), data export, evaluation, tables, plots | `envs/lyrics2audio.txt` |
| `audio` (conda) | Demucs and MMS (ctc-forced-aligner from GitHub, not the PyPI package of the same name) | `envs/audio.yml` |
| `hb` (conda, Python 3.9) | Huang & Benetos; needs espeak-ng and a clone of [their code](https://github.com/jhuang448/LyricsAlignment-Multilingual) in `external/` | `envs/hb.yml` |
| `mfa` (conda) | Montreal Forced Aligner 3 with its pretrained models | `envs/mfa.yml` |

```
python3 -m venv lyrics2audio && lyrics2audio/bin/pip install -r envs/lyrics2audio.txt
conda env create -f envs/audio.yml      # likewise hb.yml, mfa.yml
```

ffmpeg must be installed. The scripts were run on an Apple-silicon Mac (Whisper uses mlx, Demucs and
MMS the `mps` device; use `--device cpu` elsewhere).

## Running

All commands run from the repository root.

**1. Export the datasets**
```
lyrics2audio/bin/python -m data.export_jamendo
git clone --depth 1 https://github.com/tikick/jamendolyricspp.git data/raw/jamendolyricspp
lyrics2audio/bin/python -m data.export_jamendopp
lyrics2audio/bin/python -m data.export_company          # needs the company download, see below
```

**2. Run all systems and score them** (per dataset; songs already done are skipped, so an interrupted
run continues where it stopped)
```
bash experiments/run_all.sh jamendo
bash experiments/run_all.sh jamendopp
bash experiments/run_all.sh company
```
`run_all.sh` separates the vocals, runs Whisper small and large-v3, MMS, Huang & Benetos and MFA on
the mix and the vocals, scores everything and rebuilds the tables. It calls each environment's own
Python, so nothing has to be activated; set `CONDA_BASE` and `HB_DIR` if your paths differ. Each
`run_*.py` script can also be run on its own (see its docstring, e.g. `--songs` or `--limit` for a
test run).

**3. Further analyses**
```
lyrics2audio/bin/python eval/offset_loo.py                     # robustness of the offset correction
lyrics2audio/bin/python experiments/agreement_check.py         # human agreement (rhythm-game songs)
lyrics2audio/bin/python data/export_stems.py                   # the five songs with vocal tracks ...
bash experiments/run_all.sh stems && bash experiments/run_all.sh stems_iso   # ... Demucs vocals vs vocal track
lyrics2audio/bin/python visualisation/plot_pco_tolerance.py    # PCO vs tolerance (add --corrected)
lyrics2audio/bin/python eval/make_results_table.py          # LaTeX tables of the thesis
lyrics2audio/bin/python eval/labels.py --dataset jamendo --song Avercage_-_Embers --system mms-vocals
                                                               # Audacity label tracks for one song
```

## Evaluation

`eval/score.py` compares the predicted onset of every reference word with the reference onset:

- **PCO at 0.05 / 0.1 / 0.3 / 0.6 s**: share of reference words whose onset error is within the
  tolerance; words without a predicted time count as wrong. 0.1, 0.3 and 0.6 s are the game's
  perfect, good and okay windows, 0.3 s the usual tolerance in the literature.
- **Corrected PCO**: the same after subtracting one constant offset per system and dataset (the
  median signed error of the words within 1 s of their reference), i.e. a single latency calibration.
- **Coverage, AAE, offset**: share of words with a time, average absolute error (as in MIREX), and the
  constant offset itself (negative = early).

Whisper transcribes instead of placing the given lyrics, so its words are first matched to the
lyrics by text (`eval/word_match.py`). All values are macro averages: per song, then averaged over
songs; songs a system fails on count as 0.

Outputs:
```
predictions/<dataset>/<system>/<song>.json     raw output of each system
results/<dataset>/summary.csv                  all systems, per language
results/<dataset>/<system>.csv                 one row per song
results/<dataset>/words/<system>.csv           one row per reference word
results/offset_robustness.csv                  leave-one-out and k-song offset estimates
results/agreement/agreement_check.csv          human agreement, one row per pair of timings
```

## Repository layout

```
data/              dataset export (Jamendo, JamendoLyrics++, rhythm-game songs, vocal tracks),
                   company audio download
experiments/       separation, the four systems, run_all.sh, human agreement
eval/              scoring, word matching, metrics, offset robustness, Audacity labels
visualisation/     PCO-vs-tolerance plot and figures
envs/              environment files
```

## Licences of the tools

The systems keep their own licences: the MMS forced-aligner checkpoint is CC-BY-NC 4.0; Whisper, MFA,
Demucs and the Huang & Benetos code are published under their respective licences. Check them before
any commercial use.
