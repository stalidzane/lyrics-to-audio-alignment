#!/bin/bash
# Run every system on one dataset (skips songs already done), then score.
#   caffeinate -is bash experiments/run_all.sh company 2>&1 | tee -a run_all.log
# Uses each environment's own python, so nothing needs to be activated.
set -u
cd "$(dirname "$0")/.."
DS="${1:-company}"
CONDA_BASE="${CONDA_BASE:-/opt/homebrew/Caskroom/miniforge/base}"
PY=lyrics2audio/bin/python                      # Whisper, Demucs launcher, scoring
AUDIO="$CONDA_BASE/envs/audio/bin/python"       # MMS
HB="$CONDA_BASE/envs/hb/bin/python"             # Huang & Benetos
MFA="$CONDA_BASE/envs/mfa/bin/python"           # MFA
HB_DIR="${HB_DIR:-$PWD/external/LyricsAlignment-Multilingual}"
HB_DIR="$(cd "$HB_DIR" 2>/dev/null && pwd || echo "$HB_DIR")"   # make it absolute
export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib
export PATH="$CONDA_BASE/envs/mfa/bin:$PATH"     # the `mfa` command

echo "=== $(date) dataset=$DS"
$PY -u experiments/separate_demucs.py --dataset "$DS"
for m in small large-v3; do for inp in mix vocals; do
  $PY -u experiments/run_whisper.py --model $m --input $inp --dataset "$DS"
done; done
for inp in vocals mix; do
  $AUDIO -u experiments/run_mms.py --dataset "$DS" --input $inp
  $HB -u experiments/run_huang_benetos.py --hb_dir "$HB_DIR" --dataset "$DS" --input $inp
  $MFA -u experiments/run_mfa.py --dataset "$DS" --input $inp
done
$PY eval/score.py --dataset "$DS"
$PY writing/make_results_table.py
echo "=== $(date) done"
