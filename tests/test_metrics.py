"""
Unit tests for eval/metrics.py. From the repo root run either
`pytest` or `python tests/test_metrics.py`.

They validate the metrics against hand-computed values on tiny fake inputs,
so you can trust the numbers before real Jamendo predictions ever touch them.
"""

import math
import os
import sys

import numpy as np

# make the repo root importable when run as a plain script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import (
    absolute_errors,
    compute_metrics,
    evaluate_song,
    aggregate,
    DEFAULT_TOLERANCES,
)

TOL = 1e-9


# --------------------------------------------------------------------------- #
# absolute_errors
# --------------------------------------------------------------------------- #
def test_absolute_errors_basic():
    ae = absolute_errors([1.0, 2.0, 3.0], [1.1, 1.8, 3.0])
    assert np.allclose(ae, [0.1, 0.2, 0.0])


def test_absolute_errors_nan_pred_propagates():
    ae = absolute_errors([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])
    assert ae[0] == 0.0
    assert math.isnan(ae[1])
    assert ae[2] == 0.0


def test_absolute_errors_shape_mismatch_raises():
    try:
        absolute_errors([1.0, 2.0], [1.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


def test_absolute_errors_gt_nan_raises():
    try:
        absolute_errors([1.0, 2.0], [1.0, np.nan])
    except ValueError:
        return
    raise AssertionError("expected ValueError when ground truth has NaN")


# --------------------------------------------------------------------------- #
# compute_metrics — perfect prediction
# --------------------------------------------------------------------------- #
def test_perfect_prediction():
    gt = [0.5, 1.0, 1.5, 2.0]
    m = compute_metrics(gt, gt)
    assert m["aae"] == 0.0
    assert m["median_ae"] == 0.0
    assert m["coverage"] == 1.0
    for t in DEFAULT_TOLERANCES:
        assert m["pco"][t] == 1.0
        assert m["pco_matched"][t] == 1.0


# --------------------------------------------------------------------------- #
# compute_metrics — constant offset of exactly 0.15 s
# --------------------------------------------------------------------------- #
def test_constant_offset():
    gt = np.array([0.0, 1.0, 2.0, 3.0])
    pred = gt + 0.15
    m = compute_metrics(pred, gt)
    assert abs(m["aae"] - 0.15) < TOL
    assert abs(m["median_ae"] - 0.15) < TOL
    # 0.15 > 0.1  -> PCO@0.1 = 0 ; 0.15 <= 0.2 and <= 0.3 -> 1.0
    assert m["pco"][0.1] == 0.0
    assert m["pco"][0.2] == 1.0
    assert m["pco"][0.3] == 1.0


# --------------------------------------------------------------------------- #
# compute_metrics — boundary is inclusive (error == tolerance counts as correct)
# --------------------------------------------------------------------------- #
def test_tolerance_boundary_inclusive():
    gt = np.array([0.0])
    pred = np.array([0.1])  # error exactly 0.1
    m = compute_metrics(pred, gt)
    assert m["pco"][0.1] == 1.0


# --------------------------------------------------------------------------- #
# compute_metrics — missing predictions (NaN) handling
# --------------------------------------------------------------------------- #
def test_missing_predictions():
    gt = np.array([0.0, 1.0, 2.0, 3.0])
    # two of four words predicted, both within 0.1 s
    pred = np.array([0.05, np.nan, np.nan, 3.05])
    m = compute_metrics(pred, gt)
    assert m["n_words"] == 4
    assert m["n_matched"] == 2
    assert abs(m["coverage"] - 0.5) < TOL
    # AAE over matched words only: mean(|0.05|, |0.05|) = 0.05
    assert abs(m["aae"] - 0.05) < TOL
    # primary PCO penalizes missing words: 2 correct / 4 total = 0.5
    assert abs(m["pco"][0.1] - 0.5) < TOL
    # matched-only PCO: 2 correct / 2 matched = 1.0
    assert abs(m["pco_matched"][0.1] - 1.0) < TOL


def test_all_missing():
    gt = np.array([0.0, 1.0])
    pred = np.array([np.nan, np.nan])
    m = compute_metrics(pred, gt)
    assert m["n_matched"] == 0
    assert math.isnan(m["aae"])
    assert m["pco"][0.2] == 0.0          # no correct onsets
    assert math.isnan(m["pco_matched"][0.2])


# --------------------------------------------------------------------------- #
# aggregate — macro vs micro across songs of different length
# --------------------------------------------------------------------------- #
def test_aggregate_macro_vs_micro():
    # Song A: 1 word, error 0.0 ; Song B: 3 words, errors 0.3 each
    a = evaluate_song("A", "en", [0.0], [0.0])
    b = evaluate_song("B", "en", [0.3, 1.3, 2.3], [0.0, 1.0, 2.0])
    agg = aggregate([a, b])

    # macro AAE = mean(0.0, 0.3) = 0.15 (songs weighted equally)
    assert abs(agg["overall"]["macro"]["aae"] - 0.15) < TOL
    # micro AAE = mean over 4 pooled words = (0 + 0.3*3)/4 = 0.225
    assert abs(agg["overall"]["micro"]["aae"] - 0.225) < TOL


def test_aggregate_by_language():
    en = evaluate_song("e1", "en", [0.0], [0.0])
    de = evaluate_song("d1", "de", [0.5], [0.0])  # error 0.5
    agg = aggregate([en, de])
    assert set(agg["by_language"]) == {"en", "de"}
    assert abs(agg["by_language"]["en"]["macro"]["aae"] - 0.0) < TOL
    assert abs(agg["by_language"]["de"]["macro"]["aae"] - 0.5) < TOL


def test_aggregate_empty_raises():
    try:
        aggregate([])
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty results")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  ok  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
