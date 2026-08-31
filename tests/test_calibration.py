"""Calibration must mean what it says, not just rank well.

AUC is invariant to any monotonic transform of the score, so it cannot tell
you whether "0.30" corresponds to a 30% failure rate. This file is the test
that actually checks that, on real held-out data from the real pipeline - not
a synthetic toy distribution where any calibrator would trivially pass.
"""

from __future__ import annotations

import pandas as pd
import pytest

from predictor.calibrate import (
    apply_calibrator,
    expected_calibration_error,
    fit_calibrator,
)
from predictor.features import build_features
from predictor.train import (
    load_observable,
    make_splits,
    train_full_model,
)

# Loose enough to survive ordinary re-seeds and data regeneration, tight
# enough to catch a genuinely broken calibrator (e.g. one that was fit
# in-sample or applied to the wrong split). The trained pipeline measures
# ECE ~0.014 on this dataset; 0.05 leaves real headroom without being
# vacuous.
ECE_TOLERANCE = 0.05


@pytest.fixture(scope="module")
def calibration_fixture():
    cycles, mandates, merchants = load_observable()
    feature_set = build_features(cycles, mandates, merchants)
    payer_ids = mandates["payer_id"].unique().tolist()
    splits = make_splits(feature_set, payer_ids)

    result, _booster = train_full_model(splits.train)
    p_calib_raw = result.predict(splits.calibration.X)
    calibrator = fit_calibrator(p_calib_raw, splits.calibration.y)

    p_test_raw = result.predict(splits.test.X)
    p_test_calibrated = apply_calibrator(calibrator, p_test_raw)

    return {
        "y_test": splits.test.y,
        "p_test_raw": p_test_raw,
        "p_test_calibrated": p_test_calibrated,
    }


def test_calibrated_probabilities_match_observed_frequency(calibration_fixture):
    ece = expected_calibration_error(calibration_fixture["y_test"], calibration_fixture["p_test_calibrated"])
    assert ece < ECE_TOLERANCE, f"expected calibration error {ece:.4f} exceeds tolerance {ECE_TOLERANCE}"


def test_calibration_improves_or_holds_expected_calibration_error(calibration_fixture):
    """Calibration should not make things worse. A small tolerance for noise
    (~1,600 test rows across a handful of quantile bins) rather than a strict
    less-than, since sampling variance alone can occasionally nudge ECE up by
    a hair even when the calibration map is doing its job."""
    ece_raw = expected_calibration_error(calibration_fixture["y_test"], calibration_fixture["p_test_raw"])
    ece_calibrated = expected_calibration_error(
        calibration_fixture["y_test"], calibration_fixture["p_test_calibrated"]
    )
    assert ece_calibrated <= ece_raw + 0.01


def test_calibrated_probabilities_stay_in_unit_interval(calibration_fixture):
    p = calibration_fixture["p_test_calibrated"]
    assert (p >= 0.0).all()
    assert (p <= 1.0).all()
