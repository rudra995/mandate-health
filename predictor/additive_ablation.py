"""The additive cross-merchant test.

See ``docs/RESULTS.md``, "Additive cross-merchant test", for the
pre-registered hypothesis, feature spec, and decision rule - written and
committed *before* this module was run. This module implements exactly that
spec and nothing else. It is a one-off diagnostic, run once; it is not part
of the production `predictor/train.py` pipeline and `make train` does not
invoke it, unlike ``predictor/ablation.py``.

The first ablation (``predictor/ablation.py``) tested whether *replacing*
clean same-mandate history with a blended cross-merchant view helps. It found
no measurable advantage, plausibly because the blend dilutes a mature
mandate's own strong day-of-month autocorrelation. That result says nothing
about the actual claim in CLAUDE.md section 3, which is additive, not a
substitution claim: the aggregator sees things a merchant cannot, on top of
what the merchant already has. This module tests that directly by adding two
new columns - other-merchants-only versions of the two named aggregator
features - to the merchant-only feature set, and asking whether the pair
beats the single (same-mandate-only) feature.

Reuses ``predictor.features``'s private state machinery
(``_PayerState``, ``_windowed_dom_rate``) rather than re-deriving the
day-of-month formula, so any effect measured here is attributable to
information content, not to a subtly different calculation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from predictor.ablation import bootstrap_auc_gap_ci
from predictor.calibrate import evaluate_calibration, fit_calibrator
from predictor.features import (
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURE_COLUMNS,
    _PayerState,
    _merge_static_fields,
    _windowed_dom_rate,
    build_features,
)
from predictor.train import evaluate, load_observable, make_splits, train_full_model

NEW_COLUMNS = ("dom_fail_propensity_other_merchants", "concurrent_debits_same_day_other_merchants")


def _other_merchant_dom_propensity(cycles: pd.DataFrame, mandates: pd.DataFrame) -> pd.Series:
    """``dom_fail_propensity`` computed from a payer's OTHER merchants only.

    Mirrors ``build_features``'s two-pass-per-date temporal safety exactly:
    every row reads state as it stood strictly before its own date, and that
    date's outcomes are applied as one batch only after every row on that
    date has read the prior state. The difference from ``merchant_only``
    scope is what gets aggregated at read time: instead of reading this
    row's own (payer, merchant) state, every *other* merchant this payer
    holds is merged into one combined state and queried.
    """
    frame = _merge_static_fields(cycles, mandates)
    frame = frame.sort_values(["scheduled_date", "payer_id", "mandate_id"], kind="stable").reset_index(
        drop=True
    )

    payer_merchants = mandates.groupby("payer_id")["merchant_id"].apply(set).to_dict()
    substates: dict[tuple[str, str], _PayerState] = defaultdict(_PayerState)

    values = np.empty(len(frame), dtype=float)
    row_positions = {idx: pos for pos, idx in enumerate(frame.index)}

    for _scheduled_date, day_frame in frame.groupby("scheduled_date", sort=True):
        for record in day_frame.itertuples():
            other_merchants = payer_merchants[record.payer_id] - {record.merchant_id}
            merged = _PayerState()
            for merchant_id in other_merchants:
                s = substates.get((record.payer_id, merchant_id))
                if s is None:
                    continue
                merged.total += s.total
                merged.total_fail += s.total_fail
                for day, count in s.day_total.items():
                    merged.day_total[day] += count
                for day, count in s.day_fail.items():
                    merged.day_fail[day] += count

            value = (
                _windowed_dom_rate(merged, record.scheduled_date.day) if merged.total > 0 else np.nan
            )
            values[row_positions[record.Index]] = value

        for record in day_frame.itertuples():
            failed = record.outcome == "failure"
            substates[(record.payer_id, record.merchant_id)].update(
                failed=failed,
                decline_code=record.decline_code,
                day_of_month=record.scheduled_date.day,
                amount=record.amount,
            )

    return pd.Series(values, name="dom_fail_propensity_other_merchants")


def build_augmented_feature_set(cycles: pd.DataFrame, mandates: pd.DataFrame, merchants: pd.DataFrame):
    """Model B: merchant_only features plus the two other-merchants columns."""
    base = build_features(cycles, mandates, merchants, scope="merchant_only")
    cross = build_features(cycles, mandates, merchants, scope="cross_merchant")

    other_dom = _other_merchant_dom_propensity(cycles, mandates)

    # Row order must match `base` exactly for a plain column concat to be
    # valid. Both build_features and _other_merchant_dom_propensity sort by
    # the same key (scheduled_date, payer_id, mandate_id), so this asserts
    # that invariant rather than silently trusting it.
    assert base.meta["cycle_id"].tolist() == cross.meta["cycle_id"].tolist()
    assert len(other_dom) == len(base.X)

    X = base.X.copy()
    X["dom_fail_propensity_other_merchants"] = other_dom.to_numpy()
    X["concurrent_debits_same_day_other_merchants"] = cross.X["concurrent_debits_same_day"].to_numpy()

    return base.meta, X, base.y


def run_additive_test(seed: int = 42) -> dict[str, Any]:
    cycles, mandates, merchants = load_observable()
    payer_ids = mandates["payer_id"].unique().tolist()

    # Model A: baseline, merchant_only scope, unchanged.
    fs_a = build_features(cycles, mandates, merchants, scope="merchant_only")
    splits_a = make_splits(fs_a, payer_ids, seed=seed)

    # Model B: augmented with the two other-merchants columns.
    meta_b, X_b, y_b = build_augmented_feature_set(cycles, mandates, merchants)
    from predictor.features import FeatureSet

    fs_b = FeatureSet(meta=meta_b, X=X_b, y=y_b)
    payer_col = fs_b.meta["payer_id"]

    def subset_b(payer_set):
        mask = payer_col.isin(payer_set)
        return FeatureSet(
            meta=fs_b.meta.loc[mask].reset_index(drop=True),
            X=fs_b.X.loc[mask].reset_index(drop=True),
            y=fs_b.y.loc[mask].reset_index(drop=True),
        )

    from predictor.split import make_payer_split

    split = make_payer_split(payer_ids, seed=seed)
    splits_b_train = subset_b(split.train)
    splits_b_calibration = subset_b(split.calibration)
    splits_b_test = subset_b(split.test)

    assert splits_a.test.y.equals(splits_b_test.y), "test targets differ between A and B - not comparable"

    categorical_b = list(CATEGORICAL_FEATURE_COLUMNS)
    numeric_new = list(NEW_COLUMNS)
    feature_columns_b = list(FEATURE_COLUMNS) + numeric_new

    result_a, _ = train_full_model(splits_a.train, seed=seed)

    import lightgbm as lgb

    from predictor.train import LGBM_MAX_ROUNDS, LGBM_EARLY_STOPPING_ROUNDS, LGBM_PARAMS, _internal_early_stop_split

    fit_b, es_b = _internal_early_stop_split(splits_b_train, seed)
    dtrain_b = lgb.Dataset(
        fit_b.X[feature_columns_b], label=fit_b.y, categorical_feature=categorical_b, free_raw_data=False
    )
    dvalid_b = lgb.Dataset(
        es_b.X[feature_columns_b],
        label=es_b.y,
        reference=dtrain_b,
        categorical_feature=categorical_b,
        free_raw_data=False,
    )
    booster_b = lgb.train(
        LGBM_PARAMS,
        dtrain_b,
        num_boost_round=LGBM_MAX_ROUNDS,
        valid_sets=[dvalid_b],
        callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    def predict_b(X: pd.DataFrame) -> np.ndarray:
        return booster_b.predict(X[feature_columns_b], num_iteration=booster_b.best_iteration)

    # Calibrate and evaluate both on the identical test targets.
    p_calib_a = result_a.predict(splits_a.calibration.X)
    cal_a = fit_calibrator(p_calib_a, splits_a.calibration.y)
    p_test_raw_a = result_a.predict(splits_a.test.X)
    p_test_cal_a = cal_a.predict(p_test_raw_a)

    p_calib_b = predict_b(splits_b_calibration.X)
    cal_b = fit_calibrator(p_calib_b, splits_b_calibration.y)
    p_test_raw_b = predict_b(splits_b_test.X)
    p_test_cal_b = cal_b.predict(p_test_raw_b)

    report_a = evaluate_calibration(splits_a.test.y, p_test_raw_a, p_test_cal_a)
    report_b = evaluate_calibration(splits_b_test.y, p_test_raw_b, p_test_cal_b)

    y_test = splits_a.test.y.to_numpy()
    ci = bootstrap_auc_gap_ci(y_test, p_test_cal_b, p_test_cal_a)  # gap = B - A

    return {
        "auc_a": report_a.auc_calibrated,
        "brier_a": report_a.brier_calibrated,
        "auc_b": report_b.auc_calibrated,
        "brier_b": report_b.brier_calibrated,
        "n_test": report_a.n,
        "gap_auc": report_b.auc_calibrated - report_a.auc_calibrated,
        "gap_brier": report_a.brier_calibrated - report_b.brier_calibrated,
        "bootstrap_ci": ci,
        "other_merchants_coverage": float(X_b["dom_fail_propensity_other_merchants"].notna().mean()),
    }


def format_report(outcome: dict[str, Any]) -> str:
    ci = outcome["bootstrap_ci"]
    verdict = (
        "supported: CI excludes zero in B's favour"
        if ci["ci_low"] > 0
        else "NOT supported: CI crosses zero (or favours A) - second null result"
    )
    lines = [
        f"{'model':<45} {'AUC':>8} {'Brier':>8} {'n':>6}",
        f"{'A: merchant-only (same-mandate history)':<45} {outcome['auc_a']:>8.4f} {outcome['brier_a']:>8.4f} {outcome['n_test']:>6}",
        f"{'B: A + other-merchants columns (additive)':<45} {outcome['auc_b']:>8.4f} {outcome['brier_b']:>8.4f} {outcome['n_test']:>6}",
        "",
        f"gap (B - A): AUC {outcome['gap_auc']:+.4f}, Brier {outcome['gap_brier']:+.4f} (positive = B better)",
        f"bootstrap 95% CI on AUC gap ({ci['n_resamples']} resamples): [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]",
        f"other-merchants column coverage (non-NaN share, test set): {outcome['other_merchants_coverage']:.1%}",
        "",
        f"verdict: {verdict}",
    ]
    return "\n".join(lines)


def main() -> None:
    outcome = run_additive_test()
    print(format_report(outcome))


if __name__ == "__main__":
    main()
