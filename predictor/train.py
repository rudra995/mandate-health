"""Train the three model variants and report how they compare.

**The prediction target, and what it does not mean.** The label is the
observed outcome of the *original presentation* for each cycle - failure
labelled 1 - exactly as it happened under Phase 0's status-quo baseline
policy (fixed 24h PDN, baseline slot weights). This is the only honest choice:
a production system never observes an outcome under "no policy", only under
whatever policy was actually running, so training on anything else would be
its own form of leakage - using information (the ground-truth counterfactual)
that is not observable in production.

**Consequence, stated plainly because it is easy to miss:** ``p_fail`` from
this model estimates failure probability *under the historical baseline
policy*. When Phase 2's policy engine computes an uplift from taking an
action, that uplift is improvement relative to the baseline this model was
trained on, not relative to doing nothing at all. The ground-truth
counterfactual outcome (what would have happened with no intervention at all)
is never used here - it is reserved for Phase 5's evaluation harness, which is
the only place in the project allowed to grade against it.

**No threshold is chosen anywhere in this module.** Precision/recall at a few
probability cuts are reported purely as diagnostics. The operating threshold
is derived in Phase 2 from the expected-value formula; picking one here would
pre-empt that and hardcode a 0.5-style cut this project explicitly rejects.

Three variants, in increasing order of what they are allowed to see:

1. **Trivial** - predicts the training population's base rate for everyone.
   The floor: any model that cannot beat "always guess the average" adds
   nothing.
2. **Single-feature** - logistic regression on ``payer_fail_streak`` alone,
   cold-start rows filled with 0 for this baseline only (documented at the
   fill site; the full model instead leaves the value as NaN and lets the
   tree route it, which is the entire point of the comparison).
3. **Full model** - LightGBM gradient boosting over all fourteen features
   plus the cold-start flag. LightGBM is chosen over
   ``HistGradientBoostingClassifier`` for two concrete reasons, not general
   preference: native categorical splits (``slot_planned``,
   ``merchant_category``) without one-hot encoding, and native per-instance
   SHAP-style contributions via ``Booster.predict(pred_contrib=True)``, which
   is exactly what ``RiskModel.explain`` needs and what CLAUDE.md section 11
   asks for - a money decision must be explainable per instance, and a tree
   model's attribution is what a compliance reviewer can actually read.

Hyperparameters were chosen by comparing four modest configurations
(``num_leaves`` in {7, 15, 31}, ``learning_rate`` in {0.03, 0.05}) on an
early-stopping split carved out of the *training* payers only - never the
calibration or test payers, which stay untouched until their own step. The
smallest model (``num_leaves=7``) won on both AUC and Brier score, consistent
with a dataset this size (roughly 5,000 training rows over 14 features): a
larger tree budget started overfitting. That comparison is not re-run here -
its result is hardcoded as the final configuration, with the losing
configurations recorded in ``docs/RESULTS.md`` rather than left as an
unexplained choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_auc_score

from predictor.calibrate import (
    evaluate_calibration,
    fit_calibrator,
    plot_reliability_diagram,
)
from predictor.features import CATEGORICAL_FEATURE_COLUMNS, FEATURE_COLUMNS, FeatureSet, build_features
from predictor.diagnostics import decile_lift_table, plot_decile_lift, plot_feature_importance
from predictor.model import RiskModel
from predictor.split import make_payer_split, temporal_holdout_cycle

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "observable"
MODEL_DIR = REPO_ROOT / "data" / "model"

SPLIT_SEED = 42

# Chosen after the four-way comparison described in the module docstring.
# Kept modest deliberately: an over-tuned model on synthetic data is not an
# achievement, and this dataset (~5,000 training rows) cannot support a large
# tree budget without overfitting - the comparison confirmed that directly.
LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 7,
    "learning_rate": 0.05,
    "min_child_samples": 40,
    "verbosity": -1,
    "seed": SPLIT_SEED,
}
LGBM_MAX_ROUNDS = 400
LGBM_EARLY_STOPPING_ROUNDS = 30
# Fraction of the TRAIN payer set (never calibration or test) carved out
# purely to pick the early-stopping round. Keeping this split separate from
# calibration matters: using the calibration set for early stopping would
# tune the round count to it, undermining its use as an honest calibration
# target later.
INTERNAL_EARLY_STOP_FRACTION = 0.15


@dataclass(slots=True)
class Splits:
    """Row-level slices of one FeatureSet, keyed by the payer split."""

    train: FeatureSet
    calibration: FeatureSet
    test: FeatureSet


@dataclass(slots=True)
class ModelResult:
    name: str
    predict: Any  # Callable[[pd.DataFrame], np.ndarray]
    p_test: np.ndarray


def load_observable(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cycles = pd.read_parquet(data_dir / "cycles.parquet")
    mandates = pd.read_parquet(data_dir / "mandates.parquet")
    merchants = pd.read_parquet(data_dir / "merchants.parquet")
    return cycles, mandates, merchants


def make_splits(feature_set: FeatureSet, payer_ids: list[str], seed: int = SPLIT_SEED) -> Splits:
    """Slice one FeatureSet into train/calibration/test by payer.

    A single ``build_features`` call over the whole dataset is correct and
    sufficient - the temporal boundary is already enforced per-row inside
    ``build_features`` itself (each row only ever saw its own history), so
    slicing the result by payer afterward introduces no leakage. Building
    three separate feature sets would only triple the work for the same
    result.
    """
    split = make_payer_split(payer_ids, seed=seed)

    def subset(payer_set: frozenset[str]) -> FeatureSet:
        mask = feature_set.meta["payer_id"].isin(payer_set)
        return FeatureSet(
            meta=feature_set.meta.loc[mask].reset_index(drop=True),
            X=feature_set.X.loc[mask].reset_index(drop=True),
            y=feature_set.y.loc[mask].reset_index(drop=True),
        )

    return Splits(train=subset(split.train), calibration=subset(split.calibration), test=subset(split.test))


# ---------------------------------------------------------------------------
# Variant 1: trivial baseline
# ---------------------------------------------------------------------------


def train_trivial_baseline(train: FeatureSet) -> ModelResult:
    base_rate = float(train.y.mean())

    def predict(X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), base_rate, dtype=float)

    return ModelResult(name="trivial", predict=predict, p_test=np.array([]))


# ---------------------------------------------------------------------------
# Variant 2: single-feature baseline
# ---------------------------------------------------------------------------


def train_single_feature_baseline(train: FeatureSet) -> ModelResult:
    """Logistic regression on ``payer_fail_streak`` alone.

    Cold-start rows have no streak; filled with 0 here only, because
    ``LogisticRegression`` cannot route NaN the way a tree can. This is a
    deliberate simplification specific to this baseline - the full model
    leaves the same rows as NaN so LightGBM can route them by a learned split,
    which is exactly the comparison this baseline exists to set up.
    """
    x_train = train.X[["payer_fail_streak"]].fillna(0.0)
    clf = LogisticRegression()
    clf.fit(x_train, train.y)

    def predict(X: pd.DataFrame) -> np.ndarray:
        x = X[["payer_fail_streak"]].fillna(0.0)
        return clf.predict_proba(x)[:, 1]

    return ModelResult(name="single_feature", predict=predict, p_test=np.array([]))


# ---------------------------------------------------------------------------
# Variant 3: full model
# ---------------------------------------------------------------------------


def _internal_early_stop_split(train: FeatureSet, seed: int) -> tuple[FeatureSet, FeatureSet]:
    """Carve a fit/early-stop split out of TRAIN payers only.

    Never touches calibration or test payers - see the module docstring for
    why sharing the calibration set with early stopping would be a problem.
    """
    payer_ids = sorted(train.meta["payer_id"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(payer_ids)
    cut = int(round(len(shuffled) * (1 - INTERNAL_EARLY_STOP_FRACTION)))
    fit_payers = set(shuffled[:cut])
    es_payers = set(shuffled[cut:])

    def subset(payer_set: set[str]) -> FeatureSet:
        mask = train.meta["payer_id"].isin(payer_set)
        return FeatureSet(
            meta=train.meta.loc[mask].reset_index(drop=True),
            X=train.X.loc[mask].reset_index(drop=True),
            y=train.y.loc[mask].reset_index(drop=True),
        )

    return subset(fit_payers), subset(es_payers)


def train_full_model(train: FeatureSet, seed: int = SPLIT_SEED) -> tuple[ModelResult, lgb.Booster]:
    fit_set, es_set = _internal_early_stop_split(train, seed)

    dtrain = lgb.Dataset(
        fit_set.X, label=fit_set.y, categorical_feature=list(CATEGORICAL_FEATURE_COLUMNS), free_raw_data=False
    )
    dvalid = lgb.Dataset(
        es_set.X,
        label=es_set.y,
        reference=dtrain,
        categorical_feature=list(CATEGORICAL_FEATURE_COLUMNS),
        free_raw_data=False,
    )
    booster = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=LGBM_MAX_ROUNDS,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    def predict(X: pd.DataFrame) -> np.ndarray:
        return booster.predict(X[list(FEATURE_COLUMNS)], num_iteration=booster.best_iteration)

    return ModelResult(name="full_model", predict=predict, p_test=np.array([])), booster


# ---------------------------------------------------------------------------
# Diagnostics (reporting only - no threshold is selected here)
# ---------------------------------------------------------------------------


def evaluate(name: str, y_true: pd.Series, p: np.ndarray) -> dict[str, Any]:
    auc = roc_auc_score(y_true, p) if y_true.nunique() > 1 else float("nan")
    brier = brier_score_loss(y_true, p)

    precision, recall, thresholds = precision_recall_curve(y_true, p)
    # Report at a few illustrative cut points for diagnostics only - none of
    # these is "the" threshold. Phase 2 derives the operating threshold from
    # the expected-value formula, not from this table.
    report_points = [0.1, 0.2, 0.3, 0.5]
    pr_at: dict[float, dict[str, float]] = {}
    for cut in report_points:
        idx = np.searchsorted(thresholds, cut)
        idx = min(idx, len(precision) - 1)
        pr_at[cut] = {"precision": float(precision[idx]), "recall": float(recall[idx])}

    return {"name": name, "auc": float(auc), "brier": float(brier), "pr_at": pr_at, "n": len(y_true)}


def format_report(results: list[dict[str, Any]]) -> str:
    lines = [f"{'model':<20} {'AUC':>8} {'Brier':>8} {'n':>6}"]
    for r in results:
        lines.append(f"{r['name']:<20} {r['auc']:>8.4f} {r['brier']:>8.4f} {r['n']:>6}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Temporal holdout - a sanity check, not a headline result
# ---------------------------------------------------------------------------


def temporal_holdout_evaluation(feature_set: FeatureSet, n_cycles: int, seed: int = SPLIT_SEED) -> dict[str, Any]:
    """Train on early cycles, evaluate on later ones, across ALL payers.

    A different split axis from the payer split: the same payer can appear on
    both sides here, because this checks whether performance holds forward in
    time, not whether it generalises to unseen payers. With only six cycles
    this is explicitly a sanity check (CLAUDE.md's own framing) - report it as
    one, not as a second headline number.
    """
    cutoff = temporal_holdout_cycle(n_cycles)
    train_mask = feature_set.meta["cycle_number"] <= cutoff
    holdout_mask = ~train_mask

    train_X, train_y = feature_set.X.loc[train_mask], feature_set.y.loc[train_mask]
    holdout_X, holdout_y = feature_set.X.loc[holdout_mask], feature_set.y.loc[holdout_mask]

    dtrain = lgb.Dataset(train_X, label=train_y, categorical_feature=list(CATEGORICAL_FEATURE_COLUMNS))
    params = dict(LGBM_PARAMS)
    params["seed"] = seed
    booster = lgb.train(params, dtrain, num_boost_round=100)

    p_holdout = booster.predict(holdout_X)
    result = evaluate("temporal_holdout", holdout_y, p_holdout)
    result["cutoff_cycle"] = cutoff
    result["n_train"] = int(train_mask.sum())
    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_pipeline() -> dict[str, Any]:
    """Everything Phase 1 produces: three model variants, calibration,
    diagnostics, a saved RiskModel, and a reliability diagram. Returns a dict
    of every number ``docs/RESULTS.md`` reports, so the doc and the run that
    produced it can never silently drift apart.
    """
    cycles, mandates, merchants = load_observable()
    feature_set = build_features(cycles, mandates, merchants)
    payer_ids = mandates["payer_id"].unique().tolist()
    splits = make_splits(feature_set, payer_ids)

    trivial = train_trivial_baseline(splits.train)
    single_feature = train_single_feature_baseline(splits.train)
    full_result, booster = train_full_model(splits.train)

    baseline_results = []
    for result in (trivial, single_feature, full_result):
        p_test = result.predict(splits.test.X)
        baseline_results.append(evaluate(result.name, splits.test.y, p_test))

    # Calibration: fit on the calibration payer split, evaluate before/after
    # on the untouched test split.
    p_calib_raw = full_result.predict(splits.calibration.X)
    calibrator = fit_calibrator(p_calib_raw, splits.calibration.y)

    p_test_raw = full_result.predict(splits.test.X)
    p_test_calibrated = calibrator.predict(p_test_raw)
    calibration_report = evaluate_calibration(splits.test.y, p_test_raw, p_test_calibrated)
    figure_path = plot_reliability_diagram(splits.test.y, p_test_raw, p_test_calibrated)

    calibrated_eval = evaluate("full_model_calibrated", splits.test.y, p_test_calibrated)

    risk_model = RiskModel.fit_from(booster, calibrator, splits.train.X)
    risk_model.save(MODEL_DIR)
    importances = risk_model.feature_importance()
    importance_figure = plot_feature_importance(importances)

    lift_table = decile_lift_table(splits.test.y, p_test_raw, p_test_calibrated)
    lift_figure = plot_decile_lift(lift_table, overall_rate=float(splits.train.y.mean()))

    temporal = temporal_holdout_evaluation(feature_set, n_cycles=int(feature_set.meta["cycle_number"].max()))

    from predictor.ablation import run_cross_merchant_ablation  # deferred: avoids a training-time import cycle

    ablation = run_cross_merchant_ablation(seed=SPLIT_SEED)

    return {
        "baseline_results": baseline_results,
        "calibration_report": calibration_report,
        "calibrated_eval": calibrated_eval,
        "reliability_figure": figure_path,
        "feature_importance": importances,
        "importance_figure": importance_figure,
        "lift_table": lift_table,
        "lift_figure": lift_figure,
        "temporal_holdout": temporal,
        "ablation": ablation,
        "model_version": risk_model.version,
        "model_dir": MODEL_DIR,
        "split_seed": SPLIT_SEED,
        "n_train": len(splits.train.X),
        "n_calibration": len(splits.calibration.X),
        "n_test": len(splits.test.X),
    }


def main() -> None:
    outcome = run_pipeline()

    print(format_report(outcome["baseline_results"]))
    print(format_report([outcome["calibrated_eval"]]))
    print()
    cr = outcome["calibration_report"]
    print(f"calibration (test set, n={cr.n}):")
    print(f"  AUC        raw={cr.auc_raw:.4f}  calibrated={cr.auc_calibrated:.4f}")
    print(f"  Brier      raw={cr.brier_raw:.4f}  calibrated={cr.brier_calibrated:.4f}")
    print(f"  reliability diagram -> {outcome['reliability_figure']}")
    print()
    print("feature importance (gain), top 10:")
    print(outcome["feature_importance"].head(10).to_string())
    print()
    t = outcome["temporal_holdout"]
    print(
        f"temporal holdout: train cycles<={t['cutoff_cycle']} (n={t['n_train']}), "
        f"holdout cycles>{t['cutoff_cycle']} (n={t['n']}) "
        f"-> AUC={t['auc']:.4f} Brier={t['brier']:.4f}"
    )
    print()
    print(f"decile lift table (test set):\n{outcome['lift_table'].to_string()}")
    print(f"  figure -> {outcome['lift_figure']}")
    print()
    print("cross-merchant ablation:")
    from predictor.ablation import format_report as format_ablation_report

    print(format_ablation_report(outcome["ablation"]))
    print()
    print(f"model version: {outcome['model_version']}  saved to {outcome['model_dir']}")
    print(f"split sizes: train={outcome['n_train']} calibration={outcome['n_calibration']} test={outcome['n_test']}")


if __name__ == "__main__":
    main()
