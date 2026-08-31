"""The aggregator argument, measured rather than asserted.

CLAUDE.md's central claim (section 3) is that this prediction can only be
built at the aggregator layer - a single merchant, seeing only its own
mandate's history with a payer, cannot build the same model. Section 10 names
``dom_fail_propensity`` and ``concurrent_debits_same_day`` as the two features
that carry this claim. Ranking #1 in feature importance (see
``docs/RESULTS.md``) is suggestive, not proof - it says the feature is useful,
not that the *scope* it requires is what makes it useful.

This module measures it directly: train the identical model, on the identical
split, on the identical feature *names*, twice - once with
``predictor.features.build_features(..., scope="cross_merchant")`` (the real
pipeline, pooling a payer's history across every merchant they hold) and once
with ``scope="merchant_only"`` (the same code, re-scoped to what a single
merchant's own book could see). Nothing else changes: same LightGBM
hyperparameters, same payer split (same seed, so train/calibration/test
contain the exact same payers in both runs), same target definition, same
calibration procedure. The only difference between the two numbers this
module reports is how much of the network a party is allowed to see.

**The result was not what the pitch expected, and this module reports that
plainly rather than around it.** The point-estimate AUC gap on the test set
slightly favours *merchant-only* (0.7558 vs 0.7484 cross-merchant) - the
wrong direction for the thesis. A paired bootstrap over the test set (2,000
resamples) puts a 95% CI on the gap at roughly [-0.038, +0.024]: it crosses
zero, and about a third of resampled draws favour the aggregator direction.
A subgroup check built specifically to favour cross-merchant scope - rows
where the *mandate* is brand new (no history of its own) but the *payer*
already has history elsewhere - shows no advantage there either (0.557 vs
0.568, n=216, itself a small and therefore noisy slice). See
``docs/RESULTS.md`` for the full writeup, the likely mechanism (a shrinkage
choice in ``dom_fail_propensity`` that may dilute a mature mandate's own
strong day-of-month autocorrelation with a broader cross-merchant average
rather than adding to it), and what this does and does not mean for the
aggregator argument as a whole - which rests on data *visibility*
(CLAUDE.md section 3), not solely on this one measured gap.

Run as ``python -m predictor.ablation``. Also invoked from
``predictor/train.py::run_pipeline`` so ``make train`` produces the
comparison, including the bootstrap CI, as part of the normal Phase 1 run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from predictor.calibrate import evaluate_calibration, fit_calibrator
from predictor.features import build_features
from predictor.train import load_observable, make_splits, train_full_model

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "docs" / "figures"

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0


@dataclass(slots=True)
class ScopeResult:
    scope: str
    label: str
    auc_raw: float
    brier_raw: float
    auc_calibrated: float
    brier_calibrated: float
    n_test: int


def run_cross_merchant_ablation(seed: int = 42) -> dict[str, Any]:
    cycles, mandates, merchants = load_observable()
    payer_ids = mandates["payer_id"].unique().tolist()

    results: dict[str, ScopeResult] = {}
    test_targets: dict[str, Any] = {}
    test_predictions: dict[str, np.ndarray] = {}
    test_X: dict[str, pd.DataFrame] = {}

    for scope, label in (
        ("merchant_only", "Merchant-only history"),
        ("cross_merchant", "Cross-merchant history (aggregator)"),
    ):
        feature_set = build_features(cycles, mandates, merchants, scope=scope)
        splits = make_splits(feature_set, payer_ids, seed=seed)

        result, _booster = train_full_model(splits.train, seed=seed)

        p_calib_raw = result.predict(splits.calibration.X)
        calibrator = fit_calibrator(p_calib_raw, splits.calibration.y)

        p_test_raw = result.predict(splits.test.X)
        p_test_calibrated = calibrator.predict(p_test_raw)

        report = evaluate_calibration(splits.test.y, p_test_raw, p_test_calibrated)
        results[scope] = ScopeResult(
            scope=scope,
            label=label,
            auc_raw=report.auc_raw,
            brier_raw=report.brier_raw,
            auc_calibrated=report.auc_calibrated,
            brier_calibrated=report.brier_calibrated,
            n_test=report.n,
        )
        test_targets[scope] = splits.test.y
        test_predictions[scope] = p_test_calibrated
        test_X[scope] = splits.test.X

    # Self-check: the two runs must differ only in feature scope. If the
    # target vectors themselves differ, the split seeded the same way
    # produced different payer sets, which would invalidate the comparison -
    # this is not a leakage risk, it is a "did I actually hold everything
    # else constant" check on the ablation itself.
    assert test_targets["merchant_only"].equals(test_targets["cross_merchant"]), (
        "test-set targets differ between ablation scopes - the two runs are "
        "not comparable; check that both used the same seed and payer_ids"
    )
    y_test = test_targets["cross_merchant"].to_numpy()

    gap_auc = results["cross_merchant"].auc_calibrated - results["merchant_only"].auc_calibrated
    gap_brier = results["merchant_only"].brier_calibrated - results["cross_merchant"].brier_calibrated

    ci = bootstrap_auc_gap_ci(y_test, test_predictions["cross_merchant"], test_predictions["merchant_only"])
    subgroup = new_mandate_existing_payer_subgroup(
        y_test, test_predictions["cross_merchant"], test_predictions["merchant_only"], test_X
    )

    figure_path = _plot(results)

    return {
        "merchant_only": results["merchant_only"],
        "cross_merchant": results["cross_merchant"],
        "gap_auc": gap_auc,
        "gap_brier": gap_brier,
        "bootstrap_ci": ci,
        "subgroup": subgroup,
        "figure": figure_path,
    }


def bootstrap_auc_gap_ci(
    y_test: np.ndarray,
    p_cross_merchant: np.ndarray,
    p_merchant_only: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Paired bootstrap CI on the AUC gap between the two scopes.

    Paired, not independent: every resample draws the *same* row indices for
    both scopes' predictions, so the comparison isolates the effect of scope
    rather than adding independent sampling noise from two separate
    resamplings. A point-estimate gap with no interval is not evidence either
    way - this is what turns "0.7484 vs 0.7558" into an answer to "is that
    difference real or could it be this seed's luck."
    """
    rng = np.random.default_rng(seed)
    n = len(y_test)
    gaps = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if y_test[idx].sum() in (0, n):
            continue  # a resample with no positives (or all positives) has no defined AUC
        auc_cm = roc_auc_score(y_test[idx], p_cross_merchant[idx])
        auc_mo = roc_auc_score(y_test[idx], p_merchant_only[idx])
        gaps.append(auc_cm - auc_mo)
    gaps_arr = np.array(gaps)
    return {
        "mean_gap": float(gaps_arr.mean()),
        "ci_low": float(np.percentile(gaps_arr, 2.5)),
        "ci_high": float(np.percentile(gaps_arr, 97.5)),
        "share_favoring_aggregator": float((gaps_arr > 0).mean()),
        "n_resamples": len(gaps_arr),
    }


def new_mandate_existing_payer_subgroup(
    y_test: np.ndarray,
    p_cross_merchant: np.ndarray,
    p_merchant_only: np.ndarray,
    test_X: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """The subgroup built specifically to favour cross-merchant scope.

    Rows where the *mandate itself* is cold-start under merchant-only scope
    (no history of its own yet) but the *payer* is not cold-start under
    cross-merchant scope (they have history via other merchants). This is
    exactly the situation the aggregator argument claims an advantage in - a
    new mandate for an existing, known payer - so if the advantage exists
    anywhere in this dataset, it should be clearest here.
    """
    is_new_mandate = test_X["merchant_only"]["is_cold_start"].to_numpy() == 1
    payer_has_history = test_X["cross_merchant"]["is_cold_start"].to_numpy() == 0
    mask = is_new_mandate & payer_has_history

    n = int(mask.sum())
    if n < 20 or y_test[mask].sum() in (0, n):
        return {"n": n, "auc_cross_merchant": float("nan"), "auc_merchant_only": float("nan")}

    return {
        "n": n,
        "positive_rate": float(y_test[mask].mean()),
        "auc_cross_merchant": float(roc_auc_score(y_test[mask], p_cross_merchant[mask])),
        "auc_merchant_only": float(roc_auc_score(y_test[mask], p_merchant_only[mask])),
    }


def format_report(outcome: dict[str, Any]) -> str:
    mo, cm = outcome["merchant_only"], outcome["cross_merchant"]
    ci = outcome["bootstrap_ci"]
    sub = outcome["subgroup"]
    lines = [
        f"{'model':<38} {'AUC':>8} {'Brier':>8} {'n':>6}",
        f"{mo.label:<38} {mo.auc_calibrated:>8.4f} {mo.brier_calibrated:>8.4f} {mo.n_test:>6}",
        f"{cm.label:<38} {cm.auc_calibrated:>8.4f} {cm.brier_calibrated:>8.4f} {cm.n_test:>6}",
        "",
        f"point-estimate gap (cross_merchant - merchant_only): AUC {outcome['gap_auc']:+.4f}, "
        f"Brier {outcome['gap_brier']:+.4f} (positive Brier = aggregator better)",
        f"bootstrap 95% CI on AUC gap ({ci['n_resamples']} resamples): "
        f"[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]  "
        f"(share favouring aggregator: {ci['share_favoring_aggregator']:.2f})",
        f"  -> {'crosses zero: NOT statistically distinguishable at this sample size' if ci['ci_low'] < 0 < ci['ci_high'] else 'does not cross zero'}",
        "",
        f"subgroup - new mandate, payer already has cross-merchant history (n={sub['n']}):",
        f"  cross_merchant AUC={sub['auc_cross_merchant']:.4f}  merchant_only AUC={sub['auc_merchant_only']:.4f}",
    ]
    return "\n".join(lines)


def _plot(results: dict[str, ScopeResult]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "cross_merchant_ablation.png"

    labels = [results["merchant_only"].label, results["cross_merchant"].label]
    auc = [results["merchant_only"].auc_calibrated, results["cross_merchant"].auc_calibrated]
    brier = [results["merchant_only"].brier_calibrated, results["cross_merchant"].brier_calibrated]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = ["#c0504d", "#3b6ea5"]

    ax1.bar(labels, auc, color=colors)
    ax1.set_ylabel("AUC (calibrated, test set)")
    ax1.set_ylim(0.5, max(auc) * 1.1)
    ax1.set_title("Higher is better")
    for i, v in enumerate(auc):
        ax1.text(i, v + 0.005, f"{v:.3f}", ha="center")

    ax2.bar(labels, brier, color=colors)
    ax2.set_ylabel("Brier score (calibrated, test set)")
    ax2.set_ylim(0, max(brier) * 1.3)
    ax2.set_title("Lower is better")
    for i, v in enumerate(brier):
        ax2.text(i, v + 0.002, f"{v:.4f}", ha="center")

    fig.suptitle("Same model, same features, same split - only visible history scope changes")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelrotation=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    outcome = run_cross_merchant_ablation()
    print(format_report(outcome))
    print(f"\nfigure -> {outcome['figure']}")


if __name__ == "__main__":
    main()
