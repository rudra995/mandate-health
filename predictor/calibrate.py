"""Probability calibration - mandatory, not optional.

Phase 2's policy engine multiplies ``p_fail`` by a rupee amount to compute
expected value. A gradient-boosted classifier's raw output is a *score* that
ranks well (that is what AUC measures) but is not guaranteed to mean what it
says as a probability - if the model says 0.30 for a group of mandates, the
true failure rate in that group is not guaranteed to be 30%. Every downstream
EV number is wrong by exactly however wrong that assumption is, so this step
is where the gap gets closed before anything acts on the number.

**Method: isotonic regression**, fit on the calibration payer split - never
the training payers (would calibrate the model to its own overfit) and never
the test payers (would spend the one set reserved for an honest final
number). Isotonic was chosen over Platt scaling (a single sigmoid) because it
makes no assumption about the *shape* of the miscalibration curve. Platt
scaling can only correct a sigmoid-shaped bias; if a gradient-boosted model on
this dataset is miscalibrated in some other shape, Platt scaling could not
fully fix it. The tradeoff is that isotonic regression needs more calibration
data to avoid overfitting the calibration map itself - it is a step function
fit to the data, so a very small calibration set can memorise pointwise noise
instead of learning a real trend. At roughly 80 payers / 1,600+ rows held out
for calibration, there is enough data for this to be a reasonable choice; it
would not be at a materially smaller n, and this is a real constraint anyone
scaling the payer count down should know about.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "docs" / "figures"


def fit_calibrator(raw_probabilities: np.ndarray, y_true: pd.Series) -> IsotonicRegression:
    """Fit isotonic regression mapping raw model score -> calibrated probability.

    ``out_of_bounds="clip"`` so a raw score outside the range seen during
    calibration fitting still returns a valid probability rather than
    extrapolating the step function past its support.
    """
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probabilities, y_true.to_numpy())
    return calibrator


def apply_calibrator(calibrator: IsotonicRegression, raw_probabilities: np.ndarray) -> np.ndarray:
    return calibrator.predict(raw_probabilities)


@dataclass(slots=True)
class CalibrationReport:
    auc_raw: float
    auc_calibrated: float
    brier_raw: float
    brier_calibrated: float
    n: int


def evaluate_calibration(y_true: pd.Series, p_raw: np.ndarray, p_calibrated: np.ndarray) -> CalibrationReport:
    """Brier score and AUC before/after, measured on held-out data (the test
    set in normal use - never the calibration set the map itself was fit on,
    which would just report how well isotonic regression fit its own input).
    """
    return CalibrationReport(
        auc_raw=float(roc_auc_score(y_true, p_raw)) if y_true.nunique() > 1 else float("nan"),
        auc_calibrated=float(roc_auc_score(y_true, p_calibrated)) if y_true.nunique() > 1 else float("nan"),
        brier_raw=float(brier_score_loss(y_true, p_raw)),
        brier_calibrated=float(brier_score_loss(y_true, p_calibrated)),
        n=len(y_true),
    )


def reliability_bins(y_true: pd.Series, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile-bin predicted probabilities and compare to observed frequency.

    Bins are by *rank* (``pd.qcut``), not by fixed-width probability range -
    with most predictions clustered under 0.3 (base rate ~12%), fixed-width
    bins above 0.5 would be nearly empty and produce meaningless, noisy rows.
    Quantile bins keep every bin populated, at the cost of bins not lining up
    to round probability numbers - the right tradeoff for a diagnostic table.
    """
    frame = pd.DataFrame({"p": p, "y": y_true.to_numpy()})
    frame["bin"] = pd.qcut(frame["p"], q=min(n_bins, frame["p"].nunique()), duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        predicted_mean=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")
    )
    grouped["abs_gap"] = (grouped["predicted_mean"] - grouped["observed_rate"]).abs()
    return grouped.reset_index(drop=True)


def expected_calibration_error(y_true: pd.Series, p: np.ndarray, n_bins: int = 10) -> float:
    """Weighted-average |predicted - observed| across quantile bins.

    Used as a single-number pass/fail gate in ``tests/test_calibration.py``
    rather than asserting every individual bin, because with ~1,600 test rows
    across 10 bins (~160 rows/bin) a single bin can swing several points from
    sampling noise alone even for a well-calibrated model. ECE averages that
    noise out and is the standard summary metric for exactly this reason.
    """
    bins = reliability_bins(y_true, p, n_bins=n_bins)
    return float(np.average(bins["abs_gap"], weights=bins["n"]))


def plot_reliability_diagram(
    y_true: pd.Series,
    p_raw: np.ndarray,
    p_calibrated: np.ndarray,
    out_path: Path = FIGURES_DIR / "reliability_diagram.png",
) -> Path:
    """Save a before/after reliability diagram. Import matplotlib lazily so
    nothing outside this one function pays for the import at module load."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_bins = reliability_bins(y_true, p_raw)
    cal_bins = reliability_bins(y_true, p_calibrated)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    ax.plot(raw_bins["predicted_mean"], raw_bins["observed_rate"], marker="o", label="raw model")
    ax.plot(cal_bins["predicted_mean"], cal_bins["observed_rate"], marker="s", label="isotonic-calibrated")
    ax.set_xlabel("mean predicted probability (decile bin)")
    ax.set_ylabel("observed failure rate")
    ax.set_title("Reliability diagram - test set")
    ax.legend()
    ax.set_xlim(0, max(0.5, raw_bins["predicted_mean"].max() * 1.1))
    ax.set_ylim(0, max(0.5, raw_bins["observed_rate"].max() * 1.1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
