"""Plotting helpers for docs/RESULTS.md that need nothing beyond a trained
RiskModel - no ground truth, no raw simulator state. Kept separate from
``calibrate.py`` (which owns the reliability diagram specifically) because
this file's only job is turning a fitted model's own summary statistics into
a picture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "docs" / "figures"


def decile_lift_table(
    y_true: pd.Series, p_rank: np.ndarray, p_report: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Rank rows into risk deciles and report actual failure rate per decile.

    Phase 2's EV formula multiplies ``p_fail`` by a rupee amount; whether
    intervention ever clears its cost depends entirely on how concentrated
    risk is in the top deciles, not on the overall base rate. This table is
    what lets Phase 2's cost parameters (`config/policy.yaml`) get set
    against a real number instead of a guess.

    ``p_rank`` (typically the *raw*, pre-calibration score) decides which
    decile a row falls into - raw scores have more distinct values than the
    calibrated output (isotonic regression is a step function with limited
    unique levels), so ranking by raw score gives more evenly sized bins.
    ``p_report`` (typically the *calibrated* probability) is what gets
    averaged and shown per bin - the number that is supposed to mean what it
    says. Decile 1 is the highest-risk group, matching how "top decile" is
    read in the eventual policy discussion.
    """
    frame = pd.DataFrame({"rank_score": p_rank, "reported_p": p_report, "y": y_true.to_numpy()})
    frame["decile"] = pd.qcut(
        frame["rank_score"], q=min(n_bins, frame["rank_score"].nunique()), duplicates="drop"
    )
    # Highest risk = decile 1. qcut's bins sort ascending by score, so reverse
    # the bin-rank order rather than the score itself.
    bin_rank = frame["decile"].cat.codes
    frame["decile_label"] = (bin_rank.max() - bin_rank) + 1

    overall_rate = frame["y"].mean()
    table = (
        frame.groupby("decile_label")
        .agg(
            n=("y", "size"),
            mean_predicted=("reported_p", "mean"),
            observed_rate=("y", "mean"),
            score_min=("rank_score", "min"),
            score_max=("rank_score", "max"),
        )
        .sort_index()
    )
    table["lift"] = table["observed_rate"] / overall_rate
    table.index.name = "decile (1=highest risk)"
    return table


def plot_decile_lift(
    table: pd.DataFrame, overall_rate: float, out_path: Path = FIGURES_DIR / "decile_lift.png"
) -> Path:
    """Bar chart of observed failure rate per risk decile, with the
    population base rate marked - the visual companion to
    ``decile_lift_table``, decile 1 (highest risk) on the left."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(table.index.astype(str), table["observed_rate"], color="#c0504d")
    ax.axhline(overall_rate, color="grey", linestyle="--", label=f"base rate ({overall_rate:.1%})")
    ax.set_xlabel("risk decile (1 = highest predicted risk)")
    ax.set_ylabel("observed failure rate")
    ax.set_title("Decile lift - test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_feature_importance(
    importances: pd.Series, out_path: Path = FIGURES_DIR / "feature_importance.png"
) -> Path:
    """Horizontal bar chart of whole-model feature importance (gain), most
    important at the top. Whichever features rank highest here are the
    direct evidence for (or against) the aggregator argument - see
    ``docs/RESULTS.md`` for the reading."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ordered = importances.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(ordered.index, ordered.values, color="#3b6ea5")
    ax.set_xlabel("importance (total gain)")
    ax.set_title("Feature importance - full model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
