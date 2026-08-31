"""Validation-only diagnostic: does the day-of-month failure signal exist?

This script exists **outside** ``predictor/`` deliberately, and reads
``data/ground_truth/payers.parquet`` deliberately. It answers a question the
production pipeline is not allowed to ask - "what is this payer's true
payday?" - because answering it is exactly how you check, from the outside,
whether the signal the predictor is being asked to recover (via
``dom_fail_propensity``, using only observable history) actually exists in
the world.

``tests/test_leakage.py`` asserts nothing under ``predictor/`` imports hidden
simulator modules or mentions a ``ground_truth`` path. This script is not
under ``predictor/`` and is never imported by it - running it and reading its
output is a one-time validation step for ``docs/RESULTS.md``, not part of the
prediction pipeline. Run it again any time the world is regenerated to keep
the figure honest.

Usage::

    python scripts/day_of_month_diagnostic.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "docs" / "figures"


def main() -> None:
    cycles = pd.read_parquet(REPO_ROOT / "data/observable/cycles.parquet")
    payers = pd.read_parquet(REPO_ROOT / "data/ground_truth/payers.parquet")

    cycles = cycles.copy()
    cycles["dom"] = pd.to_datetime(cycles["scheduled_date"]).dt.day
    cycles["is_failure"] = cycles["outcome"] == "failure"

    raw = (
        cycles.groupby("dom")
        .agg(failure_rate=("is_failure", "mean"), n=("is_failure", "size"))
        .query("n >= 30")
    )

    income_day = payers.set_index("payer_id")["income_day"].to_dict()
    cycles["income_day"] = cycles["payer_id"].map(income_day)
    cycles["days_until_payday"] = (cycles["income_day"] - cycles["dom"]) % 28
    cycles["is_insufficient_funds"] = cycles["decline_code"] == "insufficient_funds"

    by_payday_gap = (
        cycles.groupby("days_until_payday")
        .agg(if_rate=("is_insufficient_funds", "mean"), n=("is_insufficient_funds", "size"))
        .query("n >= 30")
    )

    _plot(raw, by_payday_gap)

    print("Raw day-of-month failure rate (n>=30):")
    print(raw)
    print()
    print("insufficient_funds rate by true days-until-payday (ground truth, n>=30):")
    print(by_payday_gap)


def _plot(raw: pd.DataFrame, by_payday_gap: pd.DataFrame) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "day_of_month_curve.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.bar(raw.index, raw["failure_rate"], color="#c0504d")
    ax1.set_xlabel("scheduled day of month")
    ax1.set_ylabel("failure rate (all causes)")
    ax1.set_title("Pooled across payers - noisy\n(different payers, different true paydays)")

    ax2.plot(by_payday_gap.index, by_payday_gap["if_rate"], marker="o", color="#3b6ea5")
    ax2.set_xlabel("days until this payer's true payday (ground truth)")
    ax2.set_ylabel("insufficient_funds rate")
    ax2.set_title("Re-cut by true payday - clean decay\n(never seen by the predictor)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    main()
