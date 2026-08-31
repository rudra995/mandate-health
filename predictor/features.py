"""Feature engineering, computable from ``data/observable/`` alone.

Every feature here follows CLAUDE.md section 10. Two of them carry the whole
aggregator argument and are marked accordingly at the point they are computed:
``dom_fail_propensity`` and ``concurrent_debits_same_day``. Both are built from
a payer's history *across all of that payer's merchants*, not per merchant -
that is the one fact a single merchant could never reproduce on its own.

This module must never import ``simulator.balance_model`` or
``simulator.pdn_model``, and must never read anything under
``data/ground_truth/``. Importing ``simulator.entities`` and
``simulator.config`` is fine - they carry no hidden state, only schema and the
decline taxonomy.

**Temporal boundary, enforced by construction, not by convention.** Every
per-payer and per-PSP running statistic is updated in two passes per calendar
date: first every row scheduled on that date reads the state *as it stood the
day before*, then - only after every row for that date has read it - the
state is updated with that date's outcomes as a single batch. Two mandates for
the same payer presenting on the same date therefore never see each other's
outcome, because in the real world they are presented together and neither
outcome is known before the other. Getting this ordering wrong is the single
easiest way to leak the future into a "past" feature, so it is structural
here rather than left to a sort-and-hope. ``tests/test_leakage.py`` checks it
directly: corrupt the outcome of cycle T and every cycle after it, rebuild,
and every feature for cycle <= T must be byte-identical.

**Cold start.** A payer's very first presentation - by calendar date, across
all of that payer's mandates - has no history. Every payer-derived feature is
left as ``NaN`` for that row, plus an explicit ``is_cold_start`` flag, rather
than silently filled with zero (which would read as "this payer has a perfect
record") or with a population constant (which would hide that nothing is
known yet). LightGBM and ``HistGradientBoostingClassifier`` both route missing
values through a learned split direction per tree node, so the model decides
how to treat "unknown" rather than being told a single substitute value.
``psp_health_index`` gets the same treatment for the same reason: very early
in the simulation window, a PSP handle may have no prior presentations at all
across *any* payer.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd

from simulator.config import load_decline_codes
from simulator.entities import DeclineCode

# ---------------------------------------------------------------------------
# Feature list - author-owned. Extend deliberately, not silently.
# ---------------------------------------------------------------------------

#: Exactly the fourteen features named in CLAUDE.md section 10, in that order.
NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "payer_fail_streak",
    "payer_fail_rate_3c",
    "dom_fail_propensity",
    "dom_success_gap",
    "decline_mix_balance",
    "decline_mix_technical",
    "amount_vs_payer_max_success",
    "amount_pct_of_cap",
    "mandate_age_cycles",
    "cycles_to_validity_end",
    "psp_health_index",
    "concurrent_debits_same_day",
)

#: Categorical features, left as pandas ``category`` dtype for LightGBM's
#: native categorical split handling rather than one-hot encoded.
CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "slot_planned",
    "merchant_category",
)

#: One addition beyond the section 10 list: a cold-start flag. Declared here
#: rather than folded silently into the feature list above, per the working
#: rule that the feature set is author-owned - propose additions, do not
#: rewrite the list quietly. See the module docstring for why it exists.
AUXILIARY_FEATURE_COLUMNS: tuple[str, ...] = ("is_cold_start",)

FEATURE_COLUMNS: tuple[str, ...] = (
    NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS + AUXILIARY_FEATURE_COLUMNS
)

# Decline-mix buckets. insufficient_funds is "balance"; every other
# recoverable code is "technical". Terminal codes count toward neither bucket
# (mirrors CLAUDE.md section 10: only two decline_mix_* features are named).
_TAXONOMY = load_decline_codes()
_TECHNICAL_CODES = frozenset(
    code.value for code in (_TAXONOMY.recoverable - {DeclineCode.INSUFFICIENT_FUNDS})
)

# Bayesian shrinkage strength for dom_fail_propensity: how many "virtual"
# observations the payer's own overall rate is worth against the windowed
# day-of-month count. A judgement call, not a spec requirement - flagged here
# rather than buried as a bare literal.
_DOM_WINDOW_DAYS = 3
_DOM_SHRINKAGE_ALPHA = 3.0

# Trailing-window size for the PSP rolling success rate. Short enough that a
# multi-day degradation event (config: psp_degradation.duration_days, 1-4
# days) visibly moves the index; long enough not to be single-debit noise.
_PSP_WINDOW = 30

_MONTH_DAYS = 28  # mandate debit days and income days are drawn on 1-28 only

#: "cross_merchant" (default) is the production scope: every payer-derived
#: feature pools that payer's history across every merchant they hold - this
#: is the aggregator's actual vantage point.
#: "merchant_only" is an ablation, not a production mode: it re-scopes the
#: same features as if a single merchant had built this alone, seeing only
#: its own mandate's history with that payer. Used by ``predictor/ablation.py``
#: to measure the aggregator argument directly rather than assert it - see
#: that module's docstring.
HistoryScope = Literal["cross_merchant", "merchant_only"]


class FeatureSet(NamedTuple):
    """Output of :func:`build_features`.

    Returned as a triple rather than the two-tuple CLAUDE.md's Phase 1 prompt
    sketches, because every downstream consumer (split, leakage tests, audit
    joins) needs ``payer_id`` / ``cycle_id`` alongside the numeric matrix, and
    smuggling identifiers into ``X`` would make "assert zero hidden columns"
    ambiguous about which columns are even candidates. Flagged as a deviation
    rather than made silently.
    """

    meta: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series


def build_features(
    cycles: pd.DataFrame,
    mandates: pd.DataFrame,
    merchants: pd.DataFrame,
    as_of_cycle: int | None = None,
    scope: HistoryScope = "cross_merchant",
) -> FeatureSet:
    """Build the feature matrix and target vector from observable data only.

    ``cycles`` must be ``data/observable/cycles.parquet`` (or a subset of its
    rows/columns) - the *original presentation* per mandate cycle. Retries
    live in a separate artifact and are never mixed in here, so the target is
    always "did the original presentation fail", exactly as specified.

    ``as_of_cycle``, if given, restricts the *returned* rows to
    ``cycle_number <= as_of_cycle`` after the full computation. It never
    changes what any individual row's features were allowed to see - that is
    always "this row's own history, strictly before its own scheduled date"
    - it only trims which rows come back. Used for the temporal-holdout
    sanity check and by the leakage test's future-corruption probe.

    ``scope`` controls how much history each row's features are allowed to
    pool. ``"cross_merchant"`` (default, production) pools a payer's history
    across every merchant they hold - the aggregator's real vantage point.
    ``"merchant_only"`` re-scopes the *same* features, computed by the *same*
    code, to what a single merchant could see of its own book alone - see
    ``predictor/ablation.py`` for why this exists and what it measures.
    """
    if scope not in ("cross_merchant", "merchant_only"):
        raise ValueError(f"unknown scope: {scope!r}")

    frame = _merge_static_fields(cycles, mandates)
    frame = frame.sort_values(["scheduled_date", "payer_id", "mandate_id"], kind="stable")
    frame = frame.reset_index(drop=True)

    if scope == "cross_merchant":
        # concurrent_debits_same_day uses only the debit CALENDAR (payer_id +
        # scheduled_date), never an outcome column, so it is safe to compute
        # across the whole frame regardless of time order - the schedule is
        # fixed when the mandate is created, unlike outcomes, which resolve
        # later.
        frame["concurrent_debits_same_day"] = (
            frame.groupby(["payer_id", "scheduled_date"])["mandate_id"].transform("count") - 1
        )
    else:
        # A single merchant has exactly one mandate per payer in this world,
        # so it structurally cannot observe another merchant's debit landing
        # the same day - there is no "concurrent" debit in its own book to
        # count. Forced to 0 rather than approximated, because the honest
        # answer for a lone merchant is "this signal does not exist for me",
        # not a noisier estimate of it.
        frame["concurrent_debits_same_day"] = 0.0

    payer_states: dict[Any, _PayerState] = defaultdict(_PayerState)
    psp_states: dict[Any, deque[int]] = defaultdict(lambda: deque(maxlen=_PSP_WINDOW))
    mandate_prior_count: dict[str, int] = defaultdict(int)

    def history_key(record: Any) -> Any:
        # cross_merchant: keyed by payer alone, pooling every merchant they
        # hold. merchant_only: keyed by (payer, merchant), so a mandate's
        # rolling state only ever sees that one merchant's own presentations
        # with this payer - exactly what that merchant alone could observe.
        return record.payer_id if scope == "cross_merchant" else (record.payer_id, record.merchant_id)

    def psp_key(record: Any) -> Any:
        # Mirrors history_key's reasoning for PSP health: cross_merchant
        # pools every merchant's transactions on a handle (the aggregator's
        # network-wide telemetry); merchant_only restricts the same rolling
        # window to transactions this one merchant itself presented, across
        # its own customer base - still a "recent PSP success rate" signal a
        # lone merchant with enough volume could plausibly build, just from a
        # narrower slice of the network.
        return record.psp_handle if scope == "cross_merchant" else (record.merchant_id, record.psp_handle)

    rows: list[dict[str, Any]] = []
    for _scheduled_date, day_frame in frame.groupby("scheduled_date", sort=True):
        # Pass 1: every row presented on this date reads state as it stood
        # strictly before this date. No row on this date can see any other
        # row on this date - they are simultaneous, not ordered.
        for record in day_frame.itertuples(index=False):
            ps = payer_states[history_key(record)]
            psp_history = psp_states[psp_key(record)]
            rows.append(
                _compute_row_features(
                    ps=ps,
                    psp_history=psp_history,
                    mandate_prior_cycles=mandate_prior_count[record.mandate_id],
                    day_of_month=record.scheduled_date.day,
                    amount=record.amount,
                    max_cap=record.max_cap,
                    validity_end=record.validity_end,
                    scheduled_date=record.scheduled_date,
                    slot=record.execution_slot,
                    merchant_category=record.merchant_category,
                    concurrent_debits_same_day=record.concurrent_debits_same_day,
                )
            )

        # Pass 2: now apply this date's outcomes, as one batch, so the next
        # date's Pass 1 sees them but this date's rows never did.
        for record in day_frame.itertuples(index=False):
            failed = record.outcome == "failure"
            payer_states[history_key(record)].update(
                failed=failed,
                decline_code=record.decline_code,
                day_of_month=record.scheduled_date.day,
                amount=record.amount,
            )
            psp_states[psp_key(record)].append(1 if failed else 0)
            mandate_prior_count[record.mandate_id] += 1

    X = pd.DataFrame(rows)
    for col in CATEGORICAL_FEATURE_COLUMNS:
        X[col] = X[col].astype("category")
    X = X[list(FEATURE_COLUMNS)]

    meta = frame[["cycle_id", "mandate_id", "payer_id", "cycle_number", "scheduled_date"]].reset_index(
        drop=True
    )
    y = (frame["outcome"] == "failure").astype(int).rename("failed").reset_index(drop=True)

    if as_of_cycle is not None:
        keep = meta["cycle_number"] <= as_of_cycle
        meta = meta.loc[keep].reset_index(drop=True)
        X = X.loc[keep].reset_index(drop=True)
        y = y.loc[keep].reset_index(drop=True)

    return FeatureSet(meta=meta, X=X, y=y)


def _merge_static_fields(cycles: pd.DataFrame, mandates: pd.DataFrame) -> pd.DataFrame:
    """Join in the mandate-level fields every row needs, nothing else.

    Only columns that are static per mandate (never derived from any
    outcome) are pulled in: amount, cap, category, PSP handle, validity end.
    ``merchant_id`` is included alongside ``merchant_category`` because the
    merchant-only ablation scope needs the actual merchant identity to key
    per-merchant history - two mandates can share a category (e.g. two
    "streaming" merchants) while being visible to entirely different
    merchants.
    """
    static_cols = [
        "mandate_id",
        "merchant_id",
        "merchant_category",
        "amount",
        "max_cap",
        "validity_end",
        "psp_handle",
    ]
    merged = cycles.merge(mandates[static_cols], on="mandate_id", how="left", validate="many_to_one")
    if merged["merchant_category"].isna().any():
        raise ValueError("a cycle references a mandate_id missing from `mandates`")
    return merged


# ---------------------------------------------------------------------------
# Per-payer running state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PayerState:
    """Everything needed to featurise this payer's next presentation.

    Every field here is derived only from presentations this payer has
    already had, strictly before the date currently being processed.
    """

    total: int = 0
    total_fail: int = 0
    fail_streak: int = 0
    recent_outcomes: deque[int] = field(default_factory=lambda: deque(maxlen=3))
    day_fail: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    day_total: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    decline_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    max_success_amount: float | None = None

    @property
    def is_cold_start(self) -> bool:
        return self.total == 0

    def update(self, *, failed: bool, decline_code: str | None, day_of_month: int, amount: float) -> None:
        self.total += 1
        self.day_total[day_of_month] += 1
        if failed:
            self.total_fail += 1
            self.fail_streak += 1
            self.day_fail[day_of_month] += 1
            if decline_code is not None:
                self.decline_counts[decline_code] += 1
        else:
            self.fail_streak = 0
            if self.max_success_amount is None or amount > self.max_success_amount:
                self.max_success_amount = amount
        self.recent_outcomes.append(1 if failed else 0)


def _cyclic_distance(a: int, b: int, modulus: int = _MONTH_DAYS) -> int:
    """Shortest distance between two days-of-month on a 28-day wheel."""
    diff = abs(a - b) % modulus
    return min(diff, modulus - diff)


def _windowed_dom_rate(state: _PayerState, day_of_month: int) -> float:
    """Bayesian-shrunk failure rate for days near ``day_of_month``.

    Sums failures and totals across every observed day within
    ``_DOM_WINDOW_DAYS`` of the target day (cyclic), then shrinks toward this
    payer's own overall rate by ``_DOM_SHRINKAGE_ALPHA`` virtual observations.
    Shrinkage target is the payer's own rate, never a population figure - a
    cold-start payer gets NaN for this feature entirely (see
    ``_compute_row_features``), not a population fallback, per the chosen
    cold-start strategy.

    A single mandate's history alone would make this indistinguishable from a
    per-mandate feature. What makes it the aggregator's feature and not any
    one merchant's is that ``state`` is accumulated across *all* of this
    payer's mandates, across every merchant they hold - a window failure rate
    built from one merchant's own debits could never see the days that
    merchant doesn't bill on.
    """
    n_fail = 0
    n_total = 0
    for day, total in state.day_total.items():
        if _cyclic_distance(day, day_of_month) <= _DOM_WINDOW_DAYS:
            n_total += total
            n_fail += state.day_fail.get(day, 0)
    prior_rate = state.total_fail / state.total
    return (n_fail + _DOM_SHRINKAGE_ALPHA * prior_rate) / (n_total + _DOM_SHRINKAGE_ALPHA)


def _best_day_gap(state: _PayerState, day_of_month: int) -> float:
    """Cyclic distance from ``day_of_month`` to this payer's best-observed day."""
    if not state.day_total:
        return float("nan")
    best_day = min(state.day_total, key=lambda d: state.day_fail.get(d, 0) / state.day_total[d])
    return float(_cyclic_distance(day_of_month, best_day))


def _compute_row_features(
    *,
    ps: _PayerState,
    psp_history: deque[int],
    mandate_prior_cycles: int,
    day_of_month: int,
    amount: float,
    max_cap: float,
    validity_end,
    scheduled_date,
    slot: str,
    merchant_category: str,
    concurrent_debits_same_day: int,
) -> dict[str, Any]:
    cold = ps.is_cold_start

    if cold:
        payer_fail_streak = np.nan
        payer_fail_rate_3c = np.nan
        dom_fail_propensity = np.nan
        dom_success_gap = np.nan
        decline_mix_balance = np.nan
        decline_mix_technical = np.nan
        amount_vs_payer_max_success = np.nan
    else:
        payer_fail_streak = float(ps.fail_streak)
        payer_fail_rate_3c = (
            float(np.mean(ps.recent_outcomes)) if ps.recent_outcomes else np.nan
        )
        dom_fail_propensity = _windowed_dom_rate(ps, day_of_month)
        dom_success_gap = _best_day_gap(ps, day_of_month)
        if ps.total_fail > 0:
            decline_mix_balance = ps.decline_counts.get(DeclineCode.INSUFFICIENT_FUNDS.value, 0) / ps.total_fail
            decline_mix_technical = (
                sum(ps.decline_counts.get(c, 0) for c in _TECHNICAL_CODES) / ps.total_fail
            )
        else:
            decline_mix_balance = np.nan
            decline_mix_technical = np.nan
        amount_vs_payer_max_success = (
            amount / ps.max_success_amount if ps.max_success_amount else np.nan
        )

    psp_health_index = float(1.0 - np.mean(psp_history)) if psp_history else np.nan

    cycles_to_validity_end = (validity_end - scheduled_date).days / 30.0

    return {
        "payer_fail_streak": payer_fail_streak,
        "payer_fail_rate_3c": payer_fail_rate_3c,
        "dom_fail_propensity": dom_fail_propensity,
        "dom_success_gap": dom_success_gap,
        "decline_mix_balance": decline_mix_balance,
        "decline_mix_technical": decline_mix_technical,
        "amount_vs_payer_max_success": amount_vs_payer_max_success,
        "amount_pct_of_cap": amount / max_cap,
        "mandate_age_cycles": float(mandate_prior_cycles),
        "cycles_to_validity_end": cycles_to_validity_end,
        "psp_health_index": psp_health_index,
        "concurrent_debits_same_day": float(concurrent_debits_same_day),
        "slot_planned": slot,
        "merchant_category": merchant_category,
        "is_cold_start": int(cold),
    }
