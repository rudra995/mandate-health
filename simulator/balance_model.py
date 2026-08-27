"""Hidden income and spending dynamics.

This module owns everything a real payment aggregator cannot see: how much a
payer earns, when it lands, how fast it drains, and what the balance therefore
is on any given day.

The daily process is

    balance[t] = balance[t-1]
               + income_credit(t)        # on the payer's income day
               - savings_sweep(t)        # surplus above the payer's buffer
               - debits(t)               # applied by the caller, in id order
               - discretionary_spend(t)  # lognormal draw, weekend-weighted

with a hard floor at zero: no overdraft exists, and a debit that would overdraw
simply fails.

Two design points are load-bearing.

**Thinness before payday is not a rule.** Nothing here says "fail near the end
of the month". Income arrives as a lump, spend drains roughly evenly, and the
trough therefore lands just before the next credit. The predictor's signature
feature (cross-merchant day-of-month failure propensity) has to find that shape
in outcomes; it was never written down as a rule for it to rediscover.

**Competing debits are a mechanism, not decoration.** All of a payer's mandates
draw on this one ledger. When two debits land on the same date, the first one
through genuinely reduces the balance available to the second. That is what
makes ``concurrent_debits_same_day`` a feature with a real causal referent, and
it is the reason the same payer's failures correlate across unrelated
merchants.

**Common random numbers.** Every stochastic draw is precomputed at construction
from a payer-scoped RNG, before any debit is presented. Two evaluation arms
that make different decisions therefore face the *same* underlying luck, so a
difference between them is a difference in policy rather than in dice. Phase 5
depends on this.

This module must never import from ``predictor``, ``policy``, ``retry``,
``audit``, or ``eval``.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterator

import numpy as np

from simulator.entities import (
    IncomeBand,
    IncomeSegment,
    Payer,
    SpendSegment,
)

# ---------------------------------------------------------------------------
# Payer trait sampling
# ---------------------------------------------------------------------------


def sample_payer_traits(rng: np.random.Generator, config: dict[str, Any], payer_id: str) -> Payer:
    """Draw one payer's hidden traits plus its observable PSP handle.

    Order of draws is fixed. Adding a draw in the middle of this function
    reshuffles every payer generated afterwards, so new traits go at the end.
    """
    income_cfg = config["income"]
    bands = income_cfg["bands"]
    band = bands[_choice_index(rng, [b["weight"] for b in bands])]
    monthly_income = float(band["monthly_amount"]) * float(
        rng.lognormal(mean=0.0, sigma=income_cfg["jitter_lognormal_sigma"])
    )

    segments = config["income_day"]["segments"]
    segment = segments[_choice_index(rng, [s["weight"] for s in segments])]
    if "days" in segment:
        income_day = int(segment["days"][_choice_index(rng, segment["day_weights"])])
    else:
        lo, hi = segment["uniform_range"]
        income_day = int(rng.integers(lo, hi + 1))

    spend_segments = config["spend"]["segments"]
    spend_segment = spend_segments[_choice_index(rng, [s["weight"] for s in spend_segments])]
    spend_ratio = _scaled_beta(rng, spend_segment)

    vol = config["spend"]["daily_volatility_sigma"]
    spend_volatility = float(rng.uniform(vol["lo"], vol["hi"]))

    responsiveness = _scaled_beta(rng, config["pdn"]["responsiveness"])

    opening = config["spend"]["opening_balance_fraction"]
    opening_balance = monthly_income * float(rng.uniform(opening["lo"], opening["hi"]))

    psps = config["psps"]
    psp = psps[_choice_index(rng, [p["weight"] for p in psps])]

    return Payer(
        payer_id=payer_id,
        psp_handle=str(psp["handle"]),
        income_band=IncomeBand(band["name"]),
        income_segment=IncomeSegment(segment["name"]),
        income_day=income_day,
        monthly_income=monthly_income,
        spend_segment=SpendSegment(spend_segment["name"]),
        spend_ratio=spend_ratio,
        spend_volatility=spend_volatility,
        responsiveness=responsiveness,
        opening_balance=opening_balance,
    )


def _choice_index(rng: np.random.Generator, weights: list[float]) -> int:
    """Weighted index draw. One uniform per call, so draw order stays stable."""
    cumulative = np.cumsum(np.asarray(weights, dtype=float))
    return int(np.searchsorted(cumulative, rng.random() * cumulative[-1]))


def _scaled_beta(rng: np.random.Generator, spec: dict[str, Any]) -> float:
    """Beta draw rescaled onto ``[lo, hi]``."""
    raw = float(rng.beta(spec["alpha"], spec["beta"]))
    return float(spec["lo"] + raw * (spec["hi"] - spec["lo"]))


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _MonthPlan:
    """Precomputed income facts for one calendar month."""

    credit_day: date
    credit_amount: float


class BalanceLedger:
    """A payer's daily current-account balance over the simulation window.

    The ledger is advanced day by day by the caller. Within a day the order is

        income credit -> savings sweep -> debits -> discretionary spend

    Debits are presented against the post-credit balance because autopay
    debits are presented early in the banking day, ahead of most discretionary
    spending. Intraday timing is deliberately not modelled: see
    ``docs/ASSUMPTIONS.md`` for why the execution slot affects technical
    decline risk only and not the balance seen at presentation.
    """

    __slots__ = (
        "payer",
        "start_date",
        "end_date",
        "_config",
        "_credit_by_day",
        "_spend_by_day",
        "_buffer_target",
        "_sweep_rate",
        "_balance",
        "_cursor",
        "_series",
        "_finalised",
    )

    def __init__(
        self,
        payer: Payer,
        start_date: date,
        end_date: date,
        rng: np.random.Generator,
        config: dict[str, Any],
    ) -> None:
        self.payer = payer
        self.start_date = start_date
        self.end_date = end_date
        self._config = config

        sweep = config["spend"]["savings_sweep"]
        self._sweep_rate = float(sweep["rate"])
        self._buffer_target = payer.monthly_income * float(
            rng.uniform(sweep["buffer_months"]["lo"], sweep["buffer_months"]["hi"])
        )

        # Everything stochastic is drawn here, up front, before any debit is
        # known. See the module docstring on common random numbers.
        self._credit_by_day = self._plan_income(rng, config)
        self._spend_by_day = self._plan_spend(rng, config)

        self._balance = float(payer.opening_balance)
        self._series: dict[date, float] = {}
        self._cursor = start_date - timedelta(days=1)
        self._finalised = False

    # -- planning ----------------------------------------------------------

    def _plan_income(self, rng: np.random.Generator, config: dict[str, Any]) -> dict[date, float]:
        """Resolve the payer's income day and amount for every month in range.

        Salaried payers get the same day every month. Irregular payers get a
        jittered day and a materially variable amount - they are meant to be
        genuinely harder to predict, and if a model later appears to nail them,
        that is a signal something has leaked rather than a signal of skill.
        """
        segment_cfg = self._income_segment_config(config)
        jitter = int(segment_cfg.get("day_jitter", 0))
        variation = float(segment_cfg.get("monthly_variation_sigma", 0.0))

        credits: dict[date, float] = {}
        for year, month in _months_between(self.start_date, self.end_date):
            day = self.payer.income_day
            if jitter:
                day += int(rng.integers(-jitter, jitter + 1))
            day = max(1, min(28, day))
            credit_date = date(year, month, day)
            if not (self.start_date <= credit_date <= self.end_date):
                continue
            amount = self.payer.monthly_income
            if variation:
                amount *= float(rng.lognormal(mean=0.0, sigma=variation))
            credits[credit_date] = amount
        return credits

    def _income_segment_config(self, config: dict[str, Any]) -> dict[str, Any]:
        for segment in config["income_day"]["segments"]:
            if segment["name"] == self.payer.income_segment.value:
                return segment
        raise KeyError(f"no income_day segment named {self.payer.income_segment}")

    def _plan_spend(self, rng: np.random.Generator, config: dict[str, Any]) -> dict[date, float]:
        """Draw discretionary spend for every day in range.

        Monthly budget is ``spend_ratio * monthly_income``, distributed across
        the month's days in proportion to a weekend-weighted shape and then
        multiplied by unit-mean lognormal noise. Noise is deliberately *not*
        renormalised back to the budget: months should genuinely differ, and
        renormalising would make the monthly total deterministic.
        """
        spend_cfg = config["spend"]
        weekend_multiplier = float(spend_cfg["weekend_multiplier"])
        sigma = self.payer.spend_volatility

        spend: dict[date, float] = {}
        for year, month in _months_between(self.start_date, self.end_date):
            days_in_month = calendar.monthrange(year, month)[1]
            days = [date(year, month, d) for d in range(1, days_in_month + 1)]
            shape = np.array(
                [weekend_multiplier if d.weekday() >= 5 else 1.0 for d in days],
                dtype=float,
            )
            shape /= shape.sum()
            budget = self.payer.spend_ratio * self.payer.monthly_income
            # exp(sigma*Z - sigma^2/2) has mean 1, so the budget is preserved
            # in expectation while individual days and months vary.
            noise = np.exp(rng.normal(0.0, sigma, size=len(days)) - 0.5 * sigma**2)
            for day, share, factor in zip(days, shape, noise):
                if self.start_date <= day <= self.end_date:
                    spend[day] = float(budget * share * factor)
        return spend

    # -- advancing ---------------------------------------------------------

    @property
    def current_day(self) -> date:
        return self._cursor

    @property
    def current_balance(self) -> float:
        return self._balance

    def advance_to(self, target: date) -> None:
        """Roll the ledger forward so that ``target`` is the open day.

        On return, ``target``'s income credit and savings sweep have been
        applied and its discretionary spend has not. Debits for ``target`` are
        presented in that window.
        """
        if self._finalised:
            raise RuntimeError("ledger already finalised")
        if target < self._cursor:
            raise ValueError(f"cannot rewind ledger from {self._cursor} to {target}")
        if target > self.end_date:
            raise ValueError(f"{target} is past the ledger end date {self.end_date}")

        while self._cursor < target:
            if self._cursor >= self.start_date:
                self._close_day(self._cursor)
            self._cursor += timedelta(days=1)
            self._open_day(self._cursor)

    def _open_day(self, day: date) -> None:
        credit = self._credit_by_day.get(day)
        if credit is None:
            return
        self._balance += credit
        surplus = self._balance - self._buffer_target
        if surplus > 0:
            self._balance -= surplus * self._sweep_rate

    def _close_day(self, day: date) -> None:
        spend = self._spend_by_day.get(day, 0.0)
        self._balance = max(0.0, self._balance - spend)
        self._series[day] = self._balance

    def finalise(self) -> None:
        """Close the final open day so the series covers the whole window."""
        if self._finalised:
            return
        if self._cursor >= self.start_date:
            self._close_day(min(self._cursor, self.end_date))
        self._finalised = True

    # -- money movement ----------------------------------------------------

    def withdraw(self, amount: float) -> None:
        """Deduct a successful debit. Callers check sufficiency first."""
        if amount > self._balance + 1e-9:
            raise ValueError(
                f"withdraw {amount:.2f} exceeds balance {self._balance:.2f}; "
                "the outcome model should have declined this presentation"
            )
        self._balance = max(0.0, self._balance - amount)

    def deposit(self, amount: float) -> None:
        """Credit a payer-initiated top-up made in response to a PDN."""
        if amount < 0:
            raise ValueError("deposit amount must be non-negative")
        self._balance += amount

    # -- output ------------------------------------------------------------

    def series(self) -> dict[date, float]:
        """End-of-day balances. Ground truth; never published as observable."""
        if not self._finalised:
            raise RuntimeError("call finalise() before reading the series")
        return dict(self._series)


def _months_between(start: date, end: date) -> Iterator[tuple[int, int]]:
    """Yield ``(year, month)`` for every month touched by ``[start, end]``."""
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        month += 1
        if month == 13:
            year, month = year + 1, 1
