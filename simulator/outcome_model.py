"""Resolution of a single debit presentation to an outcome and decline code.

The resolution order below is deliberate and fixed:

1. **Mandate state.** A revoked, expired, paused mandate or a closed account
   cannot execute. Terminal.
2. **Slot.** A presentation inside the restricted 10:00-12:59 IST peak draws a
   rail-level decline, independent of balance. This is what lets the naive
   baseline be punished in Phase 5 for choosing a slot badly.
3. **PSP health.** The payer's PSP leg may be degraded. Degradation events give
   this temporal structure, which is what ``psp_health_index`` later detects.
4. **Balance.** Insufficient funds. The dominant cause, and the only one that
   PDN timing can address.
5. **Residual technical noise.** A small per-slot failure rate applied even to
   an otherwise-successful debit.

One consequence of that order is worth stating plainly rather than discovering
later: because the PSP check precedes the balance check, a debit that *would*
have failed on balance can be recorded as ``psp_unavailable``. The observed
``insufficient_funds`` share is therefore slightly below its true causal share.
This is realistic - the rail returns the first failure it hits, not the most
fundamental one - but it means the decline mix is a measurement of what was
observed, not of what was ultimately responsible.

This module must never import from ``predictor``, ``policy``, ``retry``,
``audit``, or ``eval``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np

from simulator.entities import (
    DeclineCode,
    ExecutionSlot,
    MandateStatus,
    Outcome,
)

# ---------------------------------------------------------------------------
# PSP health
# ---------------------------------------------------------------------------


class PspHealthModel:
    """Per-PSP daily success probability, with degradation events.

    Without this, ``psp_health_index`` would be a constant per handle and the
    feature would carry no information beyond the handle's identity. The
    events give it something to detect: a handle that was fine last month and
    is degraded today.

    Health is internal to the simulator. It is never published: the observable
    world sees only the outcomes it produced, exactly as a real aggregator
    would have to infer issuer health from its own traffic.
    """

    __slots__ = ("_health", "_base", "_start", "_end")

    def __init__(
        self,
        config: dict[str, Any],
        start_date: date,
        end_date: date,
        rng: np.random.Generator,
    ) -> None:
        self._start = start_date
        self._end = end_date
        self._base = {str(p["handle"]): float(p["base_health"]) for p in config["psps"]}
        self._health: dict[str, dict[date, float]] = {}

        degradation = config["psp_degradation"]
        n_days = (end_date - start_date).days + 1
        months = n_days / 30.0
        recovery_days = int(degradation["recovery_days"])

        for handle, base in self._base.items():
            series = {start_date + timedelta(days=i): base for i in range(n_days)}
            n_events = int(rng.poisson(float(degradation["events_per_psp_per_month"]) * months))
            for _ in range(n_events):
                offset = int(rng.integers(0, n_days))
                duration = int(
                    rng.integers(degradation["duration_days"]["lo"], degradation["duration_days"]["hi"] + 1)
                )
                drop = float(
                    rng.uniform(degradation["health_drop"]["lo"], degradation["health_drop"]["hi"])
                )
                for d in range(duration):
                    idx = offset + d
                    if idx >= n_days:
                        break
                    day = start_date + timedelta(days=idx)
                    series[day] = min(series[day], base - drop)
                # Linear ramp back to base, so health recovers rather than snaps.
                for r in range(1, recovery_days + 1):
                    idx = offset + duration + r - 1
                    if idx >= n_days:
                        break
                    day = start_date + timedelta(days=idx)
                    recovered = base - drop * (1.0 - r / (recovery_days + 1))
                    series[day] = min(series[day], recovered)
            self._health[handle] = series

    def health_on(self, handle: str, day: date) -> float:
        """Probability the PSP leg completes for ``handle`` on ``day``."""
        try:
            return self._health[handle][day]
        except KeyError as exc:
            raise KeyError(f"no PSP health for {handle} on {day}") from exc


# ---------------------------------------------------------------------------
# Presentation resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Presentation:
    """Everything needed to resolve one debit presentation."""

    day: date
    slot: ExecutionSlot
    amount: float
    balance: float
    mandate_status: MandateStatus
    validity_end: date
    psp_handle: str
    account_closed: bool


@dataclass(frozen=True, slots=True)
class PresentationDraws:
    """Pre-drawn uniforms for one presentation.

    Drawn ahead of resolution and reused across the actual and counterfactual
    evaluation of the same presentation. That is what makes the counterfactual
    a comparison of *decisions* rather than a comparison of dice: the only
    thing that differs between the two is whether an intervention happened.
    """

    slot_restricted: float
    psp: float
    residual: float

    @classmethod
    def draw(cls, rng: np.random.Generator) -> "PresentationDraws":
        return cls(
            slot_restricted=float(rng.random()),
            psp=float(rng.random()),
            residual=float(rng.random()),
        )


@dataclass(frozen=True, slots=True)
class PresentationResult:
    outcome: Outcome
    decline_code: DeclineCode | None
    stage: str

    @property
    def succeeded(self) -> bool:
        return self.outcome is Outcome.SUCCESS


def resolve_presentation(
    presentation: Presentation,
    draws: PresentationDraws,
    psp_health: PspHealthModel,
    config: dict[str, Any],
) -> PresentationResult:
    """Resolve one presentation. Pure: no I/O, no clock, no global state."""
    # 1. Mandate and account state.
    if presentation.account_closed:
        return PresentationResult(Outcome.FAILURE, DeclineCode.ACCOUNT_CLOSED, "state")
    if presentation.mandate_status is MandateStatus.REVOKED:
        return PresentationResult(Outcome.FAILURE, DeclineCode.MANDATE_REVOKED, "state")
    if presentation.mandate_status is MandateStatus.PAUSED:
        return PresentationResult(Outcome.FAILURE, DeclineCode.MANDATE_PAUSED, "state")
    if (
        presentation.mandate_status is MandateStatus.EXPIRED
        or presentation.day > presentation.validity_end
    ):
        return PresentationResult(Outcome.FAILURE, DeclineCode.MANDATE_EXPIRED, "state")

    slot_cfg = config["slots"][presentation.slot.value]

    # 2. Restricted execution window.
    restricted_rate = float(slot_cfg["restricted_decline_rate"])
    if restricted_rate > 0.0 and draws.slot_restricted < restricted_rate:
        return PresentationResult(Outcome.FAILURE, DeclineCode.SLOT_RESTRICTED, "slot")

    # 3. PSP leg.
    health = psp_health.health_on(presentation.psp_handle, presentation.day)
    if draws.psp > health:
        return PresentationResult(Outcome.FAILURE, DeclineCode.PSP_UNAVAILABLE, "psp")

    # 4. Balance.
    if presentation.balance + 1e-9 < presentation.amount:
        return PresentationResult(Outcome.FAILURE, DeclineCode.INSUFFICIENT_FUNDS, "balance")

    # 5. Residual technical noise.
    if draws.residual < float(slot_cfg["residual_technical_rate"]):
        return PresentationResult(Outcome.FAILURE, DeclineCode.TECHNICAL_TIMEOUT, "residual")

    return PresentationResult(Outcome.SUCCESS, None, "success")
