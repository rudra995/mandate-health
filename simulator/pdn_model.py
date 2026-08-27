"""Payer response to a pre-debit notification.

The whole project rests on interventions having a real effect in the world, so
this model has to be more than a constant uplift. It has two halves, and the
second one matters as much as the first.

**Top-up.** A payer who would otherwise fail on insufficient funds may move
money in before the debit:

    p_topup = responsiveness x lead_factor(lead_hours) x shortfall_factor(gap)

- ``responsiveness`` is a hidden per-payer trait. It is per-payer rather than
  global on purpose: if every payer responded identically, blanket notification
  would be strictly optimal and the Phase 5 comparison would be decided before
  it ran. Targeting can only beat blanket if payers differ.
- ``lead_factor`` rises with notice and flattens: the gain from 24h to 36h is
  larger than the gain from 36h to 48h.
- ``shortfall_factor`` decays as the gap grows relative to monthly income.
  Being short Rs 200 is easy to cover; being short Rs 8,000 usually is not.
  Without this term the model would imply a notification can conjure money.

**Cancellation.** A PDN also reminds the payer the subscription exists, and a
small share cancel or pause. Longer lead time gives slightly more time to act
on that impulse. This is the empirical basis for the false-positive penalty in
the Phase 2 EV formula: notifying a mandate that would have succeeded anyway is
not free, and the cost is measured here rather than asserted there.

This module must never import from ``predictor``, ``policy``, ``retry``,
``audit``, or ``eval``.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

from simulator.entities import ExecutionSlot, MandateStatus


def presentation_datetime(day: date, slot: ExecutionSlot, config: dict[str, Any]) -> datetime:
    """The IST timestamp at which a debit in ``slot`` is presented on ``day``."""
    raw = str(config["slots"][slot.value]["present_at"])
    hour, minute = (int(part) for part in raw.split(":"))
    return datetime.combine(day, time(hour=hour, minute=minute))


def pdn_send_time(day: date, slot: ExecutionSlot, lead_hours: int, config: dict[str, Any]) -> datetime:
    """When a PDN with ``lead_hours`` of notice would be sent."""
    return presentation_datetime(day, slot, config) - timedelta(hours=lead_hours)


def is_lead_permitted(lead_hours: int, config: dict[str, Any]) -> bool:
    """Regulatory check: at least 24h of notice, at most the 48h ceiling.

    The simulator only ever uses the configured baseline lead, but the check
    lives here so the world and the Phase 2 compliance guards are testing the
    same rule rather than two copies of it.
    """
    pdn = config["pdn"]
    return int(pdn["min_lead_hours"]) <= lead_hours <= int(pdn["max_lead_hours"])


def is_send_time_permitted(sent_at: datetime, config: dict[str, Any]) -> bool:
    """Requests at or after the late-night cutoff for a next-day debit are rejected."""
    raw = str(config["pdn"]["late_night_cutoff"])
    hour, minute = (int(part) for part in raw.split(":"))
    return sent_at.time() < time(hour=hour, minute=minute)


def lead_factor(lead_hours: int, config: dict[str, Any]) -> float:
    """Diminishing-returns multiplier on top-up probability."""
    factors = config["pdn"]["lead_factor"]
    if lead_hours not in factors:
        raise KeyError(f"no pdn.lead_factor entry for {lead_hours}h")
    return float(factors[lead_hours])


def shortfall_factor(shortfall: float, monthly_income: float, config: dict[str, Any]) -> float:
    """How coverable the gap is, on a 0-1 scale.

    ``1 / (1 + (shortfall / reference) ** shape)``, where the reference is a
    fraction of monthly income. At a shortfall equal to the reference the
    factor is exactly 0.5.
    """
    pdn = config["pdn"]
    reference = monthly_income * float(pdn["shortfall_reference_ratio"])
    if reference <= 0:
        return 0.0
    if shortfall <= 0:
        return 1.0
    ratio = shortfall / reference
    return float(1.0 / (1.0 + ratio ** float(pdn["shortfall_decay_shape"])))


def topup_probability(
    responsiveness: float,
    lead_hours: int,
    shortfall: float,
    monthly_income: float,
    config: dict[str, Any],
) -> float:
    """Probability the payer covers the gap before the debit presents."""
    return float(
        responsiveness
        * lead_factor(lead_hours, config)
        * shortfall_factor(shortfall, monthly_income, config)
    )


def topup_amount(amount_due: float, balance: float, config: dict[str, Any]) -> float:
    """How much a responding payer moves in.

    Enough to cover the debit plus a little headroom - a payer topping up does
    not transfer the exact rupee figure.
    """
    headroom = float(config["pdn"]["topup_headroom_multiplier"])
    return max(0.0, amount_due * headroom - balance)


def cancel_hazard(lead_hours: int, config: dict[str, Any]) -> float:
    """Probability the PDN prompts the payer to revoke or pause the mandate."""
    hazards = config["pdn"]["cancel_hazard"]
    if lead_hours not in hazards:
        raise KeyError(f"no pdn.cancel_hazard entry for {lead_hours}h")
    return float(hazards[lead_hours])


def cancellation_status(draw: float, split_draw: float, lead_hours: int, config: dict[str, Any]) -> MandateStatus | None:
    """Resolve whether a PDN triggers cancellation, and of which kind.

    Returns ``None`` when the payer does nothing, which is the common case.
    """
    if draw >= cancel_hazard(lead_hours, config):
        return None
    revoke_share = float(config["pdn"]["cancel_revoke_share"])
    return MandateStatus.REVOKED if split_draw < revoke_share else MandateStatus.PAUSED
