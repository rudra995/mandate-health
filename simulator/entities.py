"""Core entities of the synthetic world.

Schemas follow CLAUDE.md section 8 field for field. The addition here is that
every entity carries an explicit ``HIDDEN_FIELDS`` constant naming the fields
that must never cross the observability boundary described in CLAUDE.md
section 9.

That constant is not documentation. ``tests/test_determinism.py`` asserts that
no observable artifact contains any field listed in it, and the Phase 1 leakage
test reads the same constants. If a field is hidden in the world model but
missing from ``HIDDEN_FIELDS``, the boundary silently stops being enforced,
so these lists are the load-bearing part of this module.

This module must never import from ``predictor``, ``policy``, ``retry``,
``audit``, or ``eval``. The world is defined without reference to anything that
later tries to learn it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import date, datetime
from enum import StrEnum
from typing import Any, ClassVar, Iterable

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class IncomeBand(StrEnum):
    """Coarse monthly-income bucket. Hidden: an aggregator cannot see income."""

    LOW = "low"
    LOWER_MID = "lower_mid"
    MID = "mid"
    UPPER = "upper"


class IncomeSegment(StrEnum):
    """Which payday pattern the payer follows. Hidden."""

    SALARIED_EARLY = "salaried_early"
    SALARIED_MIDMONTH = "salaried_midmonth"
    IRREGULAR = "irregular"


class SpendSegment(StrEnum):
    """Structural spending posture of the payer. Hidden."""

    STRETCHED = "stretched"
    COMFORTABLE = "comfortable"


class MerchantCategory(StrEnum):
    STREAMING = "streaming"
    FITNESS = "fitness"
    SAAS = "saas"
    EDUCATION = "education"
    INSURANCE = "insurance"
    UTILITIES = "utilities"


class Frequency(StrEnum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class MandateStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    REVOKED = "revoked"
    EXPIRED = "expired"


class ExecutionSlot(StrEnum):
    """NPCI presentation windows (IST).

    ``PEAK_RESTRICTED`` is the 10:00-12:59 band. It is modelled because the
    world contains processors that present there, not because the agent is ever
    permitted to choose it. ``policy/bounds.py`` (Phase 2) rejects it.
    """

    MORNING = "morning"
    AFTERNOON = "afternoon"
    NIGHT = "night"
    PEAK_RESTRICTED = "peak_restricted"

    @property
    def is_permitted(self) -> bool:
        return self is not ExecutionSlot.PEAK_RESTRICTED


class Outcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


class DeclineClass(StrEnum):
    RECOVERABLE = "recoverable"
    TERMINAL = "terminal"


class DeclineCode(StrEnum):
    """Mirrors ``config/decline_codes.yaml``.

    The YAML file is the source of truth for classification and retry
    eligibility; this enum exists only for type safety inside the simulator.
    ``simulator.outcome_model`` asserts the two agree at load time, so the pair
    cannot drift apart silently.
    """

    INSUFFICIENT_FUNDS = "insufficient_funds"
    TECHNICAL_TIMEOUT = "technical_timeout"
    PSP_UNAVAILABLE = "psp_unavailable"
    SLOT_RESTRICTED = "slot_restricted"
    MANDATE_REVOKED = "mandate_revoked"
    MANDATE_EXPIRED = "mandate_expired"
    ACCOUNT_CLOSED = "account_closed"
    MANDATE_PAUSED = "mandate_paused"


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Payer:
    """A person holding several mandates across unrelated merchants.

    Only ``payer_id`` and ``psp_handle`` are observable. Everything describing
    *why* this payer succeeds or fails - income size, payday, spending posture,
    responsiveness to a notification, the balance series itself - is hidden.

    This is the crux of the project's honesty claim: the predictor is asked to
    infer the shape of these traits from outcome history alone, exactly as a
    real aggregator would have to.
    """

    payer_id: str
    psp_handle: str

    # -- ground truth ------------------------------------------------------
    income_band: IncomeBand
    income_segment: IncomeSegment
    income_day: int                    # 1-28
    monthly_income: float
    spend_segment: SpendSegment
    spend_ratio: float                 # share of income consumed by spend
    spend_volatility: float            # lognormal sigma on daily spend
    responsiveness: float              # 0-1 propensity to act on a PDN
    opening_balance: float
    balance_series: dict[date, float] = field(default_factory=dict, repr=False)

    HIDDEN_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "income_band",
            "income_segment",
            "income_day",
            "monthly_income",
            "spend_segment",
            "spend_ratio",
            "spend_volatility",
            "responsiveness",
            "opening_balance",
            "balance_series",
        }
    )


@dataclass(slots=True)
class Mandate:
    """A standing authorisation to debit a payer on a recurring schedule.

    Every field is observable. A mandate is a contract the aggregator issued
    and administers; there is nothing secret in it. ``amount`` and ``max_cap``
    are set by the merchant and can never be changed by this system - doing so
    would require payer UPI PIN re-authorisation (CLAUDE.md section 4.4).
    """

    mandate_id: str
    payer_id: str
    merchant_id: str
    merchant_category: MerchantCategory
    amount: float
    max_cap: float
    debit_day_of_month: int            # 1-28
    frequency: Frequency
    created_at: date
    validity_end: date
    status: MandateStatus = MandateStatus.ACTIVE

    HIDDEN_FIELDS: ClassVar[frozenset[str]] = frozenset()


@dataclass(slots=True)
class DebitCycle:
    """One scheduled presentation of one mandate.

    ``outcome`` and ``decline_code`` are observable only *after* the fact -
    they are the label, and any feature built from them must come from prior
    cycles. Enforcing that is the Phase 1 feature builder's job.

    The counterfactual fields record what would have happened with no
    intervention. Phase 5 needs them to measure prevention by comparison rather
    than by inference; without them, "failures prevented" would be a story
    rather than a measurement.
    """

    cycle_id: str
    mandate_id: str
    payer_id: str
    cycle_number: int
    scheduled_date: date
    execution_slot: ExecutionSlot
    pdn_lead_hours: int | None = None
    pdn_sent_at: datetime | None = None
    outcome: Outcome | None = None
    decline_code: DeclineCode | None = None

    # -- ground truth ------------------------------------------------------
    balance_at_debit: float | None = None
    shortfall_at_debit: float | None = None
    topped_up: bool = False
    pdn_triggered_cancellation: bool = False
    counterfactual_outcome: Outcome | None = None
    counterfactual_decline_code: DeclineCode | None = None

    HIDDEN_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "balance_at_debit",
            "shortfall_at_debit",
            "topped_up",
            "pdn_triggered_cancellation",
            "counterfactual_outcome",
            "counterfactual_decline_code",
        }
    )


@dataclass(slots=True)
class RetryAttempt:
    """One retry against a failed cycle.

    ``attempt_number`` runs 1-3. The original presentation is not an attempt;
    NPCI's cap is one original plus three retries, so attempt_number never
    exceeds 3. Enforcement lives in ``retry/budget.py`` (Phase 3); the
    simulator only records what was attempted.
    """

    attempt_id: str
    cycle_id: str
    mandate_id: str
    payer_id: str
    attempt_number: int
    scheduled_at: datetime
    slot: ExecutionSlot
    outcome: Outcome | None = None
    decline_code: DeclineCode | None = None
    cost_incurred: float = 0.0

    HIDDEN_FIELDS: ClassVar[frozenset[str]] = frozenset()


# ---------------------------------------------------------------------------
# Observability helpers
# ---------------------------------------------------------------------------

#: Every entity that participates in the observability boundary. Tests iterate
#: this rather than hardcoding a list, so a new entity is covered by default.
ENTITIES: tuple[type, ...] = (Payer, Mandate, DebitCycle, RetryAttempt)


def hidden_fields(entity: type) -> frozenset[str]:
    """Return the fields of ``entity`` that must never be observable."""
    return getattr(entity, "HIDDEN_FIELDS", frozenset())


def observable_fields(entity: type) -> tuple[str, ...]:
    """Return the field names of ``entity`` that may be published."""
    blocked = hidden_fields(entity)
    return tuple(f.name for f in fields(entity) if f.name not in blocked)


def all_hidden_field_names() -> frozenset[str]:
    """Union of hidden field names across every entity.

    Used by the leakage checks: any observable artifact containing one of these
    column names is a boundary violation regardless of which entity it came
    from.
    """
    result: set[str] = set()
    for entity in ENTITIES:
        result |= hidden_fields(entity)
    return frozenset(result)


def to_observable_dict(obj: Any) -> dict[str, Any]:
    """Serialise ``obj`` with its hidden fields removed.

    This is the only sanctioned way for simulator objects to leave the
    simulator. Building the dict by hand elsewhere would bypass the boundary.
    """
    blocked = hidden_fields(type(obj))
    return {
        f.name: getattr(obj, f.name)
        for f in fields(obj)
        if f.name not in blocked
    }


def to_ground_truth_dict(obj: Any, include: Iterable[str] | None = None) -> dict[str, Any]:
    """Serialise the hidden side of ``obj`` for the ground-truth artifact.

    ``include`` optionally restricts the output to named fields, which keeps
    the large ``balance_series`` out of row-oriented tables.
    """
    wanted = set(include) if include is not None else hidden_fields(type(obj))
    return {f.name: getattr(obj, f.name) for f in fields(obj) if f.name in wanted}
