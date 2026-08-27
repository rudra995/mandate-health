"""CLI entry point: build the synthetic world and write it to disk.

    python -m simulator.generate --seed 42 --payers 400 --cycles 6 --out data/

Two artifacts come out, and the separation between them is the point:

``data/observable/``
    Everything the rest of the system may read. This is the merchant-facing
    view: mandates, cycles, attempts, merchants. No balance, no income, no
    payday, no responsiveness.

``data/ground_truth/``
    The hidden world. Readable by ``simulator/`` (which writes it) and
    ``eval/`` (which needs the counterfactuals to measure prevention by
    comparison). Never by ``predictor/``, ``policy/``, or ``retry/``.

The split is enforced structurally rather than by discipline: observable rows
are built with ``entities.to_observable_dict``, which drops every field named
in an entity's ``HIDDEN_FIELDS``. Adding a hidden field to an entity therefore
excludes it from the observable artifact automatically, and the tests fail if
one ever shows up there.

**Determinism.** Every random draw is derived from ``--seed`` plus a stable
hash of the entity it belongs to, never from iteration order. The same seed
produces byte-identical parquet; a different seed produces a different world.

**Common random numbers.** Presentation draws are keyed by
``(mandate_id, cycle_number, attempt_number)``, so an evaluation arm that makes
different decisions still faces the same underlying luck. Phase 5 needs that to
attribute a difference in results to policy rather than to variance.
"""

from __future__ import annotations

import argparse
import calendar
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:  # pandas is imported lazily: building a world does not need it,
    import pandas as pd  # only writing the artifacts does, and it is slow to import.

import numpy as np

from simulator.balance_model import BalanceLedger, sample_payer_traits
from simulator.config import (
    DeclineTaxonomy,
    load_decline_codes,
    load_simulator_config,
)
from simulator.entities import (
    DebitCycle,
    DeclineCode,
    ExecutionSlot,
    Frequency,
    Mandate,
    MandateStatus,
    MerchantCategory,
    Outcome,
    Payer,
    RetryAttempt,
    to_observable_dict,
)
from simulator.outcome_model import (
    Presentation,
    PresentationDraws,
    PspHealthModel,
    resolve_presentation,
)
from simulator import pdn_model

# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def _stable_hash(value: str) -> int:
    """Order-independent, process-independent hash.

    ``hash()`` is salted per process and would break reproducibility across
    runs, so CRC32 is used instead. It is not a good hash; it does not need to
    be. It needs to be the same number every time.
    """
    return zlib.crc32(value.encode("utf-8"))


def _rng(seed: int, *parts: str | int) -> np.random.Generator:
    """A generator keyed by ``seed`` and the identity of what it is drawing for."""
    entropy = [int(seed)]
    for part in parts:
        entropy.append(_stable_hash(part) if isinstance(part, str) else int(part))
    return np.random.default_rng(np.random.SeedSequence(entropy))


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------


def _add_months(anchor: date, months: int) -> date:
    """Shift by whole months, clamping the day to the target month's length."""
    total = anchor.month - 1 + months
    year = anchor.year + total // 12
    month = total % 12 + 1
    day = min(anchor.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _cycle_date(first_cycle_month: date, cycle_number: int, debit_day: int) -> date:
    """Scheduled presentation date for ``cycle_number`` (1-indexed)."""
    month_anchor = _add_months(first_cycle_month.replace(day=1), cycle_number - 1)
    day = min(debit_day, calendar.monthrange(month_anchor.year, month_anchor.month)[1])
    return date(month_anchor.year, month_anchor.month, day)


def _presents_on_cycle(frequency: Frequency, cycle_number: int, config: dict[str, Any]) -> bool:
    modulo = int(config["mandate"]["frequency"]["cycle_modulo"][frequency.value])
    return (cycle_number - 1) % modulo == 0


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class World:
    seed: int
    config: dict[str, Any]
    start_date: date
    first_cycle_month: date
    end_date: date
    n_cycles: int
    payers: list[Payer] = field(default_factory=list)
    mandates: list[Mandate] = field(default_factory=list)
    cycles: list[DebitCycle] = field(default_factory=list)
    attempts: list[RetryAttempt] = field(default_factory=list)
    balances: dict[str, dict[date, float]] = field(default_factory=dict)


@dataclass(slots=True)
class _Task:
    """One scheduled presentation: either the original or a retry of it."""

    mandate_id: str
    cycle_number: int
    attempt_number: int          # 0 = original presentation
    original_date: date
    scheduled_date: date


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def _build_mandates(
    payer: Payer,
    payer_index: int,
    seed: int,
    config: dict[str, Any],
    first_cycle_month: date,
) -> list[Mandate]:
    """Give a payer 3-5 mandates, each with a distinct merchant.

    Distinct merchants are required, not preferred: the entire aggregator
    argument is that this payer is visible across *unrelated* merchants. Two
    mandates with the same merchant would be one merchant's own data.
    """
    rng = _rng(seed, "mandates", payer.payer_id)
    catalogue: list[dict[str, Any]] = config["merchants"]
    bounds = config["population"]["mandates_per_payer"]
    n = int(rng.integers(bounds["min"], bounds["max"] + 1))
    picks = rng.choice(len(catalogue), size=n, replace=False)

    mandate_cfg = config["mandate"]
    freq_cfg = mandate_cfg["frequency"]
    freq_names = list(freq_cfg["weights"])
    freq_weights = [float(freq_cfg["weights"][f]) for f in freq_names]
    day_cfg = mandate_cfg["debit_day"]

    mandates: list[Mandate] = []
    for slot_index, merchant_index in enumerate(sorted(int(p) for p in picks), start=1):
        merchant = catalogue[merchant_index]

        plan = float(merchant["plans"][_weighted_index(rng, merchant["plan_weights"])])
        frequency = Frequency(freq_names[_weighted_index(rng, freq_weights)])
        amount = plan * float(freq_cfg["amount_multiplier"][frequency.value])

        cap_cfg = mandate_cfg["max_cap_multiplier"]
        rounding = float(mandate_cfg["cap_rounding"])
        raw_cap = amount * float(rng.uniform(cap_cfg["lo"], cap_cfg["hi"]))
        max_cap = float(np.ceil(raw_cap / rounding) * rounding)

        if rng.random() < float(day_cfg["clustered_share"]):
            debit_day = int(day_cfg["days"][_weighted_index(rng, day_cfg["day_weights"])])
        else:
            lo, hi = day_cfg["uniform_range"]
            debit_day = int(rng.integers(lo, hi + 1))

        lookback = mandate_cfg["created_at_lookback_months"]
        created_at = _add_months(
            first_cycle_month.replace(day=1),
            -int(rng.integers(lookback["lo"], lookback["hi"] + 1)),
        )
        validity = mandate_cfg["remaining_validity_months"]
        validity_end = _add_months(
            first_cycle_month.replace(day=1),
            int(rng.integers(validity["lo"], validity["hi"] + 1)),
        )

        mandates.append(
            Mandate(
                mandate_id=f"MND-{payer_index:04d}-{slot_index}",
                payer_id=payer.payer_id,
                merchant_id=str(merchant["merchant_id"]),
                merchant_category=MerchantCategory(merchant["category"]),
                amount=round(amount, 2),
                max_cap=max_cap,
                debit_day_of_month=debit_day,
                frequency=frequency,
                created_at=created_at,
                validity_end=validity_end,
                status=MandateStatus.ACTIVE,
            )
        )
    return mandates


def _weighted_index(rng: np.random.Generator, weights: Sequence[float]) -> int:
    cumulative = np.cumsum(np.asarray(weights, dtype=float))
    return int(np.searchsorted(cumulative, rng.random() * cumulative[-1]))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _simulate_payer(
    payer: Payer,
    mandates: list[Mandate],
    world: World,
    psp_health: PspHealthModel,
    taxonomy: DeclineTaxonomy,
) -> tuple[list[DebitCycle], list[RetryAttempt], dict[date, float]]:
    """Run one payer's whole timeline: ledger, cycles, and status-quo retries.

    Payers are independent, so this is the natural unit of simulation. It is
    also what makes competing debits real: all of this payer's mandates draw on
    one ledger, in mandate-id order within a day, so the first debit through
    genuinely reduces what is left for the second.
    """
    config = world.config
    seed = world.seed

    ledger = BalanceLedger(
        payer=payer,
        start_date=world.start_date,
        end_date=world.end_date,
        rng=_rng(seed, "ledger", payer.payer_id),
        config=config,
    )

    by_id = {m.mandate_id: m for m in mandates}
    schedule: dict[date, list[_Task]] = defaultdict(list)
    for mandate in mandates:
        for cycle_number in range(1, world.n_cycles + 1):
            if not _presents_on_cycle(mandate.frequency, cycle_number, config):
                continue
            due = _cycle_date(world.first_cycle_month, cycle_number, mandate.debit_day_of_month)
            schedule[due].append(
                _Task(mandate.mandate_id, cycle_number, 0, due, due)
            )
            if due > mandate.validity_end:
                # Present once past validity - the merchant learns the mandate
                # is dead from that decline - then stop. Continuing to present
                # against an expired mandate every month would manufacture
                # failures rather than model them.
                break

    retry_cfg = config["baseline_retry_policy"]
    retry_offsets: list[int] = list(retry_cfg["offsets_days"])
    max_attempts = int(retry_cfg["max_attempts"])
    retry_slot_names = list(retry_cfg["slot_weights"])
    retry_slot_weights = [float(retry_cfg["slot_weights"][s]) for s in retry_slot_names]
    slot_names = list(config["baseline_slot_policy"]["weights"])
    slot_weights = [float(config["baseline_slot_policy"]["weights"][s]) for s in slot_names]

    baseline_lead = int(config["pdn"]["baseline_lead_hours"])
    lifecycle = config["mandate"]["lifecycle"]

    cycles: dict[tuple[str, int], DebitCycle] = {}
    attempts: list[RetryAttempt] = []
    resolved: set[tuple[str, int]] = set()
    recent_failure: dict[str, bool] = {m.mandate_id: False for m in mandates}
    # A mandate that has already returned a terminal decline is not presented
    # again. The merchant learns from that one decline and stops; the mandate
    # can only come back through a fresh authorisation, which is out of scope.
    dead: set[str] = set()
    account_closed = False

    day = world.start_date
    while day <= world.end_date:
        ledger.advance_to(day)

        tasks = schedule.get(day)
        if tasks:
            for task in sorted(tasks, key=lambda t: (t.mandate_id, t.attempt_number)):
                key = (task.mandate_id, task.cycle_number)
                if task.attempt_number > 0 and key in resolved:
                    continue
                mandate = by_id[task.mandate_id]

                if task.attempt_number == 0:
                    if task.mandate_id in dead:
                        continue
                    account_closed = account_closed or _draw_account_closure(
                        payer, task.cycle_number, seed, lifecycle
                    )
                    _apply_lifecycle_hazards(
                        mandate, task.cycle_number, seed, lifecycle, recent_failure[mandate.mandate_id]
                    )
                    cycle, succeeded = _present_original(
                        payer=payer,
                        mandate=mandate,
                        cycle_number=task.cycle_number,
                        due=task.scheduled_date,
                        ledger=ledger,
                        psp_health=psp_health,
                        world=world,
                        slot_names=slot_names,
                        slot_weights=slot_weights,
                        baseline_lead=baseline_lead,
                        account_closed=account_closed,
                    )
                    cycles[key] = cycle
                    recent_failure[mandate.mandate_id] = not succeeded
                    if succeeded:
                        resolved.add(key)
                    else:
                        if cycle.decline_code is not None and taxonomy.is_terminal(
                            cycle.decline_code
                        ):
                            dead.add(mandate.mandate_id)
                        _queue_retries(
                            schedule=schedule,
                            task=task,
                            offsets=retry_offsets,
                            max_attempts=max_attempts,
                            end_date=world.end_date,
                            decline_code=cycle.decline_code,
                            taxonomy=taxonomy,
                            retry_terminal=bool(retry_cfg["retry_terminal_codes"]),
                        )
                else:
                    attempt, succeeded = _present_retry(
                        payer=payer,
                        mandate=mandate,
                        cycle=cycles[key],
                        task=task,
                        ledger=ledger,
                        psp_health=psp_health,
                        world=world,
                        slot_names=retry_slot_names,
                        slot_weights=retry_slot_weights,
                        account_closed=account_closed,
                    )
                    attempts.append(attempt)
                    if succeeded:
                        resolved.add(key)
                        recent_failure[mandate.mandate_id] = False

        day += timedelta(days=1)

    ledger.finalise()
    ordered = [cycles[k] for k in sorted(cycles)]
    return ordered, attempts, ledger.series()


def _draw_account_closure(payer: Payer, cycle_number: int, seed: int, lifecycle: dict[str, Any]) -> bool:
    """Account closure is a payer-level event: it kills every mandate at once."""
    rng = _rng(seed, "account", payer.payer_id, cycle_number)
    return bool(rng.random() < float(lifecycle["account_closed_hazard_per_cycle"]))


def _apply_lifecycle_hazards(
    mandate: Mandate,
    cycle_number: int,
    seed: int,
    lifecycle: dict[str, Any],
    after_failure: bool,
) -> None:
    """Voluntary churn between cycles.

    A payer whose debit just bounced is likelier to walk away, so the hazard is
    multiplied after a failure. Direction is well supported in subscription
    churn work; the magnitude is an assumption and is labelled as one.
    """
    if mandate.status is not MandateStatus.ACTIVE:
        return
    multiplier = float(lifecycle["post_failure_hazard_multiplier"]) if after_failure else 1.0
    rng = _rng(seed, "lifecycle", mandate.mandate_id, cycle_number)
    if rng.random() < float(lifecycle["revoke_hazard_per_cycle"]) * multiplier:
        mandate.status = MandateStatus.REVOKED
        return
    if rng.random() < float(lifecycle["pause_hazard_per_cycle"]) * multiplier:
        mandate.status = MandateStatus.PAUSED


def _present_original(
    *,
    payer: Payer,
    mandate: Mandate,
    cycle_number: int,
    due: date,
    ledger: BalanceLedger,
    psp_health: PspHealthModel,
    world: World,
    slot_names: list[str],
    slot_weights: list[float],
    baseline_lead: int,
    account_closed: bool,
) -> tuple[DebitCycle, bool]:
    """Resolve the original presentation of one cycle, PDN included.

    The sequence is: send the PDN, let the payer react (cancel, or top up if
    they were going to be short), then present. The counterfactual is computed
    from the *same* draws with the PDN removed, which is what makes it a
    measurement of the intervention rather than of variance.
    """
    config = world.config
    seed = world.seed
    rng = _rng(seed, "cycle", mandate.mandate_id, cycle_number)

    slot = ExecutionSlot(slot_names[_weighted_index(rng, slot_weights)])
    if mandate.status is MandateStatus.ACTIVE and due > mandate.validity_end:
        mandate.status = MandateStatus.EXPIRED

    status_before_pdn = mandate.status
    sent_at = pdn_model.pdn_send_time(due, slot, baseline_lead, config)
    lead_ok = pdn_model.is_lead_permitted(baseline_lead, config) and pdn_model.is_send_time_permitted(
        sent_at, config
    )
    pdn_lead = baseline_lead if lead_ok else None
    pdn_sent_at = sent_at if lead_ok else None

    # PDN reaction: cancellation first, then top-up. Draws are taken
    # unconditionally so that draw order does not depend on the decisions made.
    u_cancel, u_split, u_topup = (float(rng.random()) for _ in range(3))
    draws = PresentationDraws.draw(rng)

    triggered_cancellation = False
    if pdn_lead is not None and mandate.status is MandateStatus.ACTIVE:
        new_status = pdn_model.cancellation_status(u_cancel, u_split, pdn_lead, config)
        if new_status is not None:
            mandate.status = new_status
            triggered_cancellation = True

    balance_before = ledger.current_balance

    counterfactual = resolve_presentation(
        Presentation(
            day=due,
            slot=slot,
            amount=mandate.amount,
            balance=balance_before,
            mandate_status=status_before_pdn,
            validity_end=mandate.validity_end,
            psp_handle=payer.psp_handle,
            account_closed=account_closed,
        ),
        draws,
        psp_health,
        config,
    )

    topped_up = False
    if (
        pdn_lead is not None
        and mandate.status is MandateStatus.ACTIVE
        and counterfactual.decline_code is DeclineCode.INSUFFICIENT_FUNDS
    ):
        shortfall = mandate.amount - balance_before
        probability = pdn_model.topup_probability(
            payer.responsiveness, pdn_lead, shortfall, payer.monthly_income, config
        )
        if u_topup < probability:
            ledger.deposit(pdn_model.topup_amount(mandate.amount, balance_before, config))
            topped_up = True

    balance_at_debit = ledger.current_balance
    actual = resolve_presentation(
        Presentation(
            day=due,
            slot=slot,
            amount=mandate.amount,
            balance=balance_at_debit,
            mandate_status=mandate.status,
            validity_end=mandate.validity_end,
            psp_handle=payer.psp_handle,
            account_closed=account_closed,
        ),
        draws,
        psp_health,
        config,
    )
    if actual.succeeded:
        ledger.withdraw(mandate.amount)

    cycle = DebitCycle(
        cycle_id=f"CYC-{mandate.mandate_id}-{cycle_number}",
        mandate_id=mandate.mandate_id,
        payer_id=payer.payer_id,
        cycle_number=cycle_number,
        scheduled_date=due,
        execution_slot=slot,
        pdn_lead_hours=pdn_lead,
        pdn_sent_at=pdn_sent_at,
        outcome=actual.outcome,
        decline_code=actual.decline_code,
        balance_at_debit=balance_at_debit,
        shortfall_at_debit=max(0.0, mandate.amount - balance_at_debit),
        topped_up=topped_up,
        pdn_triggered_cancellation=triggered_cancellation,
        counterfactual_outcome=counterfactual.outcome,
        counterfactual_decline_code=counterfactual.decline_code,
    )
    return cycle, actual.succeeded


def _queue_retries(
    *,
    schedule: dict[date, list[_Task]],
    task: _Task,
    offsets: list[int],
    max_attempts: int,
    end_date: date,
    decline_code: DeclineCode | None,
    taxonomy: DeclineTaxonomy,
    retry_terminal: bool,
) -> None:
    """Queue the status-quo dunning cron: a fixed T+1 / T+3 / T+7 schedule.

    It fires regardless of decline code when ``retry_terminal`` is set, which
    is what the industry default does. The waste that produces is not an
    oversight in this simulator - it is the thing the Phase 3 allocator exists
    to remove, so the historical data has to contain it.
    """
    if decline_code is not None and not retry_terminal and taxonomy.is_terminal(decline_code):
        return
    for attempt_number, offset in enumerate(offsets[:max_attempts], start=1):
        retry_date = task.original_date + timedelta(days=int(offset))
        if retry_date > end_date:
            break
        schedule[retry_date].append(
            _Task(task.mandate_id, task.cycle_number, attempt_number, task.original_date, retry_date)
        )


def _present_retry(
    *,
    payer: Payer,
    mandate: Mandate,
    cycle: DebitCycle,
    task: _Task,
    ledger: BalanceLedger,
    psp_health: PspHealthModel,
    world: World,
    slot_names: list[str],
    slot_weights: list[float],
    account_closed: bool,
) -> tuple[RetryAttempt, bool]:
    """Resolve one retry. No PDN is re-sent: the budget runs off the original."""
    config = world.config
    rng = _rng(world.seed, "retry", mandate.mandate_id, task.cycle_number, task.attempt_number)
    slot = ExecutionSlot(slot_names[_weighted_index(rng, slot_weights)])
    draws = PresentationDraws.draw(rng)

    if mandate.status is MandateStatus.ACTIVE and task.scheduled_date > mandate.validity_end:
        mandate.status = MandateStatus.EXPIRED

    result = resolve_presentation(
        Presentation(
            day=task.scheduled_date,
            slot=slot,
            amount=mandate.amount,
            balance=ledger.current_balance,
            mandate_status=mandate.status,
            validity_end=mandate.validity_end,
            psp_handle=payer.psp_handle,
            account_closed=account_closed,
        ),
        draws,
        psp_health,
        config,
    )
    if result.succeeded:
        ledger.withdraw(mandate.amount)

    attempt = RetryAttempt(
        attempt_id=f"ATT-{cycle.cycle_id}-{task.attempt_number}",
        cycle_id=cycle.cycle_id,
        mandate_id=mandate.mandate_id,
        payer_id=payer.payer_id,
        attempt_number=task.attempt_number,
        scheduled_at=pdn_model.presentation_datetime(task.scheduled_date, slot, config),
        slot=slot,
        outcome=result.outcome,
        decline_code=result.decline_code,
        # Per-attempt cost is a policy number, not a world number. It is priced
        # in config/policy.yaml from Phase 2 onward, so it stays zero here.
        cost_incurred=0.0,
    )
    return attempt, result.succeeded


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_world(
    seed: int,
    n_payers: int,
    n_cycles: int,
    config: dict[str, Any],
    taxonomy: DeclineTaxonomy,
) -> World:
    """Build the whole world for one seed."""
    first_cycle_month = date.fromisoformat(str(config["simulation"]["start_date"]))
    start_date = first_cycle_month - timedelta(days=int(config["simulation"]["warmup_days"]))
    last_month = _add_months(first_cycle_month.replace(day=1), n_cycles - 1)
    last_day = calendar.monthrange(last_month.year, last_month.month)[1]
    tail = max(int(o) for o in config["baseline_retry_policy"]["offsets_days"]) + 7
    end_date = date(last_month.year, last_month.month, last_day) + timedelta(days=tail)

    world = World(
        seed=seed,
        config=config,
        start_date=start_date,
        first_cycle_month=first_cycle_month,
        end_date=end_date,
        n_cycles=n_cycles,
    )

    psp_health = PspHealthModel(config, start_date, end_date, _rng(seed, "psp_health"))

    for index in range(n_payers):
        payer_id = f"PAY-{index:05d}"
        payer = sample_payer_traits(_rng(seed, "payer", payer_id), config, payer_id)
        mandates = _build_mandates(payer, index, seed, config, first_cycle_month)

        cycles, attempts, series = _simulate_payer(payer, mandates, world, psp_health, taxonomy)

        payer.balance_series = series
        world.payers.append(payer)
        world.mandates.extend(mandates)
        world.cycles.extend(cycles)
        world.attempts.extend(attempts)
        world.balances[payer_id] = series

    return world


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def _frame(rows: Iterable[dict[str, Any]], sort_by: list[str]) -> "pd.DataFrame":
    """Build a deterministically ordered frame with enums flattened to strings."""
    import pandas as pd

    materialised = [
        {k: (str(v) if isinstance(v, ExecutionSlot | Outcome | DeclineCode | MandateStatus | Frequency | MerchantCategory) else v)
         for k, v in row.items()}
        for row in rows
    ]
    frame = pd.DataFrame(materialised)
    if not frame.empty:
        frame = frame.sort_values(sort_by, kind="stable").reset_index(drop=True)
    return frame


def write_artifacts(world: World, out_dir: Path) -> dict[str, Path]:
    """Write the observable and ground-truth artifacts, cleanly separated."""
    observable_dir = out_dir / "observable"
    truth_dir = out_dir / "ground_truth"
    observable_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    psp_by_payer = {p.payer_id: p.psp_handle for p in world.payers}

    # -- observable --------------------------------------------------------
    # psp_handle is denormalised onto mandates: it is genuinely observable
    # (the aggregator routes through it) and the observable artifact set has no
    # payer table to carry it.
    mandate_rows = []
    for mandate in world.mandates:
        row = to_observable_dict(mandate)
        row["psp_handle"] = psp_by_payer[mandate.payer_id]
        mandate_rows.append(row)

    frames = {
        "observable/merchants": _frame(
            (
                {
                    "merchant_id": m["merchant_id"],
                    "name": m["name"],
                    "category": m["category"],
                }
                for m in world.config["merchants"]
            ),
            ["merchant_id"],
        ),
        "observable/mandates": _frame(mandate_rows, ["mandate_id"]),
        "observable/cycles": _frame(
            (to_observable_dict(c) for c in world.cycles), ["cycle_id"]
        ),
        "observable/attempts": _frame(
            (to_observable_dict(a) for a in world.attempts), ["attempt_id"]
        ),
        # -- ground truth --------------------------------------------------
        "ground_truth/payers": _frame(
            (
                {
                    "payer_id": p.payer_id,
                    "income_band": str(p.income_band),
                    "income_segment": str(p.income_segment),
                    "income_day": p.income_day,
                    "monthly_income": p.monthly_income,
                    "spend_segment": str(p.spend_segment),
                    "spend_ratio": p.spend_ratio,
                    "spend_volatility": p.spend_volatility,
                    "responsiveness": p.responsiveness,
                    "opening_balance": p.opening_balance,
                }
                for p in world.payers
            ),
            ["payer_id"],
        ),
        "ground_truth/balances": _frame(
            (
                {"payer_id": payer_id, "date": day, "balance": balance}
                for payer_id, series in world.balances.items()
                for day, balance in series.items()
            ),
            ["payer_id", "date"],
        ),
        "ground_truth/cycle_truth": _frame(
            (
                {
                    "cycle_id": c.cycle_id,
                    "mandate_id": c.mandate_id,
                    "payer_id": c.payer_id,
                    "cycle_number": c.cycle_number,
                    "scheduled_date": c.scheduled_date,
                    "balance_at_debit": c.balance_at_debit,
                    "shortfall_at_debit": c.shortfall_at_debit,
                    "topped_up": c.topped_up,
                    "pdn_triggered_cancellation": c.pdn_triggered_cancellation,
                    "counterfactual_outcome": str(c.counterfactual_outcome)
                    if c.counterfactual_outcome
                    else None,
                    "counterfactual_decline_code": str(c.counterfactual_decline_code)
                    if c.counterfactual_decline_code
                    else None,
                }
                for c in world.cycles
            ),
            ["cycle_id"],
        ),
    }

    written: dict[str, Path] = {}
    for name, frame in frames.items():
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        written[name] = path
    return written


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarise(world: World) -> str:
    """Human-readable summary, printed on every generation run."""
    n_cycles = len(world.cycles)
    failures = [c for c in world.cycles if c.outcome is Outcome.FAILURE]
    failure_rate = len(failures) / n_cycles if n_cycles else 0.0

    decline_counts = Counter(str(c.decline_code) for c in failures)
    attempt_counts = Counter(str(a.outcome) for a in world.attempts)

    recovered = {
        (a.mandate_id, a.cycle_id) for a in world.attempts if a.outcome is Outcome.SUCCESS
    }
    unrecovered = len(failures) - len(recovered)

    lines = [
        "",
        "=" * 68,
        f"  synthetic world generated  |  seed {world.seed}",
        "=" * 68,
        f"  payers                     {len(world.payers):>10,}",
        f"  mandates                   {len(world.mandates):>10,}",
        f"  cycles (presentations)     {n_cycles:>10,}",
        f"  window                     {world.start_date} to {world.end_date}",
        "",
        f"  first-attempt failure rate {failure_rate:>9.2%}   (target band 8-15%)",
        f"  failures                   {len(failures):>10,}",
        "",
        "  decline mix (first attempt)",
    ]
    for code, count in decline_counts.most_common():
        share = count / len(failures) if failures else 0.0
        lines.append(f"    {code:<22} {count:>7,}  {share:>7.2%}")

    lines += [
        "",
        "  status-quo retries (naive T+1 / T+3 / T+7)",
        f"    attempts consumed        {len(world.attempts):>10,}",
        f"    recovered by retry       {len(recovered):>10,}",
        f"    still unrecovered        {unrecovered:>10,}",
        f"    attempt outcomes         {dict(attempt_counts)}",
        "=" * 68,
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="simulator.generate",
        description="Generate the synthetic UPI Autopay world.",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: config base_seed)")
    parser.add_argument("--payers", type=int, default=None, help="number of payers")
    parser.add_argument("--cycles", type=int, default=None, help="number of monthly cycles")
    parser.add_argument("--out", type=Path, default=Path("data"), help="output directory")
    parser.add_argument("--config", type=Path, default=None, help="path to simulator.yaml")
    parser.add_argument("--quiet", action="store_true", help="suppress the summary")
    args = parser.parse_args(argv)

    config = load_simulator_config(args.config)
    taxonomy = load_decline_codes()

    seed = args.seed if args.seed is not None else int(config["simulation"]["base_seed"])
    n_payers = args.payers if args.payers is not None else int(config["population"]["n_payers"])
    n_cycles = args.cycles if args.cycles is not None else int(config["simulation"]["n_cycles"])

    world = generate_world(seed, n_payers, n_cycles, config, taxonomy)
    write_artifacts(world, args.out)

    if not args.quiet:
        print(summarise(world))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
