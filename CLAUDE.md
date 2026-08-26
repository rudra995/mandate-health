# CLAUDE.md — Mandate Health Service

> **This file is the single source of truth for this project.**
> Claude Code: read this before any task. If something here conflicts with an
> instruction in a prompt, flag the conflict instead of silently picking one.
> Update the checklist in §16 as work completes — that section is live state,
> not documentation.

---

## 1. Project identity

**Name:** Mandate Health Service (working title)
**One-liner:** A pre-debit failure prevention and retry-budget allocation service for UPI Autopay mandates, built at the payment-aggregator layer.
**Thesis:** *The cheapest recovered rupee is the one that never failed.*

**Event:** Razorpay Buildathon — Track 03, "AI Revenue Recovery"
**Deliverable:** Completed public repository
**Deadline:** 5 September 2026
**Start:** 27 August 2026
**Stated reward:** internship consideration / interview

**Track brief (verbatim intent):** Build an agent that detects revenue at risk, determines the right intervention, and executes a bounded recovery workflow. The bar: don't just identify the problem — show measured money recovered across a batch, with compliant escalation, stopping rules, and an audit trail.

**Judging criteria (all four must be visibly satisfied):**

| Criterion | What they said | How this project answers it |
|---|---|---|
| Problem taste | did you pick something that actually matters | Silent, pre-failure revenue loss that only the aggregator can see |
| Build quality | does it run, is it structured, would you trust it | Deterministic core, tested, reproducible seeds, clean module boundaries |
| AI judgment | the right tool in the right place, and where you chose *not* to use one | ML for prediction, deterministic rules for money decisions, LLM only for explanation |
| Failure recovery | what broke, and what you did about it | `docs/FAILURES.md`, plus a system explicitly designed for its own predictor being wrong |

---

## 2. The problem

UPI Autopay debits fail at roughly **8–15%** in production. The dominant cause is insufficient balance at the moment of presentation.

Today the industry response is **reactive**: the debit fires, it fails, and only then does dunning begin — retries, emails, in-app nudges, eventual involuntary churn. Every actor in the chain treats the failure as the starting point.

But the failure is **predictable before it happens**, and the rails already provide a window in which to act:

- NPCI requires a **pre-debit notification (PDN)** to reach the payer at least **24 hours** before the debit. Processors send it in a **24–48 hour** window ahead of execution.
- NPCI enforces **execution windows** — debits run before 10:00 IST, 13:00–17:00 IST, or after 21:30 IST. Presenting in the blocked 10:00–13:00 peak now produces technical declines.
- NPCI caps recovery at **1 original attempt + 3 retries**. Once exhausted, the merchant must start over with a fresh PDN.

Nobody optimises inside these constraints. PDNs fire at a fixed offset. Slots are picked arbitrarily. Retries run on a fixed cron (T+1, T+3, T+7) regardless of whether the payer could plausibly pay on those dates.

**The loss is invisible by construction.** A failure that was preventable never appears in any dashboard as "preventable" — it appears as a normal failure, indistinguishable from an unavoidable one. Nobody measures the gap.

---

## 3. Why this must live at the aggregator, not the merchant

This is the core defensibility argument. It must appear in the README and be sayable in an interview in 30 seconds.

A single merchant — say a music streaming service — sees only its own debits against a given payer. One data point per month. From that, no useful failure model can be built.

Razorpay is a **payment aggregator**. The same payer transacts across many merchants on its rails. That gives platform-level signal a merchant structurally cannot have:

- **Cross-merchant payer outcome history.** This payer's mandate debits fail on the 29th–2nd across three unrelated merchants and succeed on the 8th. We never learn *why*. We learn the **shape**, and the shape is enough.
- **Decline-code mix per payer** — balance-driven vs technical vs mandate-state.
- **Issuer/PSP health in aggregate**, in near real time.
- **Full attempt-level history** across every cycle and every merchant.

Additionally, Razorpay already **owns every lever**: it triggers the PDN, chooses the execution slot, and schedules the retries. The prediction and the actuation live in the same place.

> **The line:** Apple Music cannot build this. Razorpay is the only party that can see the payer across the network *and* control the timing levers. That is why it is a platform service.

---

## 4. Domain primer — hard rules (do not violate, do not invent around)

Claude Code: these are researched regulatory facts, not design preferences. Any code path that violates one is a bug.

### 4.1 Pre-debit notification (PDN)
- Must reach the payer **at least 24 hours** before debit execution.
- Practical send window: **24–48 hours** ahead. Some processors standardise on 36–48h.
- PDN must contain amount, debit date, purpose, merchant name, mandate reference.
- The payer may **pause or cancel** during this window. That is a legitimate outcome, not a system failure.
- Requests submitted very late at night (at/after 23:50) for a next-day debit are rejected.

### 4.2 Execution windows (IST)
Permitted presentation slots:
- Before **10:00**
- **13:00 – 17:00**
- After **21:30**

The 10:00–13:00 peak is restricted. Presenting there produces elevated **technical declines**, independent of balance.

### 4.3 Retry budget
- **1 original attempt + maximum 3 retries.** Hard cap.
- After the budget is exhausted, mandate execution must restart **with a fresh PDN**.
- Retry timing must stay inside permitted execution windows.

### 4.4 Mandate modification — NOT AVAILABLE TO US
- **Any** change to a live mandate (amount, validity, pause, cancel, port) requires **UPI PIN re-authorisation by the payer**.
- Therefore: we **cannot** change the debit date. We **cannot** change the amount. We **cannot** split a debit.
- A mandate can be ported at most once per 90 days.
- Mandate data may not be repurposed beyond display.

### 4.5 Amount limits (context only)
- Standard per-debit cap **₹15,000** without per-transaction PIN.
- Elevated cap up to **₹1,00,000** for specified categories (mutual funds, insurance, credit card bills).
- Amount is set by the **merchant**, never by the aggregator.

### 4.6 What we never touch
- Bank balance — we do not have it and must never model having it.
- Salary credit dates — not visible to an aggregator.
- Any payer PII beyond what mandate metadata already carries.

---

## 5. Action space — everything the agent may do

Every action below is executable by an aggregator without payer re-authorisation.

| Action | Lever | Cost | Constraint |
|---|---|---|---|
| `set_pdn_timing` | Send PDN at 24h / 36h / 48h before debit | ~0 | Must be ≥24h, ≤48h |
| `select_execution_slot` | Choose morning / afternoon / night slot | ~0 | Permitted windows only |
| `schedule_retry` | Place a retry attempt at a chosen slot/date | per-attempt cost + issuer trust cost | Max 3 total |
| `skip_retry` | Spend zero retries on a terminal decline | 0 (saves cost) | Terminal codes only |
| `escalate_to_merchant` | Emit `mandate.at_risk` webhook | ops cost | — |
| `do_nothing` | Deliberate restraint | 0 | Must still be logged with reasoning |

**Advisory-only output** (a report, never an executed action):
- Cohort-level pricing/cadence insight to the merchant, e.g. "your annual ₹1,499 debit fails ~3× more often than the monthly ₹149 for the same payer cohort." The merchant decides; we only surface.

**Explicitly cut** (and the README should say why they were cut — it shows we checked):
- ~~Change debit date~~ — requires UPI PIN
- ~~Change amount~~ — merchant's decision, not ours
- ~~Split debit~~ — mandate modification, requires UPI PIN
- ~~Direct customer balance nudge based on known balance~~ — we have no balance

---

## 6. System architecture

```
                 SYNTHETIC WORLD (hidden ground truth)
                            │
                 ┌──────────┴──────────┐
                 │   L0  simulator     │  payers, mandates, balances*, outcomes
                 └──────────┬──────────┘
                            │  merchant-observable fields ONLY
                            ▼
                 ┌─────────────────────┐
                 │   L1  predictor     │  P(fail) at T-48h
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │   L2  policy engine │  EV-based, deterministic
                 └──────────┬──────────┘
                            │  PDN timing + slot chosen
                            ▼
                      ══ DEBIT FIRES ══
                            │
                ┌───────────┴───────────┐
             success                 failure
                │                       ▼
                │           ┌─────────────────────┐
                │           │  L3  retry allocator│  budget ≤3, EV-gated
                │           └───────────┬─────────┘
                │                       │
                └───────────┬───────────┘
                            ▼
                 ┌─────────────────────┐
                 │   L4  audit + LLM   │  record → human explanation
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │   L5  eval harness  │  agent vs 3 baselines
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │   L6  dashboard     │  4 screens
                 └─────────────────────┘

*balance exists only inside L0 and is never exposed past the boundary.
```

**Build order is bottom-up by dependency:** L0 → L1 → L2 → L3 → L4 → L5 → L6.
Nothing downstream is testable until the simulator is trustworthy.

---

## 7. Repository structure

```
mandate-health/
├── CLAUDE.md                    ← this file
├── README.md                    ← public-facing pitch + results
├── requirements.txt
├── Makefile                     ← make data / train / eval / dash
├── config/
│   ├── simulator.yaml           world parameters
│   ├── policy.yaml              EV costs, thresholds
│   └── decline_codes.yaml       taxonomy: terminal vs recoverable
├── simulator/
│   ├── entities.py              Payer, Mandate, DebitAttempt
│   ├── balance_model.py         hidden income/spend dynamics
│   ├── outcome_model.py         balance + slot → outcome + decline code
│   └── generate.py              CLI entrypoint, seeded
├── predictor/
│   ├── features.py              observable-only feature builder
│   ├── train.py
│   ├── calibrate.py
│   └── model.py                 predict_proba interface
├── policy/
│   ├── ev.py                    expected-value formula
│   ├── engine.py                action selection, deterministic
│   └── bounds.py                compliance guards (windows, caps)
├── retry/
│   ├── budget.py                ≤3 enforcement
│   ├── trust.py                 issuer trust decay model
│   └── allocator.py             which attempts, which slots
├── audit/
│   ├── record.py                structured decision record
│   └── explain.py               LLM: record → prose
├── eval/
│   ├── baselines.py             do_nothing / naive / blanket
│   ├── run.py                   multi-seed harness
│   └── metrics.py
├── dashboard/
│   └── app.py                   Streamlit (or Next.js if time allows)
├── tests/
│   ├── test_determinism.py
│   ├── test_leakage.py          ← critical
│   ├── test_compliance.py       ← windows, retry cap, terminal codes
│   └── test_policy_golden.py
├── data/                        gitignored, regenerable from seed
└── docs/
    ├── ARCHITECTURE.md
    ├── ASSUMPTIONS.md           every invented number, justified
    ├── RESULTS.md               benchmark table
    └── FAILURES.md              ← judging criterion #4
```

---

## 8. Data schemas

### 8.1 `Payer` (simulator-internal)
| Field | Type | Visible to predictor? |
|---|---|---|
| `payer_id` | str | yes |
| `psp_handle` | str | yes (e.g. `@okhdfcbank`) |
| `income_band` | enum | **NO** |
| `income_day` | int 1-28 | **NO** |
| `spend_volatility` | float | **NO** |
| `balance_series` | dict[date→float] | **NO** |

### 8.2 `Mandate`
| Field | Type | Visible? |
|---|---|---|
| `mandate_id` | str | yes |
| `payer_id` | str | yes |
| `merchant_id` | str | yes |
| `merchant_category` | enum | yes |
| `amount` | float | yes |
| `max_cap` | float | yes |
| `debit_day_of_month` | int | yes |
| `frequency` | enum | yes |
| `created_at` | date | yes |
| `validity_end` | date | yes |
| `status` | enum: active/paused/revoked/expired | yes |

### 8.3 `DebitCycle`
| Field | Type | Visible? |
|---|---|---|
| `cycle_id` | str | yes |
| `mandate_id` | str | yes |
| `cycle_number` | int | yes |
| `scheduled_date` | date | yes |
| `pdn_sent_at` | datetime | yes |
| `execution_slot` | enum: morning/afternoon/night | yes |
| `balance_at_debit` | float | **NO** |
| `outcome` | enum: success/failure | yes, *after* the fact |
| `decline_code` | str | yes, after the fact |

### 8.4 `RetryAttempt`
`attempt_id`, `cycle_id`, `attempt_number` (1-3), `scheduled_at`, `slot`, `outcome`, `decline_code`, `cost_incurred`

### 8.5 `DecisionRecord` (audit)
`decision_id`, `timestamp`, `mandate_id`, `cycle_id`, `features_seen` (dict), `p_fail`, `top_feature_contributions` (list of 3), `candidate_actions` (list with EV each), `action_chosen`, `action_cost`, `compliance_checks_passed` (list), `outcome`, `model_version`, `policy_version`

---

## 9. The leakage boundary — non-negotiable

The single most likely criticism of a synthetic-data project is: *"you built a world and a model that both know the same secret."*

Defence, enforced in code:

1. The simulator has **no knowledge of the predictor.** Its parameters are set independently and never tuned to make the model look good.
2. `balance_at_debit`, `income_day`, `income_band`, `spend_volatility`, `balance_series` live in a **separate ground-truth artifact** that the `predictor/` package cannot import.
3. `tests/test_leakage.py` asserts that the feature matrix contains **zero** columns from the hidden set, and that no import path connects `predictor/` to `simulator.balance_model`.
4. Train/test split is **by payer**, not by row. A payer appearing in train never appears in test.
5. `docs/ASSUMPTIONS.md` lists every invented parameter with a stated rationale. Openly declared assumptions are credible; hidden ones are not.

**Interview answer:** "The predictor sees exactly what a merchant-facing system would see — past attempt outcomes, decline codes, amounts, dates, PSP handle. Never balance. There's a test that fails the build if that boundary is crossed."

---

## 10. Feature set (predictor) — observable only

| Feature | Why it's legitimately observable |
|---|---|
| `payer_fail_streak` | consecutive prior failures across all merchants |
| `payer_fail_rate_3c` | rolling failure rate, last 3 cycles |
| `dom_fail_propensity` | learned failure rate for this payer at this day-of-month, **cross-merchant** — the signature feature |
| `dom_success_gap` | days from this debit day to payer's best-performing day-of-month |
| `decline_mix_balance` | share of prior declines that were balance-driven |
| `decline_mix_technical` | share that were technical |
| `amount_vs_payer_max_success` | this amount ÷ largest amount that ever succeeded for this payer |
| `amount_pct_of_cap` | headroom against mandate cap |
| `mandate_age_cycles` | maturity |
| `cycles_to_validity_end` | expiry proximity |
| `psp_health_index` | recent aggregate success rate for this PSP handle |
| `slot_planned` | which permitted window |
| `merchant_category` | categorical |
| `concurrent_debits_same_day` | how many other mandates for this payer land on the same date — **aggregator-only signal** |

`concurrent_debits_same_day` and `dom_fail_propensity` are the two features that make the aggregator argument concrete. Emphasise both.

---

## 11. Model choice and calibration

**Use gradient boosting** (LightGBM or sklearn's `HistGradientBoostingClassifier`). **Do not use a neural network.**

Stated reason — this is an "AI judgment" scoring point, so say it explicitly in the README:
> This decision touches customer money and must be explainable per instance. A tree model gives per-prediction feature attribution that a compliance reviewer can read. A neural net gives a number.

**Calibration is mandatory, not optional.** The policy layer multiplies `p_fail` by a rupee amount to compute expected value. If `p_fail` is uncalibrated, every EV downstream is wrong.

- Fit isotonic or Platt calibration on a held-out payer split.
- Ship a reliability diagram in `docs/RESULTS.md`.
- Report Brier score alongside AUC.

**Threshold is not 0.5.** It is derived from the EV formula — intervene when EV is positive, not when probability crosses a round number. Say this out loud; it separates you from every submission that hardcoded 0.5.

---

## 12. Policy engine — deterministic by design

**No LLM in this layer.** This is the "where you chose not to use one" answer.

> A money decision must produce the same output for the same input, every time, and be reviewable by someone who does not trust models. Probabilistic text generation cannot satisfy that.

### EV formula
For each candidate action *a*:

```
EV(a) = p_fail × uplift(a) × amount
        − cost(a)
        − (1 − p_fail) × false_positive_penalty(a)
```

Where:
- `uplift(a)` — modelled reduction in failure probability from taking action *a*. Sourced from `config/policy.yaml`, justified in `ASSUMPTIONS.md`.
- `cost(a)` — direct cost (notification cost, retry processing cost, ops cost).
- `false_positive_penalty(a)` — harm from acting on a mandate that would have succeeded anyway: payer annoyance, notification fatigue, trust erosion.

Select `argmax EV`. If the max EV is ≤ 0, the action is `do_nothing`.

### Compliance guards (`policy/bounds.py`)
Every selected action passes through hard guards before execution:
- PDN offset ∈ [24h, 48h]
- Execution slot ∈ permitted windows
- Retry count ≤ 3
- No mandate-modification action ever emitted
- Terminal decline code → no retry

A guard rejection is logged, not silently swallowed.

### Restraint is a feature
Every `do_nothing` is logged with its reasoning and its runner-up EV. The track bar explicitly asks for stopping rules. Show them working.

---

## 13. Retry allocation — a scarce budget problem

Reframe away from "retry economics" toward **budget allocation under a hard cap of 3.** This framing is sharper and matches the actual rail constraint.

**Components:**

1. **Decline classification** (`config/decline_codes.yaml`)
   - *Terminal* (mandate revoked, account closed, mandate expired) → **zero retries**, immediate `escalate_to_merchant`.
   - *Recoverable* (insufficient funds, technical timeout, slot-related decline) → eligible for budget.

2. **Issuer trust model** (`retry/trust.py`)
   A modelled `trust_score` per payer-PSP pair that decays with each decline and recovers over time, feeding back into success probability. **This is an assumption, not a measured fact — declare it plainly in `ASSUMPTIONS.md`.** Honest labelling of a modelled assumption is worth more than pretending it is ground truth.

3. **Allocation logic**
   - Do not spend attempt #2 on a payer whose `dom_fail_propensity` says they cannot pay until day 8. Wait.
   - Choose slots to avoid technical-decline risk.
   - Stop when marginal EV of the next attempt turns negative — even if budget remains.

**Three independent brakes:** hard cap (3), EV-negative cutoff, terminal-code immediate stop.

**Expected result:** the agent uses *fewer* retries than the naive baseline while recovering more. That inversion is the headline, not a weakness.

---

## 14. Audit trail and the LLM's narrow job

Every decision emits a `DecisionRecord` (schema §8.5). No exceptions — a decision without a record is a bug.

**The LLM does exactly one thing:** turn a structured record into readable prose.

Example output:
> Mandate M-4471 was scored at 78% failure risk for the 29 August cycle. This payer has failed on three separate merchants when debits land between the 28th and the 2nd, and two other mandates hit the same date. The notification was sent 48 hours ahead rather than the default 24, and the debit was presented in the evening slot. Expected value of the intervention: ₹1,299 preserved against ₹4 of notification cost.

**Rules:**
- The LLM reads the record. It never reads raw data.
- The LLM never selects an action, never computes an EV, never sets a threshold.
- The explanation must be regenerable from the record alone — there is a test for this.

**README line:** *The LLM writes prose from the decision. It never makes the decision.*

---

## 15. Evaluation design — this is what wins

A single demo run proves nothing. The track bar says so explicitly.

### Arms
1. **`do_nothing`** — no intervention, no retries. Establishes the floor.
2. **`naive_retry`** — fixed T+1 / T+3 / T+7, retry everything including terminal codes, fixed 24h PDN, fixed slot. This is the industry default.
3. **`blanket_intervention`** — maximum PDN lead time for *every* mandate, full retry budget always. Proves that targeting has value and that spam is costly.
4. **`agent`** — the system.

### Scale
≥ 400 payers, ≥ 1,500 mandates, 6 cycles, **5 random seeds**. Report mean ± spread. A single lucky run is not a result.

### Metrics
| Metric | Direction |
|---|---|
| Failures prevented (%) | ↑ |
| Revenue preserved (₹) | ↑ |
| PDN notifications sent | ↓ at equal recovery |
| False-positive interventions | ↓ |
| Retry attempts consumed | ↓ |
| Terminal-code retries wasted | → 0 |
| **Net value** = revenue − notification cost − retry cost − trust damage | ↑ |

### Target headline
> Prevented **N%** of failures using **X× fewer** customer notifications than blanket intervention and **Y% fewer** retry attempts than the naive baseline, across 5 seeds.

An **honest exception list** ships alongside: mandates the system could not save, and why. The track bar rewards this directly.

---

## 16. Live checklist

> Claude Code: update this section as work completes. Mark `[x]`, add the date, and note anything that deviated from spec. This is state, not a plan — keep it accurate.

### Phase 0 — Foundation · target 27–28 Aug
- [ ] Repo skeleton, `requirements.txt`, `Makefile`
- [ ] `config/simulator.yaml` with all world parameters
- [ ] `config/decline_codes.yaml` taxonomy (terminal vs recoverable)
- [ ] `simulator/entities.py` — Payer, Mandate, DebitCycle, RetryAttempt
- [ ] `simulator/balance_model.py` — income cycle, spend drift, competing debits
- [ ] `simulator/outcome_model.py` — balance + slot → outcome + decline code
- [ ] Cross-merchant structure: each payer holds 3–5 mandates across merchants
- [ ] `simulator/generate.py` CLI, fully seeded
- [ ] Ground truth written to a **separate** artifact from observable data
- [ ] `tests/test_determinism.py` — same seed, same output
- [ ] Sanity check: baseline failure rate lands in 8–15%
- [ ] `docs/ASSUMPTIONS.md` started

### Phase 1 — Predictor · target 29–30 Aug
- [ ] `predictor/features.py` — all §10 features, observable only
- [ ] `tests/test_leakage.py` passing (no hidden columns, no forbidden imports)
- [ ] Payer-level train/test split
- [ ] `predictor/train.py` — gradient boosting
- [ ] `predictor/calibrate.py` — isotonic/Platt
- [ ] Reliability diagram generated
- [ ] AUC + Brier score recorded in `docs/RESULTS.md`
- [ ] Feature importance chart
- [ ] `predictor/model.py` stable `predict_proba` interface

### Phase 2 — Policy engine · target 31 Aug – 1 Sep
- [ ] `config/policy.yaml` — uplifts, costs, penalties, all justified in ASSUMPTIONS
- [ ] `policy/ev.py` — EV formula
- [ ] `policy/bounds.py` — compliance guards (PDN window, slots, cap, no-modification)
- [ ] `policy/engine.py` — argmax selection, `do_nothing` fallback
- [ ] `tests/test_policy_golden.py` — fixed input → fixed action
- [ ] `tests/test_compliance.py` — guards reject every out-of-bounds action
- [ ] Restraint verified: low-risk mandates receive `do_nothing`

### Phase 3 — Retry allocator · target 2 Sep
- [ ] `retry/budget.py` — hard cap of 3 enforced
- [ ] Terminal decline codes → zero retries, escalate
- [ ] `retry/trust.py` — issuer trust decay/recovery, declared as assumption
- [ ] `retry/allocator.py` — slot-aware, EV-gated scheduling
- [ ] Test: cap never exceeded, terminal codes never retried
- [ ] Test: agent retry count < naive retry count

### Phase 4 — Audit + explainer · target 2–3 Sep
- [ ] `audit/record.py` — DecisionRecord schema
- [ ] Every decision path emits a record (test asserts coverage)
- [ ] `audit/explain.py` — LLM record → prose
- [ ] Test: explanation regenerable from record alone
- [ ] Verify no LLM call exists in `policy/` or `retry/`

### Phase 5 — Eval harness · target 3 Sep
- [ ] `eval/baselines.py` — three baseline arms
- [ ] `eval/metrics.py` — full metric set
- [ ] `eval/run.py` — 5 seeds, mean ± spread
- [ ] Results table written to `docs/RESULTS.md`
- [ ] Exception list generated (unsaveable mandates + reasons)
- [ ] Headline number confirmed and defensible

### Phase 6 — Dashboard + docs · target 4 Sep
- [ ] Screen 1 — Overview (₹ at risk, ₹ preserved, prevention %, retry budget used)
- [ ] Screen 2 — At-Risk Queue (sortable table, action taken, status)
- [ ] Screen 3 — Mandate Replay (full timeline + LLM explanation) ← the demo moment
- [ ] Screen 4 — Benchmark (agent vs 3 baselines, table + chart)
- [ ] `README.md` complete
- [ ] `docs/ARCHITECTURE.md`
- [ ] `docs/ASSUMPTIONS.md` finalised
- [ ] `docs/FAILURES.md` — what broke, what changed
- [ ] Clean commit history with meaningful messages

### 5 Sep — Buffer and submit
- [ ] Full clean-clone run: `make data && make train && make eval && make dash`
- [ ] All tests green
- [ ] README results match a real run, not a stale copy
- [ ] Submit

---

## 17. Frontend specification

**Stack:** Streamlit by default. Next.js only if Phase 5 finishes early. A working Streamlit app beats a half-built React app.

### Screen 1 — Overview
Large-number cards: revenue at risk this cycle, revenue preserved, failures prevented %, retry budget consumed vs available, notifications sent. One trend line across cycles.

### Screen 2 — At-Risk Queue
The operational screen. Sortable table: `mandate_id`, merchant, amount, `p_fail` (with a risk badge), action taken, PDN timing chosen, slot chosen, current status. Filter by risk band and by merchant.

### Screen 3 — Mandate Replay — **the demo moment**
Select one mandate. Show the full timeline as a vertical trail:

```
T-48h   scored 0.78 · top drivers: dom_fail_propensity, concurrent_debits, fail_streak
T-48h   action: PDN sent early (48h) · EV +₹1,295 · runner-up: 24h PDN, EV +₹840
T-0     presented in evening slot (21:45)
T-0     FAILED · decline: insufficient_funds (recoverable)
T+0     retry budget: 3 available · allocator holds — payer's success window is day 8
T+6d    retry 1 · morning slot · SUCCESS · ₹1,299 recovered
        budget remaining: 2 (unspent)
```

Below it, the LLM explanation paragraph. Beside it, the raw `DecisionRecord` in a collapsible JSON block — showing the record *and* the prose makes the "LLM explains, doesn't decide" claim visible rather than merely asserted.

### Screen 4 — Benchmark
The four-arm comparison table with mean ± spread across seeds, plus a grouped bar chart on net value. The exception list sits underneath.

---

## 18. Engineering conventions

- **Python 3.11+.** Type hints on all public functions.
- **Config over constants.** No magic numbers in code — they live in `config/*.yaml` and are explained in `ASSUMPTIONS.md`.
- **Seeded everywhere.** Every stochastic call takes an explicit seed. `make data SEED=42` must reproduce byte-identically.
- **Pure functions in `policy/` and `retry/`.** No I/O, no global state, no clock reads — pass time in. This is what makes golden tests possible.
- **One module, one responsibility.** If a file needs "and" to describe it, split it.
- **Commit per phase**, with messages describing the decision, not the diff.
- **Tests are not optional** in `tests/test_leakage.py` and `tests/test_compliance.py`. Those two encode the project's credibility.
- **No network calls** except the LLM explanation call in `audit/explain.py`.

### Working rules for Claude Code sessions
1. One module per session. Do not attempt a whole phase in one pass.
2. After writing a module, explain it back: what each function does and why.
3. The EV formula and the feature list are **author-owned** — propose changes, do not silently rewrite them.
4. Update §16 after each completed item.
5. Flag any conflict between a prompt and this file rather than resolving it silently.

---

## 19. Glossary

| Term | Meaning |
|---|---|
| **Mandate** | Standing authorisation from a payer allowing recurring debits |
| **UPI Autopay** | NPCI's recurring-payment mandate rail |
| **PDN** | Pre-debit notification — the ≥24h advance alert |
| **Presentation** | The act of submitting a debit for execution |
| **Execution window** | NPCI-permitted time slot for presenting a debit |
| **PSP** | Payment service provider — the payer's UPI app/bank side |
| **Aggregator** | Payment platform serving many merchants (Razorpay's role) |
| **Terminal decline** | Decline that cannot be fixed by retrying (revoked, closed) |
| **Involuntary churn** | Subscription lost to payment failure, not customer intent |
| **Dunning** | The post-failure chase process |

---

## 20. Interview defence — answer these cold

Each phase must leave you able to answer without notes.

| # | Question | Where the answer lives |
|---|---|---|
| 1 | Why is this a Razorpay feature and not a merchant feature? | §3 |
| 2 | How do you predict failure without seeing bank balance? | §10 — cross-merchant day-of-month propensity |
| 3 | How do you know your synthetic data isn't rigged? | §9 — leakage boundary, independent simulator, leakage test |
| 4 | Why gradient boosting and not a neural net? | §11 — per-instance explainability |
| 5 | What is calibration and why does it matter here? | §11 — EV multiplies probability by rupees |
| 6 | Where did you choose *not* to use AI, and why? | §12 — money decisions must be deterministic and reviewable |
| 7 | What can this system legally do to a live mandate? | §4.4, §5 — nothing requiring UPI PIN |
| 8 | Why does your agent use fewer retries than the baseline? | §13 — scarce budget, EV-gated, terminal-code skip |
| 9 | What is your stopping rule? | §12, §13 — three independent brakes |
| 10 | What broke and what did you do? | `docs/FAILURES.md` |

---

## 21. Known limitations — state these before a judge finds them

Declaring limitations is a credibility move. `README.md` must carry this section.

1. **Synthetic data.** No production access exists for a hackathon. The simulator is parameterised from published failure rates (8–15%) and NPCI's operational rules, not from real Razorpay traffic. Every assumption is listed in `ASSUMPTIONS.md`.
2. **Uplift values are modelled.** The effect of PDN timing and slot choice on success probability is an assumption, not a measured causal effect. In production this would require an A/B holdout.
3. **Issuer trust decay is a hypothesis.** Plausible and widely believed in payments, but not something we measured.
4. **No payer-side behavioural model.** We assume a fraction of payers act on an early PDN. Real response rates would need measurement.
5. **Single rail.** UPI Autopay only. eNACH and card mandates have different constraints and are out of scope.
6. **No live integration.** The service is demonstrated against the simulator, not against Razorpay's API.

---

*End of CLAUDE.md. Keep §16 current.*
