# Assumptions

Every invented number in this project has a row here. The point of this file is
not completeness for its own sake — it is that a synthetic-data project is only
as credible as its willingness to say which numbers are facts and which are
guesses. A guess labelled as a guess is defensible. A guess presented as a fact
is not.

**Confidence scale**

| Level | Meaning |
|---|---|
| **Regulatory** | An NPCI / RBI rule. Not a choice. Violating it is a bug, not a tuning decision. |
| **Published** | Traceable to a published industry figure or widely reported range. |
| **Reasoned** | Not measured, but derived from a stated argument about how the world works. |
| **Guess** | Invented. Plausible, unmeasured, and would need a real-data check before anyone acted on it. |

**Standing caveat for the whole simulator:** no parameter below was chosen by
looking at predictor performance. The world was parameterised first and is not
re-tuned when a downstream model does poorly. The one exception is the
calibration pass described in §Calibration, which targets the *published*
industry failure rate — not any model metric.

---

## 1. Regulatory constraints (not assumptions)

These are researched facts. They are listed so the reader can see they were
encoded rather than invented.

| Parameter | Value | Basis | Confidence |
|---|---|---|---|
| PDN minimum lead time | 24 h | NPCI mandate: pre-debit notification must reach the payer at least 24 h before execution | Regulatory |
| PDN practical maximum lead | 48 h | Processor practice; PDNs are sent in a 24–48 h window | Published |
| Late-night submission cutoff | 23:50 | Requests at or after this time for a next-day debit are rejected | Regulatory |
| Permitted execution windows (IST) | before 10:00 · 13:00–17:00 · after 21:30 | NPCI execution-window rules | Regulatory |
| Restricted peak window | 10:00–12:59 IST | NPCI peak restriction; presentation here draws technical declines | Regulatory |
| Retry cap | 1 original + 3 retries | NPCI hard cap; exhausting it forces a fresh PDN and fresh mandate execution | Regulatory |
| Per-debit ceiling without per-transaction PIN | ₹15,000 | NPCI standard cap (`mandate.regulatory_cap`) | Regulatory |
| Mandate modification requires UPI PIN | — | Any change to amount, date, validity, pause or cancel needs payer re-authorisation, so no such action exists in this system | Regulatory |

---

## 2. Population and income

| Parameter | Value | Basis | Confidence |
|---|---|---|---|
| `population.n_payers` | 400 (default) | Evaluation floor set in CLAUDE.md §15; large enough for stable per-seed metrics, small enough to regenerate in seconds | Reasoned |
| `population.mandates_per_payer` | 3–5 | The cross-merchant claim needs several merchants per payer. Below 3, `concurrent_debits_same_day` is almost always 0 and the aggregator argument stops being demonstrable | Reasoned |
| `simulation.n_cycles` | 6 | Six monthly cycles gives enough history for rolling features (`payer_fail_rate_3c` needs 3) while leaving cycles to evaluate on | Reasoned |
| `simulation.warmup_days` | 45 | Balance must have a history before the first debit, otherwise cycle 1 is debited off an artificial starting balance and cycle 1 failures are an artifact | Reasoned |
| `income.bands` amounts | ₹25k / ₹45k / ₹80k / ₹150k monthly net | Coarse buckets spanning the range of Indian urban salaried payers likely to hold 3–5 subscriptions | Guess |
| `income.bands` weights | 0.30 / 0.30 / 0.25 / 0.15 | Skewed toward the lower bands so thin-balance cases are common enough to matter | Guess |
| `income.jitter_lognormal_sigma` | 0.15 | Prevents two payers in one band being numerically identical, which would create artificial ties in any learned model | Reasoned |

### Income day — the load-bearing distribution

This is one of the three parameters that most shapes results, and it was an
explicit design decision rather than a default.

| Parameter | Value | Basis | Confidence |
|---|---|---|---|
| `income_day.segments` mixture | 60% salaried-early (days 1–7) · 20% salaried-midmonth (days 14–18) · 20% irregular (uniform 1–28) | Reasoned. A **single** cluster would make day-of-month failure a population-wide calendar effect that any individual merchant could learn from its own book — which would quietly destroy the aggregator argument. The mixture makes the payday a *per-payer* trait, so knowing this payer's day-of-month failure shape genuinely requires cross-merchant history | Reasoned |
| Salaried-early day weights | peak at days 1–2, decaying to day 7 | Indian salary credits concentrate at month start | Published |
| Salaried-midmonth segment | 20% at days 14–18 | Some employers pay mid-month; included to prevent the world having one universal payday | Guess |
| `irregular.day_jitter` | ±3 days | Gig and self-employed income does not arrive on a fixed date. Also serves as an irreducible-noise floor: this segment should be genuinely harder to predict, and if the model appears to predict it perfectly, something is leaking | Reasoned |
| `irregular.monthly_variation_sigma` | 0.22 | Irregular earners see materially variable monthly inflow | Guess |
| Salaried `monthly_variation_sigma` | 0.03 | Salary is near-constant month to month | Reasoned |

---

## 3. Spending and balance dynamics

| Parameter | Value | Basis | Confidence |
|---|---|---|---|
| `spend.segments` split | 85% stretched · 15% comfortable | The comfortable segment is deliberate: without a genuinely low-risk class in the world, `do_nothing` would never be *correct*, only cheap, and the restraint result in Phase 2 would be vacuous | Reasoned |
| Stretched `spend_ratio` | Beta(3.4, 2.0) on [0.55, 0.98], mean ≈ 0.82 | Reasoned from the prompt's 0.70–0.95 guidance, widened at both ends so the distribution has a natural tail of chronically thin payers rather than a hard floor | Guess |
| Comfortable `spend_ratio` | Beta(2, 2) on [0.45, 0.60] | A payer spending under 60% of income essentially never runs short before payday | Guess |
| `spend.daily_volatility_sigma` | 0.35–0.75 (lognormal σ, per payer) | Daily discretionary spend is bursty. The per-payer range means volatility itself is a hidden trait, so two payers with the same spend ratio still differ in risk | Guess |
| `spend.weekend_multiplier` | 1.25 | Discretionary spend rises at weekends. Adds day-of-week texture that is *not* directly observable, so it acts as noise rather than signal | Guess |
| `spend.opening_balance_fraction` | 0.05–0.45 of monthly income | Payers start the simulation with varying buffers. The low end is what makes early cycles risky for some payers | Guess |
| `spend.savings_sweep.rate` | 0.35 of the surplus above target | **Necessary, not cosmetic.** Any payer with `spend_ratio < 1` accumulates indefinitely, so without a sweep the failure rate would fall monotonically from cycle 1 to cycle 6 — a generation artifact a reader would spot instantly and correctly distrust. Real payers move surplus out of the current account. Started at 0.60, revised to 0.35 during calibration (§9) — the original was aggressive enough to pin most balances near zero for most of the month | Reasoned |
| `spend.savings_sweep.buffer_months` | 0.25–1.00 months of income, per payer | The cushion a payer keeps in the current account. Drawn per payer, so buffer size is itself a hidden trait: two payers with the same spend ratio still differ in risk. Started at 0.10–0.60, widened during calibration (§9) | Guess |
| Balance floor | 0 | Balance never goes negative; a debit that would overdraw simply fails. No overdraft is modelled | Reasoned |
| Intraday order | income credit → savings sweep → debits → discretionary spend | Autopay debits present early in the banking day, ahead of most discretionary spending | Reasoned |
| Same-day debit ordering | ascending `mandate_id` | Arbitrary but deterministic. Real presentation order within a day is not something an aggregator controls precisely. It matters because whichever debit goes first gets first claim on a thin balance | Guess |
| Competing debits | drawn from the payer's *other* mandates landing that day | Not decoration — this is the mechanism that creates genuine cross-merchant correlation. Without it, `concurrent_debits_same_day` would be a feature with no causal referent | Reasoned |

**Deliberately not modelled: intraday timing.** The execution slot affects
technical decline risk only; it does not change the balance seen at
presentation. A richer model (salary credited at a per-payer hour, spend
draining through the day) would make slot choice interact with payday and was
considered, then cut. Two reasons: it would have meant inventing an entire
intraday layer with no published basis, and the project's central timing claim
is about *day of month*, not time of day. Recorded as a limitation rather than
quietly omitted.

**Deliberate non-assumption:** thin-before-payday behaviour is *not* coded as a
rule. It emerges from spend ratio, spend timing and competing debits interacting
with the income date. If it were hardcoded, the predictor would be learning a
rule the author wrote rather than structure in a world.

---

## 4. Merchants and mandates

| Parameter | Value | Basis | Confidence |
|---|---|---|---|
| Merchant catalogue | 10 merchants across streaming, fitness, SaaS, education, insurance, utilities | Covers the recurring-payment categories that dominate UPI Autopay volume | Reasoned |
| Plan amounts | ₹129–₹4,999, category-appropriate | Anchored to observed Indian subscription pricing | Published |
| `mandate.max_cap_multiplier` | 1.2–2.5× amount, rounded to ₹100 | Merchants set a cap above the current plan price to leave upgrade headroom. Drives the `amount_pct_of_cap` feature | Guess |
| `frequency.weights` | 87% monthly · 10% quarterly · 3% annual | Monthly dominates recurring subscriptions; the non-monthly tail exists so the advisory cohort insight (CLAUDE.md §5) is measurable rather than asserted | Guess |
| `frequency.amount_multiplier` | ×2.7 quarterly · ×10 annual | Longer billing periods bill more per debit, at a discount to the monthly rate | Reasoned |
| `debit_day.clustered_share` | 0.75 on days [1,2,3,5,7,10,15,20,25,28] | Real subscription debit dates bunch on round dates. This clustering is what makes `concurrent_debits_same_day` a signal rather than noise | Reasoned |
| `created_at_lookback_months` | 1–30 | Gives a spread of mandate ages so `mandate_age_cycles` varies | Reasoned |
| `remaining_validity_months` | 3–60 at simulation start | Validity remaining when the window opens, rather than total validity from creation. Drawn this way so roughly 7% of mandates expire during the six cycles, producing a small stream of genuine `mandate_expired` declines | Guess |
| **Dead mandates stop presenting** | One terminal decline, then no further cycles | **This was a bug found during calibration, not a design choice made up front.** The first implementation kept presenting revoked and expired mandates every cycle, which drove the failure rate to 65% with `mandate_expired` at 78% of failures. A real merchant system presents once, reads the terminal decline, and stops. Recorded here rather than quietly fixed — see `docs/FAILURES.md` | Reasoned |
| `lifecycle.revoke_hazard_per_cycle` | 0.0060 | Voluntary churn per monthly cycle | Guess |
| `lifecycle.pause_hazard_per_cycle` | 0.0040 | Payer-initiated pauses | Guess |
| `lifecycle.account_closed_hazard_per_cycle` | 0.0015 | Rare, but must exist so a truly unrecoverable terminal code is present in the data | Guess |
| `lifecycle.post_failure_hazard_multiplier` | 2.5 | A payer who just had a debit bounce is likelier to walk away. Directionally well supported in subscription churn literature; the magnitude is invented | Guess |

---

## 5. PSPs

| Parameter | Value | Basis | Confidence |
|---|---|---|---|
| PSP handles and weights | 7 handles, HDFC/SBI-weighted | Reflects the concentration of UPI handle share; exact weights invented | Guess |
| `base_health` | 0.986–0.994 | Implies a 0.6–1.4% PSP-leg failure rate. **Revised during calibration** from an initial 0.968–0.985: at those values PSP declines alone were 2.8% of all cycles, above published UPI technical decline rates and large enough to crowd out the balance-driven failures the project is about. See §9 for why this counts as correcting a parameter against external evidence rather than tuning toward a result | Reasoned |
| `psp_degradation.events_per_psp_per_month` | 0.35 (Poisson) | Roughly one degradation event per PSP per three months. Exists so `psp_health_index` has temporal structure to detect; a constant health value would make the feature useless | Guess |
| `psp_degradation.duration_days` | 1–4 | Outages measured in days, not weeks | Guess |
| `psp_degradation.health_drop` | 0.05–0.25 | A bad span is meaningfully bad but not a full outage | Guess |
| `psp_degradation.recovery_days` | 2 (linear ramp) | Health returns gradually rather than snapping back | Guess |

---

## 6. Execution slots

Each slot carries **two** rates rather than one, deliberately: a
`restricted_decline_rate` (the rail refusing a presentation for being in the
blocked peak) and a `residual_technical_rate` (background noise present in every
slot). Merging them into a single number would double-count the peak penalty,
since the resolution order applies the slot check and the residual check
separately.

| Parameter | Value | Basis | Confidence |
|---|---|---|---|
| `slots.*.residual_technical_rate` | 0.006 morning · 0.008 afternoon · 0.010 night | Small background technical noise, slightly worse later in the day as batch load accumulates. The *ordering* is the point; the magnitudes are invented | Guess |
| `slots.peak_restricted.restricted_decline_rate` | 0.180 | Presenting in the restricted peak must be materially punishing, or slot choice is a lever with no consequence and the agent's advantage over the naive baseline would be fake. Set high enough to matter, low enough that the peak is not a guaranteed failure. **This is the weakest-supported number that materially sizes an agent lever** | Guess |
| `baseline_slot_policy.weights` | 50% morning · 28% afternoon · 14% night · 8% peak | The status quo: processors mostly present in permitted windows, and sometimes get it wrong. The 8% peak share is what the agent later removes | Guess |

---

## 7. PDN response model

The second of the three load-bearing design decisions. The model is

```
p_topup = responsiveness × lead_factor(lead_hours) × shortfall_factor(shortfall, income)
shortfall_factor = 1 / (1 + (shortfall / (monthly_income × 0.12)) ** 1.0)
```

| Parameter | Value | Basis | Confidence |
|---|---|---|---|
| `pdn.lead_factor` | 24 h → 0.55 · 36 h → 0.78 · 48 h → 0.90 | More notice helps, with diminishing returns: the gain from 24→36 h exceeds the gain from 36→48 h. Shape is reasoned; values are invented | Guess |
| `pdn.responsiveness` | Beta(2, 2.3) on [0, 0.75], mean ≈ 0.35 | Only a minority of payers act on a notification. Capped below 1 because no notification converts everyone. Per-payer so that *targeting* PDN timing can beat blanket PDN timing — with a constant, blanket would be strictly better and Phase 5 would be a foregone conclusion | Guess |
| `pdn.shortfall_reference_ratio` | 0.12 of monthly income | The shortfall at which top-up probability halves. Being short ₹200 is easy to cover; being short ₹8,000 usually is not. Including this term is what stops the model implying a notification can conjure money | Reasoned |
| `pdn.shortfall_decay_shape` | 1.0 | Simple hyperbolic decay; no evidence for a sharper knee | Guess |
| **Measured, not assumed:** net effect of the blanket 24h status-quo PDN (seed 42, 400 payers, 8,349 cycles) | **+37 net** (82 counterfactual-failures prevented by top-up, 45 successes turned into failures by PDN-triggered cancellation) | Confirms the false-positive penalty in §12's EV formula has a real, non-trivial cost behind it — sending a notification to everyone barely breaks even. One topped-up cycle still failed on residual technical noise after the balance check cleared; that is correct, since the counterfactual isolates the *balance-driven* effect only, not every failure mode | Measured |
| `pdn.topup_headroom_multiplier` | 1.15 | A payer who tops up moves slightly more than the exact amount due | Guess |
| `pdn.cancel_hazard` | 0.008 at 24 h → 0.012 at 48 h | **Intervention is not free.** A PDN reminds the payer the subscription exists, and some cancel or pause. Longer lead gives more time to act on that impulse. This is the empirical basis for the false-positive penalty in the Phase 2 EV formula — without it, that penalty would be an invented number defending an invented risk | Reasoned |
| `pdn.cancel_revoke_share` | 0.55 | Of payers who act against the mandate after a PDN, slightly more revoke than pause | Guess |
| `pdn.baseline_lead_hours` | 24 | The status quo: a fixed offset with no targeting, which is what the industry does today | Published |

---

## 8. Status-quo retry behaviour

Used only to populate historical `attempts.parquet` with a realistic record of
how the world retries today. It constrains nothing the agent does; Phase 5
re-simulates under its own arms.

| Parameter | Value | Basis | Confidence |
|---|---|---|---|
| `baseline_retry_policy.offsets_days` | T+1, T+3, T+7 | The industry-default dunning cron named in CLAUDE.md §2 | Published |
| `baseline_retry_policy.max_attempts` | 3 | NPCI cap | Regulatory |
| `baseline_retry_policy.retry_terminal_codes` | `true` | The naive world spends attempts on revoked mandates. This is the waste the agent eliminates, so the historical data has to contain it | Reasoned |

---

## 8b. Methodology choices that are not parameters

These shape results as much as any number does, so they are declared rather
than left implicit in the code.

| Choice | What it means | Why | Confidence |
|---|---|---|---|
| **Resolution order** | state → slot → PSP → balance → residual noise | The rail returns the first failure it encounters, not the most fundamental one. **Consequence:** a debit that would have failed on balance can be recorded as `psp_unavailable`, so the observed `insufficient_funds` share sits slightly *below* its true causal share. The decline mix is a measurement of what was observed, not of what was ultimately responsible | Reasoned |
| **Local counterfactual** | `counterfactual_outcome` = what this presentation would have returned had no PDN been sent *for this cycle*, holding all prior history fixed | A true global counterfactual needs a parallel world simulated from day zero. The local version is well defined, cheap, and is exactly what per-cycle prevention attribution requires. It does **not** capture second-order effects (a top-up in cycle 3 changing the balance in cycle 4) | Reasoned |
| **Common random numbers** | Presentation draws keyed by `(mandate_id, cycle_number, attempt_number)`; ledger draws precomputed per payer before any debit | Two evaluation arms making different decisions face the same underlying luck, so a Phase 5 difference is attributable to policy rather than variance. Without this, arm comparison across 5 seeds would be far noisier | Reasoned |
| **Lifecycle hazards evaluated at the cycle's presentation date** | Rather than continuously between cycles | A simplification. It means a mandate revoked "between" cycles is observed as revoked at the next presentation, which is when an aggregator would learn of it anyway | Reasoned |
| **`RetryAttempt.cost_incurred` = 0.0 in Phase 0** | Attempt costs are not priced by the simulator | Cost is a policy number, not a world number. It belongs in `config/policy.yaml` and is populated from Phase 2 onward. Pricing it here would mean inventing EV inputs outside the phase that owns them | Reasoned |

---

## 9. Calibration

**Status: complete.** Run at 400 payers × 6 cycles across seeds 42, 7, 101,
2024 and 5.

Target: overall first-presentation failure rate between **8% and 15%**
(published UPI Autopay range), with `insufficient_funds` the dominant decline
code.

The rules were written down **before** the pass was run, so the pass could not
become an unprincipled fudge after the fact:

- Only **balance-side** parameters are adjusted. PSP health, slot decline rates
  and lifecycle hazards are held fixed, because those are anchored to reasoning
  about the rails rather than to a target rate.
- The failure rate is **never clamped** in code. If the emergent rate is out of
  band, the underlying parameter changes and the world is regenerated.
- Nothing is tuned against predictor performance. No model existed when this
  ran.

### What actually changed

Two parameters moved, both in `spend.savings_sweep`:

| Parameter | Before | After |
|---|---|---|
| `savings_sweep.rate` | 0.60 | **0.35** |
| `savings_sweep.buffer_months` | 0.10–0.60 | **0.25–1.00** |

`spend.segments` was **not** touched: the stretched Beta(3.4, 2.0) on
[0.55, 0.98] is exactly as first written. The original sweep was too aggressive
— it removed money payers needed, pinning most balances at zero for most of the
month, which produced a 29% failure rate rather than a realistic pre-payday
trough.

### One rule was broken, deliberately

`psps.*.base_health` was revised from 0.968–0.985 to 0.986–0.994, which the
pre-committed rule said would be held fixed. The reason: at the original values
PSP declines were 2.8% of all cycles, which is above published UPI technical
decline rates and left `insufficient_funds` unable to reach a dominant share
without pushing the overall rate past 15%.

This is a change of kind, not of convenience: the parameter was corrected
against an *external* anchor (published technical decline rates), not moved to
make a downstream result look better. It is recorded here rather than made
quietly, because a pre-committed procedure that gets silently amended is worth
less than one that is amended in public.

### Achieved

| Quantity | Target | Achieved (5 seeds, 400 payers) |
|---|---|---|
| Overall first-attempt failure rate | 8–15% | **12.59% mean** (11.69% – 13.28%) |
| `insufficient_funds` share of failures | dominant | **46–52%**, ~4× the next code |
| Terminal-code share of failures | present, minority | **21–24%** |
| Failure rate drift across cycles 1→6 | flat | flat (no monotonic trend) |

Decline mix at seed 42, as a share of all cycles:

| Code | Share of cycles | Share of failures |
|---|---|---|
| `insufficient_funds` | 6.48% | 50.2% |
| `slot_restricted` | 1.58% | 12.3% |
| `psp_unavailable` | 1.37% | 10.6% |
| `mandate_revoked` | 1.01% | 7.8% |
| `mandate_expired` | 0.82% | 6.3% |
| `technical_timeout` | 0.70% | 5.4% |
| `mandate_paused` | 0.66% | 5.1% |
| `account_closed` | 0.30% | 2.3% |

### Structural checks (diagnostics, not calibration targets)

These were inspected to confirm the world is what was intended. Nothing was
tuned to move them.

**Day-of-month structure emerged on its own.** Insufficient-funds rate by
scheduled debit day, seed 42: 12.9% on day 1, 9.6% on day 3, 2.4% on day 10,
3.0% on day 15, 17.6% on day 27, 6.1% on day 28. The pre-payday trough is
visible without any rule in the code saying "fail near the end of the month" —
it falls out of a lump income credit draining against roughly even spend.

**Failure is a payer trait, not a coin flip.** Per-payer failure rates at seed
42: p10 = 0%, p50 = 8.3%, p90 = 29.4%, with 16% of payers failing zero times
across all six cycles. That spread is what a predictor has to find, and the
zero-failure group is what makes `do_nothing` sometimes *correct* rather than
merely cheap.

---

## 9b. Post-calibration verification pass

Five specific claims were checked against the calibrated world (400 payers,
seed 42 unless stated) before building on top of it. Recorded here because each
one is a claim the README or an interview answer will make out loud.

**Counterfactual outcome is present and correct.** `ground_truth/cycle_truth.parquet`
carries `counterfactual_outcome` / `counterfactual_decline_code` since the first
build (§8 deviations already noted the field; this is the correctness check).
Of 83 top-ups, 82 flip a counterfactual `insufficient_funds` failure into an
actual success; the 1 exception still fails on residual technical noise after
the balance check clears, which is correct — the counterfactual isolates the
*balance-driven* effect only, not every failure mode a debit can hit. Of 51
PDN-triggered cancellations, 45 flip the outcome; the other 6 would have failed
for an unrelated reason regardless. Net measured effect of the blanket 24h
status-quo PDN: **+37** (82 prevented − 45 caused) across 8,349 cycles — see the
new row in §7. A blanket notification to everyone barely breaks even, which is
the empirical case for targeting it.

**Terminal attrition is real but not a population collapse.** 217 of 1,601
mandates (13.6%) hit a terminal decline within the 6-cycle window — on the high
side of plausible, consistent with lifecycle hazards plus the ~7% expiry rate
plus the 2.5× post-failure hazard multiplier compounding together. Raw
per-cycle presentation counts (1,601 → 1,388 → 1,358 → 1,471 → 1,283 → 1,248)
look like a steep decline but are not: quarterly and annual mandates only
present on their modulo cycles, so the count swings with billing frequency, not
just mortality. Isolating monthly-only mandates, cycle 6 still carries roughly
90% of cycle 1's volume. Later cycles are not thin.

**The day-of-month signal is real, the right shape, and not directly visible to
the predictor.** Pooled across all payers by raw scheduled day-of-month, the
insufficient-funds curve is noisy and spiky (0% at day 11, 22% at day 27) —
expected, since payers with different paydays are mixed together and several
days have under 50 observations. Re-cut by *days until this payer's true
payday* (ground truth only), the curve is clean: 26% at 2 days out, 18% at 4
days, 5–7% at 7–9 days, under 1% from day 11 through day 26. Smooth decay, not
a flat line, not a cliff. The catch that matters: `income_day` is hidden, so
the predictor can never compute "days until payday" directly. It has to
recover an equivalent signal through `dom_fail_propensity` — per-payer,
per-day-of-month failure rate learned from that payer's own cross-merchant
history. This check confirms the underlying signal it is chasing is strong
enough to be worth finding, not that finding it is easy.

**Savings-sweep thinness varies across payers, with two expected clusters, not
one.** Checked "thinnest day in November 2025" per payer across all 400:
observed on 30 of 30 possible days. Two concentrations — 92 payers thinnest on
day 1, 45 on day 30 — both explained by the 60% early-month payday segment
(§2): a payer paid on the 1st or 2nd is naturally thinnest just before that
date. The remaining ~66% of payers spread across the other 26 days, 2–25
payers each. Real per-payer variation, not a single shared trough.

**PSP degradation produces genuine multi-day clustering, not per-debit noise.**
Directly inspected the generated health series: `@okhdfcbank` had three
contiguous degraded runs of 6, 5, and 6 days (example: 26–31 Oct, health
0.994 → 0.844, ramping back to 0.944 over the two recovery days);
`@oksbi` had one 3-day run. `psp_health_index` has real temporal structure to
detect — a handle that was healthy last week and degraded this week — rather
than a value that is constant in expectation.

---

## 10. Deviations from CLAUDE.md §8 schemas

Recorded so the difference is visible rather than discovered.

| Entity | Deviation | Why |
|---|---|---|
| `Payer` | Adds hidden `income_segment`, `monthly_income`, `spend_segment`, `spend_ratio`, `responsiveness`, `opening_balance` | §8.1 lists the hidden traits abstractly; the balance process and the PDN response model need them concretely. All are marked hidden and none reaches the observable artifact |
| `DebitCycle` | Adds `payer_id` (observable) and hidden `shortfall_at_debit`, `topped_up`, `pdn_triggered_cancellation`, `counterfactual_outcome`, `counterfactual_decline_code` | `payer_id` is denormalised for join convenience and is already observable via the mandate. The counterfactual pair is required by CLAUDE.md §15 to measure prevention by comparison rather than inference |
| `DebitCycle` | Adds observable `pdn_lead_hours` alongside `pdn_sent_at` | The lead time is the decision the system actually made; storing it directly avoids every consumer re-deriving it from two timestamps |
| `RetryAttempt` | Adds `mandate_id`, `payer_id` | Denormalised for join convenience; both already observable |
| `config/policy.yaml` | Not created in Phase 0 | Phase 2 owns it. Creating it now would mean inventing EV costs and uplifts outside this phase's scope |
| `simulator/config.py` | Module not in the CLAUDE.md §7 tree | YAML loading and validation in one place, so "no magic numbers in code" has exactly one place it can be broken. It also validates `decline_codes.yaml` against the `DeclineCode` enum, which stops the taxonomy drifting between the simulator (enum) and the Phase 3 allocator (YAML) |
| `simulator/pdn_model.py` | Module not in the §7 tree | Payer response to a notification is a distinct responsibility from resolving a presentation. Folding it into `outcome_model.py` would have made that file need "and" to describe it |
| `observable/mandates.parquet` | Carries `psp_handle` | `psp_handle` is genuinely observable and `psp_health_index` depends on it, but the specified observable artifact set has no payer table to carry it. Denormalised onto mandates |
| `conftest.py` | Added at the repo root | Puts the repo root on `sys.path` so `pytest` works from anywhere without an installed package |
| Status-quo retries generated in Phase 0 | `attempts.parquet` is populated by a naive T+1/T+3/T+7 cron | The spec asks for the artifact but not its contents. Leaving it empty would mean the historical data contains no retry behaviour at all, and the wasteful default is precisely what Phase 3 exists to improve on. Phase 5 re-simulates under its own arms, so nothing here constrains the agent |
| Quarterly / annual mandates | 10% / 3% of mandates, with ×2.7 / ×10 amounts | Not requested. Included so the advisory cohort insight in CLAUDE.md §5 ("your annual debit fails more often than the monthly one") is a measurement rather than an assertion |
