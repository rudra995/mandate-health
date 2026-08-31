# What broke

A running log of things that went wrong and what changed as a result. Started
in Phase 0 rather than written retrospectively at the end, because a failure
log assembled from memory after the fact is a different document — a tidier and
less useful one.

Each entry records what broke, how it was caught, and what changed. Entries are
not removed once fixed.

---

## Phase 0

### 1. Dead mandates were presented forever, and the failure rate hit 65%

**Symptom.** The first full generation run produced a 57.3% failure rate at
cycle 1, rising monotonically to 72.8% by cycle 6, for an overall rate of
64.7%. `mandate_expired` was 78% of all failures.

**How it was caught.** A calibration probe that printed the failure rate *by
cycle number* rather than only in aggregate. The aggregate number alone would
have looked merely "too high"; the monotonic rise from 57% to 73% is what
identified the mechanism. This is worth keeping in mind: the aggregate would
have sent me tuning spend parameters, which would not have fixed anything.

**Cause.** Two compounding mistakes, both the same shape.

1. `validity_end` was drawn as `created_at + 12..24 months`, with `created_at`
   up to 30 months in the past. A large share of mandates were therefore
   already expired when the simulation window opened.
2. More fundamentally, a mandate that reached a terminal state — expired,
   revoked, paused, or on a closed account — kept being scheduled and presented
   *every subsequent cycle*, each time producing another terminal decline. One
   dead mandate manufactured up to six failures.

The second is the real bug. A real merchant system presents once, reads the
terminal decline, and stops. Modelling it otherwise inflates the failure rate
with an artifact and, worse, would have made the terminal-decline class look
enormous — which would have made the Phase 3 `skip_retry` rule look far more
valuable than it honestly is.

**Fix.**
- `validity_months` (measured from creation) replaced by
  `remaining_validity_months` (measured from the start of the simulation
  window), so mandates are live when the window opens and roughly 7% expire
  during it.
- Cycle scheduling now stops at the first presentation past `validity_end` —
  one decline, then no further cycles.
- A `dead` set in the per-payer simulation loop: once a mandate returns any
  terminal decline, it is not presented again. Retries already queued against
  that cycle still fire, because the naive status-quo policy genuinely does
  waste attempts on terminal codes, and that waste is what Phase 3 removes.

**Result.** Failure rate fell from 64.7% to 29.1%, terminal declines from 83%
of failures to 9%, and the per-cycle drift disappeared.

**What it changed about how I work.** Diagnostics on generated data are now
broken out by cycle number by default, not just reported in aggregate. A flat
number can hide a trend that names the bug.

---

### 2. Calibration required breaking a rule I had written down in advance

**Symptom.** After fix #1, the failure rate was 29%, well above the published
8–15% band. Balance-side parameters could bring the total into band, but only
by suppressing `insufficient_funds` so far that it stopped being the dominant
decline — which would have contradicted the premise of the entire project.

**Cause.** The non-balance failure floor was too high. PSP declines alone were
2.8% of all cycles, which is above published UPI technical decline rates. With
a ~7% floor of technical and terminal declines, an in-band total left no room
for balance failures to dominate.

**The awkward part.** `docs/ASSUMPTIONS.md` §9 had already committed, in
writing, to adjusting *only* balance-side parameters during calibration, with
PSP health held fixed. Fixing this meant breaking that commitment.

**What I did.** Revised `psps.*.base_health` from 0.968–0.985 to 0.986–0.994,
and recorded the deviation explicitly in ASSUMPTIONS §9 under the heading "One
rule was broken, deliberately", with the reasoning: the parameter was corrected
against an external anchor (published technical decline rates), not moved to
make a downstream result look better.

The alternative was to keep a number I now believed was wrong purely to honour
a procedural commitment, or to change it quietly. Both are worse. A
pre-committed procedure that gets silently amended is worth less than one
amended in public.

**Result.** 12.59% mean failure rate across 5 seeds (11.69–13.28%), with
`insufficient_funds` at ~50% of failures.

---

### 3. `HIDDEN_FIELDS` was silently becoming a dataclass field

**Symptom.** Caught on the first import of `simulator/entities.py`, before any
data was generated.

**Cause.** `HIDDEN_FIELDS: frozenset[str] = frozenset({...})` inside a
`@dataclass` is an *annotated* class attribute, so `dataclass` treats it as a
field with a default rather than a constant. It would have appeared in
`fields()`, become a per-instance attribute, and — the part that matters — been
included in the observable serialisation it exists to restrict.

**Fix.** `HIDDEN_FIELDS: ClassVar[frozenset[str]]`.

**Why it is logged despite being a one-line fix.** The leakage boundary is the
project's central credibility claim, and this failure mode is silent: the code
runs, the tests that existed at the time would have passed, and the boundary
would simply not have been enforced. `test_every_entity_declares_its_hidden_fields_explicitly`
now asserts `HIDDEN_FIELDS` is present in each entity's own `vars()`, so the
same mistake fails loudly.

---

### 4. Environment: `python -m venv` produced a venv with no pip

**Symptom.** `.venv/Scripts/` contained only `python.exe` and `pythonw.exe`.
`python -m pip install -r requirements.txt` then hung indefinitely with no
output.

**Cause.** `ensurepip` did not complete during venv creation on this machine.
Package imports here are extraordinarily slow — a first `import pandas` took
over two minutes — which appears to be antivirus scanning, and pip bootstrap
was affected by the same thing.

**Fix.** Recreate with `python -m venv .venv --without-pip` followed by an
explicit `python -m ensurepip --default-pip`, so the bootstrap is a separate
step that either succeeds or fails visibly rather than hanging inside venv
creation.

**Knock-on change worth keeping.** While diagnosing this, `pandas` was moved
from a module-level import in `simulator/generate.py` to a lazy import inside
`write_artifacts`. Building a world needs numpy only; only writing parquet
needs pandas. The calibration loop went from unrunnable to 0.7 seconds per
400-payer world, which is what made the parameter sweeps in entry 2 practical
at all. A performance problem that looked like an environment nuisance was
partly an unnecessary import.

---

### 5. `make` is not installed on the development machine

**Symptom.** `make data SEED=42` — the command named in the Phase 0 acceptance
criteria — cannot run here. `which make` returns nothing; this is Git Bash on
Windows without MinGW's `make` on `PATH`.

**Status: not fixed, and deliberately so.** The `Makefile` itself is correct and
was reviewed by hand: the `data` target expands to exactly the generator
invocation used for verification, and `PYTHON` auto-detects `.venv`. What is
missing is the tool, not the target.

Phase 0 was therefore verified with the underlying command:

```
.venv/Scripts/python.exe -m simulator.generate --seed 42 --payers 400 --cycles 6 --out data
```

**Why it is logged rather than worked around.** Adding a `make.bat` shim or a
`tasks.py` runner would hide the fact that the documented entry point has never
actually been executed. It needs to run once on a machine that has `make`
before the submission claims a clean-clone run, and that check is still
outstanding. **Update, Phase 1:** `make train` was added to the Makefile
during Phase 1 and is subject to the same gap — verified via
`.venv/Scripts/python.exe -m predictor.train` directly, not via `make`.

---

## Phase 1

### 6. The temporal-leakage test's own cutoff logic was wrong, not the pipeline

**Symptom.** `tests/test_leakage.py::test_corrupting_future_outcomes_does_not_change_past_features`
failed on first run: `payer_fail_streak` differed for 67% of rows at the
cutoff cycle between the baseline and future-corrupted builds.

**How it was caught.** Writing the test itself, before `train.py` existed —
per the working method's "tests first" instruction. If this had been written
after the model, a temporal-leakage bug of this shape could easily have been
mistaken for a modelling quirk instead of investigated directly.

**Cause — a bug in the test, not in `predictor/features.py`.** The test cut
"past" from "future" using `cycle_number >= cutoff`. But `cycle_number` is
per-mandate, not calendar-aligned across mandates: a payer's several mandates
all reach, say, `cycle_number == 4` within the same billing month, but on
different calendar days (different `debit_day_of_month`). Checked directly:
147 of 148 payers in a test-sized world had mandates whose `cycle_number == 4`
rows landed on different dates. `build_features` correctly orders everything
by `scheduled_date`, so a mandate presenting on the 3rd of that month
legitimately feeds a same-`cycle_number` sibling mandate presenting on the
20th — that is real history, not leakage. The test's assumption that
same-`cycle_number` rows were mutually invariant to each other's corruption
was simply false.

**Fix.** Rewrote the cutoff to use a calendar `scheduled_date` boundary
instead of `cycle_number`, which is an unambiguous partition matching how the
pipeline actually orders rows. Re-ran: 8 passed.

**Why it is logged despite being a test-only bug.** This is exactly the kind
of mistake that, in the other direction, could have hidden a real leakage bug
behind a false failure someone patches by weakening the assertion instead of
fixing the cutoff. Caught here because the corrected test's own sanity check
(`assert not changed.empty` for genuinely-future rows) forced verifying the
mechanism rather than just relaxing tolerances until the test passed.

### 7. The leakage test's own docstring tripped its own check

**Symptom.** `test_predictor_source_never_mentions_ground_truth_paths` failed
against `predictor/features.py` and `predictor/split.py` — not because either
file reads `data/ground_truth/`, but because their module docstrings *explain*
the rule that they must not, using the words "ground_truth" in prose.

**Fix.** Rewrote the check to walk the AST and only inspect `Constant` string
nodes that are not a module/class/function docstring — comments are already
outside the AST entirely, so only genuine string literals (the kind that could
be a path passed to `open()` or `read_parquet()`) are checked. Docstrings
explaining the rule no longer trip the rule.

**Why it is logged.** A plain substring scan is the obvious first
implementation of this check, and it would have stayed broken (or been
"fixed" by deleting the explanatory prose) if it hadn't been run before
anything else was built on top of it.

### 8. The cross-merchant ablation did not confirm the aggregator thesis

**Context.** The whole project's central claim (CLAUDE.md §3) is that this
prediction can only be built at the aggregator layer. After Phase 1's initial
results showed `dom_fail_propensity` ranking #1 in feature importance, the
natural next step — proposed and asked for directly — was to measure that
claim rather than assert it: build the identical model, on the identical
split, on the identical feature *names*, twice — once pooling a payer's
history across every merchant they hold (the real pipeline), once re-scoped
to what a single merchant's own book alone could see. A draft README headline
was prepared in advance: "restricted to a single merchant's own history it
scores X, with aggregator-level history it scores 0.748."

**What happened.** The measured gap ran the wrong direction. Merchant-only
scope scored *higher* on the test set — AUC 0.7558 vs 0.7484 for
cross-merchant — and a paired bootstrap 95% CI on the gap, [-0.038, +0.024],
crosses zero: at this sample size (1,618 test rows, ~200 positives) the two
scopes are not statistically distinguishable. A subgroup built specifically
to favour cross-merchant scope (a brand-new mandate for a payer who already
has history through other merchants — exactly the scenario CLAUDE.md §3
describes) showed no advantage either, though at n=216 that slice is itself
too small to be conclusive on its own.

**Root-caused, not just reported.** The likely mechanism: a mature monthly
mandate's `debit_day_of_month` never changes, so its own past presentations
all land on (almost exactly) the same day — an unusually clean, low-noise
autocorrelation signal. `dom_fail_propensity`'s cross-merchant version pools a
±3-day window across *all* of a payer's mandates and shrinks toward that
payer's *overall* rate across every merchant, which can dilute that precise
same-day signal with less-relevant nearby-day data rather than adding
genuinely new information on top of it. Under merchant-only scope, a
mandate's own history and its day-of-month window collapse onto the same
narrow, high-precision quantity by construction, which plausibly explains why
it held up as well as it did.

**What was NOT done, and why.** The feature's shrinkage design was not
rebuilt to "fix" this result. Redesigning `dom_fail_propensity` after seeing
which answer it produced would be exactly the kind of tuning-toward-a-result
this project's Phase 0 calibration procedure was explicit about refusing to
do (see FAILURES #2, which drew the same line the other way — a rule broken
in public when a measurement demanded it, never bent quietly toward a
preferred number). The honest move here is the opposite: report the actual
measured gap, and let it stand as a finding about the current implementation
rather than adjusting the implementation until it produces the desired
finding.

**What changed as a result.**
- The draft README headline is **not used**. `docs/RESULTS.md` states plainly
  that the measurement does not support it.
- `docs/RESULTS.md`'s earlier "what the model learned" paragraph, which had
  read `dom_fail_propensity`'s #1 ranking as direct evidence for the
  aggregator argument, was rewritten to point at this section instead of
  asserting the conclusion the ranking alone could not support.
- `docs/ASSUMPTIONS.md` §9c records the result and the mechanism.
- The aggregator argument itself is not abandoned — CLAUDE.md §3's claim is
  about which *data* is visible to which party (cross-merchant payer history,
  decline-code mix, PSP health in aggregate), a fact about the world that this
  one measurement cannot undo. What changed is the specific, falsifiable claim
  that *this feature design, at this dataset scale* turns that visibility gap
  into a measurably better prediction — which this measurement does not
  support, and which the project now says so about, rather than implying
  otherwise.

**Why this is worth logging as loudly as any bug.** A synthetic-data project
whose only self-tests are ones that confirm its own thesis is not credible.
This is the test that could have gone the other way and made a clean pitch
line — it did not, and the response was to report that rather than to keep
looking for a cut of the data that would.

### 9. The additive follow-up test was also a null result — and that changed the pitch, not the code

**Context.** Entry #8's ablation tested substitution — replacing same-mandate
history with a blended cross-merchant view — and found no measurable gain.
That test had a real confound: it never tested whether cross-merchant
information helps *on top of* same-mandate history, which is what CLAUDE.md
§3 actually claims. A follow-up was pre-registered in `docs/RESULTS.md`
("Additive cross-merchant test") before being run: hypothesis, exact feature
spec (two new other-merchants-only columns added to the merchant-only
baseline), and decision rule, all committed in writing first, with an
explicit one-run, no-parameter-search constraint.

**What happened.** Also null — and the point estimate ran further against the
hypothesis than the first test did. Adding the two other-merchants columns
scored *lower* than the same-mandate-only baseline (AUC 0.7410 vs 0.7558,
Brier 0.0878 vs 0.0875), with a bootstrap 95% CI on the gap of
[-0.035, +0.007]. Column coverage was checked and was 94.2% non-null on the
test set, ruling out "the new feature was mostly missing" as an explanation.

**What was not done, on purpose.** Per the pre-registered rule: no second run,
no hyperparameter adjustment, no redesign of the new features after seeing
this number. The temptation at this point — two ablations in a row failing to
support the project's headline claim — is real, and the discipline was to
report the second null exactly as flatly as the first rather than searching
for a version of the experiment that would land differently.

**What changed as a result — and this is the important part.** Not the
model, not the features, not the world. The *argument*. CLAUDE.md §3 always
named two things a merchant cannot do: see a payer across merchants, and
control the timing levers (PDN offset, execution slot, retry budget
allocation). Both ablations tested only the first half and found no
measurable effect at this scale. The second half was never a claim that
needed a model to confirm it — it is a structural fact about who can call
which API, true regardless of any AUC number. `docs/RESULTS.md`'s "Additive
cross-merchant test" section states the reframed argument explicitly: the
demonstrated aggregator advantage in this build is in *actuation*, not in
producing a measurably better risk score than a well-built single-merchant
baseline. This is a stronger, not a weaker, position than the one the project
started with, because it no longer rests on a number that a bigger dataset or
a different seed could flip.

**Why two failed experiments in a row is being logged as progress, not as
damage control.** The alternative history where this looks worse is the one
where only the first ablation ran, showed a null result, and the project
quietly moved on without checking whether the test itself was well-posed.
Running the correctly-specified follow-up, pre-registering it before results
existed, and accepting a second null under that same discipline is what makes
both numbers trustworthy enough to put in front of a judge at all.
