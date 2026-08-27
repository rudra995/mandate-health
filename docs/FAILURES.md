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
outstanding.
