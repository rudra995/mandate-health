# Ground truth — off limits

This directory holds the hidden state of the synthetic world: per-payer income
band, payday, spend ratio and PDN responsiveness; the full daily balance
series; and per-cycle balance at debit, top-up flags and counterfactual
outcomes. None of it is visible to a real payment aggregator, so none of it may
be read by `predictor/`, `policy/`, or `retry/` — those packages must see
exactly what `data/observable/` exposes and nothing more. Only `simulator/`
(which writes it) and `eval/` (which needs the counterfactuals to measure
prevention by comparison rather than by inference) may read this directory.
This is enforced, not merely requested: `tests/test_leakage.py` fails the build
if a hidden column appears in an observable artifact or if an import path
connects the restricted packages to this data or to `simulator.balance_model`.
