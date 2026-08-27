"""Phase 0 credibility tests.

These are not unit tests of convenience. Each one guards a claim the project
makes out loud:

* *"regenerable from a seed"* - byte-identical output for the same seed.
* *"a world, not a fixture"* - a different seed is a different world.
* *"parameterised from published failure rates"* - the emergent first-attempt
  failure rate lands in the published 8-15% band without being clamped.
* *"no overdraft is modelled"* - the balance process never goes negative.
* *"cross-merchant by construction"* - every payer holds 3-5 mandates with
  distinct merchants.
* *"the leakage boundary is enforced in code"* - no hidden field reaches an
  observable artifact.

If one of these fails, the corresponding claim in the README stops being true,
so none of them should be relaxed to make a build pass.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from simulator.config import load_decline_codes, load_simulator_config
from simulator.entities import ENTITIES, all_hidden_field_names, hidden_fields
from simulator.generate import generate_world, write_artifacts

# Smaller than the evaluation population, large enough for the failure-rate
# assertion to be stable. The full 400-payer run is exercised by `make data`.
TEST_PAYERS = 250
TEST_CYCLES = 6
TEST_SEED = 42

OBSERVABLE_ARTIFACTS = ("mandates", "cycles", "attempts", "merchants")


@pytest.fixture(scope="module")
def config() -> dict:
    return load_simulator_config()


@pytest.fixture(scope="module")
def taxonomy():
    return load_decline_codes()


@pytest.fixture(scope="module")
def world(config, taxonomy):
    return generate_world(TEST_SEED, TEST_PAYERS, TEST_CYCLES, config, taxonomy)


@pytest.fixture(scope="module")
def written(tmp_path_factory, world) -> Path:
    out = tmp_path_factory.mktemp("world")
    write_artifacts(world, out)
    return out


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_produces_byte_identical_observable_output(tmp_path, config, taxonomy):
    """`make data SEED=42` must reproduce byte for byte, not merely in aggregate."""
    first = tmp_path / "run_a"
    second = tmp_path / "run_b"

    for target in (first, second):
        world = generate_world(TEST_SEED, TEST_PAYERS, TEST_CYCLES, config, taxonomy)
        write_artifacts(world, target)

    for name in OBSERVABLE_ARTIFACTS:
        rel = f"observable/{name}.parquet"
        assert _digest(first / rel) == _digest(second / rel), f"{rel} is not reproducible"

    for name in ("payers", "balances", "cycle_truth"):
        rel = f"ground_truth/{name}.parquet"
        assert _digest(first / rel) == _digest(second / rel), f"{rel} is not reproducible"


def test_different_seeds_produce_different_worlds(tmp_path, config, taxonomy):
    """A seed has to be a seed. Identical output across seeds would mean the
    stochastic layer is not wired to it at all."""
    first = tmp_path / "seed_42"
    second = tmp_path / "seed_7"

    write_artifacts(generate_world(42, TEST_PAYERS, TEST_CYCLES, config, taxonomy), first)
    write_artifacts(generate_world(7, TEST_PAYERS, TEST_CYCLES, config, taxonomy), second)

    differing = [
        name
        for name in OBSERVABLE_ARTIFACTS
        if _digest(first / f"observable/{name}.parquet")
        != _digest(second / f"observable/{name}.parquet")
    ]
    # merchants.parquet is the static catalogue and is expected to match.
    assert set(differing) >= {"mandates", "cycles"}


# ---------------------------------------------------------------------------
# World plausibility
# ---------------------------------------------------------------------------


def test_first_attempt_failure_rate_is_in_the_published_band(written):
    """8-15% is the published UPI Autopay range.

    The rate is emergent: nothing in the simulator clamps it. If this fails,
    the fix is a balance-side parameter in config/simulator.yaml, never a
    clamp in code.
    """
    cycles = pd.read_parquet(written / "observable/cycles.parquet")
    failure_rate = (cycles["outcome"] == "failure").mean()
    assert 0.08 <= failure_rate <= 0.15, f"failure rate {failure_rate:.4f} is outside the 8-15% band"


def test_insufficient_funds_is_the_dominant_decline(written):
    """The premise of the whole project is that the dominant failure is a
    balance failure, and that balance failures are the ones timing can move."""
    cycles = pd.read_parquet(written / "observable/cycles.parquet")
    failures = cycles[cycles["outcome"] == "failure"]
    counts = failures["decline_code"].value_counts()
    assert counts.idxmax() == "insufficient_funds"
    assert counts["insufficient_funds"] / len(failures) > 0.4


def test_terminal_declines_exist_but_are_a_minority(written):
    """Phase 3's skip_retry rule needs terminal codes to be present in the
    data, and the world would be implausible if they dominated."""
    taxonomy = load_decline_codes()
    cycles = pd.read_parquet(written / "observable/cycles.parquet")
    failures = cycles[cycles["outcome"] == "failure"]
    terminal = failures["decline_code"].isin({c.value for c in taxonomy.terminal})
    assert terminal.sum() > 0, "no terminal declines were generated"
    assert terminal.mean() < 0.5


def test_balances_are_never_negative(world):
    """No overdraft is modelled: a debit that would overdraw simply fails."""
    for payer_id, series in world.balances.items():
        worst = min(series.values())
        assert worst >= 0.0, f"{payer_id} went negative: {worst:.2f}"


def test_balance_series_covers_the_whole_window(world):
    """A gap in the series would mean a day was silently skipped, which would
    quietly break the drain between income credits."""
    expected_days = (world.end_date - world.start_date).days + 1
    for payer_id, series in world.balances.items():
        assert len(series) == expected_days, f"{payer_id} has {len(series)} days, expected {expected_days}"


# ---------------------------------------------------------------------------
# Cross-merchant structure
# ---------------------------------------------------------------------------


def test_every_payer_holds_three_to_five_mandates_with_distinct_merchants(written, config):
    """The aggregator argument depends on seeing one payer across *unrelated*
    merchants. Two mandates on one merchant would be a merchant's own data."""
    mandates = pd.read_parquet(written / "observable/mandates.parquet")
    bounds = config["population"]["mandates_per_payer"]

    grouped = mandates.groupby("payer_id")["merchant_id"]
    counts = grouped.count()
    assert counts.min() >= bounds["min"]
    assert counts.max() <= bounds["max"]

    distinct = grouped.nunique()
    assert (distinct == counts).all(), "a payer holds two mandates with the same merchant"


def test_concurrent_same_day_debits_actually_occur(written):
    """`concurrent_debits_same_day` is one of the two features that make the
    aggregator claim concrete. If debit days never collide it is a dead
    column, and the debit-day clustering in config is not doing its job.

    The bound guards the mechanism, not a calibration: collisions land at
    roughly 9.5% of payer-days at the default config, and the threshold sits
    well below that deliberately. Pinning the test to the observed value would
    make an ordinary config change look like a regression.
    """
    cycles = pd.read_parquet(written / "observable/cycles.parquet")
    per_payer_day = cycles.groupby(["payer_id", "scheduled_date"]).size()
    assert (per_payer_day > 1).mean() > 0.05
    assert per_payer_day.max() >= 2


def test_retry_budget_cap_is_respected(written, config):
    """NPCI allows at most three retries per cycle. Even the deliberately
    wasteful status-quo policy must not exceed the rail's hard cap."""
    attempts = pd.read_parquet(written / "observable/attempts.parquet")
    if attempts.empty:
        pytest.fail("no retry attempts were generated")
    assert attempts["attempt_number"].max() <= config["baseline_retry_policy"]["max_attempts"] <= 3
    per_cycle = attempts.groupby("cycle_id").size()
    assert per_cycle.max() <= 3


# ---------------------------------------------------------------------------
# Leakage boundary
# ---------------------------------------------------------------------------


def test_no_observable_artifact_contains_a_hidden_field(written):
    """The single most important test in Phase 0.

    Every hidden column name, from every entity, checked against every
    observable artifact. Adding a hidden field to an entity extends this test
    automatically.
    """
    forbidden = all_hidden_field_names()
    assert forbidden, "HIDDEN_FIELDS are empty; the boundary would be vacuous"

    for name in OBSERVABLE_ARTIFACTS:
        frame = pd.read_parquet(written / f"observable/{name}.parquet")
        leaked = forbidden & set(frame.columns)
        assert not leaked, f"observable/{name}.parquet leaks hidden fields: {sorted(leaked)}"


def test_ground_truth_actually_contains_the_hidden_state(written):
    """The mirror of the test above: hidden state must exist *somewhere*, or
    the boundary is being satisfied by simply not modelling anything."""
    payers = pd.read_parquet(written / "ground_truth/payers.parquet")
    for column in ("income_day", "spend_ratio", "responsiveness", "monthly_income"):
        assert column in payers.columns

    truth = pd.read_parquet(written / "ground_truth/cycle_truth.parquet")
    for column in ("balance_at_debit", "topped_up", "counterfactual_outcome"):
        assert column in truth.columns


def test_every_entity_declares_its_hidden_fields_explicitly():
    """`HIDDEN_FIELDS` must be a real frozenset on every entity, not inherited
    by accident, or a new entity would default to fully observable."""
    for entity in ENTITIES:
        assert isinstance(hidden_fields(entity), frozenset)
        assert "HIDDEN_FIELDS" in vars(entity), f"{entity.__name__} does not declare HIDDEN_FIELDS"


def test_simulator_hidden_modules_are_not_imported_by_restricted_packages():
    """A crude import-path check, kept here so the boundary is guarded from
    Phase 0 rather than only from Phase 1 when predictor/ starts existing."""
    import ast

    restricted = ("predictor", "policy", "retry")
    banned_prefixes = ("simulator.balance_model", "simulator.pdn_model")
    root = Path(__file__).resolve().parent.parent

    for package in restricted:
        for path in (root / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    assert not name.startswith(banned_prefixes), (
                        f"{path.relative_to(root)} imports {name}, which crosses the leakage boundary"
                    )
