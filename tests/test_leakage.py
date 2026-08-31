"""The most important test in the repository.

Four claims, each one load-bearing for the project's central defence:

1. The feature matrix contains zero columns named after a hidden field, for
   any entity, read from ``simulator.entities`` rather than a hand-maintained
   list - so the check stays correct if the schema changes.
2. Nothing under ``predictor/`` imports the two hidden-world simulator
   modules, or references a ``data/ground_truth`` path anywhere in its
   source.
3. A feature computed for cycle T never changes when the outcome of cycle T
   or any later cycle is corrupted - the temporal boundary is structural, not
   a convention someone could accidentally break in a future edit.
4. Train, calibration, and test payer sets are pairwise disjoint and the
   split is reproducible from its seed alone.

If any of these fail, the project's leakage-boundary claim in CLAUDE.md
section 9 is false, regardless of what any other test says. Treat failures
here as build-breaking, not as something to relax.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from predictor.features import FEATURE_COLUMNS, build_features
from predictor.split import make_payer_split
from simulator.config import load_decline_codes, load_simulator_config
from simulator.entities import all_hidden_field_names
from simulator.generate import generate_world, write_artifacts

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_PAYERS = 150
TEST_CYCLES = 6
TEST_SEED = 42


@pytest.fixture(scope="module")
def observable(tmp_path_factory) -> dict[str, pd.DataFrame]:
    config = load_simulator_config()
    taxonomy = load_decline_codes()
    world = generate_world(TEST_SEED, TEST_PAYERS, TEST_CYCLES, config, taxonomy)
    out = tmp_path_factory.mktemp("leakage_world")
    write_artifacts(world, out)
    return {
        "cycles": pd.read_parquet(out / "observable/cycles.parquet"),
        "mandates": pd.read_parquet(out / "observable/mandates.parquet"),
        "merchants": pd.read_parquet(out / "observable/merchants.parquet"),
    }


@pytest.fixture(scope="module")
def feature_set(observable):
    return build_features(observable["cycles"], observable["mandates"], observable["merchants"])


# ---------------------------------------------------------------------------
# 1. No hidden field ever reaches the feature matrix
# ---------------------------------------------------------------------------


def test_feature_matrix_contains_no_hidden_field_columns(feature_set):
    forbidden = all_hidden_field_names()
    assert forbidden, "HIDDEN_FIELDS are empty; this check would be vacuous"

    leaked_in_X = forbidden & set(feature_set.X.columns)
    leaked_in_meta = forbidden & set(feature_set.meta.columns)
    assert not leaked_in_X, f"feature matrix leaks hidden fields: {sorted(leaked_in_X)}"
    assert not leaked_in_meta, f"meta frame leaks hidden fields: {sorted(leaked_in_meta)}"


def test_feature_columns_match_the_declared_list(feature_set):
    """The returned X has exactly FEATURE_COLUMNS, in that order - not a
    superset that happens to exclude hidden names by luck."""
    assert list(feature_set.X.columns) == list(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# 2. Import graph and path-string scan
# ---------------------------------------------------------------------------


def test_predictor_does_not_import_hidden_simulator_modules():
    banned_prefixes = ("simulator.balance_model", "simulator.pdn_model")
    for path in (REPO_ROOT / "predictor").rglob("*.py"):
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
                    f"{path.relative_to(REPO_ROOT)} imports {name}, "
                    "which crosses the leakage boundary"
                )


def _docstring_node_ids(tree: ast.Module) -> set[int]:
    """id() of every Constant node that is a module/class/function docstring.

    Explanatory prose in a docstring is not a leakage risk; a string literal
    used as a path in actual code is. Excluding docstrings (comments are
    already outside the AST entirely) keeps the scan below meaningful instead
    of tripping on this file's own explanation of the rule.
    """
    ids: set[int] = set()
    candidates: list[ast.AST] = [tree]
    candidates.extend(
        n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    for node in candidates:
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def test_predictor_source_never_references_ground_truth_paths():
    for path in (REPO_ROOT / "predictor").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstring_ids = _docstring_node_ids(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docstring_ids:
                continue
            assert "ground_truth" not in node.value, (
                f"{path.relative_to(REPO_ROOT)} has a non-docstring string literal "
                f"referencing 'ground_truth': {node.value!r} - predictor/ must never "
                "read that directory"
            )


# ---------------------------------------------------------------------------
# 3. Temporal boundary: corrupting the future must not change the past
# ---------------------------------------------------------------------------


def test_corrupting_future_outcomes_does_not_change_past_features(observable):
    """The cutoff here is a calendar DATE, not a ``cycle_number``.

    ``cycle_number`` is per-mandate: a payer's several mandates reach
    ``cycle_number == 4`` in the same billing month but on different days
    (different ``debit_day_of_month``), since each mandate keeps its own
    schedule. ``build_features`` correctly orders everything by
    ``scheduled_date``, so a mandate presenting on the 3rd legitimately feeds
    a same-``cycle_number`` sibling mandate presenting on the 20th for the
    same payer - that is real history, not leakage. An earlier version of
    this test cut on ``cycle_number`` and failed on exactly that case; cutting
    on date instead makes "past" and "future" an unambiguous partition that
    matches how the pipeline actually orders rows.
    """
    cycles = observable["cycles"]
    mandates = observable["mandates"]
    merchants = observable["merchants"]

    baseline = build_features(cycles, mandates, merchants)

    unique_dates = sorted(cycles["scheduled_date"].unique())
    cutoff_date = unique_dates[len(unique_dates) // 2]

    corrupted = cycles.copy()
    future_mask = corrupted["scheduled_date"] >= cutoff_date
    assert future_mask.sum() > 0 and (~future_mask).sum() > 0, "cutoff leaves no rows on one side"

    flipped_outcome = corrupted.loc[future_mask, "outcome"].map(
        {"success": "failure", "failure": "success"}
    )
    corrupted.loc[future_mask, "outcome"] = flipped_outcome
    corrupted.loc[future_mask & (corrupted["outcome"] == "failure"), "decline_code"] = "technical_timeout"
    corrupted.loc[future_mask & (corrupted["outcome"] == "success"), "decline_code"] = None

    rebuilt = build_features(corrupted, mandates, merchants)

    past = baseline.meta["scheduled_date"] < cutoff_date
    pd.testing.assert_frame_equal(
        baseline.X.loc[past].reset_index(drop=True),
        rebuilt.X.loc[past].reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        baseline.y.loc[past].reset_index(drop=True),
        rebuilt.y.loc[past].reset_index(drop=True),
    )

    # Rows scheduled exactly on cutoff_date: their own outcome was corrupted,
    # but their features must depend only on strictly earlier dates, so their
    # features (not their target) must be unchanged.
    at_cutoff = baseline.meta["scheduled_date"] == cutoff_date
    pd.testing.assert_frame_equal(
        baseline.X.loc[at_cutoff].reset_index(drop=True),
        rebuilt.X.loc[at_cutoff].reset_index(drop=True),
        check_dtype=False,
    )

    # Sanity check the probe itself: rows strictly after cutoff_date SHOULD
    # change, because their real history was altered. If nothing changes
    # here, the pipeline might not be using history at all, which would make
    # the "unchanged" assertions above trivially true rather than meaningful.
    after = baseline.meta["scheduled_date"] > cutoff_date
    changed = (
        baseline.X.loc[after]
        .reset_index(drop=True)
        .compare(rebuilt.X.loc[after].reset_index(drop=True))
    )
    assert not changed.empty, (
        "no post-cutoff feature changed after corrupting the future - the "
        "pipeline may not be reading history at all, which would make the "
        "leakage checks above vacuous"
    )


def test_as_of_cycle_never_changes_included_rows_features(observable):
    """Truncating with as_of_cycle must match building fresh and filtering -
    truncation is a row filter, not a different computation."""
    cycles = observable["cycles"]
    mandates = observable["mandates"]
    merchants = observable["merchants"]

    full = build_features(cycles, mandates, merchants)
    truncated = build_features(cycles, mandates, merchants, as_of_cycle=3)

    keep = full.meta["cycle_number"] <= 3
    pd.testing.assert_frame_equal(
        full.X.loc[keep].reset_index(drop=True),
        truncated.X.reset_index(drop=True),
        check_dtype=False,
    )


# ---------------------------------------------------------------------------
# 4. Payer split disjointness and reproducibility
# ---------------------------------------------------------------------------


def test_train_calibration_test_payer_sets_are_disjoint(observable):
    payer_ids = observable["mandates"]["payer_id"].unique().tolist()
    split = make_payer_split(payer_ids, seed=42)

    assert split.train & split.calibration == frozenset()
    assert split.train & split.test == frozenset()
    assert split.calibration & split.test == frozenset()
    assert split.train | split.calibration | split.test == frozenset(payer_ids)


def test_payer_split_is_reproducible_from_its_seed(observable):
    payer_ids = observable["mandates"]["payer_id"].unique().tolist()
    a = make_payer_split(payer_ids, seed=42)
    b = make_payer_split(payer_ids, seed=42)
    c = make_payer_split(payer_ids, seed=7)

    assert a.train == b.train and a.calibration == b.calibration and a.test == b.test
    assert a.train != c.train or a.test != c.test
