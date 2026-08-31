"""Payer-level and temporal splits.

**Primary split: by payer.** A payer appearing in training must never appear
in test or calibration. Splitting by row would let the model memorise a
payer's own outcome history through leakage of a different kind - not the
hidden-field leakage ``test_leakage.py`` guards against, but the ordinary
statistical kind where the same entity's rows land on both sides of a split.
Both are disqualifying for a project whose whole claim rests on the split
being real.

**Three-way, not two-way.** Calibration (Phase 1's isotonic regression) needs
its own held-out payer set, disjoint from both the set the model is trained
on and the set held back for final evaluation. Fitting the calibration map on
the training set would calibrate the model to its own overfit; fitting it on
the test set would spend the one set meant for an honest final number.

Everything here is a pure function of ``payer_ids`` and ``seed`` - no I/O, no
global state - so a later phase can reproduce the exact same split from the
seed alone without re-deriving it from this module's internals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Train / calibration / test payer share. 60/20/20 is a judgement call: large
# enough test and calibration sets to make Brier score and the per-decile
# calibration check stable at n=400 payers, without starving the model of
# training payers.
TRAIN_FRACTION = 0.60
CALIBRATION_FRACTION = 0.20
# TEST_FRACTION is implied: 1 - TRAIN_FRACTION - CALIBRATION_FRACTION


@dataclass(frozen=True, slots=True)
class PayerSplit:
    seed: int
    train: frozenset[str]
    calibration: frozenset[str]
    test: frozenset[str]

    def __post_init__(self) -> None:
        pairs = (
            (self.train, self.calibration),
            (self.train, self.test),
            (self.calibration, self.test),
        )
        for a, b in pairs:
            if a & b:
                raise ValueError("payer split sets are not disjoint")


def make_payer_split(payer_ids: list[str] | tuple[str, ...], seed: int) -> PayerSplit:
    """Deterministic 60/20/20 train/calibration/test split by payer.

    Same ``seed`` and same ``payer_ids`` (any order) always produce the same
    three sets - the shuffle is seeded by a fresh, independent
    ``np.random.default_rng`` rather than any process- or call-order-dependent
    state.
    """
    unique_ids = sorted(set(payer_ids))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)

    n = len(shuffled)
    n_train = int(round(n * TRAIN_FRACTION))
    n_calib = int(round(n * CALIBRATION_FRACTION))

    train = frozenset(shuffled[:n_train])
    calibration = frozenset(shuffled[n_train : n_train + n_calib])
    test = frozenset(shuffled[n_train + n_calib :])

    return PayerSplit(seed=seed, train=train, calibration=calibration, test=test)


def temporal_holdout_cycle(n_cycles: int) -> int:
    """Which cycle number is the last one used for the temporal-holdout fit.

    With six cycles this is a sanity check, not a headline result - see
    CLAUDE.md section on the split strategy. Train on cycles
    ``<= temporal_holdout_cycle(n_cycles)``, evaluate on the rest.
    """
    if n_cycles < 3:
        raise ValueError("temporal holdout needs at least 3 cycles to be meaningful")
    return n_cycles - 2
