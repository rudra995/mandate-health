"""The stable interface the rest of the system depends on.

Nothing outside ``predictor/`` should ever touch a LightGBM ``Booster`` or an
``IsotonicRegression`` object directly. Phase 2's policy engine, Phase 4's
audit record, and Phase 5's eval harness all go through ``RiskModel`` instead,
so the underlying model can change (different hyperparameters, a retrain, even
a different library) without touching any caller.

``explain`` returns SHAP-style per-feature contributions from LightGBM's
native ``pred_contrib`` support - no extra dependency, and the reason
LightGBM was chosen over ``HistGradientBoostingClassifier`` in the first
place (see ``predictor/train.py``). This is not optional decoration: Phase
4's ``DecisionRecord`` schema requires ``top_feature_contributions`` on every
decision, so a model interface without a real explanation path would block an
entire downstream phase.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from predictor.features import CATEGORICAL_FEATURE_COLUMNS, FEATURE_COLUMNS

MODEL_SEMVER = "0.1.0"


@dataclass(frozen=True, slots=True)
class _Metadata:
    feature_columns: list[str]
    categorical_columns: list[str]
    category_levels: dict[str, list[str]]
    best_iteration: int
    content_hash: str

    @property
    def version(self) -> str:
        return f"{MODEL_SEMVER}+{self.content_hash[:8]}"


class RiskModel:
    """Loads, predicts, explains. Nothing here trains - that is train.py's job."""

    def __init__(self, booster: lgb.Booster, calibrator: IsotonicRegression, metadata: _Metadata) -> None:
        self._booster = booster
        self._calibrator = calibrator
        self._metadata = metadata

    # -- construction --------------------------------------------------

    @classmethod
    def fit_from(
        cls,
        booster: lgb.Booster,
        calibrator: IsotonicRegression,
        train_X: pd.DataFrame,
    ) -> "RiskModel":
        """Build a RiskModel from an already-trained booster and calibrator.

        Captures the exact categorical levels seen during training, so a
        later ``predict_proba`` call re-encodes categoricals identically
        regardless of what categories happen to appear in a new batch -
        LightGBM's categorical split logic depends on stable category codes,
        not on category *names*, so this is not optional bookkeeping.
        """
        category_levels = {
            col: [str(c) for c in train_X[col].astype("category").cat.categories]
            for col in CATEGORICAL_FEATURE_COLUMNS
        }
        content_hash = _content_hash(booster.model_to_string(), calibrator, list(FEATURE_COLUMNS))
        metadata = _Metadata(
            feature_columns=list(FEATURE_COLUMNS),
            categorical_columns=list(CATEGORICAL_FEATURE_COLUMNS),
            category_levels=category_levels,
            best_iteration=int(booster.best_iteration) if booster.best_iteration else booster.num_trees(),
            content_hash=content_hash,
        )
        return cls(booster, calibrator, metadata)

    # -- prediction ------------------------------------------------------

    def _prepare(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = set(self._metadata.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"features frame is missing columns: {sorted(missing)}")
        X = features[self._metadata.feature_columns].copy()
        for col in self._metadata.categorical_columns:
            X[col] = pd.Categorical(X[col], categories=self._metadata.category_levels[col])
        return X

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        X = self._prepare(features)
        raw = self._booster.predict(X, num_iteration=self._metadata.best_iteration)
        return np.asarray(self._calibrator.predict(raw))

    def explain(self, features: pd.DataFrame) -> list[list[tuple[str, float]]]:
        """Top-3 feature contributions per row, most influential first.

        Contributions come from LightGBM's native SHAP-style output
        (``pred_contrib=True``): one column per input feature plus a final
        bias/expected-value column, which is dropped here since it is not
        attributable to any one feature. Ranked by absolute contribution -
        a feature that pushed risk *down* is exactly as informative to an
        auditor as one that pushed it up, and CLAUDE.md's example audit
        prose ("top drivers: ...") reads contributions in that spirit.
        """
        X = self._prepare(features)
        contributions = self._booster.predict(
            X, num_iteration=self._metadata.best_iteration, pred_contrib=True
        )
        contributions = np.asarray(contributions)[:, :-1]  # drop trailing bias column

        explanations: list[list[tuple[str, float]]] = []
        for row in contributions:
            ranked = sorted(
                zip(self._metadata.feature_columns, row.tolist()),
                key=lambda pair: abs(pair[1]),
                reverse=True,
            )
            explanations.append(ranked[:3])
        return explanations

    # -- metadata ----------------------------------------------------------

    @property
    def version(self) -> str:
        return self._metadata.version

    @property
    def best_iteration(self) -> int:
        return self._metadata.best_iteration

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """Whole-model feature importance for the diagnostics report.

        Not used for any per-decision explanation - that is ``explain``'s
        job. This is the aggregate view for ``docs/RESULTS.md``.
        """
        values = self._booster.feature_importance(importance_type=importance_type)
        names = self._booster.feature_name()
        return pd.Series(values, index=names).sort_values(ascending=False)

    # -- persistence ---------------------------------------------------

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(directory / "model.txt"))
        joblib.dump(self._calibrator, directory / "calibrator.joblib")
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "feature_columns": self._metadata.feature_columns,
                    "categorical_columns": self._metadata.categorical_columns,
                    "category_levels": self._metadata.category_levels,
                    "best_iteration": self._metadata.best_iteration,
                    "content_hash": self._metadata.content_hash,
                    "model_semver": MODEL_SEMVER,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "RiskModel":
        directory = Path(directory)
        booster = lgb.Booster(model_file=str(directory / "model.txt"))
        calibrator = joblib.load(directory / "calibrator.joblib")
        raw = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        metadata = _Metadata(
            feature_columns=raw["feature_columns"],
            categorical_columns=raw["categorical_columns"],
            category_levels=raw["category_levels"],
            best_iteration=raw["best_iteration"],
            content_hash=raw["content_hash"],
        )
        return cls(booster, calibrator, metadata)


def _content_hash(model_text: str, calibrator: IsotonicRegression, feature_columns: list[str]) -> str:
    """Deterministic hash of everything that defines this model's behaviour.

    Two trainings that produce bit-identical booster text, calibrator knots,
    and feature ordering get the same version string; anything that changes
    the model's predictions changes this hash. Written into every
    ``DecisionRecord`` in Phase 4 so a decision can always be traced back to
    exactly the model that made it.
    """
    hasher = hashlib.sha256()
    hasher.update(model_text.encode("utf-8"))
    hasher.update(json.dumps(feature_columns).encode("utf-8"))
    knots = np.concatenate([calibrator.X_thresholds_, calibrator.y_thresholds_])
    hasher.update(knots.tobytes())
    return hasher.hexdigest()
