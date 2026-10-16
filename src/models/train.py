"""
Global LightGBM training for the Store Demand Predictor.

One global model is trained across every (store, family) series rather than a
model per series: it pools statistical strength, handles cold-start series and
ships as a single artifact.

Categorical handling
--------------------
``family``, ``city``, ``state``, ``type`` are pandas ``category`` columns;
``store_nbr`` and ``cluster`` stay integer-coded (Phase 2 contract). All six are
declared to LightGBM as categorical features, so it can split on membership
rather than on an arbitrary numeric ordering, and no target encoding (a classic
leakage source) is needed.

Usage:
    from src.models import train
    result = train.train_model(train_df, val_df)     # early stopping on RMSE
    train.save_model(config.MODEL_ARTIFACT, result)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.models import evaluate

log = logging.getLogger("models.train")

# Keys of the saved artifact; load_model validates their presence.
ARTIFACT_KEYS: tuple[str, ...] = (
    "model",
    "features",
    "categorical_features",
    "params",
    "target",
    "best_iteration",
    "metrics",
    "trained_at",
)


# ===========================================================================
# Feature plumbing
# ===========================================================================
def feature_columns(
    df: pd.DataFrame,
    exclude: tuple[str, ...] | None = None,
) -> list[str]:
    """
    Feature column names: everything except id / date / target.

    Keeping this derived from the frame (rather than hard-coded) means adding a
    feature in Phase 2 is automatically picked up here.
    """
    excluded = set(config.NON_FEATURE_COLS if exclude is None else exclude)
    return [column for column in df.columns if column not in excluded]


def categorical_features(features: list[str]) -> list[str]:
    """Subset of ``features`` that LightGBM should treat as categorical."""
    available = set(features)
    return [name for name in config.CATEGORICAL_FEATURES if name in available]


def resolve_params(params: dict | None = None) -> dict:
    """Merge caller overrides on top of the project defaults (no mutation)."""
    merged = dict(config.LGBM_PARAMS)
    if params:
        merged.update(params)
    return merged


def clip_predictions(predictions, clip_non_negative: bool = config.CLIP_NON_NEGATIVE):
    """Clip regression output at zero — sales cannot be negative."""
    array = np.asarray(predictions, dtype="float64")
    return np.clip(array, 0.0, None) if clip_non_negative else array


# ===========================================================================
# Training result
# ===========================================================================
@dataclass
class TrainingResult:
    """Everything Phase 4/5 need to know about a trained model."""

    model: lgb.LGBMRegressor
    features: list[str]
    categorical_features: list[str]
    params: dict
    best_iteration: int
    metrics: dict[str, float] = field(default_factory=dict)
    train_rows: int = 0
    val_rows: int = 0
    train_start: pd.Timestamp | None = None
    train_end: pd.Timestamp | None = None
    val_start: pd.Timestamp | None = None
    val_end: pd.Timestamp | None = None
    target: str = config.TARGET_COL

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict with the trained model, applying the non-negativity clip."""
        return clip_predictions(self.model.predict(df[self.features]))

    def to_artifact(self, extra: dict | None = None) -> dict:
        """Serialisable artifact dictionary for Joblib."""
        artifact = {
            "model": self.model,
            "features": list(self.features),
            "categorical_features": list(self.categorical_features),
            "params": dict(self.params),
            "target": self.target,
            "best_iteration": int(self.best_iteration),
            "metrics": dict(self.metrics),
            "train_rows": int(self.train_rows),
            "val_rows": int(self.val_rows),
            "train_window": _window(self.train_start, self.train_end),
            "val_window": _window(self.val_start, self.val_end),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "library": f"lightgbm=={lgb.__version__}",
        }
        if extra:
            artifact.update(extra)
        return artifact


def _window(start: pd.Timestamp | None, end: pd.Timestamp | None) -> list[str] | None:
    """ISO date pair for the metadata block."""
    if start is None or end is None:
        return None
    return [pd.Timestamp(start).date().isoformat(), pd.Timestamp(end).date().isoformat()]


# ===========================================================================
# Training
# ===========================================================================
def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    params: dict | None = None,
    num_boost_round: int = config.LGBM_NUM_BOOST_ROUND,
    early_stopping_rounds: int | None = config.LGBM_EARLY_STOPPING_ROUNDS,
    features: list[str] | None = None,
    target: str = config.TARGET_COL,
    clip_non_negative: bool = config.CLIP_NON_NEGATIVE,
) -> TrainingResult:
    """
    Train the global LightGBM regressor.

    Parameters
    ----------
    train_df, val_df:
        Feature frames (Phase 2 output). Validation enables early stopping on
        validation RMSE; pass ``None`` to train a fixed number of rounds.
    params:
        Parameter overrides merged onto ``config.LGBM_PARAMS``.
    num_boost_round:
        Maximum number of trees (a ceiling when early stopping is active).
    early_stopping_rounds:
        Stop when validation RMSE does not improve for this many rounds.
    features:
        Explicit feature list; defaults to every non-id/date/target column.

    Returns
    -------
    TrainingResult
        Model plus metadata, including validation metrics (NaN-safe).
    """
    if train_df.empty:
        raise ValueError("train_df is empty — nothing to train on.")

    resolved = resolve_params(params)
    # Default exclusion is config.NON_FEATURE_COLS (id, date, target) — the
    # `exclude` argument of feature_columns() replaces that set wholesale, so
    # call it with no override here.
    feats = features or feature_columns(train_df)
    missing = [column for column in feats if column not in train_df.columns]
    if missing:
        raise KeyError(f"Training frame is missing features: {missing}")
    cats = categorical_features(feats)

    X_train = train_df[feats]
    y_train = train_df[target].to_numpy(dtype="float64")

    # Split LightGBM's own eval switch from the sklearn constructor kwargs.
    booster_params = {
        key: value
        for key, value in resolved.items()
        if key not in {"metric", "num_boost_round", "early_stopping_rounds", "n_estimators"}
    }
    model = lgb.LGBMRegressor(
        n_estimators=int(num_boost_round),
        eval_metric="rmse",
        **booster_params,
    )

    callbacks = [lgb.log_evaluation(period=0)]
    eval_set = None
    if val_df is not None and not val_df.empty:
        eval_set = [(val_df[feats], val_df[target].to_numpy(dtype="float64"))]
        if early_stopping_rounds:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=int(early_stopping_rounds),
                                   verbose=False)
            )

    log.info(
        "Training global LightGBM: %s rows, %d features (%d categorical)%s",
        f"{len(train_df):,}", len(feats), len(cats),
        f", early stopping @ {early_stopping_rounds}" if eval_set and early_stopping_rounds else "",
    )
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        categorical_feature=cats,
        callbacks=callbacks,
    )

    best_iteration = int(getattr(model, "best_iteration_", 0) or 0)
    scores: dict[str, float] = {}
    if eval_set is not None:
        val_pred = clip_predictions(model.predict(val_df[feats]), clip_non_negative)
        scores = evaluate.metrics(val_df[target], val_pred)
        log.info("Validation " + evaluate.format_metrics(scores))

    return TrainingResult(
        model=model,
        features=list(feats),
        categorical_features=cats,
        params=resolved,
        best_iteration=best_iteration,
        metrics=scores,
        train_rows=int(len(train_df)),
        val_rows=int(len(val_df)) if val_df is not None else 0,
        train_start=_first(train_df),
        train_end=_last(train_df),
        val_start=_first(val_df),
        val_end=_last(val_df),
        target=target,
    )


def _first(df: pd.DataFrame | None) -> pd.Timestamp | None:
    if df is None or df.empty or config.DATE_COL not in df.columns:
        return None
    return pd.Timestamp(df[config.DATE_COL].min())


def _last(df: pd.DataFrame | None) -> pd.Timestamp | None:
    if df is None or df.empty or config.DATE_COL not in df.columns:
        return None
    return pd.Timestamp(df[config.DATE_COL].max())


# ===========================================================================
# Feature importance
# ===========================================================================
def feature_importance(result: TrainingResult, top_n: int = 25) -> pd.DataFrame:
    """Gain-based feature importance, descending — useful for the report."""
    booster = result.model.booster_
    importance = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    )
    importance = importance.sort_values("gain", ascending=False).reset_index(drop=True)
    total = importance["gain"].sum()
    importance["gain_pct"] = (
        importance["gain"] / total * 100.0 if total > 0 else 0.0
    )
    return importance.head(top_n)


# ===========================================================================
# Persistence
# ===========================================================================
def save_model(
    path: Path | str,
    result: TrainingResult,
    extra: dict | None = None,
) -> Path:
    """Persist a :class:`TrainingResult` (with metadata) via Joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = result.to_artifact(extra=extra)
    joblib.dump(artifact, path, compress=3)
    log.info("Saved model artifact: %s (%.1f MB)", path,
             path.stat().st_size / 1e6)
    return path


def load_model(path: Path | str = config.MODEL_ARTIFACT) -> dict:
    """
    Load and validate a saved artifact.

    Raises
    ------
    FileNotFoundError
        If the artifact does not exist (with a hint on how to create it).
    ValueError
        If the file is not a valid Store Demand Predictor artifact.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}\n"
            "Train one first:  python -m src.models.build"
        )
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError(f"{path} is not a Store Demand Predictor artifact.")
    missing = [key for key in ARTIFACT_KEYS if key not in artifact]
    if missing:
        raise ValueError(f"Artifact {path} is missing keys: {missing}")
    return artifact


def artifact_summary(artifact: dict) -> str:
    """Human-readable one-block summary of a loaded artifact."""
    metrics = artifact.get("metrics") or {}
    metric_line = (
        evaluate.format_metrics(metrics) if metrics else "no validation metrics"
    )
    return (
        f"artifact trained_at={artifact.get('trained_at')} | "
        f"library={artifact.get('library')} | "
        f"best_iteration={artifact.get('best_iteration')} | "
        f"{len(artifact.get('features', []))} features | {metric_line}"
    )
