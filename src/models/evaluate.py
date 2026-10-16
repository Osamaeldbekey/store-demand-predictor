"""
Regression metrics and baseline comparison for demand forecasting.

**WAPE is the headline metric.** It is scale-free and business-readable:
``WAPE = sum(|actual - forecast|) / sum(|actual|) * 100`` answers "we are off
by X% of total demand", and unlike MAPE it does not explode when individual
days sell zero units.

RMSE is reported alongside because it is the model's training loss and
penalises large misses; MAE translates directly into "units of error per
store-day".

Usage:
    from src.models import evaluate
    evaluate.metrics(y_true, y_pred)                 # {'rmse','mae','wape','n'}
    evaluate.compare_with_baseline(df, model_pred)   # model vs last-week naive
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger("models.evaluate")

MetricDict = dict[str, float]

# Default naive baseline column: "repeat what sold on the same weekday last week"
DEFAULT_NAIVE_COL = "lag_7"
# Fallbacks used when the primary naive column is NaN (series warm-up)
DEFAULT_NAIVE_FALLBACKS: tuple[str, ...] = ("roll_mean_7", "ema_7")


# ===========================================================================
# Input handling
# ===========================================================================
def _clean_pair(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coerce to float64 arrays and drop rows where either side is NaN.

    Raises
    ------
    ValueError
        On shape mismatch or when no comparable (non-NaN) rows remain.
    """
    true = np.asarray(y_true, dtype="float64")
    pred = np.asarray(y_pred, dtype="float64")
    if true.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {true.shape} vs y_pred {pred.shape}"
        )
    mask = ~(np.isnan(true) | np.isnan(pred))
    true, pred = true[mask], pred[mask]
    if true.size == 0:
        raise ValueError("No comparable rows: all values are NaN.")
    return true, pred


# ===========================================================================
# Individual metrics
# ===========================================================================
def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error — the model's training loss."""
    true, pred = _clean_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error, in units of sales."""
    true, pred = _clean_pair(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def wape(y_true, y_pred) -> float:
    """
    Weighted Absolute Percentage Error, as a percentage of total demand.

    Guard: when total actual demand is zero the ratio is undefined — returns
    ``0.0`` if the forecast is also perfectly zero, otherwise ``inf``.
    """
    true, pred = _clean_pair(y_true, y_pred)
    return _wape(true, pred)


def _wape(true: np.ndarray, pred: np.ndarray) -> float:
    """WAPE on already-cleaned float64 arrays."""
    denominator = float(np.abs(true).sum())
    numerator = float(np.abs(true - pred).sum())
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator * 100.0


def bias(y_true, y_pred) -> float:
    """
    Mean signed error (positive = systematic over-forecast).

    Read it together with MAE: a small MAE with a large bias means the model is
    consistently wrong in *one direction* (which matters for inventory).
    """
    true, pred = _clean_pair(y_true, y_pred)
    return float(np.mean(pred - true))


# ===========================================================================
# Metric bundle
# ===========================================================================
def metrics(y_true, y_pred) -> MetricDict:
    """
    All headline metrics in one pass.

    Returns
    -------
    dict
        ``{'rmse', 'mae', 'wape', 'bias', 'n'}`` where ``n`` is the number of
        comparable rows actually scored (NaN pairs are excluded).
    """
    true, pred = _clean_pair(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(np.mean((true - pred) ** 2))),
        "mae": float(np.mean(np.abs(true - pred))),
        "wape": _wape(true, pred),
        "bias": float(np.mean(pred - true)),
        "n": int(true.size),
    }


def format_metrics(scores: MetricDict, title: str | None = None) -> str:
    """One-line human-readable summary (WAPE first — it is the headline)."""
    prefix = f"{title}: " if title else ""
    return (
        f"{prefix}WAPE={scores['wape']:.2f}% | RMSE={scores['rmse']:.3f} | "
        f"MAE={scores['mae']:.3f} | bias={scores.get('bias', 0.0):+.3f} "
        f"| n={int(scores.get('n', 0)):,}"
    )


# ===========================================================================
# Naive baseline
# ===========================================================================
def naive_predictions(
    df: pd.DataFrame,
    primary: str = DEFAULT_NAIVE_COL,
    fallbacks: tuple[str, ...] = DEFAULT_NAIVE_FALLBACKS,
    default: float = 0.0,
) -> np.ndarray:
    """
    "Last week's sales repeat" baseline, with graceful warm-up fallbacks.

    Uses ``df[primary]`` (``lag_7`` by default) where available, then each
    column in ``fallbacks``, then ``default``. Because the lag columns were
    built leak-safely in Phase 2, this baseline is a fair, information-matched
    opponent for the model.
    """
    if primary in df.columns:
        result = pd.to_numeric(df[primary], errors="coerce").astype("float64")
    else:
        raise KeyError(f"Naive baseline column '{primary}' not found in frame.")

    for column in fallbacks:
        if column in df.columns:
            result = result.fillna(pd.to_numeric(df[column], errors="coerce"))
    return result.fillna(default).to_numpy(dtype="float64")


def compare_with_baseline(
    df: pd.DataFrame,
    predictions: pd.Series | np.ndarray,
    truth: str = config.TARGET_COL,
    naive_col: str = DEFAULT_NAIVE_COL,
    baseline_name: str = "naive_last_week",
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Score the model against the naive baseline on the **same rows**.

    Rows are compared only where the truth, the model prediction and the naive
    column are all present, so neither side gets an unfair NaN advantage.

    Returns
    -------
    pd.DataFrame
        Index ``[model_name, baseline_name]`` with columns
        ``rmse, mae, wape, bias, n_rows``.
    """
    if truth not in df.columns:
        raise KeyError(f"Truth column '{truth}' not found in frame.")
    if naive_col not in df.columns:
        raise KeyError(f"Naive column '{naive_col}' not found in frame.")

    pred = np.asarray(predictions, dtype="float64")
    if pred.shape[0] != len(df):
        raise ValueError(
            f"predictions length {pred.shape[0]} != frame length {len(df)}"
        )

    truth_values = df[truth].to_numpy(dtype="float64")
    naive_values = df[naive_col].to_numpy(dtype="float64")
    mask = ~(np.isnan(truth_values) | np.isnan(pred) | np.isnan(naive_values))
    if not mask.any():
        raise ValueError("No rows with truth, prediction and naive all present.")

    scored = {
        model_name: metrics(truth_values[mask], pred[mask]),
        baseline_name: metrics(truth_values[mask], naive_values[mask]),
    }
    table = pd.DataFrame(scored).T
    table["n_rows"] = table["n"].astype(int)
    table = table.drop(columns=["n"])
    return table[["rmse", "mae", "wape", "bias", "n_rows"]]


def relative_improvement_pct(baseline_score: float, model_score: float) -> float:
    """
    Skill score: percentage reduction in a metric vs the baseline.

    ``+12.5`` means the model reduced the metric by 12.5%. Lower-is-better
    metrics only (WAPE, RMSE, MAE).
    """
    if baseline_score == 0:
        return 0.0
    return (baseline_score - model_score) / baseline_score * 100.0


# ===========================================================================
# Diagnostics by series
# ===========================================================================
def metrics_by_series(
    df: pd.DataFrame,
    predictions: pd.Series | np.ndarray,
    groups: tuple[str, ...] = tuple(config.GROUP_KEYS),
    truth: str = config.TARGET_COL,
) -> pd.DataFrame:
    """
    Per-series metrics, sorted worst WAPE first.

    The first rows are the series the business should worry about — this is
    what feeds the dashboard's "problem stores" view.
    """
    pred = np.asarray(predictions, dtype="float64")
    if pred.shape[0] != len(df):
        raise ValueError(
            f"predictions length {pred.shape[0]} != frame length {len(df)}"
        )

    frame = df[list(groups)].copy()
    frame["_truth"] = df[truth].to_numpy(dtype="float64")
    frame["_pred"] = pred

    rows = []
    for keys, group in frame.groupby(list(groups), observed=True):
        true = group["_truth"].to_numpy(dtype="float64")
        guess = group["_pred"].to_numpy(dtype="float64")
        mask = ~(np.isnan(true) | np.isnan(guess))
        if not mask.any():
            continue
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(groups, keys)),
                     **metrics(true[mask], guess[mask])})

    result = pd.DataFrame(rows)
    return result.sort_values("wape", ascending=False).reset_index(drop=True)
