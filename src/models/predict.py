"""
Recursive multi-horizon forecasting for the Store Demand Predictor.

The problem
-----------
A model trained on the feature matrix predicts ``sales[T]`` from features built
out of ``sales[T-1], sales[T-2], ...``. To forecast day ``T+1`` we therefore
need ``sales[T]`` — which we do not have. Recursive (a.k.a. "iterated") one-step
forecasting solves this by predicting day by day and feeding each prediction
back into the history as if it were an observation.

How consistency is guaranteed
-----------------------------
Rather than re-implementing lag/rolling/EMA maths (and risking an off-by-one
that silently degrades accuracy), this module **reuses the Phase 2 feature
functions** on a working frame that grows one day at a time:

    history + [T+1, NaN]  ->  recompute features  ->  predict T+1
                          ->  write prediction into sales
    history + [T+1..T+2]  ->  recompute features  ->  predict T+2   ...

Every feature for day ``d`` depends only on ``d-1`` and earlier, so appending a
row never changes an earlier row's features. The values the model sees during
recursion are therefore *identical* to the batch-computed features it was
trained on — there is a test that asserts exactly this.

Horizon independence
--------------------
Forward calendar distances (``days_to_payday``) are derived from the *frame's*
date axis in Phase 2. The working frame is therefore padded with a fixed
``CALENDAR_BUFFER_DAYS`` (45) beyond the horizon, which both populates those
distances and makes them horizon-independent: a 7-day forecast is identical to
the first 7 days of a 30-day forecast.

Promotions
----------
``onpromotion`` is unknown for a blind horizon. By default future days are
assumed **not** promoted; pass ``future_onpromotion`` (scalar, ``dict`` or
``date``-indexed ``Series``) to run a promotional scenario.

Usage:
    from src.models import predict
    outcome = predict.forecast_series(model, features, store_nbr=1, family="GROCERIES I",
                                      horizon=30)
    outcome.predictions            # date, store_nbr, family, prediction
    predict.forecast_all(model, features, horizon=30)   # every series
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.features import calendar, lags
from src.models import train

log = logging.getLogger("models.predict")

# Static / structural columns required to rebuild a series' feature rows.
REQUIRED_BASE_COLS: tuple[str, ...] = (
    config.DATE_COL,
    "store_nbr",
    "family",
    config.TARGET_COL,
    "onpromotion",
    "cluster",
    "city",
    "state",
    "type",
)

# Columns cast back to their original dtype when synthesising future rows.
_INT_COLS: tuple[str, ...] = ("store_nbr", "cluster", "onpromotion")
_CATEGORY_COLS: tuple[str, ...] = ("family", "city", "state", "type")

# Payday distances are derived from the frame's date axis (Phase 2 behaviour:
# ``payday_events = dates[is_payday]``), so a forecast frame must extend past
# the last predicted day or ``days_to_payday`` is NaN. The longest payday gap is
# ~15 days (the 15th and the last day of the month), so extending by 45 days
# guarantees every horizon sees the *same* forward distance features — which is
# what makes a 7-day forecast identical to the first 7 days of a 30-day one.
CALENDAR_BUFFER_DAYS: int = 45


@dataclass
class ForecastResult:
    """Recursive forecast for a single (store, family) series."""

    predictions: pd.DataFrame
    horizon: int
    store_nbr: int
    family: str
    features: pd.DataFrame | None = None

    def __len__(self) -> int:
        return len(self.predictions)

    def truncate(self, horizon: int) -> pd.DataFrame:
        """First ``horizon`` days — identical to a shorter forecast run."""
        return self.predictions.head(horizon).reset_index(drop=True)


# ===========================================================================
# Helpers
# ===========================================================================
def _resolve_model(model) -> object:
    """Accept either a fitted estimator or a loaded artifact dict."""
    if isinstance(model, dict):
        if "model" not in model:
            raise ValueError("Artifact dict has no 'model' key.")
        return model["model"]
    if not hasattr(model, "predict"):
        raise TypeError("model must be a fitted estimator or a model artifact dict.")
    return model


def _resolve_features(model, reference: pd.DataFrame) -> list[str]:
    """Feature list: from the artifact if available, else from the frame."""
    if isinstance(model, dict) and model.get("features"):
        return list(model["features"])
    return train.feature_columns(reference)


def _materialise_onpromotion(dates: pd.DatetimeIndex, schedule) -> np.ndarray:
    """Future promotion flags from a scalar, mapping or date-indexed Series."""
    if schedule is None:
        return np.zeros(len(dates), dtype="int8")
    if isinstance(schedule, pd.Series):
        aligned = schedule.copy()
        aligned.index = pd.DatetimeIndex(aligned.index)
        return (
            aligned.reindex(dates).fillna(0).astype("int8").to_numpy()
        )
    if isinstance(schedule, dict):
        return np.array(
            [int(schedule.get(pd.Timestamp(day), 0)) for day in dates], dtype="int8"
        )
    return np.full(len(dates), int(schedule), dtype="int8")


def _cast_like(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """
    Restore the reference dtypes after concatenation.

    Casting categorical columns back to the *exact* reference dtype is
    critical: LightGBM maps ``family``/``city``/... through their category
    codes, so a re-encoded column would silently shift every categorical split.
    """
    for column in _INT_COLS:
        if column in frame.columns and column in reference.columns:
            frame[column] = frame[column].astype(reference[column].dtype)
    for column in _CATEGORY_COLS:
        if column in frame.columns and column in reference.columns:
            frame[column] = frame[column].astype(reference[column].dtype)
    return frame


def _build_working_frame(
    group: pd.DataFrame,
    horizon: int,
    future_onpromotion,
    date_col: str = config.DATE_COL,
    buffer_days: int = CALENDAR_BUFFER_DAYS,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    History rows plus unobserved future rows covering the forecast horizon.

    The date axis is extended ``buffer_days`` beyond the horizon so that
    frame-derived calendar distances (``days_to_payday``) are populated and,
    crucially, **horizon-independent**. Only the first ``horizon`` dates are
    ever scored; buffer rows stay NaN and are ignored by the causal temporal
    features.

    Returns
    -------
    (work, forecast_dates)
        ``work`` is the padded working frame; ``forecast_dates`` are the
        ``horizon`` dates to predict, in chronological order.
    """
    reference = group.sort_values(date_col).reset_index(drop=True)
    last = reference.iloc[-1]

    base_cols = [column for column in reference.columns
                 if column in REQUIRED_BASE_COLS]
    missing = [column for column in REQUIRED_BASE_COLS if column not in base_cols]
    if missing:
        raise KeyError(f"Series is missing required columns: {missing}")

    total = int(horizon) + int(buffer_days)
    all_future_dates = pd.date_range(
        pd.Timestamp(last[date_col]) + pd.Timedelta(days=1),
        periods=total,
        freq="D",
    )
    forecast_dates = all_future_dates[:horizon]

    # Tile the final observation so every column keeps its exact dtype, then
    # overwrite the date and blank the target. (Tiling avoids a dtype-inference
    # warning that a hand-built all-NaN frame would trigger on concat.)
    future = reference[base_cols].iloc[[-1] * total].reset_index(drop=True)
    future[date_col] = all_future_dates
    future["onpromotion"] = _materialise_onpromotion(
        all_future_dates, future_onpromotion
    )
    future[config.TARGET_COL] = np.full(
        total, np.nan, dtype=reference[config.TARGET_COL].dtype
    )

    with warnings.catch_warnings():
        # Deliberate all-NaN target column for the unobserved future rows.
        warnings.simplefilter("ignore", FutureWarning)
        work = pd.concat([reference[base_cols], future], ignore_index=True)
    return _cast_like(work, reference), pd.DatetimeIndex(forecast_dates)


# ===========================================================================
# Single-series recursive forecast
# ===========================================================================
def forecast_series(
    model,
    features: pd.DataFrame,
    store_nbr: int,
    family: str,
    horizon: int = config.FORECAST_HORIZON,
    holidays: pd.DataFrame | None = None,
    feature_names: list[str] | None = None,
    future_onpromotion=None,
    clip_non_negative: bool = config.CLIP_NON_NEGATIVE,
    return_features: bool = True,
    date_col: str = config.DATE_COL,
    buffer_days: int = CALENDAR_BUFFER_DAYS,
) -> ForecastResult:
    """
    Recursively forecast ``horizon`` days for one (store, family) series.

    Parameters
    ----------
    model:
        Fitted estimator or a loaded artifact dict (uses ``['model']``).
    features:
        Historical feature matrix (Phase 2 output) covering the series.
    horizon:
        Number of future days to predict.
    holidays:
        Holiday table for future-date features; ``None`` disables them
        (``is_holiday = 0``).
    future_onpromotion:
        Promotion scenario for the horizon (see module docstring).
    clip_non_negative:
        Clip predictions at zero (sales cannot be negative).
    buffer_days:
        Extra unobserved days appended to the working frame so forward calendar
        distances (``days_to_payday``) are populated and horizon-independent.
        Leave at the default unless you know a shorter margin suffices.

    Returns
    -------
    ForecastResult
        ``.predictions`` holds ``date, store_nbr, family, prediction``;
        ``.features`` (when requested) holds the exact rows the model scored,
        which makes the forecast auditable.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    estimator = _resolve_model(model)
    names = feature_names or _resolve_features(model, features)

    mask = (features["store_nbr"] == store_nbr) & (features["family"] == family)
    group = features.loc[mask]
    if group.empty:
        raise KeyError(
            f"No history for store_nbr={store_nbr}, family='{family}'."
        )

    unknown = [column for column in names if column not in group.columns]
    if unknown:
        raise KeyError(f"Feature columns absent from history frame: {unknown}")

    holiday_table = holidays if holidays is not None else calendar.empty_holidays()

    # --- structural features: deterministic, computed once for all future days
    work, future_dates = _build_working_frame(
        group, horizon, future_onpromotion, date_col, buffer_days
    )
    work = calendar.add_calendar_features(work, date_col=date_col)
    work = calendar.add_holiday_features(work, holiday_table, date_col=date_col)

    if len(future_dates) != horizon:
        raise RuntimeError(
            f"Expected {horizon} future rows, built {len(future_dates)}."
        )

    predictions: list[dict] = []

    for day in future_dates:
        # Recompute the temporal block on the growing history. Features for
        # `day` depend only on earlier rows, so this is exact, not approximate.
        work = lags.add_all_temporal_features(work, target=config.TARGET_COL)

        row = work.loc[work[date_col] == day]
        if row.empty:
            raise RuntimeError(f"No feature row built for {day.date()}.")

        y_hat = float(estimator.predict(row[names])[0])
        if clip_non_negative:
            y_hat = max(0.0, y_hat)

        # Feed the prediction back in as pseudo-history for the next step.
        work.loc[work[date_col] == day, config.TARGET_COL] = y_hat
        predictions.append(
            {
                date_col: pd.Timestamp(day),
                "store_nbr": store_nbr,
                "family": family,
                "prediction": y_hat,
            }
        )

    log.debug("Forecast %s/%s: %d days", store_nbr, family, horizon)

    return ForecastResult(
        predictions=pd.DataFrame(predictions),
        horizon=horizon,
        store_nbr=store_nbr,
        family=family,
        features=work if return_features else None,
    )


# ===========================================================================
# Batch forecast
# ===========================================================================
def forecast_all(
    model,
    features: pd.DataFrame,
    horizon: int = config.FORECAST_HORIZON,
    stores: list[int] | None = None,
    families: list[str] | None = None,
    holidays: pd.DataFrame | None = None,
    future_onpromotion=None,
    clip_non_negative: bool = config.CLIP_NON_NEGATIVE,
    log_every: int = 50,
) -> pd.DataFrame:
    """
    Forecast every selected series and return one tidy prediction frame.

    The loop is deliberately simple (one series at a time): a dashboard
    pre-computes this batch offline, and clarity beats cleverness here.
    """
    subset = features
    if stores is not None:
        subset = subset[subset["store_nbr"].isin(stores)]
    if families is not None:
        subset = subset[subset["family"].isin(families)]

    groups = subset[config.GROUP_KEYS].drop_duplicates()
    if groups.empty:
        raise ValueError("No series matched the requested store/family filters.")

    names = _resolve_features(model, features)
    frames: list[pd.DataFrame] = []

    for position, row in enumerate(groups.itertuples(index=False), start=1):
        outcome = forecast_series(
            model,
            subset,
            store_nbr=int(row.store_nbr),
            family=str(row.family),
            horizon=horizon,
            holidays=holidays,
            feature_names=names,
            future_onpromotion=future_onpromotion,
            clip_non_negative=clip_non_negative,
            return_features=False,
        )
        frames.append(outcome.predictions)
        if log_every and position % log_every == 0:
            log.info("Forecast %d/%d series...", position, len(groups))

    result = pd.concat(frames, ignore_index=True)
    log.info("Forecast complete: %d series x %d days = %s rows",
             len(groups), horizon, f"{len(result):,}")
    return result


def truncate_horizon(predictions: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Return the first ``horizon`` days per series of a longer forecast.

    Because the recursion is self-contained, an N-day forecast is a valid
    prefix of any longer one — so the 7/14/30-day dashboard horizons never need
    three separate runs.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    ordered = predictions.sort_values([*config.GROUP_KEYS, config.DATE_COL])
    return (
        ordered.groupby(config.GROUP_KEYS, observed=True, group_keys=False)
        .head(horizon)
        .reset_index(drop=True)
    )
