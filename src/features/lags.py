"""
Temporal (lag / rolling / EMA) feature engineering for the Store Demand
Predictor.

LEAKAGE CONTRACT — read before editing
--------------------------------------
A model that predicts ``sales[T]`` may only use information available at
``T-1`` or earlier.

* **Lags** — ``lag_k[T] = sales[T-k]``. Using ``groupby.shift(k)`` is safe by
  construction: ``shift`` only moves values backwards in time.
* **Rolling statistics** — computed on the series **after** ``shift(1)``, so a
  window covering ``T-w .. T-1`` never includes the target day ``T``::
      series.groupby(group).shift(1) -> .rolling(w).mean()
  Computing ``series.rolling(w).mean()`` instead would leak ``sales[T]`` into
  its own predictor and produce beautiful, useless validation scores.
* **EMA** — same rule: the shift makes the "current" observation ``T-1``.

Warm-up
-------
Every feature is ``NaN`` for the first observations of a series (``k`` days
for ``lag_k``, ``w`` for a window ``w``). NaNs are **intentional** and left in
place: LightGBM handles missing values natively, whereas filling them with 0
would inject a false "no demand" signal. ``build.py --drop-warmup`` trims them
if wanted.

Input ordering
--------------
These functions require the panel sorted by ``config.GROUP_KEYS + [date]``
(guaranteed by the Phase 1 cleaning pipeline); ``build.py`` enforces this
defensively via ``ensure_sorted``.

Usage:
    from src.features import lags
    df = lags.add_all_temporal_features(df)
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from src import config

log = logging.getLogger("features.lags")

# --- Feature configuration -------------------------------------------------
LAG_DAYS: tuple[int, ...] = (7, 14, 21, 28, 30, 365)
ROLLING_WINDOWS: tuple[int, ...] = (7, 14, 28)
EMA_SPANS: tuple[int, ...] = (7, 14)


# ===========================================================================
# Feature-name helpers (single source of truth for build / tests / modelling)
# ===========================================================================
def lag_feature_names(lags: Sequence[int] = LAG_DAYS) -> list[str]:
    """``lag_7``, ``lag_14`` ..."""
    return [f"lag_{lag}" for lag in lags]


def rolling_feature_names(windows: Sequence[int] = ROLLING_WINDOWS) -> list[str]:
    """``roll_mean_7``, ``roll_std_7`` ... for every configured window."""
    names: list[str] = []
    for window in windows:
        names += [f"roll_mean_{window}", f"roll_std_{window}"]
    return names


def ema_feature_names(spans: Sequence[int] = EMA_SPANS) -> list[str]:
    """``ema_7``, ``ema_14`` ..."""
    return [f"ema_{span}" for span in spans]


def engineered_feature_columns(
    lags: Sequence[int] = LAG_DAYS,
    windows: Sequence[int] = ROLLING_WINDOWS,
    spans: Sequence[int] = EMA_SPANS,
) -> list[str]:
    """All temporal feature names, in generation order."""
    return (
        lag_feature_names(lags)
        + rolling_feature_names(windows)
        + ema_feature_names(spans)
    )


# ===========================================================================
# Lag features
# ===========================================================================
def add_lag_features(
    df: pd.DataFrame,
    target: str = config.TARGET_COL,
    keys: Sequence[str] = tuple(config.GROUP_KEYS),
    lags: Sequence[int] = LAG_DAYS,
) -> pd.DataFrame:
    """
    Add ``lag_k`` columns: the target value ``k`` days earlier, per series.

    Safe by construction — ``shift(k)`` only reaches backwards in time. The
    first ``k`` rows of each series are ``NaN``.
    """
    grouped = df.groupby(list(keys), observed=True)[target]
    for lag in lags:
        df[f"lag_{lag}"] = grouped.shift(lag).astype("float32")

    log.info("Added %d lag features: %s", len(lags),
             ", ".join(lag_feature_names(lags)))
    return df


# ===========================================================================
# Rolling statistics
# ===========================================================================
def _leak_safe_base(
    df: pd.DataFrame,
    target: str,
    keys: Sequence[str],
) -> pd.Series:
    """``target`` shifted by one day within each series — the only legal base
    for any rolling / EMA statistic used to predict the current day."""
    return df.groupby(list(keys), observed=True)[target].shift(1)


def add_rolling_features(
    df: pd.DataFrame,
    target: str = config.TARGET_COL,
    keys: Sequence[str] = tuple(config.GROUP_KEYS),
    windows: Sequence[int] = ROLLING_WINDOWS,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Add rolling mean and std over ``windows``, computed on the ``shift(1)``
    series (see the module leakage contract).

    Parameters
    ----------
    min_periods:
        Minimum non-NaN observations required for a value. Defaults to the
        window length, so a ``roll_mean_28`` appears only once 28 prior days
        genuinely exist. Lower it to trade history requirements for coverage.
    """
    base = _leak_safe_base(df, target, keys)
    grouped = base.groupby([df[k] for k in keys], observed=True)

    for window in windows:
        min_obs = window if min_periods is None else min_periods
        df[f"roll_mean_{window}"] = grouped.transform(
            lambda s, w=window, m=min_obs: s.rolling(w, min_periods=m).mean()
        ).astype("float32")
        df[f"roll_std_{window}"] = grouped.transform(
            lambda s, w=window, m=min_obs: s.rolling(w, min_periods=m).std()
        ).astype("float32")

    log.info("Added %d rolling features: %s", 2 * len(windows),
             ", ".join(rolling_feature_names(windows)))
    return df


# ===========================================================================
# Exponential moving averages
# ===========================================================================
def add_ema_features(
    df: pd.DataFrame,
    target: str = config.TARGET_COL,
    keys: Sequence[str] = tuple(config.GROUP_KEYS),
    spans: Sequence[int] = EMA_SPANS,
) -> pd.DataFrame:
    """
    Add exponentially weighted means (``ema_s``) over the ``shift(1)`` series.

    EMA weights the most recent observation most heavily, giving the model a
    responsive "recent level" signal that complements the flat rolling mean.
    """
    base = _leak_safe_base(df, target, keys)
    grouped = base.groupby([df[k] for k in keys], observed=True)

    for span in spans:
        df[f"ema_{span}"] = grouped.transform(
            lambda s, sp=span: s.ewm(span=sp, min_periods=sp).mean()
        ).astype("float32")

    log.info("Added %d EMA features: %s", len(spans),
             ", ".join(ema_feature_names(spans)))
    return df


# ===========================================================================
# Convenience entry point
# ===========================================================================
def add_all_temporal_features(
    df: pd.DataFrame,
    target: str = config.TARGET_COL,
    keys: Sequence[str] = tuple(config.GROUP_KEYS),
) -> pd.DataFrame:
    """Run the full temporal feature stack: lags, rolling stats, EMAs."""
    df = add_lag_features(df, target=target, keys=keys)
    df = add_rolling_features(df, target=target, keys=keys)
    df = add_ema_features(df, target=target, keys=keys)
    return df
