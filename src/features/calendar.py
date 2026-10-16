"""
Calendar and holiday feature engineering for the Store Demand Predictor.

Two families of features are produced:

1. **Pure calendar features** derived from the date itself (day of week,
   month, ISO week, weekend flag, Ecuadorian payday flags and payday
   distances). These are deterministic and known for any future date, so they
   are always safe to use in a forecasting model.

2. **Holiday features** derived from ``data/raw/holidays_events.csv``.
   Holidays are scheduled events published in advance, so distance features
   such as ``days_to_holiday`` are legitimately available at prediction time
   and do **not** constitute leakage.

Holiday policy
--------------
The Kaggle file encodes moved celebrations with ``transferred == True`` rows
that mark the *original* date of the celebration (which is a normal working
day), while the row for the actual observed day carries
``transferred == False``. We therefore drop ``transferred`` rows by default.

``type`` values treated as non-working days:
``Holiday``, ``Transfer``, ``Bridge``, ``Additional``.
``Work Day`` (an explicit normal working day) and ``Event`` (cultural, not a
day off) are excluded.

Usage:
    from src.features import calendar
    holidays = calendar.load_holidays()
    df = calendar.add_calendar_features(df)
    df = calendar.add_holiday_features(df, holidays)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make `python -m src.features.calendar` / direct execution both work.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402

log = logging.getLogger("features.calendar")

# --- Holiday taxonomy ------------------------------------------------------
HOLIDAY_TYPES: tuple[str, ...] = ("Holiday", "Transfer", "Bridge", "Additional")
NON_HOLIDAY_TYPES: tuple[str, ...] = ("Work Day", "Event")

# --- Feature column contracts (consumed by build.validate_features) ---------
CALENDAR_FEATURE_COLS: list[str] = [
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "month",
    "year",
    "week_of_year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_payday",
    "days_to_payday",
    "days_since_payday",
]

HOLIDAY_FEATURE_COLS: list[str] = [
    "is_holiday",
    "days_to_holiday",
    "days_since_holiday",
]

# Columns that must never contain NaN (distances legitimately may, see below).
NON_NULL_CALENDAR_COLS: list[str] = [
    col for col in CALENDAR_FEATURE_COLS if not col.startswith("days_")
] + ["is_holiday"]

_INT8_CALENDAR_COLS: tuple[str, ...] = (
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_payday",
    "is_holiday",
)


# ===========================================================================
# Distance helper
# ===========================================================================
def _distance_features(
    dates: pd.DatetimeIndex,
    events: pd.DatetimeIndex | np.ndarray,
) -> tuple[pd.Series, pd.Series]:
    """
    Calendar distance to the next and previous event date.

    Returns two float32 Series indexed by ``dates``:

    * ``days_to``   — days until the next event (0 on an event day, NaN when
      no event exists at or after the date).
    * ``days_since`` — days since the previous event (0 on an event day, NaN
      when no event exists at or before the date).

    Both are computed with a sorted search, so cost is O(n log k).
    """
    dates = pd.DatetimeIndex(dates)
    raw_events = pd.DatetimeIndex(np.asarray(events)).values
    event_ts = np.sort(np.unique(raw_events)) if raw_events.size else raw_events

    n = len(dates)
    days_to = np.full(n, np.nan, dtype="float64")
    days_since = np.full(n, np.nan, dtype="float64")

    if event_ts.size:
        date_values = dates.values
        # For each date, its insertion position among the sorted event dates.
        next_pos = np.searchsorted(event_ts, date_values, side="left")
        prev_pos = np.searchsorted(event_ts, date_values, side="right") - 1

        has_next = next_pos < event_ts.size
        has_prev = prev_pos >= 0

        days_to[has_next] = (
            event_ts[next_pos[has_next]] - date_values[has_next]
        ) / np.timedelta64(1, "D")
        days_since[has_prev] = (
            date_values[has_prev] - event_ts[prev_pos[has_prev]]
        ) / np.timedelta64(1, "D")

    idx = pd.Index(dates, name=config.DATE_COL)
    return (
        pd.Series(days_to.astype("float32"), index=idx, name="days_to"),
        pd.Series(days_since.astype("float32"), index=idx, name="days_since"),
    )


def _date_index(dates: pd.Series | pd.DatetimeIndex | np.ndarray) -> pd.DatetimeIndex:
    """Return a sorted, de-duplicated DatetimeIndex of the input dates."""
    values = np.unique(pd.DatetimeIndex(np.asarray(dates)).values)
    return pd.DatetimeIndex(values)


# ===========================================================================
# Pure calendar features
# ===========================================================================
def build_calendar_features(
    dates: pd.Series | pd.DatetimeIndex | np.ndarray,
) -> pd.DataFrame:
    """
    Build the calendar feature block for every unique date in ``dates``.

    The result is indexed by date so it can be joined onto any panel with a
    single ``df.join(features, on="date")`` call.
    """
    dates = _date_index(dates)
    idx = pd.Index(dates, name=config.DATE_COL)

    dayofweek = dates.dayofweek
    day = dates.day
    is_month_end = dates.is_month_end

    feat = pd.DataFrame(index=idx)
    feat["day_of_week"] = dayofweek.to_numpy().astype("int8")
    feat["day_of_month"] = day.to_numpy().astype("int8")
    feat["day_of_year"] = dates.dayofyear.to_numpy().astype("int16")
    feat["month"] = dates.month.to_numpy().astype("int8")
    feat["year"] = dates.year.to_numpy().astype("int16")
    feat["week_of_year"] = dates.isocalendar().week.to_numpy().astype("int8")
    feat["is_weekend"] = np.asarray(dayofweek >= 5).astype("int8")
    feat["is_month_start"] = np.asarray(dates.is_month_start).astype("int8")
    feat["is_month_end"] = np.asarray(is_month_end).astype("int8")

    # Ecuadorian paydays: the 15th and the last calendar day of the month.
    is_payday = np.asarray(day == 15) | np.asarray(is_month_end)
    feat["is_payday"] = is_payday.astype("int8")

    payday_events = dates[is_payday]
    days_to, days_since = _distance_features(dates, payday_events)
    feat["days_to_payday"] = days_to.to_numpy()
    feat["days_since_payday"] = days_since.to_numpy()

    return feat


def add_calendar_features(
    df: pd.DataFrame,
    date_col: str = config.DATE_COL,
) -> pd.DataFrame:
    """
    Join pure calendar features onto ``df`` (in place, returns the frame).

    Calendar features are date-level, so they are computed once per unique date
    and broadcast to all (store, family) rows — far cheaper than a groupby.
    """
    feat = build_calendar_features(df[date_col])
    out = df.join(feat, on=date_col, how="left")

    # Guarantee integer dtypes even if a date unexpectedly failed to match.
    for col in _INT8_CALENDAR_COLS:
        if col in out.columns:
            out[col] = out[col].astype("int8")
    for col in ("day_of_year", "year"):
        out[col] = out[col].astype("int16")

    log.info("Added %d calendar features: %s", len(CALENDAR_FEATURE_COLS),
             ", ".join(CALENDAR_FEATURE_COLS))
    return out


# ===========================================================================
# Holiday features
# ===========================================================================
def empty_holidays() -> pd.DataFrame:
    """An empty holiday table with the exact schema of ``holidays_events.csv``.

    Useful when forecasting beyond the downloaded file, or in tests: every date
    collapses to ``is_holiday = 0`` with NaN distance features.
    """
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "type": pd.Series(dtype="object"),
            "locale": pd.Series(dtype="object"),
            "locale_name": pd.Series(dtype="object"),
            "description": pd.Series(dtype="object"),
            "transferred": pd.Series(dtype=bool),
        }
    )


def load_holidays(csv_path: Path | str = config.HOLIDAYS_CSV) -> pd.DataFrame:
    """Load ``holidays_events.csv`` with parsed dates and category dtypes."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Holiday file not found: {csv_path}\n"
            "It ships with the Kaggle competition bundle — unzip "
            "holidays_events.csv into data/raw/."
        )
    df = pd.read_csv(csv_path, parse_dates=["date"])
    for col in ("type", "locale", "locale_name"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    log.info("Loaded holidays: %d rows spanning %s .. %s",
             len(df), df["date"].min().date(), df["date"].max().date())
    return df


def _as_bool(series: pd.Series) -> pd.Series:
    """Coerce a 'transferred' column (bool or 'True'/'False' text) to bool."""
    if series.dtype == bool:
        return series
    return (
        series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    )


def filter_holidays(
    holidays: pd.DataFrame,
    locale: str | None = "National",
    drop_transferred: bool = True,
    types: tuple[str, ...] | None = HOLIDAY_TYPES,
) -> pd.DataFrame:
    """
    Reduce the raw holiday table to *observed, non-working* days.

    Parameters
    ----------
    locale:
        ``"National"`` (default), ``"Regional"``, ``"Local"``, or ``None`` to
        keep every scope. National holidays are the safest default because the
        store scope spans many cities.
    drop_transferred:
        Drop rows where the celebration was moved. The kept row is the
        actually-observed day.
    types:
        Holiday types counted as non-working days.
    """
    out = holidays.copy()

    if drop_transferred and "transferred" in out.columns:
        transferred = _as_bool(out["transferred"])
        n_dropped = int(transferred.sum())
        out = out[~transferred]
        if n_dropped:
            log.info("Dropped %d transferred (moved) holiday rows.", n_dropped)

    if locale is not None and "locale" in out.columns:
        out = out[out["locale"].astype(str).str.casefold() == str(locale).casefold()]

    if types is not None and "type" in out.columns:
        out = out[out["type"].astype(str).isin(types)]

    return out.reset_index(drop=True)


def build_holiday_features(
    dates: pd.Series | pd.DatetimeIndex | np.ndarray,
    holidays: pd.DataFrame,
    locale: str | None = "National",
    drop_transferred: bool = True,
    types: tuple[str, ...] | None = HOLIDAY_TYPES,
) -> pd.DataFrame:
    """
    Build holiday features for every unique date in ``dates``.

    Returns a date-indexed frame with ``is_holiday``, ``days_to_holiday`` and
    ``days_since_holiday``. Distance columns are ``NaN`` only when no holiday
    exists on the relevant side of the date — LightGBM consumes that natively.
    """
    dates = _date_index(dates)
    idx = pd.Index(dates, name=config.DATE_COL)

    filtered = filter_holidays(
        holidays, locale=locale, drop_transferred=drop_transferred, types=types
    )
    event_values = np.unique(filtered["date"].to_numpy()) if len(filtered) else np.array([], dtype="datetime64[ns]")

    is_holiday = np.isin(dates.values, event_values).astype("int8")
    days_to, days_since = _distance_features(dates, event_values)

    feat = pd.DataFrame(index=idx)
    feat["is_holiday"] = is_holiday
    feat["days_to_holiday"] = days_to.to_numpy()
    feat["days_since_holiday"] = days_since.to_numpy()
    return feat


def add_holiday_features(
    df: pd.DataFrame,
    holidays: pd.DataFrame,
    date_col: str = config.DATE_COL,
    locale: str | None = "National",
    drop_transferred: bool = True,
) -> pd.DataFrame:
    """Join holiday features onto ``df`` (in place, returns the frame)."""
    feat = build_holiday_features(
        df[date_col], holidays, locale=locale, drop_transferred=drop_transferred
    )
    out = df.join(feat, on=date_col, how="left")
    out["is_holiday"] = out["is_holiday"].astype("int8")

    n_holidays = int(out.loc[~out[date_col].duplicated(), "is_holiday"].sum())
    log.info("Added %d holiday features: %s (%d observed holidays in range)",
             len(HOLIDAY_FEATURE_COLS), ", ".join(HOLIDAY_FEATURE_COLS), n_holidays)
    return out
