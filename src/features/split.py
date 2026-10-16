"""
Time-series splitting utilities: expanding-window walk-forward validation
with a fully sealed holdout test set.

Why not ``train_test_split``?
-----------------------------
Random shuffling destroys temporal order and lets the model see the future,
inflating scores that collapse in production. Forecast evaluation must always
train on the past and evaluate on the future.

Strategy implemented here
-------------------------
::

    dates:  |<-------------- development ------------->|<-- TEST (sealed) -->|
            |<--- TRAIN (expanding) --->|<- VAL f3 ->|
            |<-------- TRAIN ---------->|<- VAL f2 ->|
            |<----- TRAIN ----->|<- VAL f1 ->|

    * The final ``test_days`` days are removed entirely from cross-validation
      and used exactly once, for the final model.
    * ``n_splits`` validation windows of ``val_days`` days are carved
      backwards from the end of the development period.
    * The training window always starts at the first available date and
      expands forward (fold 1 is the earliest fold).
    * Training and validation never overlap: every fold satisfies
      ``train_end < val_start``.

Usage:
    from src.features import split
    for fold in split.walk_forward_splits(df, n_splits=3, val_days=16):
        train_df, val_df = split.fold_frames(df, fold)
    train, val, test = split.train_val_test_split(df)
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, replace

import pandas as pd

from src import config

log = logging.getLogger("features.split")


# ===========================================================================
# Fold definition
# ===========================================================================
@dataclass(frozen=True)
class Fold:
    """One expanding-window walk-forward split, expressed as date ranges."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp

    def __post_init__(self) -> None:
        if not self.train_start <= self.train_end:
            raise ValueError(f"Fold {self.fold}: train_start > train_end")
        if not self.val_start <= self.val_end:
            raise ValueError(f"Fold {self.fold}: val_start > val_end")
        if not self.train_end < self.val_start:
            raise ValueError(
                f"Fold {self.fold}: train/validation overlap "
                f"({self.train_end.date()} >= {self.val_start.date()})"
            )

    @property
    def train_days(self) -> int:
        """Number of calendar days in the training window."""
        return (self.train_end - self.train_start).days + 1

    @property
    def val_days(self) -> int:
        """Number of calendar days in the validation window."""
        return (self.val_end - self.val_start).days + 1

    def train_mask(
        self, df: pd.DataFrame, date_col: str = config.DATE_COL
    ) -> pd.Series:
        """Boolean mask selecting the training rows of this fold."""
        return (df[date_col] >= self.train_start) & (df[date_col] <= self.train_end)

    def val_mask(
        self, df: pd.DataFrame, date_col: str = config.DATE_COL
    ) -> pd.Series:
        """Boolean mask selecting the validation rows of this fold."""
        return (df[date_col] >= self.val_start) & (df[date_col] <= self.val_end)

    def describe(self) -> str:
        """One-line human-readable summary."""
        return (
            f"fold {self.fold}: train {self.train_start.date()}..{self.train_end.date()} "
            f"({self.train_days}d) | val {self.val_start.date()}..{self.val_end.date()} "
            f"({self.val_days}d)"
        )


# ===========================================================================
# Date helpers
# ===========================================================================
def unique_dates(
    df: pd.DataFrame, date_col: str = config.DATE_COL
) -> pd.DatetimeIndex:
    """Sorted, de-duplicated dates present in the panel."""
    return pd.DatetimeIndex(df[date_col].unique()).sort_values()


def _warn_if_not_daily(
    dates: pd.DatetimeIndex, date_col: str = config.DATE_COL
) -> None:
    """Warn when the calendar has gaps (window sizes are in observations)."""
    if len(dates) < 2:
        return
    gaps = dates.to_series().diff().dropna()
    if (gaps != pd.Timedelta(days=1)).any():
        log.warning(
            "Calendar gaps detected in '%s' — windows are measured in "
            "observations, so a %d-day window may span more calendar days.",
            date_col, len(dates),
        )


def sealed_test_window(
    dates: pd.DatetimeIndex, test_days: int
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last date of the sealed holdout test window."""
    if test_days < 1:
        raise ValueError("test_days must be >= 1")
    if test_days >= len(dates):
        raise ValueError(
            f"test_days={test_days} leaves no development data "
            f"(only {len(dates)} dates available)."
        )
    return dates[-test_days], dates[-1]


# ===========================================================================
# Walk-forward splits
# ===========================================================================
def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 3,
    val_days: int = 16,
    test_days: int = 16,
    min_train_days: int = 90,
    date_col: str = config.DATE_COL,
) -> Iterator[Fold]:
    """
    Yield expanding-window walk-forward folds, earliest first.

    Parameters
    ----------
    n_splits:
        Number of validation folds to carve out of the development period.
    val_days:
        Length of each validation window, in days.
    test_days:
        Length of the sealed holdout window, in days. Never yielded as a fold.
    min_train_days:
        A fold is skipped if its training window would be shorter than this.

    Raises
    ------
    ValueError
        If the panel is too short to carve ``n_splits`` folds plus the sealed
        test window.
    """
    dates = unique_dates(df, date_col)
    _warn_if_not_daily(dates, date_col)

    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    required = test_days + n_splits * val_days + max(min_train_days, 1)
    if len(dates) < required:
        raise ValueError(
            f"Not enough history: {len(dates)} dates available, but "
            f"test_days={test_days} + {n_splits} x val_days={val_days} + "
            f"min_train_days={min_train_days} requires {required}."
        )

    dev = dates[:-test_days]                      # development period
    reversed_folds: list[Fold] = []
    val_end_pos = len(dev) - 1

    for _ in range(n_splits):
        val_start_pos = val_end_pos - val_days + 1
        if val_start_pos < 1:
            break
        train_end_pos = val_start_pos - 1
        if train_end_pos + 1 < min_train_days:
            break

        reversed_folds.append(
            Fold(
                fold=0,
                train_start=dev[0],
                train_end=dev[train_end_pos],
                val_start=dev[val_start_pos],
                val_end=dev[val_end_pos],
            )
        )
        val_end_pos = val_start_pos - 1

    folds = list(reversed(reversed_folds))
    for position, fold in enumerate(folds, start=1):
        yield replace(fold, fold=position)


def fold_frames(
    df: pd.DataFrame,
    fold: Fold,
    date_col: str = config.DATE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialise ``(train_df, val_df)`` for a fold (row slices, no leakage)."""
    return (
        df.loc[fold.train_mask(df, date_col)].copy(),
        df.loc[fold.val_mask(df, date_col)].copy(),
    )


# ===========================================================================
# Final holdout split
# ===========================================================================
def train_val_test_split(
    df: pd.DataFrame,
    val_days: int = 16,
    test_days: int = 16,
    date_col: str = config.DATE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Deterministic final holdout: the most recent validation + test windows.

    Returns ``(train, val, test)`` where::

        test  = last ``test_days`` days            (touched once, at the end)
        val   = ``val_days`` days before the test  (model selection)
        train = everything before ``val``

    Guarantees no date appears in more than one split.
    """
    dates = unique_dates(df, date_col)
    _warn_if_not_daily(dates, date_col)

    if len(dates) < val_days + test_days + 1:
        raise ValueError(
            f"Need more than val_days + test_days dates; got {len(dates)}."
        )

    test_start, _ = sealed_test_window(dates, test_days)
    dev_dates = dates[:-test_days]
    val_start = dev_dates[-val_days]

    train = df.loc[df[date_col] < val_start].copy()
    val = df.loc[(df[date_col] >= val_start) & (df[date_col] < test_start)].copy()
    test = df.loc[df[date_col] >= test_start].copy()

    _assert_disjoint(train, val, test, date_col)
    log.info(
        "Holdout split: train %s..%s (%d rows) | val %s..%s (%d rows) | "
        "test %s..%s (%d rows)",
        train[date_col].min().date(), train[date_col].max().date(), len(train),
        val[date_col].min().date(), val[date_col].max().date(), len(val),
        test[date_col].min().date(), test[date_col].max().date(), len(test),
    )
    return train, val, test


def _assert_disjoint(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str,
) -> None:
    """Fail loudly if any date leaked between splits."""
    train_dates = set(train[date_col])
    val_dates = set(val[date_col])
    test_dates = set(test[date_col])
    if train_dates & val_dates:
        raise AssertionError("train/validation date overlap detected")
    if val_dates & test_dates:
        raise AssertionError("validation/test date overlap detected")
    if train_dates & test_dates:
        raise AssertionError("train/test date overlap detected")
    if train[date_col].max() >= val[date_col].min():
        raise AssertionError("train extends into the validation window")
    if val[date_col].max() >= test[date_col].min():
        raise AssertionError("validation extends into the test window")
