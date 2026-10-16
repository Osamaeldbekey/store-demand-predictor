"""
Feature-building CLI: cleaned panel -> engineered feature matrix.

Orchestrates the Phase 2 pipeline:

1. ``ensure_sorted`` — enforce the row order lag features depend on.
2. ``add_calendar_features`` — day/week/month flags, paydays, distances.
3. ``add_holiday_features`` — observed holidays + distance features.
4. ``add_all_temporal_features`` — lags, rolling stats, EMAs (leak-safe).
5. ``validate_features`` — no infinities, no unexpected nulls, correct dtypes.
6. Write ``data/processed/features.parquet``.

Usage
-----
    # from the project root
    python -m src.features.build
    python -m src.features.build --input data/processed/sales_clean.parquet \\
                                 --output data/processed/features.parquet \\
                                 --drop-warmup
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.features import calendar, lags  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("features.build")

# Columns the pipeline is expected to produce / preserve
CALENDAR_COLS = calendar.CALENDAR_FEATURE_COLS
HOLIDAY_COLS = calendar.HOLIDAY_FEATURE_COLS
TEMPORAL_COLS = lags.engineered_feature_columns()
ALL_ENGINEERED_COLS = CALENDAR_COLS + HOLIDAY_COLS + TEMPORAL_COLS

# Float features whose dtype/monotonic sanity we enforce
FLOAT_FEATURE_COLS = [*TEMPORAL_COLS, "days_to_payday", "days_since_payday",
                      "days_to_holiday", "days_since_holiday"]

# Feature-matrix dtype contract.
# String/categorical attributes are stored as pandas `category`; high-cardinality
# numeric identifiers (store_nbr, cluster) stay integer-coded — LightGBM treats
# declared `categorical_feature` columns as categories regardless of dtype, and
# keeping them integer saves memory and Parquet space.
CATEGORY_DTYPE_COLS: list[str] = ["family", "city", "state", "type"]
INTEGER_CODE_COLS: list[str] = ["store_nbr", "cluster"]


# ===========================================================================
# Ordering guard
# ===========================================================================
def is_sorted_by(df: pd.DataFrame, cols: list[str] | tuple[str, ...]) -> bool:
    """
    Lexicographic sortedness check using pandas' own ordering semantics.

    ``MultiIndex.is_monotonic_increasing`` applies the same comparison rules
    as ``sort_values`` (including category ordering), which a hand-rolled
    column comparison would get wrong for categorical keys.
    """
    if len(df) < 2:
        return True
    return bool(pd.MultiIndex.from_frame(df[list(cols)]).is_monotonic_increasing)


def ensure_sorted(
    df: pd.DataFrame,
    keys: tuple[str, ...] = tuple(config.GROUP_KEYS),
    date_col: str = config.DATE_COL,
) -> pd.DataFrame:
    """
    Guarantee the panel is sorted by ``(keys..., date)`` with unique rows.

    Lag/rolling correctness depends on chronological order within each series.
    Sorting is only performed when needed, so an already-sorted frame is not
    copied.
    """
    required = [*keys, date_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"ensure_sorted(): missing columns {missing}")

    if df.duplicated(subset=required).any():
        raise ValueError(
            f"Duplicate {'/'.join(required)} rows found — the panel must have "
            "exactly one row per (store, family, date)."
        )

    if not is_sorted_by(df, required):
        log.warning("Input not sorted by %s — sorting (required for lag safety).",
                    required)
        df = df.sort_values(required, kind="mergesort").reset_index(drop=True)
    return df


def _drop_existing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Make re-runs idempotent by removing previously engineered columns."""
    stale = [col for col in ALL_ENGINEERED_COLS if col in df.columns]
    if stale:
        log.warning("Dropping %d pre-existing engineered columns before rebuild.",
                    len(stale))
        df = df.drop(columns=stale)
    return df


# ===========================================================================
# Validation
# ===========================================================================
def validate_features(
    df: pd.DataFrame,
    target: str = config.TARGET_COL,
    date_col: str = config.DATE_COL,
) -> dict[str, object]:
    """
    Assert the invariants Phase 3 depends on. Fails loudly.

    Checks
    ------
    * every expected column is present and rows are unique per series-day
    * ``date`` is datetime64 and rows are chronologically ordered per series
    * calendar features and ``is_holiday`` contain **no nulls**
    * engineered numeric features contain **no infinities**
    * engineered float features are ``float32``; flags are integer dtypes
    * categorical features are still pandas ``category`` dtype

    Returns
    -------
    dict
        A small report (row count, feature count, per-feature NaN rates) that
        the CLI logs — NaN rates for warm-up features are expected and useful
        to record for the report.
    """
    missing = [col for col in [*config.CLEAN_COLUMN_ORDER, *ALL_ENGINEERED_COLS]
               if col not in df.columns]
    if missing:
        raise KeyError(f"validate_features(): missing columns {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise TypeError(f"'{date_col}' must be datetime64, got {df[date_col].dtype}")

    keys = [*config.GROUP_KEYS, date_col]
    if df.duplicated(subset=keys).any():
        raise ValueError("Duplicate (store, family, date) rows detected.")

    deltas = df.groupby(config.GROUP_KEYS, observed=True)[date_col].diff().dropna()
    if not (deltas == pd.Timedelta(days=1)).all():
        raise ValueError("Series are not chronologically contiguous/sorted.")

    null_flags = [col for col in calendar.NON_NULL_CALENDAR_COLS
                  if df[col].isna().any()]
    if null_flags:
        raise ValueError(f"Null values in non-nullable features: {null_flags}")

    for col in FLOAT_FEATURE_COLS:
        if np.isinf(df[col].to_numpy(dtype="float64", na_value=np.nan)).any():
            raise ValueError(f"Infinite values found in feature '{col}'.")
        if df[col].dtype != np.float32:
            raise TypeError(f"Feature '{col}' should be float32, got {df[col].dtype}")

    for col in calendar._INT8_CALENDAR_COLS:
        if df[col].dtype != np.int8:
            raise TypeError(f"Flag '{col}' should be int8, got {df[col].dtype}")

    for col in CATEGORY_DTYPE_COLS:
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            raise TypeError(
                f"Categorical feature '{col}' lost its category dtype "
                f"({df[col].dtype})."
            )

    for col in INTEGER_CODE_COLS:
        if not pd.api.types.is_integer_dtype(df[col]):
            raise TypeError(
                f"Integer-coded feature '{col}' must stay integer, "
                f"got {df[col].dtype}."
            )

    nan_rates = {
        col: round(float(df[col].isna().mean()), 4) for col in FLOAT_FEATURE_COLS
    }
    report: dict[str, object] = {
        "n_rows": int(len(df)),
        "n_features": len(ALL_ENGINEERED_COLS),
        "n_series": int(df.groupby(config.GROUP_KEYS, observed=True).ngroups),
        "nan_rates": nan_rates,
    }
    log.info(
        "Validated feature matrix: %s rows | %d series | %d engineered features",
        f"{report['n_rows']:,}", report["n_series"], report["n_features"],
    )
    highest_nan = sorted(nan_rates.items(), key=lambda kv: kv[1], reverse=True)[:5]
    log.info("Highest NaN rates (warm-up by design): %s", highest_nan)
    return report


# ===========================================================================
# Orchestration
# ===========================================================================
def build_features(
    sales_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    drop_warmup: bool = False,
    validate: bool = True,
) -> pd.DataFrame:
    """Run the full Phase 2 pipeline on a cleaned panel and return the matrix."""
    df = _drop_existing_features(sales_df)
    df = ensure_sorted(df)

    log.info("Input panel: %s rows | %s .. %s",
             f"{len(df):,}", df[config.DATE_COL].min().date(),
             df[config.DATE_COL].max().date())

    df = calendar.add_calendar_features(df)
    df = calendar.add_holiday_features(df, holidays_df)
    df = lags.add_all_temporal_features(df)

    if drop_warmup:
        before = len(df)
        df = df[df["lag_365"].notna()].reset_index(drop=True)
        log.info("Dropped warm-up rows (lag_365 NaN): %s -> %s",
                 f"{before:,}", f"{len(df):,}")

    if validate:
        validate_features(df)

    return df


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write the feature matrix as compressed Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    log.info("Saved %s (%.1f MB, zstd)", path, path.stat().st_size / 1e6)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Phase 2 feature matrix from sales_clean.parquet",
    )
    parser.add_argument("--input", type=Path, default=config.SALES_CLEAN_PARQUET,
                        help="Cleaned panel (default: data/processed/sales_clean.parquet)")
    parser.add_argument("--output", type=Path, default=config.FEATURES_PARQUET,
                        help="Output feature matrix (default: data/processed/features.parquet)")
    parser.add_argument("--holidays", type=Path, default=config.HOLIDAYS_CSV,
                        help="holidays_events.csv path (default: data/raw/holidays_events.csv)")
    parser.add_argument("--drop-warmup", action="store_true",
                        help="Drop rows lacking a full year of history (lag_365 NaN)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Cleaned panel not found: {args.input}\n"
            "Run Phase 1 first:  python -m src.data.clean"
        )

    sales_df = pd.read_parquet(args.input)
    holidays_df = calendar.load_holidays(args.holidays)

    features = build_features(sales_df, holidays_df, drop_warmup=args.drop_warmup)
    save_parquet(features, args.output)
    log.info("Feature build finished.")


if __name__ == "__main__":
    main()
