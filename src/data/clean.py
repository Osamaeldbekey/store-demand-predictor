"""
Clean the raw Kaggle Corporación Favorita dataset into a modeling-ready
Parquet artifact.

Pipeline
--------
1. Load ``train.csv`` with memory-efficient dtypes.
2. Clip negative sales (data-entry artifacts present in the raw file).
3. Filter to the demo scope: ``config.SCOPE_STORE_NBR`` stores, dropping
   ``config.EXCLUDED_FAMILIES`` (long-tail zero-sales noise).
4. Reindex every (store, family) series onto a **contiguous daily calendar**.
   The raw file omits days with zero transactions; those rows are synthesized
   and filled with ``sales=0, onpromotion=0`` (zeros are signal — the model
   must learn that items can sell nothing).
5. Merge store metadata (cluster, city, state, type).
6. Validate invariants and save compressed Parquet (``zstd``).

Memory note
-----------
Loading the full raw ``train.csv`` (~121.9M rows) peaks at roughly 1.3–1.8 GB
of RAM with the dtype map below. Close other heavy applications on small
laptops. The filtered + reindexed output is ~3M rows / well under 150 MB.

Usage
-----
    # from the project root
    python -m src.data.clean
    python -m src.data.clean --raw-dir data/raw --output data/processed/sales_clean.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Make `python src/data/clean.py` work as well as `python -m src.data.clean`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("clean")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mem_mb(df: pd.DataFrame) -> str:
    """Human-readable memory footprint of a DataFrame."""
    return f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB"


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}\n{hint}")


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def load_raw_train(csv_path: Path = config.TRAIN_CSV) -> pd.DataFrame:
    """Load train.csv with memory-efficient dtypes and parsed dates."""
    _require(
        csv_path,
        "Download the dataset first:\n"
        "  kaggle competitions download -c store-sales-time-series-forecasting\n"
        "then unzip it so that data/raw/train.csv exists.",
    )
    df = pd.read_csv(
        csv_path,
        dtype=config.DTYPE_MAP,
        parse_dates=[config.DATE_COL],
    )
    log.info(
        "Loaded raw train.csv: %s rows x %d cols | %s",
        f"{len(df):,}", df.shape[1], _mem_mb(df),
    )
    return df


def load_stores(csv_path: Path = config.STORES_CSV) -> pd.DataFrame:
    """Load stores.csv (store_nbr, cluster, city, state, type)."""
    _require(
        csv_path,
        "stores.csv ships with the same Kaggle competition bundle — unzip it "
        "into data/raw/.",
    )
    stores = pd.read_csv(csv_path, dtype=config.STORES_DTYPE_MAP)
    log.info("Loaded stores.csv: %d rows", len(stores))
    return stores


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------
def clip_negative_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Clip negative sales to zero (raw file contains data-entry artifacts)."""
    n_negative = int((df[config.TARGET_COL] < 0).sum())
    if n_negative:
        log.warning("Clipping %s negative sales values to 0.", f"{n_negative:,}")
    df[config.TARGET_COL] = df[config.TARGET_COL].clip(lower=0)
    return df


# ---------------------------------------------------------------------------
# 3. Scope filtering
# ---------------------------------------------------------------------------
def filter_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the configured stores and drop excluded product families."""
    n_before = len(df)
    df = df[df["store_nbr"].isin(config.SCOPE_STORE_NBR)]
    df = df[~df["family"].isin(config.EXCLUDED_FAMILIES)].copy()
    # Categories of dropped families linger otherwise and bloat memory
    df["family"] = df["family"].cat.remove_unused_categories()
    log.info(
        "Scope filter: %s -> %s rows (%d stores, %d families, %d excluded families)",
        f"{n_before:,}", f"{len(df):,}",
        df["store_nbr"].nunique(), df["family"].nunique(),
        len(config.EXCLUDED_FAMILIES),
    )
    return df


# ---------------------------------------------------------------------------
# 4. Calendar reindexing
# ---------------------------------------------------------------------------
def reindex_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex every (store, family) series onto a contiguous daily calendar.

    The raw Kaggle file has no row for days without transactions. Without
    this step, lag/rolling features in Phase 2 would silently skip days and
    produce wrong windows.
    """
    key_cols = config.GROUP_KEYS + [config.DATE_COL]

    duplicates = int(df.duplicated(subset=key_cols).sum())
    if duplicates:
        log.warning("Dropping %s duplicate (store, family, date) rows.", f"{duplicates:,}")
        df = df.drop_duplicates(subset=key_cols, keep="first")

    # Work with plain strings for the reindex join (categorical level matching
    # against a fresh MultiIndex is fragile); cast back to category later.
    df = df.copy()
    df["family"] = df["family"].astype(str)

    stores = sorted(df["store_nbr"].unique())
    families = sorted(df["family"].unique())
    dates = pd.date_range(df[config.DATE_COL].min(), df[config.DATE_COL].max(), freq="D")
    log.info(
        "Building full calendar: %d stores x %d families x %d days = %s cells",
        len(stores), len(families), len(dates),
        f"{len(stores) * len(families) * len(dates):,}",
    )

    full_index = pd.MultiIndex.from_product(
        [stores, families, dates], names=key_cols,
    )
    df = (
        df.set_index(key_cols)
        .reindex(full_index)
        .reset_index()
    )

    n_created = int(df[config.TARGET_COL].isna().sum())
    log.info("Reindex complete: %s previously-missing rows synthesized.", f"{n_created:,}")
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill synthesized rows: no transaction day = zero sales, no promotion."""
    df[config.TARGET_COL] = df[config.TARGET_COL].fillna(0.0).astype("float32")
    df["onpromotion"] = df["onpromotion"].fillna(0).astype("int8")
    # Synthesized rows have no original Kaggle id → nullable integer
    df[config.ID_COL] = df[config.ID_COL].astype("Int64")
    return df


# ---------------------------------------------------------------------------
# 5. Metadata merge & finalization
# ---------------------------------------------------------------------------
def merge_store_metadata(df: pd.DataFrame, stores_df: pd.DataFrame) -> pd.DataFrame:
    """Attach store attributes (cluster, city, state, type) via left join."""
    df = df.merge(stores_df, on="store_nbr", how="left", validate="many_to_one")
    unmatched = int(df["city"].isna().sum())
    if unmatched:
        raise ValueError(
            f"{unmatched} rows have no matching store metadata — "
            "stores.csv is incomplete for the configured scope."
        )
    log.info("Merged store metadata: cluster, city, state, type.")
    return df


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce dtypes, canonical column order, and deterministic sort order."""
    missing = [c for c in config.CLEAN_COLUMN_ORDER if c not in df.columns]
    if missing:
        raise KeyError(f"finalize(): missing expected columns {missing}")

    df["store_nbr"] = df["store_nbr"].astype("int16")
    df["family"] = df["family"].astype("category")
    df["cluster"] = df["cluster"].astype("int8")
    for col in ("city", "state", "type"):
        df[col] = df[col].astype("category")

    df = df.sort_values(config.GROUP_KEYS + [config.DATE_COL], kind="mergesort")
    df = df.reset_index(drop=True)
    return df[config.CLEAN_COLUMN_ORDER]


# ---------------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> None:
    """Assert the invariants downstream phases depend on. Fail loudly."""
    n_stores = df["store_nbr"].nunique()
    n_families = df["family"].nunique()
    n_days = df[config.DATE_COL].nunique()
    expected_rows = n_stores * n_families * n_days

    assert len(df) == expected_rows, (
        f"Row count {len(df):,} != {n_stores} stores x {n_families} families "
        f"x {n_days} days = {expected_rows:,}"
    )
    assert df[config.TARGET_COL].notna().all(), "Null sales survived cleaning."
    assert df["onpromotion"].notna().all(), "Null onpromotion survived cleaning."
    assert (df[config.TARGET_COL] >= 0).all(), "Negative sales survived cleaning."
    assert df[["cluster", "city", "state", "type"]].notna().all().all(), (
        "Store metadata contains nulls after merge."
    )
    assert df[config.DATE_COL].min() == df[config.DATE_COL].min().normalize(), (
        "Non-normalized dates detected."
    )

    # Every series must cover every day, contiguously, one day at a time.
    grouped = df.groupby(config.GROUP_KEYS, observed=True)
    per_series_days = grouped[config.DATE_COL].nunique()
    assert (per_series_days == n_days).all(), "Some series do not span all days."
    gaps = grouped[config.DATE_COL].diff().dropna()
    assert (gaps == pd.Timedelta(days=1)).all(), "Non-contiguous dates within a series."

    log.info(
        "Validation passed: %d stores x %d families x %d days = %s rows | %s",
        n_stores, n_families, n_days, f"{len(df):,}", _mem_mb(df),
    )


# ---------------------------------------------------------------------------
# 7. Persistence
# ---------------------------------------------------------------------------
def save_parquet(df: pd.DataFrame, path: Path = config.SALES_CLEAN_PARQUET) -> Path:
    """Write the cleaned dataset as compressed Parquet (zstd)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    size_mb = path.stat().st_size / 1e6
    log.info("Saved %s (%.1f MB, zstd)", path, size_mb)
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_clean_dataset(
    raw_dir: Path = config.RAW_DIR,
    output_path: Path = config.SALES_CLEAN_PARQUET,
) -> pd.DataFrame:
    """Run the full raw → clean → parquet pipeline and return the DataFrame."""
    train = load_raw_train(raw_dir / "train.csv")
    stores = load_stores(raw_dir / "stores.csv")

    train = clip_negative_sales(train)
    train = filter_scope(train)
    train = reindex_calendar(train)
    train = fill_missing(train)
    train = merge_store_metadata(train, stores)
    train = finalize(train)

    validate(train)
    save_parquet(train, output_path)
    return train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the raw Favorita dataset into sales_clean.parquet",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=config.RAW_DIR,
        help="Directory containing train.csv and stores.csv (default: data/raw)",
    )
    parser.add_argument(
        "--output", type=Path, default=config.SALES_CLEAN_PARQUET,
        help="Output parquet path (default: data/processed/sales_clean.parquet)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_clean_dataset(raw_dir=args.raw_dir, output_path=args.output)
    log.info("Pipeline finished.")
