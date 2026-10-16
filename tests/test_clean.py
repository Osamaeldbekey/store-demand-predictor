"""
Smoke tests for the cleaning pipeline — run on a tiny synthetic dataset so
they need no Kaggle download.

    pytest tests/test_clean.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.data import clean


# ---------------------------------------------------------------------------
# Synthetic raw data (mimics the Kaggle schema)
# ---------------------------------------------------------------------------
@pytest.fixture()
def mini_raw_dir(tmp_path):
    """
    2 stores x 2 families x 5 days (2017-01-01 .. 2017-01-05) with:
      - store 1 / GROCERIES I missing 2017-01-03 (no-transaction day)
      - a negative sales value on 2017-01-02 (data-entry artifact)
      - a BOOKS row (excluded family)
    """
    dates = pd.date_range("2017-01-01", "2017-01-05", freq="D")
    rows = [
        # store 1 — GROCERIES I (missing 01-03, negative on 01-02)
        {"id": 0, "date": dates[0], "store_nbr": 1, "family": "GROCERIES I", "sales": 10.0, "onpromotion": 0},
        {"id": 1, "date": dates[1], "store_nbr": 1, "family": "GROCERIES I", "sales": -5.0, "onpromotion": 0},
        {"id": 2, "date": dates[3], "store_nbr": 1, "family": "GROCERIES I", "sales": 8.0, "onpromotion": 1},
        {"id": 3, "date": dates[4], "store_nbr": 1, "family": "GROCERIES I", "sales": 12.0, "onpromotion": 0},
        # store 2 — GROCERIES I (complete series)
        {"id": 4, "date": dates[0], "store_nbr": 2, "family": "GROCERIES I", "sales": 20.0, "onpromotion": 0},
        {"id": 5, "date": dates[1], "store_nbr": 2, "family": "GROCERIES I", "sales": 21.0, "onpromotion": 0},
        {"id": 6, "date": dates[2], "store_nbr": 2, "family": "GROCERIES I", "sales": 22.0, "onpromotion": 0},
        {"id": 7, "date": dates[3], "store_nbr": 2, "family": "GROCERIES I", "sales": 23.0, "onpromotion": 0},
        {"id": 8, "date": dates[4], "store_nbr": 2, "family": "GROCERIES I", "sales": 24.0, "onpromotion": 0},
        # store 1 — BOOKS (must be dropped by the scope filter)
        {"id": 9, "date": dates[0], "store_nbr": 1, "family": "BOOKS", "sales": 0.0, "onpromotion": 0},
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)

    stores = pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "cluster": [1, 2],
            "city": ["Quito", "Quito"],
            "state": ["Pichincha", "Pichincha"],
            "type": ["A", "B"],
        }
    )
    stores.to_csv(tmp_path / "stores.csv", index=False)
    return tmp_path


@pytest.fixture()
def cleaned(mini_raw_dir, tmp_path):
    output = tmp_path / "processed" / "sales_clean.parquet"
    df = clean.build_clean_dataset(raw_dir=mini_raw_dir, output_path=output)
    return df, output


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_full_grid_is_synthesized(cleaned):
    """2 in-scope stores x 1 surviving family x 5 days = 10 rows."""
    df, _ = cleaned
    assert len(df) == 10
    assert df["store_nbr"].nunique() == 2
    assert df["family"].nunique() == 1


def test_excluded_family_dropped(cleaned):
    df, _ = cleaned
    assert "BOOKS" not in set(df["family"].astype(str))


def test_missing_day_zero_filled(cleaned):
    """The 2017-01-03 gap for store 1 becomes sales=0, onpromotion=0."""
    df, _ = cleaned
    gap = df[(df["store_nbr"] == 1) & (df["date"] == "2017-01-03")]
    assert len(gap) == 1
    assert gap.iloc[0]["sales"] == 0.0
    assert gap.iloc[0]["onpromotion"] == 0
    assert pd.isna(gap.iloc[0][config.ID_COL])  # synthesized row has no Kaggle id


def test_negative_sales_clipped(cleaned):
    df, _ = cleaned
    assert (df["sales"] >= 0).all()
    assert df[(df["store_nbr"] == 1) & (df["date"] == "2017-01-02")].iloc[0]["sales"] == 0.0


def test_dtypes_enforced(cleaned):
    df, _ = cleaned
    assert df["sales"].dtype == "float32"
    assert df["onpromotion"].dtype == "int8"
    assert df["store_nbr"].dtype == "int16"
    assert df["cluster"].dtype == "int8"
    assert str(df[config.ID_COL].dtype) == "Int64"
    assert isinstance(df["family"].dtype, pd.CategoricalDtype)


def test_store_metadata_merged(cleaned):
    df, _ = cleaned
    store2 = df[df["store_nbr"] == 2].iloc[0]
    assert store2["city"] == "Quito"
    assert store2["type"] == "B"
    assert not df[["cluster", "city", "state", "type"]].isna().any().any()


def test_no_nulls_and_contiguous_dates(cleaned):
    df, _ = cleaned
    assert df[["sales", "onpromotion"]].notna().all().all()
    deltas = df.groupby(["store_nbr", "family"], observed=True)["date"].diff().dropna()
    assert (deltas == pd.Timedelta(days=1)).all()


def test_parquet_written_and_reloadable(cleaned):
    df, output = cleaned
    assert output.exists()
    reloaded = pd.read_parquet(output)
    assert len(reloaded) == len(df)
    assert list(reloaded.columns) == list(config.CLEAN_COLUMN_ORDER)
