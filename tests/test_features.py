"""
Phase 2 test suite — calendar, holiday, lag/rolling/EMA and split behaviour.

The leakage tests are the important ones: they prove that no feature used to
predict ``sales[T]`` depends on ``sales[T]`` or on any later observation.

    pytest tests/test_features.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import build, calendar, lags, split

# ===========================================================================
# Fixtures / helpers
# ===========================================================================
START = "2017-01-01"
STORES = (1, 2)
FAMILIES = ("GROCERIES I", "BEVERAGES")


def store_offset(store_nbr: int) -> float:
    """Deterministic per-store shift so group isolation is observable."""
    return float((store_nbr - 1) * 1000)


def make_panel(n_days: int = 60, stores=STORES, families=FAMILIES) -> pd.DataFrame:
    """
    Synthetic cleaned panel with ``sales[i] = i + store_offset`` per series,
    so expected lag/rolling values have closed forms.
    """
    dates = pd.date_range(START, periods=n_days, freq="D")
    rows = []
    for store in stores:
        for family in families:
            for i, date in enumerate(dates):
                rows.append(
                    {
                        "id": i,
                        "date": date,
                        "store_nbr": store,
                        "family": family,
                        "sales": float(i) + store_offset(store),
                        "onpromotion": 0,
                        "cluster": 1,
                        "city": "Quito",
                        "state": "Pichincha",
                        "type": "A",
                    }
                )
    df = pd.DataFrame(rows)
    df["store_nbr"] = df["store_nbr"].astype("int16")
    df["family"] = df["family"].astype("category")
    df["sales"] = df["sales"].astype("float32")
    df["onpromotion"] = df["onpromotion"].astype("int8")
    df["cluster"] = df["cluster"].astype("int8")
    for col in ("city", "state", "type"):
        df[col] = df[col].astype("category")
    return df.sort_values([*config.GROUP_KEYS, config.DATE_COL]).reset_index(drop=True)


def make_holidays(rows: list[dict] | None = None) -> pd.DataFrame:
    """Holiday table mirroring the Kaggle schema (empty by default)."""
    if rows is None:
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
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def group_frame(df: pd.DataFrame, store: int, family: str) -> pd.DataFrame:
    """One series, in date order."""
    mask = (df["store_nbr"] == store) & (df["family"] == family)
    return df.loc[mask].sort_values(config.DATE_COL).reset_index(drop=True)


@pytest.fixture()
def panel() -> pd.DataFrame:
    return make_panel(n_days=60)


@pytest.fixture()
def features(panel: pd.DataFrame) -> pd.DataFrame:
    return build.build_features(panel, make_holidays())


# ===========================================================================
# 1. Lag features — exact shift correctness
# ===========================================================================
def test_lag_names_cover_configured_days():
    assert lags.lag_feature_names() == [f"lag_{d}" for d in lags.LAG_DAYS]
    assert lags.engineered_feature_columns() == (
        [f"lag_{d}" for d in lags.LAG_DAYS]
        + [name for w in lags.ROLLING_WINDOWS
           for name in (f"roll_mean_{w}", f"roll_std_{w}")]
        + [f"ema_{s}" for s in lags.EMA_SPANS]
    )


def test_lag_7_equals_sales_seven_days_earlier(features):
    grp = group_frame(features, store=1, family="GROCERIES I")
    offset = store_offset(1)
    sales = grp["sales"].to_numpy(dtype="float64")
    lag7 = grp["lag_7"].to_numpy(dtype="float64")

    assert np.isnan(lag7[:7]).all(), "first 7 rows must be warm-up NaN"
    for i in range(7, len(grp)):
        assert lag7[i] == sales[i - 7] == i - 7 + offset


def test_all_lags_follow_their_horizon(features):
    grp = group_frame(features, store=1, family="GROCERIES I")
    sales = grp["sales"].to_numpy(dtype="float64")
    for lag in lags.LAG_DAYS:
        if lag >= len(grp):
            continue
        actual = grp[f"lag_{lag}"].to_numpy(dtype="float64")
        assert np.isnan(actual[:lag]).all()
        np.testing.assert_allclose(actual[lag:], sales[:-lag], rtol=1e-6)


def test_lags_do_not_bleed_between_groups(features):
    g1 = group_frame(features, store=1, family="GROCERIES I")
    g2 = group_frame(features, store=2, family="GROCERIES I")
    # store 2 is offset by +1000; its own lag_7 must come from its own history
    assert g2["lag_7"].iloc[10] == g2["sales"].iloc[3]
    assert g2["lag_7"].iloc[10] != g1["sales"].iloc[3]


# ===========================================================================
# 2. Rolling statistics — closed-form values + leak safety
# ===========================================================================
def test_rolling_mean_matches_closed_form(features):
    grp = group_frame(features, store=1, family="GROCERIES I")
    expected = grp["sales"].rolling(7).mean().shift(1)  # shifted by exactly 1 day
    actual = grp["roll_mean_7"]
    np.testing.assert_allclose(
        actual.to_numpy(dtype="float64"),
        expected.to_numpy(dtype="float64"),
        equal_nan=True,
        rtol=1e-6,
    )
    # sales[i] = i -> mean of previous 7 values = i - 4
    assert actual.iloc[7] == pytest.approx(3.0)
    assert actual.iloc[10] == pytest.approx(6.0)


def test_rolling_std_is_constant_for_linear_series(features):
    grp = group_frame(features, store=1, family="GROCERIES I")
    expected_std = float(pd.Series([0, 1, 2, 3, 4, 5, 6]).std())  # ddof=1
    assert np.isnan(grp["roll_std_7"].iloc[:7]).all()
    np.testing.assert_allclose(
        grp["roll_std_7"].iloc[7:].to_numpy(dtype="float64"),
        expected_std,
        rtol=1e-5,
    )


def test_rolling_windows_respect_min_periods(features):
    grp = group_frame(features, store=1, family="GROCERIES I")
    for window in lags.ROLLING_WINDOWS:
        assert np.isnan(grp[f"roll_mean_{window}"].iloc[:window]).all()
        assert np.isfinite(grp[f"roll_mean_{window}"].iloc[window])


def test_rolling_mean_excludes_target_day_values():
    """Rolling stats must not move when sales[T] changes."""
    panel = make_panel(n_days=40)
    base = build.build_features(panel.copy(), make_holidays())

    mutated = panel.copy()
    mutated.loc[mutated[config.DATE_COL] == mutated[config.DATE_COL].max(), "sales"] = 12345.0
    changed = build.build_features(mutated, make_holidays())

    rolling_cols = lags.rolling_feature_names() + lags.ema_feature_names()
    last = base[config.DATE_COL].max()
    a = base.loc[base[config.DATE_COL] == last, rolling_cols]
    b = changed.loc[changed[config.DATE_COL] == last, rolling_cols]
    np.testing.assert_allclose(
        a.to_numpy(dtype="float64", na_value=np.nan),
        b.to_numpy(dtype="float64", na_value=np.nan),
        equal_nan=True,
    )


def test_no_future_leakage_for_any_engineered_feature(panel):
    """Mutating all sales from a cutoff onward must leave earlier rows intact."""
    engineered = lags.engineered_feature_columns()
    base = build.build_features(panel.copy(), make_holidays())

    cutoff = pd.Timestamp(START) + pd.Timedelta(days=30)
    mutated = panel.copy()
    mutated.loc[mutated[config.DATE_COL] >= cutoff, "sales"] = 999999.0
    changed = build.build_features(mutated, make_holidays())

    before = base[config.DATE_COL] < cutoff
    np.testing.assert_allclose(
        base.loc[before, engineered].to_numpy(dtype="float64", na_value=np.nan),
        changed.loc[before, engineered].to_numpy(dtype="float64", na_value=np.nan),
        equal_nan=True,
    )


# ===========================================================================
# 3. EMA features
# ===========================================================================
def test_ema_matches_independent_per_group_computation(features):
    for store in STORES:
        for family in FAMILIES:
            grp = group_frame(features, store=store, family=family)
            expected = (
                grp["sales"].shift(1).ewm(span=7, min_periods=7).mean()
            ).to_numpy(dtype="float64")
            np.testing.assert_allclose(
                grp["ema_7"].to_numpy(dtype="float64"),
                expected,
                equal_nan=True,
                rtol=1e-5,
            )


def test_ema_warmup_boundary(features):
    grp = group_frame(features, store=1, family="GROCERIES I")
    assert np.isnan(grp["ema_7"].iloc[:7]).all()
    assert np.isfinite(grp["ema_7"].iloc[7])


# ===========================================================================
# 4. Calendar features
# ===========================================================================
def test_calendar_flags_and_paydays():
    feat = calendar.build_calendar_features(pd.date_range("2017-01-01", "2017-01-31"))

    assert feat.loc["2017-01-01", "day_of_week"] == 6          # Sunday
    assert feat.loc["2017-01-01", "is_weekend"] == 1
    assert feat.loc["2017-01-02", "is_weekend"] == 0
    assert feat.loc["2017-01-01", "is_month_start"] == 1
    assert feat.loc["2017-01-31", "is_month_end"] == 1
    assert feat.loc["2017-01-15", "is_payday"] == 1
    assert feat.loc["2017-01-31", "is_payday"] == 1            # month end
    assert feat.loc["2017-01-14", "is_payday"] == 0
    assert feat.loc["2017-01-15", "month"] == 1
    assert 1 <= feat.loc["2017-01-01", "week_of_year"] <= 53


def test_payday_distance_features():
    feat = calendar.build_calendar_features(pd.date_range("2017-01-01", "2017-01-31"))

    assert feat.loc["2017-01-10", "days_to_payday"] == 5       # -> 15th
    assert feat.loc["2017-01-30", "days_to_payday"] == 1       # -> 31st
    assert feat.loc["2017-01-20", "days_since_payday"] == 5    # <- 15th
    assert feat.loc["2017-01-16", "days_since_payday"] == 1
    assert feat.loc["2017-01-15", "days_to_payday"] == 0
    assert feat.loc["2017-01-15", "days_since_payday"] == 0


# ===========================================================================
# 5. Holiday features
# ===========================================================================
def holiday_rows():
    return [
        {"date": "2017-01-06", "type": "Holiday", "locale": "National",
         "locale_name": "Ecuador", "description": "observed", "transferred": False},
        {"date": "2017-01-09", "type": "Work Day", "locale": "National",
         "locale_name": "Ecuador", "description": "working", "transferred": False},
        {"date": "2017-01-16", "type": "Holiday", "locale": "Local",
         "locale_name": "Quito", "description": "local", "transferred": False},
        {"date": "2017-01-20", "type": "Holiday", "locale": "National",
         "locale_name": "Ecuador", "description": "moved-from", "transferred": True},
    ]


def test_holiday_filtering_rules():
    hol = make_holidays(holiday_rows())
    nat = calendar.filter_holidays(hol, locale="National", drop_transferred=True)
    assert list(nat["date"]) == [pd.Timestamp("2017-01-06")]

    kept = calendar.filter_holidays(hol, locale="National", drop_transferred=False)
    assert set(kept["date"]) == {pd.Timestamp("2017-01-06"), pd.Timestamp("2017-01-20")}

    loc = calendar.filter_holidays(hol, locale="Local")
    assert list(loc["date"]) == [pd.Timestamp("2017-01-16")]


def test_holiday_flags_and_distances():
    dates = pd.date_range("2017-01-01", "2017-01-31")
    feat = calendar.build_holiday_features(dates, make_holidays(holiday_rows()))

    assert feat.loc["2017-01-06", "is_holiday"] == 1
    assert feat.loc["2017-01-05", "is_holiday"] == 0
    assert feat.loc["2017-01-09", "is_holiday"] == 0   # Work Day
    assert feat.loc["2017-01-16", "is_holiday"] == 0   # Local, filtered out
    assert feat.loc["2017-01-20", "is_holiday"] == 0   # transferred

    assert feat.loc["2017-01-05", "days_to_holiday"] == 1
    assert feat.loc["2017-01-06", "days_to_holiday"] == 0
    assert feat.loc["2017-01-07", "days_since_holiday"] == 1
    assert feat.loc["2017-01-08", "days_since_holiday"] == 2
    assert np.isnan(feat.loc["2017-01-05", "days_since_holiday"])
    assert np.isnan(feat.loc["2017-01-07", "days_to_holiday"])


def test_holiday_features_handle_empty_table():
    dates = pd.date_range("2017-01-01", "2017-01-10")
    feat = calendar.build_holiday_features(dates, make_holidays())
    assert (feat["is_holiday"] == 0).all()
    assert feat["days_to_holiday"].isna().all()


# ===========================================================================
# 6. Walk-forward splitting
# ===========================================================================
def test_walk_forward_no_overlap_and_expanding():
    panel = make_panel(n_days=200)
    folds = list(split.walk_forward_splits(panel, n_splits=3, val_days=16, test_days=16))
    dates = split.unique_dates(panel)

    assert len(folds) == 3
    for fold in folds:
        assert fold.train_start == dates[0]           # expanding window
        assert fold.train_end < fold.val_start        # no overlap
        assert fold.val_days == 16

        train_df, val_df = split.fold_frames(panel, fold)
        assert train_df[config.DATE_COL].max() < val_df[config.DATE_COL].min()
        assert not (set(train_df[config.DATE_COL]) & set(val_df[config.DATE_COL]))
        assert val_df[config.DATE_COL].nunique() == 16

    # validation windows step backwards, contiguously, without overlap
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.val_end < later.val_start
        assert earlier.train_days < later.train_days


def test_sealed_test_window_is_excluded_from_all_folds():
    panel = make_panel(n_days=200)
    dates = split.unique_dates(panel)
    test_start, test_end = split.sealed_test_window(dates, test_days=16)

    assert test_end == dates[-1]
    assert (dates >= test_start).sum() == 16

    for fold in split.walk_forward_splits(panel, n_splits=3, val_days=16, test_days=16):
        assert fold.val_end < test_start
        assert fold.train_end < test_start


def test_walk_forward_rejects_insufficient_history():
    panel = make_panel(n_days=50)
    with pytest.raises(ValueError, match="Not enough history"):
        list(split.walk_forward_splits(panel, n_splits=3, val_days=16,
                                       test_days=16, min_train_days=90))


def test_train_val_test_split_is_disjoint():
    panel = make_panel(n_days=200)
    train, val, test = split.train_val_test_split(panel, val_days=16, test_days=16)
    col = config.DATE_COL

    assert train[col].nunique() == 200 - 32
    assert val[col].nunique() == 16
    assert test[col].nunique() == 16
    assert train[col].max() < val[col].min()
    assert val[col].max() < test[col].min()
    assert not (set(train[col]) & set(val[col]) & set(test[col]))
    assert not (set(train[col]) & set(val[col]))
    assert not (set(val[col]) & set(test[col]))


def test_fold_rejects_overlapping_windows():
    with pytest.raises(ValueError, match="overlap"):
        split.Fold(
            fold=1,
            train_start=pd.Timestamp("2017-01-01"),
            train_end=pd.Timestamp("2017-01-20"),
            val_start=pd.Timestamp("2017-01-15"),
            val_end=pd.Timestamp("2017-01-30"),
        )


# ===========================================================================
# 7. build.py orchestration & validation
# ===========================================================================
def test_build_features_produces_expected_columns(features):
    for col in build.ALL_ENGINEERED_COLS:
        assert col in features.columns, f"missing engineered column {col}"
    assert len(build.ALL_ENGINEERED_COLS) == (
        len(lags.LAG_DAYS) + 2 * len(lags.ROLLING_WINDOWS) + len(lags.EMA_SPANS)
        + len(calendar.CALENDAR_FEATURE_COLS) + len(calendar.HOLIDAY_FEATURE_COLS)
    )


def test_build_features_validation_report(features):
    report = build.validate_features(features)
    assert report["n_rows"] == len(features)
    assert report["n_series"] == len(STORES) * len(FAMILIES)
    assert report["nan_rates"]["lag_7"] == pytest.approx(7 / 60, abs=1e-3)


def test_build_features_is_idempotent(panel):
    once = build.build_features(panel.copy(), make_holidays())
    twice = build.build_features(once.copy(), make_holidays())
    pd.testing.assert_frame_equal(once, twice)


def test_validate_rejects_infinite_values(features):
    broken = features.copy()
    broken.loc[broken.index[0], "lag_7"] = np.inf
    with pytest.raises(ValueError, match="Infinite values"):
        build.validate_features(broken)


def test_validate_rejects_missing_column(features):
    broken = features.drop(columns=["roll_mean_7"])
    with pytest.raises(KeyError):
        build.validate_features(broken)


def test_ensure_sorted_rejects_duplicates(panel):
    dup = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        build.ensure_sorted(dup)


def test_ensure_sorted_reorders_shuffled_input(panel):
    shuffled = panel.sample(frac=1.0, random_state=config.RANDOM_SEED)
    assert not build.is_sorted_by(shuffled, [*config.GROUP_KEYS, config.DATE_COL])
    ordered = build.ensure_sorted(shuffled)
    assert build.is_sorted_by(ordered, [*config.GROUP_KEYS, config.DATE_COL])


def test_drop_warmup_removes_lag_365_nans():
    panel = make_panel(n_days=400, stores=(1,), families=("GROCERIES I",))
    warm = build.build_features(panel.copy(), make_holidays())
    assert warm["lag_365"].isna().sum() == 365

    trimmed = build.build_features(panel.copy(), make_holidays(), drop_warmup=True)
    assert len(trimmed) == len(warm) - 365
    assert trimmed["lag_365"].notna().all()


def test_float_features_are_float32(features):
    for col in build.FLOAT_FEATURE_COLS:
        assert features[col].dtype == np.float32, col
    for col in build.CATEGORY_DTYPE_COLS:
        assert isinstance(features[col].dtype, pd.CategoricalDtype), col
    for col in build.INTEGER_CODE_COLS:
        assert pd.api.types.is_integer_dtype(features[col]), col


def test_features_round_trip_through_parquet(tmp_path, features):
    path = tmp_path / "features.parquet"
    build.save_parquet(features, path)
    reloaded = pd.read_parquet(path)
    pd.testing.assert_frame_equal(reloaded, features)
    build.validate_features(reloaded)   # categories + dtypes survive the round trip
