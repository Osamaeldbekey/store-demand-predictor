"""
Phase 3 test suite: training, tuning, evaluation, recursive forecasting and
persistence of the global LightGBM demand model.

Strategy
--------
Everything runs against a small synthetic panel pushed through the **real
Phase 2 pipeline**, so the tests exercise the true feature contract (dtypes,
categorical codes, leak-safe lags) rather than a hand-built stand-in. Expensive
objects (feature matrix, splits, trained model) are session-scoped fixtures so
the suite stays fast.

Run with:
    python -m pytest tests/test_models.py -q
"""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
import pytest

import joblib
import optuna

from src import config
from src.features import build as fbuild
from src.features import calendar, lags, split
from src.models import build as mbuild
from src.models import evaluate, predict, train, tune
from src.models import forecast as mforecast

# ---------------------------------------------------------------------------
# Synthetic panel / feature fixtures
# ---------------------------------------------------------------------------
N_STORES = 3
N_FAMILIES = 2
N_DAYS = 420
PANEL_START = "2016-01-01"
PANEL_SEED = 7
FORECAST_STORE = 1
FORECAST_FAMILY = "FAMILY_0"


def make_synthetic_panel(
    n_stores: int = N_STORES,
    n_families: int = N_FAMILIES,
    n_days: int = N_DAYS,
    start: str = PANEL_START,
    seed: int = PANEL_SEED,
) -> pd.DataFrame:
    """
    Build a cleaned multi-store panel with the exact ``CLEAN_COLUMN_ORDER``
    schema, dtypes and daily contiguity that ``validate_features`` requires.

    The signal is deliberately learnable (weekly seasonality + a weekend bump +
    store/family level effects) so metric assertions are meaningful rather than
    noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    families = [f"FAMILY_{i}" for i in range(n_families)]

    rows: list[dict] = []
    row_id = 1
    for store in range(1, n_stores + 1):
        for family_index, family in enumerate(families):
            base = 10.0 + store * 3 + family_index * 7
            for day_index, day in enumerate(dates):
                weekend = 0.5 if day.dayofweek >= 5 else 0.0
                weekly = 0.25 * np.sin(2 * np.pi * day_index / 7.0)
                sales = max(0.0, base * (1.0 + weekly + weekend) + rng.normal(0, 1.0))
                rows.append(
                    {
                        "id": row_id,
                        "date": day,
                        "store_nbr": store,
                        "family": family,
                        "sales": sales,
                        "onpromotion": int(rng.random() < 0.10),
                        "cluster": store % 5 + 1,
                        "city": f"CITY_{store % 3}",
                        "state": f"STATE_{store % 2}",
                        "type": f"TYPE_{store % 3 + 1}",
                    }
                )
                row_id += 1

    frame = pd.DataFrame(rows)[config.CLEAN_COLUMN_ORDER]
    frame["date"] = pd.to_datetime(frame["date"])
    frame["store_nbr"] = frame["store_nbr"].astype("int16")
    frame["cluster"] = frame["cluster"].astype("int8")
    frame["sales"] = frame["sales"].astype("float32")
    frame["onpromotion"] = frame["onpromotion"].astype("int8")
    for column in ("family", "city", "state", "type"):
        frame[column] = frame[column].astype("category")
    return frame


def make_synthetic_features() -> pd.DataFrame:
    """Push the synthetic panel through the Phase 2 feature pipeline."""
    panel = make_synthetic_panel()
    return fbuild.build_features(panel, calendar.empty_holidays())


@pytest.fixture(scope="session")
def features() -> pd.DataFrame:
    """Session-wide feature matrix (built once, never mutated by tests)."""
    return make_synthetic_features()


@pytest.fixture(scope="session")
def splits(features: pd.DataFrame):
    """(train, val, test) holdout split of the synthetic features."""
    return split.train_val_test_split(features)


@pytest.fixture(scope="session")
def train_df(splits):
    return splits[0]


@pytest.fixture(scope="session")
def val_df(splits):
    return splits[1]


@pytest.fixture(scope="session")
def test_df(splits):
    return splits[2]


@pytest.fixture(scope="session")
def trained(train_df: pd.DataFrame, val_df: pd.DataFrame) -> train.TrainingResult:
    """A small but real trained model, shared by the prediction tests."""
    return train.train_model(
        train_df, val_df, num_boost_round=50, early_stopping_rounds=10
    )


# ===========================================================================
# evaluate.py
# ===========================================================================
class TestEvaluate:
    def test_metrics_known_values(self):
        scores = evaluate.metrics([10, 20, 30], [12, 18, 33])
        # errors: -2, +2, +3 -> abs sum 7, true abs sum 60, squared sum 17
        assert scores["wape"] == pytest.approx(7 / 60 * 100)
        assert scores["mae"] == pytest.approx(7 / 3)
        assert scores["rmse"] == pytest.approx(np.sqrt(17 / 3))
        assert scores["bias"] == pytest.approx(1.0)
        assert scores["n"] == 3

    def test_metrics_ignore_nan_pairs(self):
        scores = evaluate.metrics([1.0, np.nan, 3.0], [1.0, 99.0, 3.0])
        assert scores["n"] == 2            # the NaN truth row is dropped
        assert scores["wape"] == pytest.approx(0.0)

    def test_wape_zero_denominator_guard(self):
        assert evaluate.wape([0, 0, 0], [0, 0, 0]) == 0.0
        assert np.isinf(evaluate.wape([0, 0, 0], [1, 0, 0]))

    def test_metrics_reject_shape_mismatch(self):
        with pytest.raises(ValueError):
            evaluate.metrics([1, 2, 3], [1, 2])

    def test_metrics_reject_all_nan(self):
        with pytest.raises(ValueError):
            evaluate.metrics([np.nan, np.nan], [np.nan, np.nan])

    def test_naive_predictions_fall_back_then_default(self):
        frame = pd.DataFrame(
            {"lag_7": [np.nan, 5.0, np.nan], "roll_mean_7": [2.0, np.nan, np.nan]}
        )
        assert evaluate.naive_predictions(frame).tolist() == [2.0, 5.0, 0.0]

    def test_relative_improvement_pct(self):
        assert evaluate.relative_improvement_pct(10.0, 8.0) == pytest.approx(20.0)
        assert evaluate.relative_improvement_pct(10.0, 12.0) == pytest.approx(-20.0)
        assert evaluate.relative_improvement_pct(0.0, 5.0) == 0.0

    def test_compare_with_baseline_structure(
        self, val_df: pd.DataFrame, trained: train.TrainingResult
    ):
        table = evaluate.compare_with_baseline(val_df, trained.predict(val_df))
        assert list(table.index) == ["model", "naive_last_week"]
        assert {"rmse", "mae", "wape", "bias", "n_rows"}.issubset(table.columns)
        assert (table["wape"] >= 0).all()
        # Both rows are scored on exactly the same rows.
        assert table.loc["model", "n_rows"] == table.loc["naive_last_week", "n_rows"]

    def test_compare_with_baseline_identical_predictions(
        self, val_df: pd.DataFrame
    ):
        frame = val_df.copy()
        naive = evaluate.naive_predictions(frame)
        table = evaluate.compare_with_baseline(frame, naive)
        assert table.loc["model", "wape"] == pytest.approx(
            table.loc["naive_last_week", "wape"]
        )
        improvement = evaluate.relative_improvement_pct(
            table.loc["naive_last_week", "wape"], table.loc["model", "wape"]
        )
        assert improvement == pytest.approx(0.0)

    def test_metrics_by_series_sorted_worst_first(
        self, val_df: pd.DataFrame, trained: train.TrainingResult
    ):
        per_series = evaluate.metrics_by_series(val_df, trained.predict(val_df))
        assert len(per_series) == val_df.groupby(
            config.GROUP_KEYS, observed=True
        ).ngroups
        assert set(config.GROUP_KEYS).issubset(per_series.columns)
        assert per_series["wape"].is_monotonic_decreasing

    def test_format_metrics_mentions_wape(self):
        text = evaluate.format_metrics(evaluate.metrics([1, 2], [1, 3]))
        assert "WAPE" in text and "RMSE" in text


# ===========================================================================
# train.py
# ===========================================================================
class TestTrain:
    def test_feature_columns_excludes_non_features(self, features: pd.DataFrame):
        columns = train.feature_columns(features)
        for excluded in config.NON_FEATURE_COLS:
            assert excluded not in columns
        assert "onpromotion" in columns
        assert "lag_7" in columns

    def test_categorical_features_are_the_declared_six(self, features: pd.DataFrame):
        cats = train.categorical_features(train.feature_columns(features))
        assert cats == config.CATEGORICAL_FEATURES

    def test_train_returns_metrics_in_range(self, features, train_df, val_df):
        result = train.train_model(train_df, val_df, num_boost_round=40,
                                   early_stopping_rounds=10)
        scores = result.metrics
        assert {"rmse", "mae", "wape", "bias", "n"}.issubset(scores)
        assert scores["rmse"] >= 0
        assert scores["mae"] >= 0
        assert 0 <= scores["wape"] < 200
        assert np.isfinite(scores["bias"])
        assert scores["n"] == len(val_df)
        assert result.best_iteration >= 1
        # No structural columns ever reach the booster.
        assert not set(config.NON_FEATURE_COLS) & set(result.features)

    def test_predictions_are_non_negative_and_finite(
        self, val_df: pd.DataFrame, trained: train.TrainingResult
    ):
        predictions = trained.predict(val_df)
        assert len(predictions) == len(val_df)
        assert np.isfinite(predictions).all()
        assert (predictions >= 0).all()

    def test_training_is_deterministic(self, train_df, val_df):
        params = dict(config.LGBM_PARAMS, n_jobs=1)
        first = train.train_model(train_df, val_df, params=params,
                                  num_boost_round=30, early_stopping_rounds=None)
        second = train.train_model(train_df, val_df, params=params,
                                   num_boost_round=30, early_stopping_rounds=None)
        assert np.array_equal(first.predict(val_df), second.predict(val_df))

    def test_train_without_validation_has_no_metrics(self, train_df):
        result = train.train_model(train_df, None, num_boost_round=15,
                                   early_stopping_rounds=None)
        assert result.metrics == {}
        assert result.val_rows == 0
        assert result.best_iteration == 0

    def test_clip_predictions(self):
        raw = np.array([-5.0, 0.0, 3.0])
        assert train.clip_predictions(raw).tolist() == [0.0, 0.0, 3.0]
        assert train.clip_predictions(raw, clip_non_negative=False).tolist() == [
            -5.0, 0.0, 3.0
        ]

    def test_feature_importance_shape(self, trained: train.TrainingResult):
        importance = train.feature_importance(trained, top_n=10)
        assert len(importance) == 10
        assert {"feature", "gain", "split", "gain_pct"}.issubset(importance.columns)
        assert importance["gain_pct"].sum() <= 100.0 + 1e-6

    def test_save_load_round_trip(
        self, tmp_path, val_df: pd.DataFrame, trained: train.TrainingResult
    ):
        path = tmp_path / "model.joblib"
        train.save_model(path, trained, extra={"note": "round-trip"})
        artifact = train.load_model(path)

        assert path.exists()
        assert artifact["note"] == "round-trip"
        for key in train.ARTIFACT_KEYS:
            assert key in artifact
        assert artifact["features"] == trained.features
        assert artifact["categorical_features"] == trained.categorical_features

        original = trained.model.predict(val_df[trained.features])
        restored = artifact["model"].predict(val_df[artifact["features"]])
        assert np.allclose(original, restored)

    def test_load_model_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            train.load_model(tmp_path / "does_not_exist.joblib")

    def test_load_model_rejects_malformed_artifact(self, tmp_path):
        path = tmp_path / "bad.joblib"
        joblib.dump({"model": None}, path)
        with pytest.raises(ValueError):
            train.load_model(path)

    def test_load_model_rejects_non_dict(self, tmp_path):
        path = tmp_path / "list.joblib"
        joblib.dump([1, 2, 3], path)
        with pytest.raises(ValueError):
            train.load_model(path)


# ===========================================================================
# tune.py
# ===========================================================================
class TestTune:
    def test_suggest_params_within_search_space(self):
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        for _ in range(3):
            params = tune.suggest_params(study.ask())
            for name, (low, high) in tune.SEARCH_SPACE.items():
                assert low <= params[name] <= high, name
            for name in tune.INT_PARAMS:
                assert isinstance(params[name], int)
            assert params["bagging_freq"] == 1

    def test_tune_returns_best_params_in_space(self, train_df, val_df):
        result = tune.tune_lightgbm(
            train_df, val_df, n_trials=2, num_boost_round=25,
            early_stopping_rounds=10, seed=0,
        )
        assert result.n_complete >= 1
        assert np.isfinite(result.best_value)
        assert set(result.best_params).issubset(set(tune.SEARCH_SPACE) | {"bagging_freq"})
        for name in tune.INT_PARAMS:
            assert isinstance(result.best_params[name], int)
        assert not result.trials.empty
        assert {"number", "value", "state"}.issubset(result.trials.columns)

    def test_best_params_actually_train(self, train_df, val_df):
        tuned = tune.tune_lightgbm(
            train_df, val_df, n_trials=2, num_boost_round=25,
            early_stopping_rounds=10, seed=1,
        )
        result = train.train_model(
            train_df, val_df, params={**config.LGBM_PARAMS, **tuned.best_params},
            num_boost_round=25, early_stopping_rounds=10,
        )
        assert 0 <= result.metrics["wape"] < 200

    def test_invalid_objective_metric_raises(self, train_df, val_df):
        with pytest.raises(ValueError):
            tune.tune_lightgbm(train_df, val_df, n_trials=1,
                               objective_metric="r2")

    def test_invalid_trial_count_raises(self, train_df, val_df):
        with pytest.raises(ValueError):
            tune.tune_lightgbm(train_df, val_df, n_trials=0)


# ===========================================================================
# predict.py
# ===========================================================================
class TestPredict:
    def test_forecast_frame_shape_and_dates(
        self, features: pd.DataFrame, trained: train.TrainingResult
    ):
        outcome = predict.forecast_series(
            trained.model, features, FORECAST_STORE, FORECAST_FAMILY, horizon=30
        )
        assert len(outcome) == 30
        assert list(outcome.predictions.columns) == [
            "date", "store_nbr", "family", "prediction"
        ]
        last_actual = features.loc[
            (features["store_nbr"] == FORECAST_STORE)
            & (features["family"] == FORECAST_FAMILY),
            "date",
        ].max()
        assert outcome.predictions["date"].min() == last_actual + pd.Timedelta(days=1)
        assert outcome.predictions["date"].max() == last_actual + pd.Timedelta(days=30)
        # strictly daily, contiguous
        deltas = outcome.predictions["date"].diff().dropna()
        assert (deltas == pd.Timedelta(days=1)).all()

    def test_predictions_are_non_negative_and_finite(
        self, features: pd.DataFrame, trained: train.TrainingResult
    ):
        outcome = predict.forecast_series(
            trained.model, features, FORECAST_STORE, FORECAST_FAMILY, horizon=30
        )
        values = outcome.predictions["prediction"]
        assert values.notna().all()
        assert np.isfinite(values).all()
        assert (values >= 0).all()

    def test_prefix_property_seven_days_matches_thirty(
        self, features: pd.DataFrame, trained: train.TrainingResult
    ):
        """
        Regression test for the horizon-dependence bug: a 7-day forecast must
        equal the first 7 days of a 30-day forecast. Forward calendar distances
        (days_to_payday) are frame-derived, so only the calendar buffer keeps
        this invariant true.
        """
        short = predict.forecast_series(
            trained.model, features, FORECAST_STORE, FORECAST_FAMILY, horizon=7
        )
        long = predict.forecast_series(
            trained.model, features, FORECAST_STORE, FORECAST_FAMILY, horizon=30
        )
        assert np.allclose(
            short.predictions["prediction"].to_numpy(),
            long.predictions["prediction"].head(7).to_numpy(),
        )

    def test_recursive_features_match_one_shot_batch(
        self, features: pd.DataFrame, trained: train.TrainingResult
    ):
        """
        The features the model scores during recursion must equal the features
        the Phase 2 pipeline produces in one batch once the same predictions are
        in place — i.e. the recursion is exact, not approximate.
        """
        outcome = predict.forecast_series(
            trained.model, features, FORECAST_STORE, FORECAST_FAMILY, horizon=12,
            return_features=True,
        )
        engineered = lags.engineered_feature_columns()
        batch = lags.add_all_temporal_features(outcome.features.copy())
        dates = sorted(set(outcome.predictions["date"]))

        recursive_rows = (
            outcome.features.set_index("date").loc[dates, engineered]
            .to_numpy(dtype="float64")
        )
        batch_rows = (
            batch.set_index("date").loc[dates, engineered].to_numpy(dtype="float64")
        )
        assert np.allclose(recursive_rows, batch_rows, equal_nan=True)

    def test_future_onpromotion_schedule_is_applied(
        self, features: pd.DataFrame, trained: train.TrainingResult
    ):
        outcome = predict.forecast_series(
            trained.model, features, FORECAST_STORE, FORECAST_FAMILY, horizon=5,
            future_onpromotion=1, return_features=True,
        )
        dates = set(outcome.predictions["date"])
        rows = outcome.features[outcome.features["date"].isin(dates)]
        assert (rows["onpromotion"] == 1).all()

    def test_default_future_onpromotion_is_zero(
        self, features: pd.DataFrame, trained: train.TrainingResult
    ):
        outcome = predict.forecast_series(
            trained.model, features, FORECAST_STORE, FORECAST_FAMILY, horizon=5,
            return_features=True,
        )
        dates = set(outcome.predictions["date"])
        rows = outcome.features[outcome.features["date"].isin(dates)]
        assert (rows["onpromotion"] == 0).all()

    def test_forecast_all_covers_every_series(
        self, features: pd.DataFrame, trained: train.TrainingResult
    ):
        all_predictions = predict.forecast_all(
            trained.model, features, horizon=7, log_every=0
        )
        expected = features.groupby(config.GROUP_KEYS, observed=True).ngroups
        assert len(all_predictions) == expected * 7
        assert all_predictions.groupby(
            config.GROUP_KEYS, observed=True
        ).size().eq(7).all()

    def test_forecast_all_respects_store_filter(
        self, features: pd.DataFrame, trained: train.TrainingResult
    ):
        all_predictions = predict.forecast_all(
            trained.model, features, horizon=3, stores=[1], log_every=0
        )
        assert set(all_predictions["store_nbr"].unique()) == {1}

    def test_truncate_horizon(self):
        frame = pd.DataFrame(
            {
                "store_nbr": np.repeat([1, 2], 30),
                "family": np.repeat(["A", "B"], 30),
                "date": list(pd.date_range("2020-01-01", periods=30)) * 2,
                "prediction": np.arange(60, dtype="float64"),
            }
        )
        truncated = predict.truncate_horizon(frame, 7)
        assert len(truncated) == 14
        assert truncated.groupby(
            config.GROUP_KEYS, observed=True
        ).size().eq(7).all()

    def test_unknown_series_raises(
        self, features: pd.DataFrame, trained: train.TrainingResult
    ):
        with pytest.raises(KeyError):
            predict.forecast_series(
                trained.model, features, 999, FORECAST_FAMILY, horizon=3
            )

    def test_invalid_horizon_raises(
        self, features: pd.DataFrame, trained: train.TrainingResult
    ):
        with pytest.raises(ValueError):
            predict.forecast_series(
                trained.model, features, FORECAST_STORE, FORECAST_FAMILY, horizon=0
            )

    def test_accepts_loaded_artifact(
        self, features: pd.DataFrame, trained: train.TrainingResult
    ):
        artifact = trained.to_artifact()
        outcome = predict.forecast_series(
            artifact, features, FORECAST_STORE, FORECAST_FAMILY, horizon=4
        )
        assert len(outcome) == 4


# ===========================================================================
# build.py (walk-forward + sealed test)
# ===========================================================================
class TestBuild:
    def test_run_walk_forward_structure(self, features: pd.DataFrame):
        table = mbuild.run_walk_forward(
            features, n_splits=3, num_boost_round=25, early_stopping_rounds=10
        )
        assert len(table) == 3
        expected = {
            "fold", "train_end", "val_start", "val_end", "train_days",
            "best_iteration", "rmse", "mae", "wape", "naive_wape",
            "wape_improvement_pct",
        }
        assert expected.issubset(table.columns)
        assert (table["wape"] >= 0).all()
        assert (table["naive_wape"] >= 0).all()
        assert np.isfinite(table["wape_improvement_pct"]).all()
        # 24 days reserved (16 val + 16 test overlap-free by construction)
        assert (table["train_days"] > 0).all()

    def test_fit_final_model_scores_sealed_test(self, features: pd.DataFrame):
        result, test_frame, comparison = mbuild.fit_final_model(
            features, num_boost_round=25, early_stopping_rounds=10
        )
        assert "model_prediction" in test_frame.columns
        assert (test_frame["model_prediction"] >= 0).all()
        assert list(comparison.index) == ["model", "naive_last_week"]
        assert comparison.loc["model", "n_rows"] == len(test_frame)
        # The sealed test window is never used for training.
        assert test_frame["date"].min() > result.train_end

    def test_artifact_summary_round_trip(self, trained: train.TrainingResult):
        text = train.artifact_summary(trained.to_artifact())
        assert "features" in text
        assert "trained_at" in text


# ===========================================================================
# forecast.py (the offline tier-1 batch)
# ===========================================================================
class TestForecastExport:
    """Contract tests for the artifact ``app/_data.py`` tier 1 consumes."""

    HORIZON = 6

    @staticmethod
    def _scope(frame: pd.DataFrame) -> tuple[list[int], list[str]]:
        """Store/family lists the frame claims to cover."""
        return (
            sorted(int(v) for v in frame["store_nbr"].unique()),
            sorted(str(v) for v in frame["family"].astype(str).unique()),
        )

    @pytest.fixture(scope="class")
    def offline(
        self, features: pd.DataFrame, trained: train.TrainingResult
    ) -> pd.DataFrame:
        return mforecast.build_offline_forecasts(
            trained.model, features, horizon=self.HORIZON
        )

    def test_columns_are_exactly_what_the_dashboard_loader_expects(
        self, offline: pd.DataFrame
    ):
        assert list(offline.columns) == mforecast.FORECAST_COLUMNS
        assert set(mforecast.FORECAST_COLUMNS) == {
            *config.GROUP_KEYS, config.DATE_COL, "prediction",
        }

    def test_every_series_is_covered_for_the_whole_horizon(
        self, offline: pd.DataFrame, features: pd.DataFrame
    ):
        expected = features.groupby(config.GROUP_KEYS, observed=True).ngroups
        counts = offline.groupby(config.GROUP_KEYS, observed=True)[
            config.DATE_COL
        ].nunique()
        assert len(counts) == expected
        assert counts.eq(self.HORIZON).all()
        assert len(offline) == expected * self.HORIZON

    def test_predictions_are_complete_and_plausible(self, offline: pd.DataFrame):
        assert not offline.isna().any().any()
        assert (offline["prediction"] >= 0).all()
        assert (offline["prediction"] > 0).any()

    def test_rows_are_sorted_by_series_then_date(self, offline: pd.DataFrame):
        keys = [*config.GROUP_KEYS, config.DATE_COL]
        expected = offline.sort_values(keys, kind="stable").reset_index(drop=True)
        pd.testing.assert_frame_equal(offline, expected)

    def test_batch_equals_the_library_call_it_wraps(
        self, offline: pd.DataFrame, features: pd.DataFrame, trained
    ):
        """The exporter must not drift from `forecast_all`'s numbers."""
        direct = predict.forecast_all(
            trained.model, features, horizon=self.HORIZON, log_every=0
        )
        keys = [*config.GROUP_KEYS, config.DATE_COL]
        merged = offline.merge(
            direct[keys + ["prediction"]], on=keys, suffixes=("", "_direct")
        )
        assert len(merged) == len(offline)
        assert np.allclose(
            merged["prediction"], merged["prediction_direct"], rtol=1e-6, atol=1e-6
        )

    def test_scope_filters_narrow_the_batch(
        self, features: pd.DataFrame, trained
    ):
        frame = mforecast.build_offline_forecasts(
            trained.model, features, horizon=3, stores=[1], families=[FORECAST_FAMILY]
        )
        assert set(frame["store_nbr"].unique()) == {1}
        assert set(frame["family"].astype(str).unique()) == {FORECAST_FAMILY}
        assert len(frame) == 3

    def test_round_trips_through_parquet_and_still_satisfies_tier_one(
        self, offline: pd.DataFrame, tmp_path
    ):
        path = tmp_path / "forecasts.parquet"
        offline.to_parquet(path, index=False)
        reloaded = pd.read_parquet(path)
        assert {*config.GROUP_KEYS, config.DATE_COL, "prediction"}.issubset(
            reloaded.columns
        )
        stores, families = self._scope(reloaded)
        mforecast.validate_offline_forecasts(reloaded, stores, families, self.HORIZON)

    def test_rejects_an_empty_scope(self, features: pd.DataFrame, trained):
        with pytest.raises(ValueError, match="No series matched"):
            mforecast.build_offline_forecasts(
                trained.model, features, horizon=3, families=["NOT_A_FAMILY"]
            )

    def test_rejects_a_non_positive_horizon(self, features: pd.DataFrame, trained):
        with pytest.raises(ValueError, match="horizon must be"):
            mforecast.build_offline_forecasts(trained.model, features, horizon=0)

    def test_default_horizon_covers_every_configured_horizon(self):
        assert mforecast.DEFAULT_HORIZON >= max(config.FORECAST_HORIZONS)
        assert mforecast.DEFAULT_HORIZON >= config.FORECAST_HORIZON

    # --- validate_offline_forecasts ------------------------------------
    def test_validator_accepts_a_sound_artifact(self, offline: pd.DataFrame):
        stores, families = self._scope(offline)
        mforecast.validate_offline_forecasts(offline, stores, families, self.HORIZON)

    def test_validator_rejects_missing_columns(self, offline: pd.DataFrame):
        stores, families = self._scope(offline)
        with pytest.raises(ValueError, match="missing columns"):
            mforecast.validate_offline_forecasts(
                offline.drop(columns=["prediction"]), stores, families, self.HORIZON
            )

    def test_validator_rejects_a_missing_series(self, offline: pd.DataFrame):
        stores, families = self._scope(offline)
        first = stores[0]
        truncated = offline[offline["store_nbr"] != first]
        with pytest.raises(ValueError, match="series in the artifact"):
            mforecast.validate_offline_forecasts(
                truncated, stores, families, self.HORIZON
            )

    def test_validator_rejects_a_short_series(self, offline: pd.DataFrame):
        """A silently short batch would make tier 1 fall through to tier 3."""
        stores, families = self._scope(offline)
        short = offline[offline[config.DATE_COL] < offline[config.DATE_COL].max()]
        with pytest.raises(ValueError, match="forecast days"):
            mforecast.validate_offline_forecasts(short, stores, families, self.HORIZON)

    def test_validator_rejects_null_predictions(self, offline: pd.DataFrame):
        stores, families = self._scope(offline)
        corrupted = offline.copy()
        corrupted.loc[corrupted.index[0], "prediction"] = np.nan
        with pytest.raises(ValueError, match="null predictions"):
            mforecast.validate_offline_forecasts(
                corrupted, stores, families, self.HORIZON
            )

    def test_missing_holiday_calendar_falls_back_to_empty(
        self, tmp_path, caplog
    ):
        """The degraded-deploy case: no raw CSV shipped, so holidays are zeroed."""
        with caplog.at_level(logging.WARNING):
            table = mforecast.load_holidays(tmp_path / "absent.csv")
        assert table.empty
        assert list(table.columns) == list(calendar.empty_holidays().columns)
        assert "holiday features" in caplog.text

    def test_cli_writes_the_artifact(
        self, features, trained, tmp_path, monkeypatch
    ):
        features_path = tmp_path / "features.parquet"
        model_path = tmp_path / "lgbm_v1.joblib"
        output_path = tmp_path / "forecasts.parquet"
        features.to_parquet(features_path, index=False)
        train.save_model(model_path, trained, extra={})

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "forecast",
                "--features", str(features_path),
                "--model", str(model_path),
                "--output", str(output_path),
                "--holidays", str(tmp_path / "absent.csv"),
                "--horizon", "4",
            ],
        )
        mforecast.main()

        frame = pd.read_parquet(output_path)
        assert {*config.GROUP_KEYS, config.DATE_COL, "prediction"}.issubset(
            frame.columns
        )
        counts = frame.groupby(config.GROUP_KEYS, observed=True)[
            config.DATE_COL
        ].nunique()
        assert counts.eq(4).all()
        assert not frame.isna().any().any()

    def test_cli_reports_missing_inputs(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "forecast",
                "--features", str(tmp_path / "absent.parquet"),
                "--model", str(tmp_path / "absent.joblib"),
                "--output", str(tmp_path / "out.parquet"),
            ],
        )
        with pytest.raises(FileNotFoundError, match="Phase 2"):
            mforecast.main()
