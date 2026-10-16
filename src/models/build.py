"""
Phase 3 model-building CLI: feature matrix -> walk-forward validation ->
tuned global LightGBM artifact.

Pipeline
--------
1. Load ``data/processed/features.parquet``.
2. **Walk-forward validation** (the folds from Phase 2) — train on the past,
   score on the next window, across every fold. Reports mean WAPE/RMSE/MAE.
3. **Optional Optuna tuning** on the most recent train/validation window.
4. **Final model** trained on the full development period, scored **once** on
   the sealed test window, and compared against the naive baseline.
5. Save ``models/lgbm_v1.joblib`` (model + features + params + metrics).

Usage
-----
    python -m src.models.build                       # baseline params
    python -m src.models.build --tune --trials 50    # Optuna, 50 trials
    python -m src.models.build --refit-all           # deploy on 100% of data
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.features import split  # noqa: E402
from src.models import evaluate, train, tune  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("models.build")


# ===========================================================================
# Walk-forward validation
# ===========================================================================
def run_walk_forward(
    df: pd.DataFrame,
    params: dict | None = None,
    n_splits: int = config.WALK_FORWARD_SPLITS,
    val_days: int = config.VALIDATION_DAYS,
    test_days: int = config.TEST_DAYS,
    min_train_days: int = config.MIN_TRAIN_DAYS,
    num_boost_round: int = config.LGBM_NUM_BOOST_ROUND,
    early_stopping_rounds: int | None = config.LGBM_EARLY_STOPPING_ROUNDS,
) -> pd.DataFrame:
    """
    Train and score one model per walk-forward fold.

    Returns
    -------
    pd.DataFrame
        One row per fold with train/val windows and model-vs-naive metrics.
        The mean WAPE across folds is the honest estimate of out-of-sample
        performance, because every fold is scored on unseen future days.
    """
    records: list[dict] = []
    for fold in split.walk_forward_splits(
        df, n_splits=n_splits, val_days=val_days,
        test_days=test_days, min_train_days=min_train_days,
    ):
        train_df, val_df = split.fold_frames(df, fold)
        result = train.train_model(
            train_df, val_df, params=params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        predictions = result.predict(val_df)
        comparison = evaluate.compare_with_baseline(val_df, predictions)

        record = {
            "fold": fold.fold,
            "train_end": fold.train_end.date(),
            "val_start": fold.val_start.date(),
            "val_end": fold.val_end.date(),
            "train_days": (pd.Timestamp(fold.train_end)
                           - pd.Timestamp(fold.train_start)).days + 1,
            "best_iteration": result.best_iteration,
            "rmse": comparison.loc["model", "rmse"],
            "mae": comparison.loc["model", "mae"],
            "wape": comparison.loc["model", "wape"],
            "naive_wape": comparison.loc["naive_last_week", "wape"],
        }
        record["wape_improvement_pct"] = evaluate.relative_improvement_pct(
            record["naive_wape"], record["wape"]
        )
        records.append(record)
        log.info(
            "Fold %d: model WAPE=%.2f%% vs naive %.2f%% (+%.1f%%) | RMSE=%.3f",
            fold.fold, record["wape"], record["naive_wape"],
            record["wape_improvement_pct"], record["rmse"],
        )

    table = pd.DataFrame(records)
    log.info(
        "Walk-forward mean: WAPE=%.2f%% | RMSE=%.3f | MAE=%.3f | "
        "vs naive WAPE=%.2f%% (+%.1f%%)",
        table["wape"].mean(), table["rmse"].mean(), table["mae"].mean(),
        table["naive_wape"].mean(), table["wape_improvement_pct"].mean(),
    )
    return table


# ===========================================================================
# Final model
# ===========================================================================
def fit_final_model(
    df: pd.DataFrame,
    params: dict | None = None,
    val_days: int = config.VALIDATION_DAYS,
    test_days: int = config.TEST_DAYS,
    num_boost_round: int = config.LGBM_NUM_BOOST_ROUND,
    early_stopping_rounds: int | None = config.LGBM_EARLY_STOPPING_ROUNDS,
) -> tuple[train.TrainingResult, pd.DataFrame, pd.DataFrame]:
    """
    Train on the development period and score once on the sealed test window.

    Returns
    -------
    (result, test_frame, comparison)
        ``test_frame`` carries the test rows plus a ``model_prediction`` column
        so the caller can persist or plot them.
    """
    dev_train, dev_val, test = split.train_val_test_split(
        df, val_days=val_days, test_days=test_days
    )
    result = train.train_model(
        dev_train, dev_val, params=params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    test_frame = test.copy()
    test_frame["model_prediction"] = result.predict(test_frame)
    comparison = evaluate.compare_with_baseline(test_frame, test_frame["model_prediction"])

    skill = evaluate.relative_improvement_pct(
        comparison.loc["naive_last_week", "wape"],
        comparison.loc["model", "wape"],
    )
    log.info("SEALED TEST " + evaluate.format_metrics(
        {**evaluate.metrics(test_frame[config.TARGET_COL],
                            test_frame["model_prediction"]),
         "n": len(test_frame)}))
    log.info("Sealed test: model WAPE=%.2f%% vs naive %.2f%% (+%.1f%%)",
             comparison.loc["model", "wape"],
             comparison.loc["naive_last_week", "wape"], skill)
    return result, test_frame, comparison


# ===========================================================================
# CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and persist the global LightGBM demand model.",
    )
    parser.add_argument("--input", type=Path, default=config.FEATURES_PARQUET,
                        help="Feature matrix (default: data/processed/features.parquet)")
    parser.add_argument("--output", type=Path, default=config.MODEL_ARTIFACT,
                        help="Model artifact (default: models/lgbm_v1.joblib)")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter search before final training")
    parser.add_argument("--trials", type=int, default=config.OPTUNA_N_TRIALS,
                        help=f"Optuna trials (default: {config.OPTUNA_N_TRIALS})")
    parser.add_argument("--refit-all", action="store_true",
                        help="After scoring the sealed test, refit on all data for deployment")
    parser.add_argument("--no-cv", action="store_true",
                        help="Skip walk-forward validation (faster)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {args.input}\n"
            "Run Phase 2 first:  python -m src.features.build"
        )

    df = pd.read_parquet(args.input)
    log.info("Loaded features: %s rows | %s .. %s | %d columns",
             f"{len(df):,}", df[config.DATE_COL].min().date(),
             df[config.DATE_COL].max().date(), df.shape[1])

    params = dict(config.LGBM_PARAMS)
    cv_table: pd.DataFrame | None = None

    # --- 1. Walk-forward validation ---------------------------------------
    if not args.no_cv:
        try:
            cv_table = run_walk_forward(df, params=params)
        except ValueError as error:
            log.warning("Skipping walk-forward validation: %s", error)

    # --- 2. Optional tuning ------------------------------------------------
    tuning_report = None
    if args.tune:
        try:
            tuning_train, tuning_val, _ = split.train_val_test_split(
                df, val_days=config.VALIDATION_DAYS, test_days=config.TEST_DAYS
            )
            tuned = tune.tune_lightgbm(
                tuning_train, tuning_val, n_trials=args.trials,
                objective_metric=config.TUNE_OBJECTIVE_METRIC,
            )
            params = {**params, **tuned.best_params}
            tuning_report = tuned.summary()
            log.info("Using tuned params: %s", tuned.best_params)
        except ValueError as error:
            log.warning("Skipping tuning: %s", error)

    # --- 3. Final model + sealed test -------------------------------------
    result, test_frame, comparison = fit_final_model(df, params=params)
    log.info("\n%s", comparison.round(4).to_string())

    # --- 4. Optional deployment refit -------------------------------------
    refit_scope = "development"
    if args.refit_all:
        log.info("Refitting on 100%% of the data for deployment...")
        refit = train.train_model(
            df, None, params=params,
            num_boost_round=result.best_iteration or config.LGBM_NUM_BOOST_ROUND,
            early_stopping_rounds=None,
        )
        # Keep the held-out scores on the deployment artifact for reference.
        refit.metrics = dict(result.metrics)
        refit.val_start, refit.val_end = result.val_start, result.val_end
        refit.val_rows = result.val_rows
        result = refit
        refit_scope = "all"

    # --- 5. Persist --------------------------------------------------------
    extra = {
        "cv_metrics": None if cv_table is None else cv_table.to_dict(orient="records"),
        "test_metrics": comparison.to_dict(orient="index"),
        "refit_scope": refit_scope,
        "tuning_report": tuning_report,
        "objective_metric": config.TUNE_OBJECTIVE_METRIC,
        "horizon": config.FORECAST_HORIZON,
        "clip_non_negative": config.CLIP_NON_NEGATIVE,
        "data_rows": int(len(df)),
        "data_window": [
            df[config.DATE_COL].min().date().isoformat(),
            df[config.DATE_COL].max().date().isoformat(),
        ],
    }
    artifact = train.save_model(args.output, result, extra=extra)

    # Persist the test predictions next to the artifact for reporting/plots.
    predictions_path = args.output.parent / "test_predictions.parquet"
    keep = [*config.GROUP_KEYS, config.DATE_COL, config.TARGET_COL, "model_prediction"]
    test_frame[keep].to_parquet(predictions_path, engine="pyarrow",
                                compression="zstd", index=False)
    log.info("Saved test predictions: %s", predictions_path)

    log.info("Artifact summary: %s", train.artifact_summary(result.to_artifact(extra)))
    log.info("Model build finished -> %s", artifact)


if __name__ == "__main__":
    main()
