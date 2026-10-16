"""
Optuna hyperparameter tuning for the global LightGBM model.

Design notes
------------
* **Objective metric.** Defaults to WAPE (the business headline), not RMSE.
  Early stopping still uses validation RMSE internally, so the two goals are
  decoupled: train to a good fit, then select the configuration that minimises
  the business error. Pass ``objective_metric="rmse"`` to tune the loss instead.
* **Fixed validation split.** Every trial is scored on the same train/val
  window, otherwise trial values are not comparable.
* **Reproducibility.** The study uses a seeded TPE sampler, so re-running the
  same command reproduces the same search trajectory.

Usage:
    from src.models import tune
    best = tune.tune_lightgbm(train_df, val_df, n_trials=50)
    train_df_model = train.train_model(train_df, val_df, params=best.best_params)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import optuna
import pandas as pd

from src import config
from src.models import train

log = logging.getLogger("models.tune")

# Search space bounds (also asserted by the test suite).
SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "num_leaves": (31, 511),
    "learning_rate": (0.01, 0.1),
    "feature_fraction": (0.5, 1.0),
    "min_data_in_leaf": (10, 100),
    "lambda_l1": (1e-3, 10.0),
    "lambda_l2": (1e-3, 10.0),
    "bagging_fraction": (0.5, 1.0),
}

# Parameters LightGBM requires to be plain Python scalars for the booster.
INT_PARAMS: tuple[str, ...] = (
    "num_leaves",
    "min_data_in_leaf",
    "bagging_freq",
)


@dataclass
class TuneResult:
    """Outcome of an Optuna study."""

    best_params: dict
    best_value: float
    objective_metric: str
    n_trials: int
    n_complete: int
    trials: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        return (
            f"Optuna: {self.n_complete}/{self.n_trials} trials complete | "
            f"best {self.objective_metric.upper()}={self.best_value:.4f} | "
            f"params={self.best_params}"
        )


# ===========================================================================
# Search space
# ===========================================================================
def suggest_params(trial: optuna.Trial) -> dict:
    """
    Sample one configuration from the search space.

    ``learning_rate`` and the L1/L2 penalties are sampled log-uniformly: their
    useful range spans orders of magnitude, so a linear sampler would waste
    most trials in the uninteresting region.
    """
    params = {
        "num_leaves": trial.suggest_int("num_leaves", *SEARCH_SPACE["num_leaves"]),
        "learning_rate": trial.suggest_float(
            "learning_rate", *SEARCH_SPACE["learning_rate"], log=True
        ),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", *SEARCH_SPACE["feature_fraction"]
        ),
        "min_data_in_leaf": trial.suggest_int(
            "min_data_in_leaf", *SEARCH_SPACE["min_data_in_leaf"]
        ),
        "lambda_l1": trial.suggest_float("lambda_l1", *SEARCH_SPACE["lambda_l1"], log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", *SEARCH_SPACE["lambda_l2"], log=True),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", *SEARCH_SPACE["bagging_fraction"]
        ),
    }
    # bagging_freq > 0 is required for bagging_fraction to have any effect.
    params["bagging_freq"] = 1
    return params


# ===========================================================================
# Tuning
# ===========================================================================
def tune_lightgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_trials: int = config.OPTUNA_N_TRIALS,
    objective_metric: str = config.TUNE_OBJECTIVE_METRIC,
    num_boost_round: int = config.LGBM_NUM_BOOST_ROUND,
    early_stopping_rounds: int | None = config.LGBM_EARLY_STOPPING_ROUNDS,
    base_params: dict | None = None,
    seed: int = config.RANDOM_SEED,
    timeout: int | None = config.OPTUNA_TIMEOUT_SECONDS,
) -> TuneResult:
    """
    Search for LightGBM hyperparameters that minimise ``objective_metric``.

    Parameters
    ----------
    train_df, val_df:
        Fixed train/validation split used for every trial.
    n_trials:
        Number of configurations to evaluate (the brief calls for 40–50).
    objective_metric:
        ``"wape"`` (default headline) or ``"rmse"`` / ``"mae"``. The metric
        must be a key returned by :func:`src.models.evaluate.metrics`.
    base_params:
        Parameters held fixed across trials (e.g. ``objective``, ``seed``).

    Returns
    -------
    TuneResult
        Best parameters plus the full trials table for the report.
    """
    if objective_metric not in {"wape", "rmse", "mae", "bias"}:
        raise ValueError(
            f"Unsupported objective_metric '{objective_metric}'. "
            "Use 'wape', 'rmse', 'mae' or 'bias'."
        )
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    fixed = train.resolve_params(base_params)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.set_user_attr("objective_metric", objective_metric)

    def objective(trial: optuna.Trial) -> float:
        params = {**fixed, **suggest_params(trial)}
        try:
            result = train.train_model(
                train_df,
                val_df,
                params=params,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
            )
        except Exception as error:  # noqa: BLE001 - a bad config must not kill the study
            log.warning("Trial %d failed (%s) — pruning.", trial.number, error)
            raise optuna.TrialPruned() from error
        score = float(result.metrics[objective_metric])
        trial.set_user_attr("best_iteration", result.best_iteration)
        log.info("Trial %d: %s=%.4f (%s)", trial.number, objective_metric, score,
                 {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in params.items() if k in SEARCH_SPACE})
        return score

    log.info("Starting Optuna: %d trials, objective=%s", n_trials, objective_metric.upper())
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   catch=(Exception,))

    # Only the search-space keys are carried forward; fixed params live in
    # config and are re-merged by train_model.
    best_params = {key: study.best_params[key] for key in SEARCH_SPACE
                   if key in study.best_params}
    for name in INT_PARAMS:
        if name in best_params:
            best_params[name] = int(best_params[name])
    best_params["bagging_freq"] = 1

    result = TuneResult(
        best_params=best_params,
        best_value=float(study.best_value),
        objective_metric=objective_metric,
        n_trials=n_trials,
        n_complete=len(study.trials),
        trials=study.trials_dataframe(),
    )
    log.info(result.summary())
    return result
