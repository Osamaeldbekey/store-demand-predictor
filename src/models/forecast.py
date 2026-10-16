"""
Phase 3b CLI: pre-compute the offline forecast batch the dashboard serves.

Why this exists
---------------
``app/_data.py`` resolves a forecast request in three tiers:

1. **Offline batch** — ``data/processed/forecasts.parquet``, written here.
2. **Canonical recursion** — one whole-bundle pass, only when the bundle has
   ``<= CANONICAL_MAX_SERIES`` (60) series.
3. **Direct subset** — a per-request recursion, otherwise.

Tier 1 is the only one that is *fast* at the real Favorita scope. The recursion
rebuilds the Phase 2 feature block once per forecast day **per series**, which
measures at roughly 0.7 s/series -- about 6 minutes for 20 stores x 26 families.
Baking the batch once turns every subsequent page load into an instant slice.

The artifact is deliberately tiny (one row per series-day: 520 series x 30 days
is ~15.6k rows) and it is *invalidated* by a promotion scenario, because the
dashboard then cannot reuse a forecast computed with the default promotion
assumption and falls through to tier 2/3.

Usage
-----
    python -m src.models.forecast                       # whole scope, 30 days
    python -m src.models.forecast --horizon 14
    python -m src.models.forecast --stores 1 2 --families BEVERAGES DAIRY
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.features import calendar  # noqa: E402
from src.models import predict, train  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("models.forecast")

#: Columns the dashboard's tier-1 loader requires (``app/_data.py``).
FORECAST_COLUMNS: list[str] = [*config.GROUP_KEYS, config.DATE_COL, "prediction"]

#: Cover every configured horizon so shorter horizons are prefixes, not misses.
DEFAULT_HORIZON: int = max(*config.FORECAST_HORIZONS, config.FORECAST_HORIZON)


# ===========================================================================
# Inputs
# ===========================================================================
def load_holidays(path: Path = config.HOLIDAYS_CSV) -> pd.DataFrame:
    """
    Holiday table for the forward calendar.

    Mirrors ``app/_data.py``: the real Ecuadorian calendar when the raw CSV is
    present, otherwise an empty table. An empty table zeroes the model's
    ``is_holiday`` / ``days_to_holiday`` / ``days_since_holiday`` features, which
    measurably shifts the forecast, so the fallback is logged as a warning.
    """
    if path.exists():
        holidays = calendar.filter_holidays(calendar.load_holidays(path))
        log.info("Loaded %d holiday rows from %s", len(holidays), path)
        return holidays

    log.warning(
        "Holiday calendar not found at %s -- forecasting with holiday features "
        "disabled (is_holiday=0). Predictions will be biased; provide the raw "
        "holidays_events.csv for production accuracy.",
        path,
    )
    return calendar.empty_holidays()


# ===========================================================================
# Build
# ===========================================================================
def build_offline_forecasts(
    model,
    features: pd.DataFrame,
    holidays: pd.DataFrame | None = None,
    horizon: int = DEFAULT_HORIZON,
    stores: list[int] | None = None,
    families: list[str] | None = None,
) -> pd.DataFrame:
    """
    Forecast every series in scope and return the tidy tier-1 frame.

    Parameters
    ----------
    model:
        Fitted estimator or a loaded artifact dict (``forecast_all`` unwraps it).
    features:
        The Phase 2 feature matrix over which history is rebuilt.
    holidays:
        Forward holiday calendar; ``None`` disables holiday features.
    horizon, stores, families:
        Forwarded to :func:`src.models.predict.forecast_all`. ``None`` scopes
        mean "every store/family present in ``features``".

    Returns
    -------
    pd.DataFrame
        ``store_nbr, family, date, prediction`` sorted by series then date.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    scoped = features
    if stores is not None:
        scoped = scoped[scoped["store_nbr"].isin(stores)]
    if families is not None:
        scoped = scoped[scoped["family"].isin(families)]
    if scoped.empty:
        raise ValueError("No series matched the requested store/family filters.")

    resolved_stores = sorted(int(v) for v in scoped["store_nbr"].unique())
    resolved_families = sorted(str(v) for v in scoped["family"].astype(str).unique())
    n_series = len(resolved_stores) * len(resolved_families)
    log.info(
        "Forecasting %d series (%d stores x %d families) x %d days...",
        n_series, len(resolved_stores), len(resolved_families), horizon,
    )

    frame = predict.forecast_all(
        model,
        features,
        horizon=horizon,
        stores=stores,
        families=families,
        holidays=holidays,
    )
    frame = frame[FORECAST_COLUMNS].sort_values(
        [*config.GROUP_KEYS, config.DATE_COL], kind="stable"
    )
    validate_offline_forecasts(
        frame, resolved_stores, resolved_families, horizon
    )
    return frame.reset_index(drop=True)


def validate_offline_forecasts(
    frame: pd.DataFrame,
    stores: list[int],
    families: list[str],
    horizon: int,
) -> None:
    """
    Assert the tier-1 contract the dashboard's loader relies on.

    ``app/_data.py`` accepts the artifact only when every requested series is
    present with at least the requested number of distinct dates; otherwise it
    silently falls through to a slow recursion. Failing loudly here is much
    better than shipping an artifact that quietly never gets used.
    """
    missing = [column for column in FORECAST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Forecast frame is missing columns: {missing}")

    expected = len(stores) * len(families)
    counts = frame.groupby(config.GROUP_KEYS, observed=True)[config.DATE_COL].nunique()
    if len(counts) != expected:
        raise ValueError(
            f"Expected {expected} series in the artifact, found {len(counts)}."
        )
    if counts.min() < horizon:
        raise ValueError(
            f"Every series needs >= {horizon} forecast days; shortest has "
            f"{int(counts.min())}."
        )
    if frame[["prediction"]].isna().any().any():
        raise ValueError("Forecast artifact contains null predictions.")


# ===========================================================================
# CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-compute the offline forecast batch the dashboard serves "
            "(tier 1 of app/_data.py's resolution order)."
        ),
    )
    parser.add_argument("--features", type=Path, default=config.FEATURES_PARQUET,
                        help="Feature matrix (default: data/processed/features.parquet)")
    parser.add_argument("--model", type=Path, default=config.MODEL_ARTIFACT,
                        help="Model artifact (default: models/lgbm_v1.joblib)")
    parser.add_argument("--output", type=Path, default=config.FORECASTS_PARQUET,
                        help="Output parquet (default: data/processed/forecasts.parquet)")
    parser.add_argument("--holidays", type=Path, default=config.HOLIDAYS_CSV,
                        help="Holiday calendar (default: data/raw/holidays_events.csv)")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help=f"Forecast days to store (default: {DEFAULT_HORIZON})")
    parser.add_argument("--stores", type=int, nargs="+", default=None,
                        help="Restrict to these store numbers (default: all)")
    parser.add_argument("--families", type=str, nargs="+", default=None,
                        help="Restrict to these families (default: all)")
    parser.add_argument("--compression", type=str, default="zstd",
                        help="Parquet compression codec (default: zstd)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.features.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {args.features}\n"
            "Run Phase 2 first:  python -m src.features.build"
        )
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {args.model}\n"
            "Run Phase 3 first:  python -m src.models.build"
        )

    features = pd.read_parquet(args.features)
    log.info(
        "Loaded features: %s rows | %s .. %s | %d columns",
        f"{len(features):,}",
        features[config.DATE_COL].min().date(),
        features[config.DATE_COL].max().date(),
        features.shape[1],
    )

    model = train.load_model(args.model)
    holidays = load_holidays(args.holidays)

    frame = build_offline_forecasts(
        model,
        features,
        holidays=holidays,
        horizon=args.horizon,
        stores=args.stores,
        families=args.families,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        args.output, engine="pyarrow", compression=args.compression, index=False
    )

    first, last = frame[config.DATE_COL].min(), frame[config.DATE_COL].max()
    log.info(
        "Saved %s rows (%d series x %d days) -> %s",
        f"{len(frame):,}",
        frame.groupby(config.GROUP_KEYS, observed=True).ngroups,
        args.horizon,
        args.output,
    )
    log.info("Forecast window: %s .. %s", first.date(), last.date())
    log.info("Total predicted demand: %s units", f"{frame['prediction'].sum():,.0f}")
    log.info("Forecast export finished -> %s", args.output)


if __name__ == "__main__":
    main()
