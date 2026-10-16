"""
Shared data-access layer for the Store Demand Predictor dashboard (Phase 5/6).

Design
------
Every page imports from this module instead of touching files directly. The
layer has two modes and picks automatically:

``trained``  ``data/processed/features.parquet`` + ``models/lgbm_v1.joblib``
             produced by ``python -m src.features.build`` and
             ``python -m src.models.build``. This is the production path.
``demo``     Neither artifact exists (the repository is checked out, or the
             Kaggle credentials have not been configured yet). A small
             synthetic panel is generated **through the real Phase 2 pipeline**
             and a compact LightGBM is trained on it, so every page exercises
             the genuine feature / model / engine code paths.
``forced``   ``SDP_FORCE_DEMO=1`` pins the layer to the synthetic corpus even
             when the real artifacts are present. The test-suite sets it so
             page tests stay hermetic and fast (they must not depend on 37 MB
             of local parquet, nor spend minutes forecasting a 20-store scope),
             and it doubles as a way to demo the UI without loading them.

Both paths are cached with ``st.cache_resource`` (features + model, loaded once
per process) and ``st.cache_data`` (derived frames, cheap to recompute).

Honesty note: the Kaggle Favorita dataset contains **no on-hand inventory
table**. Current stock levels are therefore a business assumption (or an input
from an ERP in production). :func:`get_stock_state` synthesises a deterministic
stock position so the inventory / transfer / deadstock engines have something to
work on, and every page surfaces a banner saying so.

Usage
-----
    from _data import get_data, get_forecast        # inside app/ pages
"""
from __future__ import annotations

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap — make this module importable from any CWD (pages, tests,
# ``streamlit run app/main.py``, AppTest, ...).
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (parent for parent in APP_DIR.parents if (parent / "src" / "config.py").exists()),
    APP_DIR.parent,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.engines import deadstock as engine_deadstock  # noqa: E402
from src.engines import inventory as engine_inventory  # noqa: E402
from src.engines import transfers as engine_transfers  # noqa: E402
from src.features import build as features_build  # noqa: E402
from src.features import calendar  # noqa: E402
from src.models import predict as model_predict  # noqa: E402
from src.models import train as model_train  # noqa: E402

log = logging.getLogger("app.data")

# ===========================================================================
# Demo corpus (used only when the real artifacts are missing)
# ===========================================================================
DEMO_SEED = 20240815
DEMO_STORES: tuple[int, ...] = (1, 2, 3)
DEMO_FAMILIES: tuple[str, ...] = (
    "BEVERAGES",
    "CLEANING",
    "DAIRY",
    "PRODUCE",
    "SNACKS",
    "TOYS",
)
DEMO_HISTORY_DAYS = 200
DEMO_END_DATE = pd.Timestamp("2017-08-15")

_DEMO_STORES_TABLE: dict[int, dict[str, Any]] = {
    1: {"cluster": 11, "city": "Quito", "state": "Pichincha", "type": "A"},
    2: {"cluster": 6, "city": "Guayaquil", "state": "Guayas", "type": "B"},
    3: {"cluster": 3, "city": "Cuenca", "state": "Azuay", "type": "C"},
}

# Rough average selling price per unit (Favorita ships no cost column).
DEMO_UNIT_COSTS: dict[str, float] = {
    "BEVERAGES": 3.40,
    "CLEANING": 5.60,
    "DAIRY": 4.10,
    "PRODUCE": 2.20,
    "SNACKS": 2.60,
    "TOYS": 12.50,
}

_DEMO_BASE_DEMAND: dict[str, float] = {
    "BEVERAGES": 150.0,
    "CLEANING": 18.0,
    "DAIRY": 90.0,
    "PRODUCE": 48.0,
    "SNACKS": 26.0,
    "TOYS": 8.0,
}
_DEMO_STORE_WEIGHT: dict[int, float] = {1: 1.15, 2: 0.95, 3: 0.75}
# TOYS is a seasonal line that stops selling this many days into the panel:
# it sells hard, then never sells again, which is exactly the deadstock shape.
_DEMO_STALL_AFTER_DAY = 80

# Per-family on-hand cover rotation, assigned in store order. Every active
# family gets a donor, a receiver and a healthy store so the transfer engine
# has a real rebalancing problem; the stalled family is stocked everywhere
# (clearance, not rebalancing).
_DEMO_COVER_TRIO: dict[str, tuple[float, float, float]] = {
    "BEVERAGES": (0.95, 2.40, 0.16),
    "CLEANING": (1.10, 2.20, 0.00),
    "DAIRY": (2.60, 1.05, 0.30),
    "PRODUCE": (0.00, 1.75, 2.40),
    "SNACKS": (0.20, 1.20, 2.30),
    "TOYS": (2.40, 2.10, 2.60),
}
_DEMO_COVER_FALLBACK: tuple[float, float, float] = (0.95, 2.40, 0.16)  # donor in the middle


def cover_multiplier(
    family: str,
    store_nbr: int,
    stores: Sequence[int] = DEMO_STORES,
) -> float:
    """
    Days-of-supply multiplier a demo series is stocked at (× target cover).

    Public rather than a closure so the demo corpus's structure is directly
    testable: every active family has a **donor** store (high cover), a
    **receiver** store (low or zero cover) and a healthy one, while the stalled
    line is overstocked everywhere — a clearance case, not a rebalancing one.
    Unknown families fall back to the healthy/donor/receiver trio.
    """
    trio = _DEMO_COVER_TRIO.get(str(family), _DEMO_COVER_FALLBACK)
    try:
        position = list(stores).index(int(store_nbr))
    except (TypeError, ValueError):
        position = 0
    return float(trio[position % 3])

# Ecuadorian national holidays, roughly, for the demo calendar.
_DEMO_HOLIDAY_MARKS: tuple[str, ...] = (
    "01-01", "02-27", "04-14", "05-01", "05-24",
    "08-10", "10-09", "11-02", "11-03", "12-25",
)


# ===========================================================================
# Policy knobs shared by the inventory / transfer / deadstock pages
# ===========================================================================
@dataclass(frozen=True)
class PolicyParams:
    """
    Scalar inputs the pages expose as widgets.

    Exposed as a frozen dataclass so a page can build one object from its
    sidebar and splat it into any cached accessor — Streamlit hashes the field
    values, so changing a slider invalidates exactly the caches it should.
    """

    # inventory
    service_level: float = config.SERVICE_LEVEL
    lead_time_days: float = config.LEAD_TIME_DAYS
    review_period_days: float = config.REVIEW_PERIOD_DAYS
    lead_time_std_days: float = config.LEAD_TIME_STD_DAYS
    # transfers
    min_transfer_units: int = config.MIN_TRANSFER_UNITS
    target_dos: float = config.TRANSFER_TARGET_DOS
    max_transfer_units: int = config.MAX_TRANSFER_UNITS
    # deadstock
    window_days: int = config.DEADSTOCK_WINDOW_DAYS
    velocity_threshold: float = config.DEADSTOCK_VELOCITY_THRESHOLD
    min_days_since_last_sale: int = config.DEADSTOCK_MIN_DAYS_SINCE_LAST_SALE
    discount_base: float = config.DEADSTOCK_DISCOUNT_BASE
    top_n_complements: int = config.BUNDLE_TOP_N_COMPLEMENTS

    def flat(self) -> dict[str, Any]:
        """Field name -> value, for ``f(**params.flat())`` into cached fns."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


# ===========================================================================
# Demo corpus builders (pure functions — no Streamlit state)
# ===========================================================================
def build_demo_sales() -> pd.DataFrame:
    """
    Deterministic synthetic cleaned panel in :data:`config.CLEAN_COLUMN_ORDER`.

    The series are engineered so every downstream engine has real signal to
    find: weekend seasonality, a mild upward trend, promotional lifts, a
    long-tail item that stops selling (deadstock), and store-level size
    differences (transfers).
    """
    dates = pd.date_range(
        DEMO_END_DATE - pd.Timedelta(days=DEMO_HISTORY_DAYS - 1),
        DEMO_END_DATE,
        freq="D",
    )
    n_days = len(dates)
    rng = np.random.default_rng(DEMO_SEED)

    # Monday..Sunday — the Favorita dataset peaks at the weekend.
    weekday = np.array([1.16, 0.90, 0.88, 0.93, 0.99, 1.20, 1.34])
    season = weekday[dates.dayofweek.to_numpy()]
    trend = 1.0 + 0.0018 * np.arange(n_days)

    frames: list[pd.DataFrame] = []
    for store in DEMO_STORES:
        attrs = _DEMO_STORES_TABLE[store]
        weight = _DEMO_STORE_WEIGHT[store]
        for family in DEMO_FAMILIES:
            base = _DEMO_BASE_DEMAND[family] * weight
            promo = rng.random(n_days) < 0.12
            lift = np.where(promo, 1.35, 1.0)
            noise = rng.lognormal(mean=0.0, sigma=0.16, size=n_days)
            level = base * season * trend * lift * noise
            if family == "TOYS":
                level = level * (np.arange(n_days) <= _DEMO_STALL_AFTER_DAY)

            frames.append(
                pd.DataFrame(
                    {
                        config.DATE_COL: dates,
                        "store_nbr": np.int16(store),
                        "family": family,
                        config.TARGET_COL: np.round(
                            np.clip(level, 0.0, None), 2
                        ).astype("float32"),
                        "onpromotion": promo.astype("int8"),
                        "cluster": np.int8(attrs["cluster"]),
                        "city": attrs["city"],
                        "state": attrs["state"],
                        "type": attrs["type"],
                    }
                )
            )

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(
        [*config.GROUP_KEYS, config.DATE_COL], kind="mergesort"
    ).reset_index(drop=True)
    out[config.ID_COL] = np.arange(len(out), dtype="int64")
    for column in ("family", "city", "state", "type"):
        out[column] = out[column].astype("category")
    return out[config.CLEAN_COLUMN_ORDER]


def build_demo_holidays() -> pd.DataFrame:
    """National-holiday calendar with the exact ``holidays_events.csv`` schema."""
    rows: list[dict[str, Any]] = []
    for year in (2016, 2017, 2018):
        for mark in _DEMO_HOLIDAY_MARKS:
            rows.append(
                {
                    "date": pd.Timestamp(f"{year}-{mark}"),
                    "type": "Holiday",
                    "locale": "National",
                    "locale_name": "Ecuador",
                    "description": f"Demo holiday {year}-{mark}",
                    "transferred": False,
                }
            )
    frame = pd.DataFrame(rows)
    for column in ("type", "locale", "locale_name"):
        frame[column] = frame[column].astype("category")
    return frame


def build_demo_stock(
    sales: pd.DataFrame,
    unit_costs: dict[str, float] | None = None,
    target_dos: float = config.TRANSFER_TARGET_DOS,
) -> pd.DataFrame:
    """
    Synthetic on-hand balance per ``(store, family)``.

    Cover is rotated through healthy / overstocked / understocked levels so the
    transfer engine finds a rebalancing problem worth solving, and the deadstock
    engine finds stock worth clearing.
    """
    keys = list(config.GROUP_KEYS)
    last_date = sales[config.DATE_COL].max()

    full_rate = (
        sales.groupby(keys, observed=True, sort=False)[config.TARGET_COL]
        .mean()
        .rename("full_rate")
    )
    recent = sales[sales[config.DATE_COL] > last_date - pd.Timedelta(days=60)]
    recent_rate = (
        recent.groupby(keys, observed=True, sort=False)[config.TARGET_COL]
        .mean()
        .rename("recent_rate")
    )

    frame = full_rate.to_frame().join(recent_rate, how="left").reset_index()
    # A stalled item has ~0 recent demand but still sits on stock; the policy
    # must size against the rate at which it *used* to sell.
    frame["avg_daily_demand"] = frame[["full_rate", "recent_rate"]].max(axis=1).clip(lower=0.4)

    stores = [int(v) for v in sorted(frame["store_nbr"].unique().tolist())]
    frame["dos_multiplier"] = [
        cover_multiplier(row.family, row.store_nbr, stores)
        for row in frame.itertuples(index=False)
    ]
    frame["on_hand"] = (
        frame["avg_daily_demand"] * target_dos * frame["dos_multiplier"]
    ).round(1)
    frame["on_order"] = 0.0

    costs = unit_costs or DEMO_UNIT_COSTS
    frame["unit_cost"] = (
        frame["family"].astype(str).map(costs).fillna(config.DEFAULT_UNIT_COST)
    )
    return frame[[*keys, "on_hand", "on_order", "unit_cost"]].astype(
        {"on_hand": "float64", "on_order": "float64", "unit_cost": "float64"}
    )


# ===========================================================================
# The bundle: features + model, loaded once per process
# ===========================================================================
@dataclass
class DataBundle:
    """Everything the dashboard needs, loaded (or built) exactly once."""

    features: pd.DataFrame
    sales: pd.DataFrame
    holidays: pd.DataFrame
    model: dict[str, Any]
    offline_forecasts: pd.DataFrame | None
    model_source: str
    is_demo: bool
    source_note: str
    stores: tuple[int, ...]
    families: tuple[str, ...]
    unit_costs: dict[str, float]
    last_history_date: pd.Timestamp
    feature_report: dict[str, Any]


def _force_demo() -> bool:
    """
    ``SDP_FORCE_DEMO=1`` pins the layer to the synthetic corpus.

    Read at call time so a test process (or a demo session) can flip it with an
    environment variable. :mod:`tests.test_app` sets it so the page tests stay
    hermetic: they must not depend on 37 MB of local artifacts, nor spend
    minutes forecasting the full 20-store scope.
    """
    return os.environ.get("SDP_FORCE_DEMO", "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_real_bundle() -> DataBundle | None:
    """Load Phase 2/3 artifacts when they exist, else ``None``."""
    if not (config.FEATURES_PARQUET.exists() and config.MODEL_ARTIFACT.exists()):
        return None

    features = pd.read_parquet(config.FEATURES_PARQUET)
    sales = (
        pd.read_parquet(config.SALES_CLEAN_PARQUET)
        if config.SALES_CLEAN_PARQUET.exists()
        else features[config.CLEAN_COLUMN_ORDER].copy()
    )
    if config.HOLIDAYS_CSV.exists():
        # Pre-filter once: forecast_series re-filters internally on every
        # recursion step, and this is a pure (idempotent) row filter.
        holidays = calendar.filter_holidays(calendar.load_holidays(config.HOLIDAYS_CSV))
    else:
        holidays = calendar.empty_holidays()
    artifact = model_train.load_model(config.MODEL_ARTIFACT)

    offline = None
    if config.FORECASTS_PARQUET.exists():
        candidate = pd.read_parquet(config.FORECASTS_PARQUET)
        required = {*config.GROUP_KEYS, config.DATE_COL, "prediction"}
        if required.issubset(candidate.columns):
            offline = candidate

    return DataBundle(
        features=features,
        sales=sales,
        holidays=holidays,
        model=artifact,
        offline_forecasts=offline,
        model_source=(
            f"trained artifact ({config.MODEL_ARTIFACT.name})"
        ),
        is_demo=False,
        source_note="Kaggle Corporación Favorita data (real artifacts).",
        stores=tuple(int(v) for v in sorted(features["store_nbr"].unique())),
        families=tuple(str(v) for v in sorted(features["family"].unique())),
        unit_costs={},
        last_history_date=features[config.DATE_COL].max(),
        feature_report={"n_rows": int(len(features)), "source": "features.parquet"},
    )


def _build_demo_bundle() -> DataBundle:
    """Synthetic panel through the real pipeline + a compact LightGBM."""
    sales = build_demo_sales()
    holidays = calendar.filter_holidays(build_demo_holidays())
    features = features_build.build_features(sales, holidays, validate=True)

    # A short, fixed-budget fit: the demo model only has to be a real LightGBM
    # that the forecast engine can drive, not an accurate one.
    result = model_train.train_model(
        features,
        params={
            "num_leaves": 31,
            "min_data_in_leaf": 5,
            "learning_rate": 0.08,
            "feature_fraction": 0.9,
        },
        num_boost_round=200,
        early_stopping_rounds=None,
    )
    artifact = result.to_artifact(extra={"source": "dashboard-demo"})

    return DataBundle(
        features=features,
        sales=sales,
        holidays=holidays,
        model=artifact,
        offline_forecasts=None,
        model_source="demo LightGBM (trained in-process on bundled sample data)",
        is_demo=True,
        source_note=(
            "Synthetic sample panel: no Kaggle download is configured yet. "
            "Run Phase 1–3 to switch this dashboard to the real dataset."
        ),
        stores=DEMO_STORES,
        families=DEMO_FAMILIES,
        unit_costs=dict(DEMO_UNIT_COSTS),
        last_history_date=features[config.DATE_COL].max(),
        feature_report={"n_rows": int(len(features)), "source": "synthetic"},
    )


# ===========================================================================
# Cached accessors
# ===========================================================================
@st.cache_resource(show_spinner=False, max_entries=2)
def get_data() -> DataBundle:
    """
    Load features + model exactly once per Streamlit process.

    Tries the real Phase 2/3 artifacts first and falls back to the in-process
    demo corpus, so ``streamlit run app/main.py`` works on a fresh checkout.
    """
    if _force_demo():
        return _build_demo_bundle()
    try:
        bundle = _resolve_real_bundle()
        if bundle is None:
            bundle = _build_demo_bundle()
    except Exception:  # pragma: no cover - corrupted/partial artifacts
        log.exception("Real artifacts failed to load; falling back to demo data.")
        bundle = _build_demo_bundle()
    return bundle


@st.cache_data(show_spinner=False, max_entries=8)
def get_stock_state() -> pd.DataFrame:
    """
    Current on-hand balance per series.

    Favourita ships no inventory table, so stock is derived from observed
    demand (see :func:`build_demo_stock`). The frame always carries
    ``on_hand`` / ``on_order`` / ``unit_cost`` columns so it can be fed
    straight to :func:`src.engines.inventory.inventory_policy_table`.
    """
    bundle = get_data()
    return build_demo_stock(bundle.sales, bundle.unit_costs)


@st.cache_data(show_spinner=False, max_entries=16)
def _band_factors(stores: tuple[int, ...], families: tuple[str, ...]) -> pd.DataFrame:
    """Per-series day-to-day demand sigma (drives the forecast band)."""
    bundle = get_data()
    frame = bundle.features
    if stores:
        frame = frame[frame["store_nbr"].isin(stores)]
    if families:
        frame = frame[frame["family"].isin(families)]
    if frame.empty:
        return pd.DataFrame(columns=[*config.GROUP_KEYS, "sigma_1"])

    tail = frame[frame[config.DATE_COL] > frame[config.DATE_COL].max() - pd.Timedelta(days=90)]
    if tail.empty:
        tail = frame
    diffs = tail.sort_values([*config.GROUP_KEYS, config.DATE_COL]).copy()
    diffs["_diff"] = diffs.groupby(config.GROUP_KEYS, observed=True)[config.TARGET_COL].diff()
    sigma = (
        diffs.groupby(config.GROUP_KEYS, observed=True, sort=False)["_diff"]
        .std(ddof=1)
        .reset_index(name="sigma_1")
    )
    return sigma


def _attach_band(
    predictions: pd.DataFrame, stores: tuple[int, ...], families: tuple[int, ...]
) -> pd.DataFrame:
    """
    Add a ``lower`` / ``upper`` band to a forecast frame.

    LightGBM regression returns point estimates only, so the band is a
    documented approximation: ``prediction ± 1σ·√step`` where ``σ`` is the
    observed day-to-day demand volatility of that series (random-walk growth).
    It widens with horizon and is labelled as such in the UI — it is not a
    calibrated quantile interval.
    """
    keys = list(config.GROUP_KEYS)
    out = predictions.sort_values([*keys, config.DATE_COL]).reset_index(drop=True)
    out["step"] = out.groupby(keys, observed=True).cumcount() + 1

    sigma = _band_factors(stores, families)
    out = out.merge(sigma, on=keys, how="left")
    out["sigma_1"] = pd.to_numeric(out["sigma_1"], errors="coerce").fillna(0.0).clip(lower=0.0)

    width = out["sigma_1"] * np.sqrt(out["step"].astype("float64"))
    out["lower"] = (out["prediction"] - width).clip(lower=0.0)
    out["upper"] = out["prediction"] + width
    return out


# Scopes at or below this many series are cheaper to forecast in one go than
# to recompute per filter combination (the recursion rebuilds Phase 2 features
# once per forecast day, per series).
CANONICAL_MAX_SERIES = 60


@st.cache_data(show_spinner=True, max_entries=2)
def _canonical_forecast(future_onpromotion: int | None) -> pd.DataFrame:
    """
    One recursion over **every** series in the bundle at ``FORECAST_HORIZON``.

    This is the only expensive call in the data layer: the recursion rebuilds
    the Phase 2 feature matrix once per forecast day, per series. Caching it
    once per process means every page, filter combination and horizon
    afterwards is a *slice* rather than a recomputation.
    """
    bundle = get_data()
    return model_predict.forecast_all(
        bundle.model,
        bundle.features,
        horizon=config.FORECAST_HORIZON,
        holidays=bundle.holidays,
        future_onpromotion=future_onpromotion,
    )


@st.cache_data(show_spinner=True, max_entries=16)
def get_forecast(
    stores: tuple[int, ...],
    families: tuple[str, ...],
    horizon: int = config.FORECAST_HORIZON,
    use_offline: bool = True,
    future_onpromotion: int | None = None,
) -> pd.DataFrame:
    """
    Recursive ``horizon``-day forecast for the selected series, with a band.

    Resolution order:

    1. **Offline batch** (``data/processed/forecasts.parquet``) when it covers
       the request — the production path for a 600-series scope.
    2. **Canonical recursion** when the bundle's total scope is small enough to
       forecast whole (demo corpus, or a narrow real scope): one cached
       computation, then every filter/horizon is an instant slice.
    3. **Direct subset computation** otherwise — slow, so callers should pass
       the offline artifact for large scopes.

    ``future_onpromotion`` is the Phase 3 scenario hook: ``0``/``1`` forces the
    promotion flag for every forecast day (``None`` keeps the model's default).
    Scenarios are a forecasting what-if — the decision engines ignore them.
    """
    if not stores or not families:
        return pd.DataFrame(
            columns=[*config.GROUP_KEYS, config.DATE_COL, "prediction", "step",
                     "sigma_1", "lower", "upper"]
        )

    bundle = get_data()
    names = list(config.GROUP_KEYS)
    wanted = int(horizon)

    # 1) offline batch artifact (invalidated by a scenario)
    if use_offline and future_onpromotion is None and bundle.offline_forecasts is not None:
        source = bundle.offline_forecasts
        masked = source[
            source["store_nbr"].isin(stores) & source["family"].isin(families)
        ]
        counts = masked.groupby(names, observed=True)[config.DATE_COL].nunique()
        expected = len(stores) * len(families)
        if len(counts) >= expected and counts.min() >= wanted:
            frame = model_predict.truncate_horizon(masked.copy(), wanted)
            return _attach_band(frame[[*names, config.DATE_COL, "prediction"]],
                                stores, families)

    # 2) canonical, cached whole-bundle recursion
    total_series = len(bundle.stores) * len(bundle.families)
    if total_series <= CANONICAL_MAX_SERIES:
        whole = _canonical_forecast(future_onpromotion)
        sliced = whole[
            whole["store_nbr"].isin(stores) & whole["family"].isin(families)
        ]
        if not sliced.empty:
            sliced = model_predict.truncate_horizon(sliced.copy(), wanted)
            return _attach_band(sliced, stores, families)

    # 3) direct subset computation
    predictions = model_predict.forecast_all(
        bundle.model,
        bundle.features,
        horizon=wanted,
        stores=list(stores),
        families=list(families),
        holidays=bundle.holidays,
        future_onpromotion=future_onpromotion,
    )
    predictions = model_predict.truncate_horizon(predictions, wanted)
    return _attach_band(predictions, stores, families)


@st.cache_data(show_spinner=False, max_entries=16)
def get_history(
    stores: tuple[int, ...],
    families: tuple[str, ...],
    days: int = 90,
) -> pd.DataFrame:
    """Observed daily sales for the selected series over the trailing window."""
    if not stores or not families:
        return pd.DataFrame(columns=[*config.GROUP_KEYS, config.DATE_COL, config.TARGET_COL])

    bundle = get_data()
    frame = bundle.features
    frame = frame[
        frame["store_nbr"].isin(stores) & frame["family"].isin(families)
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=[*config.GROUP_KEYS, config.DATE_COL, config.TARGET_COL])

    cutoff = frame[config.DATE_COL].max() - pd.Timedelta(days=int(days) - 1)
    frame = frame[frame[config.DATE_COL] >= cutoff]
    return frame[[*config.GROUP_KEYS, config.DATE_COL, config.TARGET_COL]].reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=8)
def get_demand_summary(
    stores: tuple[int, ...],
    families: tuple[str, ...],
    horizon: int = config.FORECAST_HORIZON,
) -> pd.DataFrame:
    """Per-series demand statistics feeding the inventory policy."""
    forecasts = get_forecast(stores, families, horizon)
    if forecasts.empty:
        return pd.DataFrame(columns=list(engine_inventory.DEMAND_SUMMARY_COLUMNS))
    return engine_inventory.demand_summary_from_forecast(forecasts, horizon=horizon)


@st.cache_data(show_spinner=False, max_entries=8)
def get_inventory_policy(
    stores: tuple[int, ...],
    families: tuple[str, ...],
    horizon: int = config.FORECAST_HORIZON,
    service_level: float = config.SERVICE_LEVEL,
    lead_time_days: float = config.LEAD_TIME_DAYS,
    review_period_days: float = config.REVIEW_PERIOD_DAYS,
    lead_time_std_days: float = config.LEAD_TIME_STD_DAYS,
) -> pd.DataFrame:
    """
    Safety stock / reorder point table for the selected scope.

    This frame doubles as the input to :func:`get_transfers`, so the transfer
    page and the inventory page can never disagree about safety stock.
    """
    demand = get_demand_summary(stores, families, horizon)
    if demand.empty:
        return pd.DataFrame(columns=list(engine_inventory.POLICY_COLUMNS))
    return engine_inventory.inventory_policy_table(
        demand,
        get_stock_state(),
        service_level=float(service_level),
        lead_time_days=float(lead_time_days),
        review_period_days=float(review_period_days),
        lead_time_std_days=float(lead_time_std_days),
        unit_costs={},
    )


@st.cache_data(show_spinner=False, max_entries=8)
def get_position_table(
    stores: tuple[int, ...],
    families: tuple[str, ...],
    horizon: int = config.FORECAST_HORIZON,
    service_level: float = config.SERVICE_LEVEL,
    lead_time_days: float = config.LEAD_TIME_DAYS,
    review_period_days: float = config.REVIEW_PERIOD_DAYS,
    lead_time_std_days: float = config.LEAD_TIME_STD_DAYS,
) -> pd.DataFrame:
    """Policy table + unit costs, i.e. the ideal ``positions`` input."""
    policy = get_inventory_policy(
        stores,
        families,
        horizon,
        service_level,
        lead_time_days,
        review_period_days,
        lead_time_std_days,
    )
    if policy.empty:
        return policy
    bundle = get_data()
    if bundle.unit_costs:
        policy = policy.copy()
        policy["unit_cost"] = (
            policy["family"].astype(str)
            .map(bundle.unit_costs)
            .fillna(policy["unit_cost"])
        )
    return policy


@st.cache_data(show_spinner=False, max_entries=8)
def get_transfers(
    stores: tuple[int, ...],
    families: tuple[str, ...],
    horizon: int = config.FORECAST_HORIZON,
    service_level: float = config.SERVICE_LEVEL,
    lead_time_days: float = config.LEAD_TIME_DAYS,
    review_period_days: float = config.REVIEW_PERIOD_DAYS,
    lead_time_std_days: float = config.LEAD_TIME_STD_DAYS,
    min_transfer_units: int = config.MIN_TRANSFER_UNITS,
    target_dos: float = config.TRANSFER_TARGET_DOS,
    max_transfer_units: int = config.MAX_TRANSFER_UNITS,
) -> pd.DataFrame:
    """Recommended inter-store transfer legs for the selected scope."""
    positions = get_position_table(
        stores,
        families,
        horizon,
        service_level,
        lead_time_days,
        review_period_days,
        lead_time_std_days,
    )
    if positions.empty:
        return pd.DataFrame(columns=list(engine_transfers.TRANSFER_COLUMNS))

    policy = engine_transfers.TransferPolicy(
        min_transfer_units=int(min_transfer_units),
        max_transfer_units=int(max_transfer_units),
        target_dos=float(target_dos),
    )
    plan = engine_transfers.generate_transfers(positions, policy=policy)
    if plan.empty:
        return plan
    return plan.sort_values(
        ["family", "from_store", "to_store"], kind="stable"
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=8)
def get_positions(
    stores: tuple[int, ...],
    families: tuple[str, ...],
    horizon: int = config.FORECAST_HORIZON,
    service_level: float = config.SERVICE_LEVEL,
    lead_time_days: float = config.LEAD_TIME_DAYS,
    review_period_days: float = config.REVIEW_PERIOD_DAYS,
    lead_time_std_days: float = config.LEAD_TIME_STD_DAYS,
) -> pd.DataFrame:
    """Full stock classification (surplus / deficit / healthy / no-demand)."""
    positions = get_position_table(
        stores,
        families,
        horizon,
        service_level,
        lead_time_days,
        review_period_days,
        lead_time_std_days,
    )
    if positions.empty:
        return positions
    policy = engine_transfers.TransferPolicy()
    return engine_transfers.classify_stock_positions(positions, policy=policy)


@st.cache_data(show_spinner=True, max_entries=8)
def get_deadstock(
    stores: tuple[int, ...],
    families: tuple[str, ...],
    horizon: int = config.FORECAST_HORIZON,
    window_days: int = config.DEADSTOCK_WINDOW_DAYS,
    velocity_threshold: float = config.DEADSTOCK_VELOCITY_THRESHOLD,
    min_days_since_last_sale: int = config.DEADSTOCK_MIN_DAYS_SINCE_LAST_SALE,
    discount_base: float = config.DEADSTOCK_DISCOUNT_BASE,
    top_n_complements: int = config.BUNDLE_TOP_N_COMPLEMENTS,
) -> dict[str, Any]:
    """
    Detection, hero ranking, affinity matrix, bundles and clearance summary.

    Returned as one dict so a page makes a single cached call.
    """
    bundle = get_data()

    sales = bundle.sales
    if stores:
        sales = sales[sales["store_nbr"].isin(stores)]
    if families:
        sales = sales[sales["family"].isin(families)]

    stock = get_stock_state()
    if stores:
        stock = stock[stock["store_nbr"].isin(stores)]
    if families:
        stock = stock[stock["family"].isin(families)]

    policy = engine_deadstock.DeadstockPolicy(
        window_days=int(window_days),
        velocity_threshold=float(velocity_threshold),
        min_days_since_last_sale=int(min_days_since_last_sale),
        discount_base=float(discount_base),
        top_n_complements=int(top_n_complements),
    )

    detected = engine_deadstock.detect_deadstock(sales, stock, policy=policy)
    affinity = (
        engine_deadstock.build_affinity_matrix(sales, method="residual")
        if len(sales)
        else None
    )
    heroes = (
        engine_deadstock.select_hero_items(sales, window_days=int(window_days), top_n=10)
        if len(sales)
        else None
    )
    bundles = engine_deadstock.generate_bundles(
        detected, affinity, heroes, policy=policy
    )
    summary = engine_deadstock.clearance_summary(detected, bundles)

    return {
        "detection": detected,
        "affinity": affinity,
        "heroes": heroes,
        "bundles": bundles,
        "summary": summary,
        "policy": policy,
    }


# ===========================================================================
# Page-level view helpers (small, stateless)
# ===========================================================================
def scope_options() -> tuple[tuple[int, ...], tuple[str, ...]]:
    """(all stores, all families) available in the loaded bundle."""
    bundle = get_data()
    return bundle.stores, bundle.families


def is_demo() -> bool:
    """True when the dashboard is running on the bundled synthetic corpus."""
    return get_data().is_demo


def source_note() -> str:
    """Human-readable provenance line shown on every page."""
    return get_data().source_note


def model_source() -> str:
    """Human-readable model provenance line."""
    return get_data().model_source


def forecast_hygiene(stores: Sequence[Any], families: Sequence[Any]) -> bool:
    """False when a caller passed an empty scope (pages show an empty state)."""
    return bool(stores) and bool(families)
