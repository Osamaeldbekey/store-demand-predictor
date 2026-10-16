"""
Safety-stock, reorder-point and order-quantity engine.

This module turns a demand forecast into an inventory *policy* per
``(store_nbr, family)`` SKU-store combination:

* **Safety stock (SS)** — the buffer that absorbs demand and forecast
  variability over the replenishment protection period::

      SS = Z * sqrt( P * sigma_d^2 + d_bar^2 * sigma_L^2 )

  where ``P = lead_time + review_period`` is the protection period,
  ``sigma_d`` is the daily demand standard deviation (optionally inflated by
  the model's forecast error), ``d_bar`` is mean daily demand and ``sigma_L``
  is the lead-time standard deviation. With a constant lead time
  (``sigma_L = 0``) this collapses to the textbook ``Z * sigma_d * sqrt(P)``.

* **Reorder point (ROP)** — the on-hand level that triggers a purchase::

      ROP = d_bar * L + SS

* **Order-up-to level** — the target for periodic review::

      target = d_bar * (L + R) + SS

* **Recommended order quantity** — the gap between the target and the current
  inventory position (on-hand + on-order), floored to whole units, lifted to
  the minimum order quantity and rounded up to a case-pack multiple.

Design notes
------------
* The engine is a **pure function of numbers and frames** — no model, no I/O —
  so it is trivially unit-testable and reusable by the dashboard.
* ``inventory_policy`` returns a **structured dictionary** (the single-SKU
  primitive); ``inventory_policy_table`` vectorises it across a whole SKU
  portfolio and returns a DataFrame.
* Zero-demand series are first-class: they yield ``SS = ROP = 0``, a
  ``days_of_supply`` of ``inf`` and no stockout risk, instead of dividing by
  zero.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Mapping, Sequence

import pandas as pd
from scipy.stats import norm

from src import config
from src.engines._common import (
    canonicalize_keys,
    non_negative,
    require_columns,
    resolve_unit_costs,
)

log = logging.getLogger("engines.inventory")

# Ordered keys of the dictionary returned by :func:`inventory_policy`. Declared
# as a constant so the empty-table fallback and the populated table always
# expose the same schema.
POLICY_COLUMNS: tuple[str, ...] = (
    "store_nbr",
    "family",
    "avg_daily_demand",
    "demand_std",
    "forecast_error_std",
    "lead_time_days",
    "review_period_days",
    "protection_days",
    "lead_time_std_days",
    "service_level",
    "z_score",
    "safety_stock",
    "reorder_point",
    "order_up_to_level",
    "expected_lead_time_demand",
    "on_hand",
    "on_order",
    "inventory_position",
    "unit_cost",
    "days_of_supply",
    "cover_days",
    "reorder_quantity",
    "reorder_value",
    "needs_reorder",
    "stockout_risk",
    "below_safety_stock",
    "is_out_of_stock",
    "overstock_risk",
    "stockout_risk_level",
)

# Columns produced by :func:`demand_summary_from_forecast`.
DEMAND_SUMMARY_COLUMNS: tuple[str, ...] = (
    "store_nbr",
    "family",
    "n_days",
    "avg_daily_demand",
    "demand_std",
    "total_demand",
    "min_daily_demand",
    "max_daily_demand",
    "last_forecast_date",
)


# ===========================================================================
# Primitives
# ===========================================================================
def z_score(service_level: float) -> float:
    """
    Standard-normal quantile for a target cycle service level.

    Parameters
    ----------
    service_level:
        Probability of not stocking out during the protection period,
        strictly between 0 and 1 (e.g. ``0.95`` → ``1.6449``).

    Returns
    -------
    float
        The ``z`` multiplier.

    Raises
    ------
    ValueError
        If ``service_level`` is not strictly inside ``(0, 1)``.
    """
    level = float(service_level)
    if not 0.0 < level < 1.0:
        raise ValueError(
            f"service_level must be strictly between 0 and 1, got {service_level!r}."
        )
    return float(norm.ppf(level))


def _round_up_to_multiple(quantity: float, multiple: float) -> float:
    """Ceil ``quantity`` to the next ``multiple`` (whole units when multiple<=0)."""
    value = non_negative(quantity)
    if value <= 0.0:
        return 0.0
    step = non_negative(multiple)
    if step <= 0.0:
        return float(math.ceil(value))
    return float(math.ceil(value / step) * step)


def safety_stock(
    avg_daily_demand: float,
    demand_std: float,
    lead_time_days: float,
    service_level: float = config.SERVICE_LEVEL,
    review_period_days: float = config.REVIEW_PERIOD_DAYS,
    lead_time_std_days: float = config.LEAD_TIME_STD_DAYS,
    forecast_error_std: float = 0.0,
) -> float:
    """
    Buffer stock required to hit ``service_level`` over the protection period.

    Parameters
    ----------
    avg_daily_demand:
        Mean demand per day (``d_bar``); negatives are clipped to 0.
    demand_std:
        Standard deviation of *daily* demand (``sigma_d``); clipped to 0.
    lead_time_days:
        Mean replenishment lead time in days (``L``).
    service_level:
        Target cycle service level, in ``(0, 1)``.
    review_period_days:
        Review interval ``R`` for periodic review; ``0`` for continuous review.
    lead_time_std_days:
        Standard deviation of the lead time (``sigma_L``); ``0`` when the
        supplier lead time is deterministic.
    forecast_error_std:
        Optional standard deviation of the model's one-day forecast error. It
        is combined with ``demand_std`` in quadrature, because safety stock has
        to absorb both natural demand noise *and* model error.

    Returns
    -------
    float
        Safety stock in units (``>= 0``, ``0.0`` when demand is perfectly flat
        and the lead time is deterministic).
    """
    z = z_score(service_level)
    mean_daily = non_negative(avg_daily_demand)
    sigma_daily = math.sqrt(non_negative(demand_std) ** 2 + non_negative(forecast_error_std) ** 2)
    protection_days = non_negative(lead_time_days) + non_negative(review_period_days)
    sigma_lead = non_negative(lead_time_std_days)

    variance = protection_days * sigma_daily**2 + (mean_daily**2) * (sigma_lead**2)
    return float(z * math.sqrt(variance))


def expected_lead_time_demand(avg_daily_demand: float, lead_time_days: float) -> float:
    """Mean demand expected to be consumed while a replenishment is in transit."""
    return non_negative(avg_daily_demand) * non_negative(lead_time_days)


def reorder_point(
    avg_daily_demand: float,
    lead_time_days: float,
    safety_stock_value: float,
) -> float:
    """
    On-hand level that should trigger a purchase order.

    ``ROP = E[lead-time demand] + safety stock``.
    """
    return expected_lead_time_demand(avg_daily_demand, lead_time_days) + non_negative(
        safety_stock_value
    )


def order_up_to_level(
    avg_daily_demand: float,
    lead_time_days: float,
    safety_stock_value: float,
    review_period_days: float = 0.0,
) -> float:
    """
    Target inventory position under periodic review (order-up-to level).

    ``target = d_bar * (L + R) + SS``. With ``R = 0`` this equals the
    reorder point.
    """
    protection_days = non_negative(lead_time_days) + non_negative(review_period_days)
    return expected_lead_time_demand(avg_daily_demand, protection_days) + non_negative(
        safety_stock_value
    )


def days_of_supply(on_hand: float, avg_daily_demand: float) -> float:
    """
    How many days the current stock lasts at the forecast demand rate.

    Returns ``math.inf`` when demand is zero (the stock never runs out), which
    lets callers compare DOS values without special-casing slow movers.
    """
    mean_daily = non_negative(avg_daily_demand)
    if mean_daily <= 0.0:
        return math.inf
    return non_negative(on_hand) / mean_daily


def recommended_order_quantity(
    on_hand: float,
    on_order: float,
    target_level: float,
    min_order_qty: float = config.MIN_ORDER_QUANTITY,
    order_multiple: float = config.ORDER_MULTIPLE,
) -> float:
    """
    Units to order to restore the inventory position to ``target_level``.

    ``gap = target - (on_hand + on_order)``; non-positive gaps yield ``0``.
    A positive gap is lifted to ``min_order_qty`` and rounded *up* to the next
    ``order_multiple`` (case pack / pallet), so the result is always a
    physically orderable quantity.
    """
    gap = non_negative(target_level) - non_negative(on_hand) - non_negative(on_order)
    if gap <= 0.0:
        return 0.0
    gap = max(gap, non_negative(min_order_qty))
    return _round_up_to_multiple(gap, order_multiple)


# ===========================================================================
# Single-SKU policy
# ===========================================================================
def inventory_policy(
    avg_daily_demand: float,
    demand_std: float,
    on_hand: float,
    lead_time_days: float = config.LEAD_TIME_DAYS,
    service_level: float = config.SERVICE_LEVEL,
    review_period_days: float = config.REVIEW_PERIOD_DAYS,
    lead_time_std_days: float = config.LEAD_TIME_STD_DAYS,
    forecast_error_std: float = 0.0,
    on_order: float = 0.0,
    unit_cost: float = config.DEFAULT_UNIT_COST,
    min_order_qty: float = config.MIN_ORDER_QUANTITY,
    order_multiple: float = config.ORDER_MULTIPLE,
    overstock_dos_threshold: float = config.OVERSTOCK_DOS_THRESHOLD,
    store_nbr: int | None = None,
    family: str | None = None,
) -> dict[str, Any]:
    """
    Full inventory policy for one ``(store_nbr, family)`` SKU-store pair.

    Returns
    -------
    dict
        Structured result with the safety stock, reorder point, order-up-to
        level, recommended order quantity and value, coverage measures and the
        boolean operational flags:

        ``stockout_risk``
            On-hand stock is below the reorder point — a stockout is likely
            within the lead time unless replenished.
        ``below_safety_stock``
            On-hand stock has eaten into the safety buffer.
        ``needs_reorder``
            The *inventory position* (on-hand + on-order) is below the reorder
            point, i.e. another order should be raised now.
        ``is_out_of_stock``
            On-hand stock is zero (or negative).
        ``overstock_risk``
            Days of supply exceed the overstock threshold (a transfer/clearance
            candidate rather than a purchase candidate).
        ``stockout_risk_level``
            ``"none"`` / ``"medium"`` / ``"high"`` / ``"critical"`` escalation
            label for dashboards and alerting.
    """
    mean_daily = non_negative(avg_daily_demand)
    sigma_daily = non_negative(demand_std)
    error_std = non_negative(forecast_error_std)
    current_on_hand = non_negative(on_hand)
    current_on_order = non_negative(on_order)
    cost = non_negative(unit_cost)

    safety_stock_value = safety_stock(
        avg_daily_demand=mean_daily,
        demand_std=sigma_daily,
        lead_time_days=lead_time_days,
        service_level=service_level,
        review_period_days=review_period_days,
        lead_time_std_days=lead_time_std_days,
        forecast_error_std=error_std,
    )
    rop = reorder_point(mean_daily, lead_time_days, safety_stock_value)
    target = order_up_to_level(
        mean_daily, lead_time_days, safety_stock_value, review_period_days
    )
    inventory_position = current_on_hand + current_on_order
    order_quantity = recommended_order_quantity(
        on_hand=current_on_hand,
        on_order=current_on_order,
        target_level=target,
        min_order_qty=min_order_qty,
        order_multiple=order_multiple,
    )

    dos = days_of_supply(current_on_hand, mean_daily)
    cover = days_of_supply(inventory_position, mean_daily)

    has_demand = mean_daily > 0.0
    is_out_of_stock = current_on_hand <= 0.0
    stockout_risk = bool(has_demand and current_on_hand < rop)
    below_safety = bool(has_demand and current_on_hand < safety_stock_value)
    needs_reorder = bool(has_demand and inventory_position < rop)
    overstock = bool(has_demand and math.isfinite(dos) and dos > non_negative(overstock_dos_threshold))

    if not stockout_risk:
        risk_level = "none"
    elif is_out_of_stock:
        risk_level = "critical"
    elif below_safety:
        risk_level = "high"
    else:
        risk_level = "medium"

    return {
        "store_nbr": store_nbr,
        "family": family,
        "avg_daily_demand": round(mean_daily, 4),
        "demand_std": round(sigma_daily, 4),
        "forecast_error_std": round(error_std, 4),
        "lead_time_days": round(non_negative(lead_time_days), 4),
        "review_period_days": round(non_negative(review_period_days), 4),
        "protection_days": round(non_negative(lead_time_days) + non_negative(review_period_days), 4),
        "lead_time_std_days": round(non_negative(lead_time_std_days), 4),
        "service_level": float(service_level),
        "z_score": round(z_score(service_level), 4),
        "safety_stock": round(safety_stock_value, 4),
        "reorder_point": round(rop, 4),
        "order_up_to_level": round(target, 4),
        "expected_lead_time_demand": round(expected_lead_time_demand(mean_daily, lead_time_days), 4),
        "on_hand": round(current_on_hand, 4),
        "on_order": round(current_on_order, 4),
        "inventory_position": round(inventory_position, 4),
        "unit_cost": round(cost, 4),
        "days_of_supply": round(dos, 4) if math.isfinite(dos) else math.inf,
        "cover_days": round(cover, 4) if math.isfinite(cover) else math.inf,
        "reorder_quantity": round(order_quantity, 4),
        "reorder_value": round(order_quantity * cost, 2),
        "needs_reorder": needs_reorder,
        "stockout_risk": stockout_risk,
        "below_safety_stock": below_safety,
        "is_out_of_stock": is_out_of_stock,
        "overstock_risk": overstock,
        "stockout_risk_level": risk_level,
    }


# ===========================================================================
# Batch helpers
# ===========================================================================
def demand_summary_from_forecast(
    forecasts: pd.DataFrame,
    horizon: int | None = None,
    prediction_col: str = "prediction",
    date_col: str = config.DATE_COL,
    keys: Sequence[str] = tuple(config.GROUP_KEYS),
    ddof: int = 1,
) -> pd.DataFrame:
    """
    Reduce a daily forecast frame to per-series demand statistics.

    This is the bridge between Phase 3 (:mod:`src.models.predict`) and the
    inventory engine: the forecaster emits one row per series per future day,
    while the policy needs a mean and a standard deviation per series.

    Parameters
    ----------
    forecasts:
        Long frame with ``store_nbr``, ``family``, ``date`` and a prediction
        column (as produced by :func:`src.models.predict.forecast_all`).
    horizon:
        Optional cap: keep only the first ``horizon`` days of each series.
        ``None`` uses every row supplied.
    prediction_col, date_col, keys:
        Column-name overrides.
    ddof:
        Delta degrees of freedom for the daily standard deviation.

    Returns
    -------
    pandas.DataFrame
        One row per series with ``n_days``, ``avg_daily_demand``,
        ``demand_std``, ``total_demand``, ``min_daily_demand``,
        ``max_daily_demand`` and ``last_forecast_date``.
    """
    key_list = list(keys)
    require_columns(forecasts, [*key_list, date_col, prediction_col], "demand_summary_from_forecast")

    frame = canonicalize_keys(forecasts[[*key_list, date_col, prediction_col]], key_list)
    frame[prediction_col] = (
        pd.to_numeric(frame[prediction_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    frame = frame.sort_values([*key_list, date_col], kind="stable")

    if horizon is not None:
        if int(horizon) <= 0:
            raise ValueError(f"horizon must be a positive integer, got {horizon!r}.")
        frame = frame.groupby(key_list, observed=True, sort=False).head(int(horizon))

    if frame.empty:
        return pd.DataFrame(columns=list(DEMAND_SUMMARY_COLUMNS))

    grouped = frame.groupby(key_list, observed=True, sort=False)
    summary = grouped[prediction_col].agg(
        n_days="count",
        avg_daily_demand="mean",
        total_demand="sum",
        min_daily_demand="min",
        max_daily_demand="max",
    )
    std = grouped[prediction_col].std(ddof=ddof).rename("demand_std")
    last_date = grouped[date_col].max().rename("last_forecast_date")

    out = (
        summary.join(std, how="left")
        .join(last_date, how="left")
        .reset_index()
    )
    out["demand_std"] = out["demand_std"].fillna(0.0).clip(lower=0.0)
    out["avg_daily_demand"] = out["avg_daily_demand"].clip(lower=0.0)
    return out[[*DEMAND_SUMMARY_COLUMNS]]


def inventory_policy_table(
    demand: pd.DataFrame,
    stock: pd.DataFrame,
    service_level: float = config.SERVICE_LEVEL,
    lead_time_days: float = config.LEAD_TIME_DAYS,
    review_period_days: float = config.REVIEW_PERIOD_DAYS,
    lead_time_std_days: float = config.LEAD_TIME_STD_DAYS,
    forecast_error_std: float = 0.0,
    unit_costs: Mapping | pd.Series | float | None = None,
    on_hand_col: str = "on_hand",
    on_order_col: str = "on_order",
    keys: Sequence[str] = tuple(config.GROUP_KEYS),
) -> pd.DataFrame:
    """
    Apply :func:`inventory_policy` to every series in a portfolio.

    Parameters
    ----------
    demand:
        Per-series demand stats — either the output of
        :func:`demand_summary_from_forecast` or any frame with ``store_nbr``,
        ``family`` and ``avg_daily_demand`` (+ optional ``demand_std``).
    stock:
        Current stock state with ``store_nbr``, ``family`` and ``on_hand``
        (+ optional ``on_order`` and ``unit_cost``).
    service_level, lead_time_days, review_period_days, lead_time_std_days:
        Policy knobs. If a matching column exists on the merged frame it wins
        per row, which lets a caller use SKU-specific lead times.
    forecast_error_std:
        Scalar or per-series column; inflates the demand variability used for
        safety stock (see :func:`safety_stock`).
    unit_costs:
        Optional cost table (see :func:`src.engines._common.resolve_unit_costs`).

    Returns
    -------
    pandas.DataFrame
        One row per ``(store_nbr, family)`` with the columns in
        :data:`POLICY_COLUMNS`.
    """
    key_list = list(keys)
    require_columns(demand, [*key_list, "avg_daily_demand"], "inventory_policy_table")
    require_columns(stock, [*key_list, on_hand_col], "inventory_policy_table")

    demand = canonicalize_keys(demand, key_list)
    stock = canonicalize_keys(stock, key_list)
    merged = demand.merge(stock, on=key_list, how="inner", suffixes=("", "_stock"))
    if merged.empty:
        return pd.DataFrame(columns=list(POLICY_COLUMNS))

    def column(name: str, default: Any) -> pd.Series:
        """Per-row series from ``merged`` when present, else a constant."""
        if name in merged.columns:
            return merged[name]
        return pd.Series(default, index=merged.index)

    avg_daily = column("avg_daily_demand", 0.0).astype("float64").fillna(0.0).clip(lower=0.0)
    demand_std = column("demand_std", 0.0).astype("float64").fillna(0.0).clip(lower=0.0)
    error_std = column("forecast_error_std", forecast_error_std)
    error_std = pd.to_numeric(error_std, errors="coerce").fillna(non_negative(forecast_error_std)).clip(lower=0.0)
    on_hand = column(on_hand_col, 0.0)
    on_order = column(on_order_col, 0.0)
    lead_time = pd.to_numeric(column("lead_time_days", lead_time_days), errors="coerce").fillna(lead_time_days)
    review = pd.to_numeric(column("review_period_days", review_period_days), errors="coerce").fillna(review_period_days)
    lead_time_std = pd.to_numeric(column("lead_time_std_days", lead_time_std_days), errors="coerce").fillna(lead_time_std_days)
    level = pd.to_numeric(column("service_level", service_level), errors="coerce").fillna(service_level)
    costs = resolve_unit_costs(merged, unit_costs)

    records: list[dict[str, Any]] = []
    for position in range(len(merged)):
        records.append(
            inventory_policy(
                avg_daily_demand=float(avg_daily.iloc[position]),
                demand_std=float(demand_std.iloc[position]),
                on_hand=float(on_hand.iloc[position]) if pd.notna(on_hand.iloc[position]) else 0.0,
                lead_time_days=float(lead_time.iloc[position]),
                service_level=float(level.iloc[position]),
                review_period_days=float(review.iloc[position]),
                lead_time_std_days=float(lead_time_std.iloc[position]),
                forecast_error_std=float(error_std.iloc[position]),
                on_order=float(on_order.iloc[position]) if pd.notna(on_order.iloc[position]) else 0.0,
                unit_cost=float(costs.iloc[position]),
                store_nbr=merged[key_list[0]].iloc[position],
                family=merged[key_list[-1]].iloc[position],
            )
        )

    table = pd.DataFrame.from_records(records)
    return table[[*POLICY_COLUMNS]]
