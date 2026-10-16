"""
Smart deadstock clearance engine.

Two jobs live here:

**Detection** — find the SKU-store combinations that are quietly costing money:
stock still on the shelf whose sales velocity has collapsed and which has not
sold for a long time. Each one is scored, banded into a severity tier and
valued (capital tied up + annual holding cost + the saving available from
clearing it inside the clearance window).

**Bundling** — pair each deadstock item with a fast-moving, complementary
"hero" item so the discount is attached to a basket rather than given away on a
product nobody wants. Complementary demand is estimated from an **affinity
matrix**: the pairwise correlation of daily family demand *after* removing the
common daily factor (see :func:`build_affinity_matrix`), because raw sales
correlation is dominated by shared seasonality and would rank every family as
complementary to every other.

> The Favorita dataset has no basket/receipt data, so true market-basket
> co-occurrence is unavailable. :func:`build_affinity_matrix` is an explicit,
> documented proxy. If real baskets ever become available, pass in the
> co-occurrence matrix as ``affinity`` — the bundling algorithm is agnostic to
> where the affinity numbers come from.

All functions are pure and DataFrame-in / DataFrame-out.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src import config
from src.engines._common import canonicalize_keys, require_columns, resolve_unit_costs

log = logging.getLogger("engines.deadstock")

# Columns returned by :func:`detect_deadstock`.
DEADSTOCK_COLUMNS: tuple[str, ...] = (
    "store_nbr",
    "family",
    "units_sold_window",
    "avg_daily_velocity",
    "days_since_last_sale",
    "never_sold",
    "on_hand",
    "unit_cost",
    "inventory_value",
    "is_deadstock",
    "severity_score",
    "severity",
    "annual_holding_cost",
    "baseline_days_to_clear",
    "holding_cost_savings",
)

# Columns returned by :func:`generate_bundles`.
BUNDLE_COLUMNS: tuple[str, ...] = (
    "bundle_id",
    "store_nbr",
    "deadstock_family",
    "complement_family",
    "affinity",
    "selection_basis",
    "complement_units_sold",
    "deadstock_units",
    "deadstock_value",
    "severity",
    "severity_score",
    "days_since_last_sale",
    "suggested_discount_pct",
    "discount_rationale",
    "discounted_unit_cost",
    "recovery_at_discount",
    "holding_cost_savings",
)

# Severity bands applied to ``severity_score`` (descending thresholds).
_SEVERITY_BANDS: tuple[tuple[float, str], ...] = (
    (0.80, "critical"),
    (0.60, "high"),
    (0.40, "medium"),
)
_LOW_SEVERITY = "low"


@dataclass(frozen=True)
class DeadstockPolicy:
    """
    Thresholds and economics for deadstock detection and clearance.

    Attributes
    ----------
    window_days:
        Lookback used to measure current sales velocity.
    velocity_threshold:
        Average units/day at or below which a SKU counts as slow-moving.
    min_days_since_last_sale:
        Minimum age (days) of the last sale for the SKU to count as deadstock.
    holding_cost_rate:
        Annual holding cost as a fraction of inventory value (capital, space,
        insurance, obsolescence).
    clearance_window_days:
        Target number of days to clear the stock once it is promoted.
    max_days_to_clear:
        Cap on the baseline sell-through horizon, so a zero-velocity line does
        not produce an infinite holding-cost saving.
    discount_base, discount_per_30_days, discount_min, discount_max:
        Promotional discount schedule: a base percentage, increased by a fixed
        step for every full 30 days since the last sale, then clipped.
    top_n_complements:
        Default number of complementary items suggested per deadstock item.
    """

    window_days: int = config.DEADSTOCK_WINDOW_DAYS
    velocity_threshold: float = config.DEADSTOCK_VELOCITY_THRESHOLD
    min_days_since_last_sale: int = config.DEADSTOCK_MIN_DAYS_SINCE_LAST_SALE
    holding_cost_rate: float = config.DEADSTOCK_HOLDING_COST_RATE
    clearance_window_days: int = config.DEADSTOCK_CLEARANCE_WINDOW_DAYS
    max_days_to_clear: int = config.DEADSTOCK_MAX_DAYS_TO_CLEAR
    discount_base: float = config.DEADSTOCK_DISCOUNT_BASE
    discount_per_30_days: float = config.DEADSTOCK_DISCOUNT_PER_30_DAYS
    discount_min: float = config.DEADSTOCK_DISCOUNT_MIN
    discount_max: float = config.DEADSTOCK_DISCOUNT_MAX
    top_n_complements: int = config.BUNDLE_TOP_N_COMPLEMENTS


# ===========================================================================
# Detection
# ===========================================================================
def detect_deadstock(
    sales: pd.DataFrame,
    stock: pd.DataFrame,
    policy: DeadstockPolicy = DeadstockPolicy(),
    keys: Sequence[str] = tuple(config.GROUP_KEYS),
    date_col: str = config.DATE_COL,
    value_col: str = config.TARGET_COL,
    on_hand_col: str = "on_hand",
    unit_costs: Mapping | pd.Series | float | None = None,
) -> pd.DataFrame:
    """
    Flag slow-moving, long-unsold stock and quantify what it is costing.

    Parameters
    ----------
    sales:
        Historical long panel with ``store_nbr``, ``family``, ``date`` and unit
        sales (e.g. ``data/processed/sales_clean.parquet``).
    stock:
        Current stock state — ``store_nbr``, ``family``, ``on_hand`` and
        optionally ``unit_cost``. It also defines the SKU universe that is
        evaluated (a SKU with no stock row has nothing to clear).
    policy:
        :class:`DeadstockPolicy` thresholds and economics.
    keys, date_col, value_col, on_hand_col, unit_costs:
        Column/scope overrides.

    Returns
    -------
    pandas.DataFrame
        One row per SKU in ``stock`` with the columns in
        :data:`DEADSTOCK_COLUMNS`:

        ``avg_daily_velocity``
            Units/day over the trailing ``window_days``.
        ``days_since_last_sale``
            Age of the last positive sale, measured against the last date in
            ``sales``. ``never_sold`` marks series with no sale at all in the
            panel (their age is reported as the panel span plus the window, a
            documented lower bound).
        ``is_deadstock``
            Has stock, velocity ``<=`` threshold and age ``>=`` threshold.
        ``severity_score``
            A ``0..1`` blend of age (45%), lack of velocity (35%) and excess
            cover (20%); ``0`` for rows that are not deadstock.
        ``baseline_days_to_clear``
            Days to sell through the current stock at the current velocity,
            clipped to ``[clearance_window_days, max_days_to_clear]``.
        ``holding_cost_savings``
            Annual holding cost avoided by clearing inside the clearance
            window instead of at the baseline pace.

    Notes
    -----
    SKUs with no on-hand stock are never flagged: there is nothing to clear.
    """
    key_list = list(keys)
    require_columns(sales, [*key_list, date_col, value_col], "detect_deadstock")
    require_columns(stock, [*key_list, on_hand_col], "detect_deadstock")

    window_days = int(policy.window_days)
    if window_days <= 0:
        raise ValueError(f"policy.window_days must be positive, got {window_days}.")
    min_age = int(policy.min_days_since_last_sale)
    if min_age < 0:
        raise ValueError(f"policy.min_days_since_last_sale must be >= 0, got {min_age}.")

    panel = canonicalize_keys(sales[[*key_list, date_col, value_col]], key_list)
    panel[date_col] = pd.to_datetime(panel[date_col], errors="coerce")
    panel[value_col] = pd.to_numeric(panel[value_col], errors="coerce").fillna(0.0)
    panel = panel.dropna(subset=[date_col])
    if panel.empty:
        return pd.DataFrame(columns=list(DEADSTOCK_COLUMNS))

    panel_end = panel[date_col].max()
    panel_start = panel[date_col].min()
    panel_span = max(int((panel_end - panel_start).days), 1)
    cutoff = panel_end - pd.Timedelta(days=window_days - 1)

    recent = panel[panel[date_col] >= cutoff]
    units_window = (
        recent.groupby(key_list, observed=True)[value_col].sum().rename("units_sold_window")
    )
    last_sale = (
        panel.loc[panel[value_col] > 0]
        .groupby(key_list, observed=True)[date_col]
        .max()
        .rename("_last_sale")
    )

    stock_canon = canonicalize_keys(stock, key_list)
    keep = [*key_list, on_hand_col]
    if "unit_cost" in stock_canon.columns:
        keep.append("unit_cost")
    universe = stock_canon[keep].drop_duplicates(subset=key_list)

    table = universe.merge(units_window.reset_index(), on=key_list, how="left")
    table = table.merge(last_sale.reset_index(), on=key_list, how="left")
    if on_hand_col != "on_hand":
        table = table.rename(columns={on_hand_col: "on_hand"})

    table["units_sold_window"] = (
        pd.to_numeric(table["units_sold_window"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    table["on_hand"] = pd.to_numeric(table["on_hand"], errors="coerce").fillna(0.0).clip(lower=0.0)

    last_sale_days = (panel_end - table["_last_sale"]).dt.days
    table["never_sold"] = last_sale_days.isna()
    table["days_since_last_sale"] = last_sale_days.fillna(panel_span + window_days).astype("int64")
    table = table.drop(columns=["_last_sale"])

    table["avg_daily_velocity"] = table["units_sold_window"] / float(window_days)
    table["unit_cost"] = resolve_unit_costs(table, unit_costs)
    table["inventory_value"] = table["on_hand"] * table["unit_cost"]

    on_hand = table["on_hand"].to_numpy(dtype="float64")
    velocity = table["avg_daily_velocity"].to_numpy(dtype="float64")
    age = table["days_since_last_sale"].to_numpy(dtype="float64")

    deadstock = (on_hand > 0.0) & (velocity <= float(policy.velocity_threshold)) & (age >= float(min_age))
    table["is_deadstock"] = deadstock

    cap = float(policy.max_days_to_clear)
    floor = float(policy.clearance_window_days)
    safe_velocity = np.where(velocity > 0.0, velocity, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_days = np.where(velocity > 0.0, on_hand / safe_velocity, cap)
    baseline = np.clip(np.nan_to_num(raw_days, nan=cap, posinf=cap, neginf=floor), floor, cap)
    table["baseline_days_to_clear"] = baseline

    table["annual_holding_cost"] = table["inventory_value"] * float(policy.holding_cost_rate)
    days_saved = np.clip(baseline - floor, 0.0, None)
    table["holding_cost_savings"] = table["annual_holding_cost"] * days_saved / 365.0

    # --- severity -----------------------------------------------------------
    velocity_ref = max(float(policy.velocity_threshold), 1e-9)
    age_ref = max(3.0 * float(min_age), 1.0)
    excess_ref = max(4.0 * floor, 1.0)

    velocity_ratio = np.clip(velocity / velocity_ref, 0.0, 1.0)
    age_ratio = np.clip(age / age_ref, 0.0, 1.0)
    excess_ratio = np.clip(baseline / excess_ref, 0.0, 1.0)
    score = 0.45 * age_ratio + 0.35 * (1.0 - velocity_ratio) + 0.20 * excess_ratio
    score = np.where(deadstock, score, 0.0)
    table["severity_score"] = np.round(score, 4)

    severity = np.full(len(table), "none", dtype=object)
    below_bands = np.zeros(len(table), dtype=bool)
    for threshold, label in _SEVERITY_BANDS:
        mask = ~below_bands & (score >= threshold)
        severity[mask] = label
        below_bands |= mask
    severity[~below_bands & (score > 0.0)] = _LOW_SEVERITY
    table["severity"] = severity

    table = table.sort_values(
        ["severity_score", "holding_cost_savings", "inventory_value"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    return table[[*DEADSTOCK_COLUMNS]]


# ===========================================================================
# Affinity (complementarity proxy)
# ===========================================================================
def build_affinity_matrix(
    sales: pd.DataFrame,
    value_col: str = config.TARGET_COL,
    date_col: str = config.DATE_COL,
    family_key: str = "family",
    method: str = "residual",
    min_periods: int = 14,
    log_transform: bool = True,
) -> pd.DataFrame:
    """
    Estimate pairwise item affinity from co-movement of daily family demand.

    Parameters
    ----------
    sales:
        Long sales panel; rows sharing a ``(date, family)`` are summed across
        stores, so the matrix describes chain-level family demand.
    value_col, date_col, family_key:
        Column-name overrides.
    method:
        ``"residual"`` (default) removes the cross-family mean of each day
        before correlating, so what remains is idiosyncratic co-movement rather
        than shared seasonality — this is what makes two families
        *complementary* rather than merely *co-seasonal*. ``"correlation"``
        correlates raw (log) demand instead.
    min_periods:
        Minimum overlapping observations for a pairwise correlation; pairs with
        less overlap receive ``0`` (no evidence of affinity).
    log_transform:
        Apply ``log1p`` before correlating — tames the scale difference between
        a low-volume family and a high-volume one.

    Returns
    -------
    pandas.DataFrame
        Square ``family x family`` matrix of correlations in ``[-1, 1]``.
        ``+1`` on the diagonal; ``0`` where there is no evidence. Negative
        values indicate substitutes; :func:`generate_bundles` ignores them.
    """
    require_columns(sales, [date_col, family_key, value_col], "build_affinity_matrix")

    frame = sales[[date_col, family_key, value_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    frame = frame.dropna(subset=[date_col])
    if frame.empty:
        return pd.DataFrame()

    # ``observed=True`` keeps the matrix to families that actually sold in the
    # window — a family with no rows carries no evidence to correlate, and
    # ``generate_bundles`` already guards for names missing from the matrix.
    # (Stated explicitly because pandas is changing this default.)
    wide = frame.pivot_table(
        index=date_col,
        columns=family_key,
        values=value_col,
        aggfunc="sum",
        observed=True,
    )
    wide = wide.sort_index()
    families = list(wide.columns)

    if len(families) < 2:
        result = pd.DataFrame(0.0, index=families, columns=families)
        for family in families:
            result.loc[family, family] = 1.0
        return result

    if log_transform:
        wide = np.log1p(wide)

    normalised_method = str(method).lower()
    if normalised_method == "residual":
        wide = wide.sub(wide.mean(axis=1), axis=0)
    elif normalised_method != "correlation":
        raise ValueError(
            f"method must be 'residual' or 'correlation', got {method!r}."
        )

    correlation = wide.corr(min_periods=max(int(min_periods), 2))
    correlation = correlation.fillna(0.0)
    correlation.index.name = None
    correlation.columns.name = None
    return correlation


def select_hero_items(
    sales: pd.DataFrame,
    window_days: int = config.DEADSTOCK_WINDOW_DAYS,
    top_n: int = 20,
    value_col: str = config.TARGET_COL,
    date_col: str = config.DATE_COL,
    family_key: str = "family",
) -> pd.DataFrame:
    """
    Rank the fastest-selling families over the trailing window ("hero" items).

    Returns
    -------
    pandas.DataFrame
        ``family``, ``units_sold`` and 1-based ``rank``, best-seller first,
        truncated to ``top_n`` rows (``top_n=None`` keeps all).
    """
    require_columns(sales, [date_col, family_key, value_col], "select_hero_items")
    window = int(window_days)
    if window <= 0:
        raise ValueError(f"window_days must be positive, got {window_days}.")

    frame = sales[[date_col, family_key, value_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    frame = frame.dropna(subset=[date_col])
    if frame.empty:
        return pd.DataFrame(columns=[family_key, "units_sold", "rank"])

    cutoff = frame[date_col].max() - pd.Timedelta(days=window - 1)
    recent = frame[frame[date_col] >= cutoff]
    totals = (
        recent.groupby(family_key, observed=True)[value_col]
        .sum()
        .sort_values(ascending=False, kind="stable")
    )
    out = totals.rename("units_sold").reset_index()
    out["rank"] = np.arange(1, len(out) + 1)
    if top_n is not None and int(top_n) > 0:
        out = out.head(int(top_n))
    return out.reset_index(drop=True)


# ===========================================================================
# Discounting & bundling
# ===========================================================================
def suggested_discount_pct(
    days_since_last_sale: int | float,
    policy: DeadstockPolicy = DeadstockPolicy(),
) -> float:
    """
    Promotional discount (as a fraction, e.g. ``0.25``) for a deadstock item.

    The schedule starts at ``discount_base`` and adds
    ``discount_per_30_days`` for every full 30 days since the last sale, then
    is clipped to ``[discount_min, discount_max]``. Older stock is discounted
    harder, but never below the floor (a token discount does not clear stock)
    nor above the ceiling (which would destroy the margin entirely).
    """
    age = max(float(days_since_last_sale), 0.0)
    steps = math.floor(age / 30.0)
    pct = float(policy.discount_base) + float(policy.discount_per_30_days) * steps
    low = float(policy.discount_min)
    high = float(policy.discount_max)
    if high < low:
        low, high = high, low
    return float(min(max(pct, low), high))


def generate_bundles(
    deadstock: pd.DataFrame,
    affinity: pd.DataFrame | None = None,
    heroes: pd.DataFrame | None = None,
    policy: DeadstockPolicy = DeadstockPolicy(),
    top_n_complements: int | None = None,
    family_key: str = "family",
) -> pd.DataFrame:
    """
    Pair each deadstock item with complementary best-sellers and price it.

    For every row flagged ``is_deadstock`` in ``deadstock`` the engine ranks
    candidate complements by: is a hero item, then affinity, then hero sales.
    Positive affinity is required where the matrix has an opinion; when it has
    none, hero items fill the remaining slots so every deadstock item still
    gets an actionable bundle.

    Parameters
    ----------
    deadstock:
        Output of :func:`detect_deadstock`.
    affinity:
        Output of :func:`build_affinity_matrix` (or a real co-occurrence
        matrix). ``None`` falls back to ranking by hero sales only.
    heroes:
        Output of :func:`select_hero_items`. ``None`` means "consider every
        family in the affinity matrix".
    policy:
        :class:`DeadstockPolicy` (drives the discount schedule and the default
        number of complements).
    top_n_complements:
        Overrides ``policy.top_n_complements``.
    family_key:
        Name of the deadstock frame's family column.

    Returns
    -------
    pandas.DataFrame
        One row per ``(deadstock item, complement)`` bundle with the columns in
        :data:`BUNDLE_COLUMNS`, including ``suggested_discount_pct``,
        ``discount_rationale`` and ``holding_cost_savings``. Empty (but
        correctly typed) when there is no deadstock.
    """
    if deadstock is None or deadstock.empty:
        return pd.DataFrame(columns=list(BUNDLE_COLUMNS))
    require_columns(
        deadstock,
        ["store_nbr", family_key, "is_deadstock", "days_since_last_sale", "on_hand",
         "unit_cost", "severity", "severity_score", "holding_cost_savings"],
        "generate_bundles",
    )

    candidates = deadstock[deadstock["is_deadstock"].astype(bool)]
    if candidates.empty:
        return pd.DataFrame(columns=list(BUNDLE_COLUMNS))

    matrix = affinity if affinity is not None and not affinity.empty else pd.DataFrame()
    hero_families: set[str] = set()
    hero_units: dict[str, float] = {}
    if heroes is not None and not heroes.empty:
        hero_families = set(heroes[family_key].tolist())
        hero_units = dict(zip(heroes[family_key], heroes["units_sold"]))

    top_n = int(policy.top_n_complements if top_n_complements is None else top_n_complements)

    # Families that are themselves deadstock, per store: pairing two dead lines
    # together just relocates the problem, so they are never recommended as
    # complements for each other.
    deadstock_by_store: dict[Any, set] = {
        store: set(group[family_key].tolist())
        for store, group in candidates.groupby("store_nbr", observed=True, sort=False)
    }

    records: list[dict[str, Any]] = []
    bundle_counter = 0

    for _, row in candidates.iterrows():
        family = row[family_key]

        affinity_scores: dict[str, float] = {}
        if not matrix.empty and family in matrix.index:
            for other, value in matrix.loc[family].items():
                if other == family or pd.isna(value):
                    continue
                affinity_scores[other] = float(value)

        excluded = {family} | deadstock_by_store.get(row["store_nbr"], set())
        if hero_families:
            pool = [name for name in hero_families if name not in excluded]
        else:
            pool = [name for name in affinity_scores if name not in excluded]

        # Complementary demand (affinity > 0) wins; only when there is no
        # positive evidence do we fall back to raw best-seller rank, so the
        # item still gets promoted rather than dropped.
        complementary = [name for name in pool if affinity_scores.get(name, 0.0) > 0.0]
        if complementary:
            ranked = sorted(
                complementary,
                key=lambda name: (-affinity_scores[name], -hero_units.get(name, 0.0), str(name)),
            )
        else:
            ranked = sorted(pool, key=lambda name: (-hero_units.get(name, 0.0), str(name)))

        complements = ranked if top_n <= 0 else ranked[:top_n]
        if not complements:
            continue
        selection_basis = "affinity" if complementary else "best_seller_fallback"

        age = int(row["days_since_last_sale"])
        discount = suggested_discount_pct(age, policy)
        unit_cost = float(row["unit_cost"])
        units = float(row["on_hand"])
        value = units * unit_cost
        recovery = value * (1.0 - discount)
        rationale = (
            f"{discount:.0%} off: {age} days since last sale "
            f"(severity '{row['severity']}'); bundles ~{units:.0f} units of {family} "
            f"with fast-moving demand to clear it inside "
            f"{int(policy.clearance_window_days)} days."
        )

        for complement in complements:
            bundle_counter += 1
            records.append(
                {
                    "bundle_id": f"B{bundle_counter:04d}",
                    "store_nbr": row["store_nbr"],
                    "deadstock_family": family,
                    "complement_family": complement,
                    "affinity": round(affinity_scores.get(complement, 0.0), 4),
                    "selection_basis": selection_basis,
                    "complement_units_sold": float(hero_units.get(complement, 0.0)),
                    "deadstock_units": round(units, 4),
                    "deadstock_value": round(value, 2),
                    "severity": row["severity"],
                    "severity_score": float(row["severity_score"]),
                    "days_since_last_sale": age,
                    "suggested_discount_pct": round(discount, 4),
                    "discount_rationale": rationale,
                    "discounted_unit_cost": round(unit_cost * (1.0 - discount), 4),
                    "recovery_at_discount": round(recovery, 2),
                    "holding_cost_savings": round(float(row["holding_cost_savings"]), 2),
                }
            )

    if not records:
        return pd.DataFrame(columns=list(BUNDLE_COLUMNS))

    bundles = pd.DataFrame.from_records(records)
    bundles["store_nbr"] = bundles["store_nbr"].astype("int64")
    bundles = bundles.sort_values(
        ["severity_score", "holding_cost_savings", "affinity", "deadstock_family", "complement_family"],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return bundles[[*BUNDLE_COLUMNS]]


def clearance_summary(
    deadstock: pd.DataFrame,
    bundles: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Headline numbers for a clearance programme.

    Returns
    -------
    dict
        ``n_skus``, ``n_deadstock_skus``, ``n_critical_skus``,
        ``deadstock_units``, ``deadstock_value``,
        ``potential_holding_savings``, ``n_bundles``, ``mean_discount_pct`` and
        ``recovery_at_discount``. All zeros when there is nothing to clear.
    """
    empty = {
        "n_skus": 0,
        "n_deadstock_skus": 0,
        "n_critical_skus": 0,
        "deadstock_units": 0.0,
        "deadstock_value": 0.0,
        "potential_holding_savings": 0.0,
        "n_bundles": 0,
        "mean_discount_pct": 0.0,
        "recovery_at_discount": 0.0,
    }
    if deadstock is None or deadstock.empty:
        return empty

    flagged = deadstock[deadstock["is_deadstock"].astype(bool)]
    summary = dict(empty)
    summary["n_skus"] = int(len(deadstock))
    summary["n_deadstock_skus"] = int(len(flagged))
    if flagged.empty:
        return summary

    summary["n_critical_skus"] = int((flagged["severity"] == "critical").sum())
    summary["deadstock_units"] = float(flagged["on_hand"].sum())
    summary["deadstock_value"] = float(flagged["inventory_value"].sum())
    summary["potential_holding_savings"] = float(flagged["holding_cost_savings"].sum())

    if bundles is not None and not bundles.empty:
        summary["n_bundles"] = int(len(bundles))
        summary["mean_discount_pct"] = float(bundles["suggested_discount_pct"].mean())
        summary["recovery_at_discount"] = float(bundles["recovery_at_discount"].sum())
    return summary
