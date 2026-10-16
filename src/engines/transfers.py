"""
Inter-store stock transfer (rebalancing) engine.

Before raising an external purchase order, the cheapest way to fix a localised
stockout is usually to move stock that is already sitting in the network. This
module finds those opportunities:

1. **Classify** every ``(store_nbr, family)`` position by its days of supply
   (DOS) into *understocked* (short of cover), *overstocked* (excess capital
   tied up) or *healthy*.
2. **Match** each deficit store with surplus stores for the same item, in a
   greedy most-urgent-first / most-surplus-first pass, crediting each donor
   with the quantity actually moved.
3. **Guard** every recommendation with business rules: a minimum transfer
   size, a per-leg cap, a cap on the number of donor legs per recipient, an
   explicit reserve so a donor never drops below its own target/safety stock,
   and a net-value test so we never spend more moving stock than the stock is
   worth.

The engine is deterministic and side-effect free: it consumes a position table
(typically the output of :func:`src.engines.inventory.inventory_policy_table`)
and returns a list of transfer orders as a DataFrame.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from src import config
from src.engines._common import (
    canonicalize_keys,
    non_negative,
    require_columns,
    resolve_unit_costs,
    whole_units,
)
from src.engines.inventory import days_of_supply

log = logging.getLogger("engines.transfers")

# Columns returned by :func:`classify_stock_positions` (position columns plus
# the derived status/surplus/deficit measures).
POSITION_COLUMNS: tuple[str, ...] = (
    "store_nbr",
    "family",
    "on_hand",
    "avg_daily_demand",
    "safety_stock",
    "unit_cost",
    "days_of_supply",
    "target_units",
    "reserve_units",
    "surplus_units",
    "deficit_units",
    "status",
)

# Columns returned by :func:`generate_transfers`.
TRANSFER_COLUMNS: tuple[str, ...] = (
    "family",
    "from_store",
    "to_store",
    "quantity",
    "unit_cost",
    "estimated_value",
    "estimated_transfer_cost",
    "estimated_net_value",
    "donor_on_hand_before",
    "donor_on_hand_after",
    "donor_dos_before",
    "donor_dos_after",
    "recipient_on_hand_before",
    "recipient_on_hand_after",
    "recipient_dos_before",
    "recipient_dos_after",
    "recipient_deficit_before",
    "rationale",
)


@dataclass(frozen=True)
class TransferPolicy:
    """
    Business guardrails for inter-store rebalancing.

    Attributes
    ----------
    overstock_dos_threshold:
        Days of supply above which a store is treated as a donor.
    understock_dos_threshold:
        Days of supply below which a store is treated as a receiver.
    target_dos:
        Days of supply a receiver is topped up to (and the level a donor is
        left with, unless its safety stock is higher).
    min_transfer_units:
        Transfers smaller than this are not economically worthwhile.
    max_transfer_units:
        Hard cap on a single transfer leg.
    fixed_cost, cost_per_unit:
        Cost model for moving stock (handling/admin + per-unit labour/freight).
    min_net_value:
        A leg is skipped unless ``estimated_value - estimated_transfer_cost``
        is at least this value. The default (0) refuses to spend more moving
        goods than the goods are worth.
    max_donors_per_recipient:
        How many separate donor legs a single receiver may draw from.
    protect_donor_stock:
        When ``True`` (default) a donor is never pushed below
        ``max(target_dos * demand, safety_stock)``.
    """

    overstock_dos_threshold: float = config.OVERSTOCK_DOS_THRESHOLD
    understock_dos_threshold: float = config.UNDERSTOCK_DOS_THRESHOLD
    target_dos: float = config.TRANSFER_TARGET_DOS
    min_transfer_units: int = config.MIN_TRANSFER_UNITS
    max_transfer_units: int = config.MAX_TRANSFER_UNITS
    fixed_cost: float = config.TRANSFER_FIXED_COST
    cost_per_unit: float = config.TRANSFER_COST_PER_UNIT
    min_net_value: float = config.MIN_TRANSFER_NET_VALUE
    max_donors_per_recipient: int = 3
    protect_donor_stock: bool = True


# ===========================================================================
# Preparation & classification
# ===========================================================================
def _prepare_positions(
    positions: pd.DataFrame,
    keys: Sequence[str],
    on_hand_col: str,
    demand_col: str,
    safety_stock_col: str,
    unit_costs: Mapping | pd.Series | float | None,
) -> pd.DataFrame:
    """
    Validate a position table and fill in the optional columns.

    Required: ``store_nbr``, ``family``, ``on_hand``, ``avg_daily_demand``.
    Optional (defaulted): ``safety_stock`` → 0, ``unit_cost`` → config default,
    ``days_of_supply`` → recomputed from on-hand and demand.
    """
    key_list = list(keys)
    require_columns(positions, [*key_list, on_hand_col, demand_col], "_prepare_positions")

    frame = canonicalize_keys(positions, key_list)
    frame["on_hand"] = pd.to_numeric(frame[on_hand_col], errors="coerce").fillna(0.0).astype("float64")
    frame["avg_daily_demand"] = (
        pd.to_numeric(frame[demand_col], errors="coerce").fillna(0.0).clip(lower=0.0).astype("float64")
    )

    if safety_stock_col in frame.columns:
        frame["safety_stock"] = (
            pd.to_numeric(frame[safety_stock_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        )
    else:
        frame["safety_stock"] = 0.0

    frame["unit_cost"] = resolve_unit_costs(frame, unit_costs)

    if "days_of_supply" not in frame.columns:
        frame["days_of_supply"] = [
            days_of_supply(on_hand, demand)
            for on_hand, demand in zip(frame["on_hand"], frame["avg_daily_demand"])
        ]
    else:
        frame["days_of_supply"] = pd.to_numeric(frame["days_of_supply"], errors="coerce")

    return frame


def classify_stock_positions(
    positions: pd.DataFrame,
    policy: TransferPolicy = TransferPolicy(),
    keys: Sequence[str] = tuple(config.GROUP_KEYS),
    on_hand_col: str = "on_hand",
    demand_col: str = "avg_daily_demand",
    safety_stock_col: str = "safety_stock",
    unit_costs: Mapping | pd.Series | float | None = None,
) -> pd.DataFrame:
    """
    Label every position as understocked / healthy / overstocked / no-demand.

    Adds these columns:

    ``days_of_supply``
        On-hand cover in days (``inf`` when demand is zero).
    ``target_units``
        ``target_dos * avg_daily_demand`` — the level a receiver is topped up
        to.
    ``reserve_units``
        What a donor must keep: ``max(target_units, safety_stock)`` when
        ``protect_donor_stock`` is set, otherwise 0.
    ``surplus_units``
        Transferable units — positive only for overstocked stores.
    ``deficit_units``
        Units needed to reach ``target_units`` — positive only for
        understocked stores.
    ``status``
        ``"understocked"`` / ``"healthy"`` / ``"overstocked"`` / ``"no_demand"``.

    A store with no forecast demand is deliberately labelled ``"no_demand"``
    and gets neither surplus nor deficit: moving stock into a store that is not
    selling, or out of one whose demand is unknown, is not a rebalancing
    decision (it is a clearance decision — see :mod:`src.engines.deadstock`).
    """
    frame = _prepare_positions(
        positions, keys, on_hand_col, demand_col, safety_stock_col, unit_costs
    )

    demand = frame["avg_daily_demand"]
    frame["target_units"] = (non_negative(policy.target_dos) * demand).astype("float64")
    if policy.protect_donor_stock:
        frame["reserve_units"] = frame[["target_units", "safety_stock"]].max(axis=1)
    else:
        frame["reserve_units"] = 0.0

    dos = frame["days_of_supply"]
    has_demand = demand > 0.0
    under = has_demand & (dos < float(policy.understock_dos_threshold))
    over = has_demand & (dos > float(policy.overstock_dos_threshold))

    frame["status"] = "healthy"
    frame.loc[~has_demand, "status"] = "no_demand"
    frame.loc[under, "status"] = "understocked"
    frame.loc[over, "status"] = "overstocked"

    surplus = (frame["on_hand"] - frame["reserve_units"]).clip(lower=0.0)
    deficit = (frame["target_units"] - frame["on_hand"]).clip(lower=0.0)
    frame["surplus_units"] = surplus.where(over, 0.0).astype("float64")
    frame["deficit_units"] = deficit.where(under, 0.0).astype("float64")

    ordered = [*keys, "on_hand", "avg_daily_demand", "safety_stock", "unit_cost",
               "days_of_supply", "target_units", "reserve_units", "surplus_units",
               "deficit_units", "status"]
    # Keep any extra caller columns (e.g. reorder_point) after the canonical set.
    extras = [column for column in frame.columns if column not in ordered]
    return frame[[*ordered, *extras]]


def detect_imbalances(
    positions: pd.DataFrame,
    policy: TransferPolicy = TransferPolicy(),
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Convenience wrapper returning the *imbalanced* rows only.

    Equivalent to :func:`classify_stock_positions` filtered to
    ``status in {"understocked", "overstocked"}`` — the actionable subset for a
    dashboard table.
    """
    classified = classify_stock_positions(positions, policy=policy, **kwargs)
    return classified[classified["status"].isin(["understocked", "overstocked"])].reset_index(drop=True)


# ===========================================================================
# Recommendation engine
# ===========================================================================
def generate_transfers(
    positions: pd.DataFrame,
    policy: TransferPolicy = TransferPolicy(),
    keys: Sequence[str] = tuple(config.GROUP_KEYS),
    on_hand_col: str = "on_hand",
    demand_col: str = "avg_daily_demand",
    safety_stock_col: str = "safety_stock",
    unit_costs: Mapping | pd.Series | float | None = None,
) -> pd.DataFrame:
    """
    Recommend inter-store transfers that fix deficits from internal surplus.

    The algorithm is a deterministic greedy matching, run **per item**
    (``family``):

    1. Sort receivers by ascending days of supply (most urgent first) and
       donors by descending surplus (most overstocked first).
    2. For each receiver, draw from donors until its deficit is covered, the
       donor pool is exhausted, or ``max_donors_per_recipient`` legs have been
       opened.
    3. Each leg is floored to whole units and capped at ``max_transfer_units``,
       then accepted only if it clears ``min_transfer_units`` and
       ``min_net_value`` (value of goods moved minus the cost of moving them).
    4. Accepted legs debit the donor's remaining surplus and the receiver's
       remaining deficit, so one unit of stock can never be promised twice.

    Parameters
    ----------
    positions:
        Position table — at minimum ``store_nbr``, ``family``, ``on_hand`` and
        ``avg_daily_demand``. The output of
        :func:`src.engines.inventory.inventory_policy_table` is ideal because
        it also carries ``safety_stock`` and ``days_of_supply``.
    policy:
        :class:`TransferPolicy` guardrails.
    keys, on_hand_col, demand_col, safety_stock_col, unit_costs:
        Column/scope overrides.

    Returns
    -------
    pandas.DataFrame
        One row per recommended transfer leg, with the columns listed in
        :data:`TRANSFER_COLUMNS` (empty — but correctly typed — when there is
        nothing to move).
    """
    classified = classify_stock_positions(
        positions,
        policy=policy,
        keys=keys,
        on_hand_col=on_hand_col,
        demand_col=demand_col,
        safety_stock_col=safety_stock_col,
        unit_costs=unit_costs,
    )
    key_list = list(keys)
    family_key = key_list[-1]
    store_key = key_list[0]

    records: list[dict[str, Any]] = []
    min_units = int(max(policy.min_transfer_units, 0))
    max_units = int(max(policy.max_transfer_units, 0))

    for family, group in classified.groupby(family_key, observed=True, sort=False):
        donors = group[group["status"] == "overstocked"].sort_values(
            "surplus_units", ascending=False, kind="stable"
        )
        recipients = group[group["status"] == "understocked"].sort_values(
            "days_of_supply", ascending=True, kind="stable"
        )
        if donors.empty or recipients.empty:
            continue

        # Remaining transferable units per donor store, mutated as legs are cut.
        remaining: dict[Any, float] = dict(
            zip(donors[store_key].tolist(), donors["surplus_units"].astype("float64").tolist())
        )
        donor_index = donors.set_index(store_key, drop=False)

        for _, recipient in recipients.iterrows():
            need = float(recipient["deficit_units"])
            if need <= 0.0:
                continue
            legs = 0
            recipient_store = recipient[store_key]
            recipient_demand = float(recipient["avg_daily_demand"])

            for donor_store in donors[store_key].tolist():
                if need < min_units or legs >= int(policy.max_donors_per_recipient):
                    break

                available = remaining.get(donor_store, 0.0)
                if available < min_units:
                    continue

                donor = donor_index.loc[donor_store]
                quantity = whole_units(min(need, available, max_units))
                if quantity < min_units:
                    continue

                cost = non_negative(policy.fixed_cost) + non_negative(policy.cost_per_unit) * quantity
                value = quantity * float(donor["unit_cost"])
                net_value = value - cost
                if net_value < float(policy.min_net_value):
                    continue

                donor_demand = float(donor["avg_daily_demand"])
                donor_before = float(donor["on_hand"])
                donor_after = donor_before - quantity
                recipient_before = float(recipient["on_hand"])
                recipient_after = recipient_before + quantity

                records.append(
                    {
                        "family": family,
                        "from_store": donor_store,
                        "to_store": recipient_store,
                        "quantity": quantity,
                        "unit_cost": round(float(donor["unit_cost"]), 4),
                        "estimated_value": round(value, 2),
                        "estimated_transfer_cost": round(cost, 2),
                        "estimated_net_value": round(net_value, 2),
                        "donor_on_hand_before": round(donor_before, 4),
                        "donor_on_hand_after": round(donor_after, 4),
                        "donor_dos_before": round(days_of_supply(donor_before, donor_demand), 4),
                        "donor_dos_after": round(days_of_supply(donor_after, donor_demand), 4),
                        "recipient_on_hand_before": round(recipient_before, 4),
                        "recipient_on_hand_after": round(recipient_after, 4),
                        "recipient_dos_before": round(days_of_supply(recipient_before, recipient_demand), 4),
                        "recipient_dos_after": round(days_of_supply(recipient_after, recipient_demand), 4),
                        "recipient_deficit_before": round(need, 4),
                        "rationale": (
                            f"Move {quantity} units of {family} from store {donor_store} "
                            f"({days_of_supply(donor_before, donor_demand):.1f}d cover -> "
                            f"{days_of_supply(donor_after, donor_demand):.1f}d) to store "
                            f"{recipient_store} ({days_of_supply(recipient_before, recipient_demand):.1f}d -> "
                            f"{days_of_supply(recipient_after, recipient_demand):.1f}d); "
                            f"avoids an external purchase at store {recipient_store}."
                        ),
                    }
                )

                remaining[donor_store] = available - quantity
                need -= quantity
                legs += 1

    if not records:
        return pd.DataFrame(columns=list(TRANSFER_COLUMNS))

    transfers = pd.DataFrame.from_records(records)
    transfers["quantity"] = transfers["quantity"].astype("int64")
    transfers = transfers.sort_values(
        ["estimated_value", "family", "from_store", "to_store"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return transfers[[*TRANSFER_COLUMNS]]


def transfer_summary(transfers: pd.DataFrame) -> dict[str, Any]:
    """
    Aggregate a transfer plan into headline numbers for dashboards.

    Returns
    -------
    dict
        ``n_transfers``, ``total_units``, ``total_value``,
        ``total_transfer_cost``, ``total_net_value``, ``n_items``,
        ``n_donor_stores``, ``n_recipient_stores``, ``mean_transfer_units``.
        All zeros for an empty plan.
    """
    if transfers is None or transfers.empty:
        return {
            "n_transfers": 0,
            "total_units": 0.0,
            "total_value": 0.0,
            "total_transfer_cost": 0.0,
            "total_net_value": 0.0,
            "n_items": 0,
            "n_donor_stores": 0,
            "n_recipient_stores": 0,
            "mean_transfer_units": 0.0,
        }

    quantity = pd.to_numeric(transfers["quantity"], errors="coerce").fillna(0.0)
    return {
        "n_transfers": int(len(transfers)),
        "total_units": float(quantity.sum()),
        "total_value": float(pd.to_numeric(transfers["estimated_value"], errors="coerce").fillna(0.0).sum()),
        "total_transfer_cost": float(
            pd.to_numeric(transfers["estimated_transfer_cost"], errors="coerce").fillna(0.0).sum()
        ),
        "total_net_value": float(
            pd.to_numeric(transfers["estimated_net_value"], errors="coerce").fillna(0.0).sum()
        ),
        "n_items": int(transfers["family"].nunique()),
        "n_donor_stores": int(transfers["from_store"].nunique()),
        "n_recipient_stores": int(transfers["to_store"].nunique()),
        "mean_transfer_units": float(quantity.mean()) if len(transfers) else 0.0,
    }
