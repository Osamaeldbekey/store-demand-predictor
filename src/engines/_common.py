"""
Internal helpers shared by the Phase 4 decision engines.

This module is private (leading underscore): the public API lives in
``inventory``, ``transfers`` and ``deadstock``. Consolidating column validation
and unit-cost resolution here keeps those engines focused on business logic and
guarantees that all three resolve costs and units identically.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger("engines.common")


# ===========================================================================
# Validation
# ===========================================================================
def require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    context: str,
) -> None:
    """
    Raise a descriptive ``KeyError`` when required columns are missing.

    Parameters
    ----------
    frame:
        Frame to validate.
    columns:
        Columns the caller depends on.
    context:
        Function name or description, used in the error message.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"{context}: missing required column(s) {missing}. "
            f"Present columns: {list(frame.columns)}"
        )


def canonicalize_keys(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """
    Return a copy of ``frame`` with the group/merge keys cast to stable dtypes.

    Numeric keys (``store_nbr``) become ``int64``; every other key
    (``family``, held as a ``category`` in the processed parquet) becomes a
    plain string. Without this, merging a parquet-loaded frame (``category``)
    with an ad-hoc frame built from a dict (``object``) can fail with
    "merging on categorical columns with different categories", and category
    levels would leak mismatched ordering into ``groupby`` results.
    """
    frame = frame.copy()
    for key in keys:
        if key not in frame.columns:
            continue
        column = frame[key]
        if isinstance(column.dtype, pd.CategoricalDtype):
            target = "int64" if pd.api.types.is_numeric_dtype(column.cat.categories.dtype) else object
            frame[key] = column.astype(target)
        elif pd.api.types.is_numeric_dtype(column):
            frame[key] = column.astype("int64")
        else:
            frame[key] = column.astype("string").fillna("").astype(object)
    return frame


# ===========================================================================
# Unit costs
# ===========================================================================
def resolve_unit_costs(
    frame: pd.DataFrame,
    unit_costs: Mapping | pd.Series | float | None = None,
    keys: Sequence[str] = tuple(config.GROUP_KEYS),
    column: str = "unit_cost",
    default: float = config.DEFAULT_UNIT_COST,
) -> pd.Series:
    """
    Return a non-negative unit-cost series aligned to ``frame.index``.

    Resolution order (first match wins), so a caller can supply costs at
    whatever granularity their data supports:

    1. explicit ``unit_costs`` mapping/Series keyed by ``(store_nbr, family)``
    2. ... keyed by ``family``
    3. a scalar ``unit_costs``
    4. an existing ``unit_cost`` column on ``frame``
    5. ``default``

    The Favorita dataset ships no price column, so ``default`` (a config
    constant) is the normal path; the mapping hooks exist so the dashboard can
    inject real costs later without touching engine code.
    """
    if unit_costs is not None and not isinstance(unit_costs, (Mapping, pd.Series, int, float)):
        raise TypeError(
            "unit_costs must be a mapping, a pandas Series, a scalar or None; "
            f"got {type(unit_costs).__name__}."
        )

    # --- scalar override ---------------------------------------------------
    if isinstance(unit_costs, (int, float)) and not isinstance(unit_costs, bool):
        return pd.Series(float(unit_costs), index=frame.index, dtype="float64")

    # --- explicit table lookup --------------------------------------------
    if unit_costs is not None:
        table = unit_costs.to_dict() if isinstance(unit_costs, pd.Series) else dict(unit_costs)
        families = frame[keys[-1]] if keys and keys[-1] in frame.columns else None
        stores = frame[keys[0]] if keys and keys[0] in frame.columns else None

        values: list[float] = []
        for position, _ in enumerate(frame.index):
            family = families.iloc[position] if families is not None else None
            store = stores.iloc[position] if stores is not None else None
            value = table.get((store, family))
            if value is None:
                value = table.get(family)
            if value is None:
                value = default
            values.append(float(value))
        return pd.Series(values, index=frame.index, dtype="float64").clip(lower=0.0)

    # --- fall back to a frame column, then to the default ------------------
    if column in frame.columns:
        costs = pd.to_numeric(frame[column], errors="coerce")
        return costs.fillna(default).clip(lower=0.0).astype("float64")

    return pd.Series(float(default), index=frame.index, dtype="float64")


# ===========================================================================
# Numeric helpers
# ===========================================================================
def non_negative(value: float | int | None) -> float:
    """Coerce to a finite non-negative float (``None``/NaN -> 0.0)."""
    if value is None:
        return 0.0
    number = float(value)
    if not np.isfinite(number):
        return 0.0
    return max(number, 0.0)


def whole_units(value: float) -> int:
    """Floor a quantity to whole units (stock moves in indivisible pieces)."""
    if value is None or not np.isfinite(float(value)):
        return 0
    return int(np.floor(max(float(value), 0.0)))
