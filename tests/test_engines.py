"""
Phase 4 test suite — business logic & decision engines.

Covers the three engines in :mod:`src.engines`:

* ``inventory``  — safety stock, reorder point, order quantity
* ``transfers``  — inter-store rebalancing
* ``deadstock``  — slow-mover detection and clearance bundling

The suite deliberately targets the awkward cases first: zero demand, exact
safety-stock boundaries, empty surplus/deficit pools and zero-deadstock
portfolios. No Kaggle data is required — every fixture is synthetic.

Run with::

    python -m pytest tests/test_engines.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src import config
from src.engines import deadstock, inventory, transfers

# ===========================================================================
# Fixtures & helpers
# ===========================================================================
DEFAULT_LEAD_TIME = 7
DEFAULT_REVIEW = 7


def _assert_columns(frame: pd.DataFrame, expected: tuple[str, ...]) -> None:
    assert list(frame.columns) == list(expected)


@pytest.fixture(scope="module")
def positions() -> pd.DataFrame:
    """
    Four stores of one family at a flat demand of 10/day.

    Stores 1–2 are overstocked (DOS 90/100), stores 3–4 understocked (DOS 6/4);
    safety stock is 30, so the target of 300 leaves surpluses of 600/700 and
    deficits of 240/260.
    """
    return pd.DataFrame(
        {
            "store_nbr": [1, 2, 3, 4],
            "family": ["X", "X", "X", "X"],
            "on_hand": [900.0, 1000.0, 60.0, 40.0],
            "avg_daily_demand": [10.0, 10.0, 10.0, 10.0],
            "safety_stock": [30.0, 30.0, 30.0, 30.0],
            "unit_cost": [5.0, 5.0, 5.0, 5.0],
        }
    )


@pytest.fixture(scope="module")
def sales_panel() -> pd.DataFrame:
    """
    120 days (2017-06-01 → 2017-09-28) for three families in store 1.

    FRESH sells every day (healthy), SLOW stopped selling on 2017-07-14 and
    only moved 28 units inside the 90-day window, DEAD never sold at all.
    """
    rng = np.random.default_rng(11)
    dates = pd.date_range("2017-06-01", periods=120, freq="D")
    stop = pd.Timestamp("2017-07-15")
    rows: list[dict[str, object]] = []
    for date in dates:
        rows.append({"date": date, "store_nbr": 1, "family": "FRESH", "sales": float(rng.poisson(5))})
        rows.append({"date": date, "store_nbr": 1, "family": "SLOW", "sales": 2.0 if date < stop else 0.0})
        rows.append({"date": date, "store_nbr": 1, "family": "DEAD", "sales": 0.0})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def stock_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_nbr": [1, 1, 1],
            "family": ["FRESH", "SLOW", "DEAD"],
            "on_hand": [20.0, 300.0, 150.0],
            "unit_cost": [3.0, 4.0, 6.0],
        }
    )


@pytest.fixture(scope="module")
def detected(sales_panel: pd.DataFrame, stock_frame: pd.DataFrame) -> pd.DataFrame:
    return deadstock.detect_deadstock(sales_panel, stock_frame)


@pytest.fixture(scope="module")
def sales_affinity(sales_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Affinity matrix built from the *sales* panel, so its family scope
    (FRESH/SLOW/DEAD) matches the detected deadstock rows.
    """
    return deadstock.build_affinity_matrix(sales_panel, method="residual")


@pytest.fixture(scope="module")
def affinity() -> pd.DataFrame:
    """
    A/B move together (shared common + idiosyncratic factor), C only shares the
    common factor — the residual method must separate them.
    """
    rng = np.random.default_rng(3)
    days = 60
    dates = pd.date_range("2017-07-01", periods=days, freq="D")
    common = rng.normal(0.0, 1.0, days)
    idiosyncratic = rng.normal(0.0, 1.0, days)
    series = {
        "A": 10.0 + 5.0 * common + 3.0 * idiosyncratic + 0.5 * rng.normal(0, 1, days),
        "B": 10.0 + 5.0 * common + 3.0 * idiosyncratic + 0.5 * rng.normal(0, 1, days),
        "C": 10.0 + 5.0 * common + 0.5 * rng.normal(0, 1, days),
    }
    rows = []
    for family, values in series.items():
        for date, value in zip(dates, values):
            rows.append({"date": date, "store_nbr": 1, "family": family, "sales": float(max(value, 0.0))})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def affinity_matrix(affinity: pd.DataFrame) -> pd.DataFrame:
    return deadstock.build_affinity_matrix(affinity)


@pytest.fixture(scope="module")
def heroes(sales_panel: pd.DataFrame) -> pd.DataFrame:
    return deadstock.select_hero_items(sales_panel, top_n=3)


# ===========================================================================
# inventory — z-score
# ===========================================================================
def test_z_score_known_values() -> None:
    assert inventory.z_score(0.95) == pytest.approx(1.6448536, abs=1e-6)
    assert inventory.z_score(0.975) == pytest.approx(1.959964, abs=1e-6)
    assert inventory.z_score(0.5) == pytest.approx(0.0, abs=1e-9)


def test_z_score_is_monotonic_in_service_level() -> None:
    levels = [0.5, 0.8, 0.9, 0.95, 0.99]
    scores = [inventory.z_score(level) for level in levels]
    assert all(later > earlier for earlier, later in zip(scores, scores[1:]))


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_z_score_rejects_invalid_service_levels(bad: float) -> None:
    with pytest.raises(ValueError):
        inventory.z_score(bad)


# ===========================================================================
# inventory — safety stock
# ===========================================================================
def test_safety_stock_continuous_review_formula() -> None:
    # SS = Z * sigma_d * sqrt(L)
    expected = 1.6448536 * 2.0 * math.sqrt(7.0)
    value = inventory.safety_stock(10.0, 2.0, 7, 0.95, review_period_days=0)
    assert value == pytest.approx(expected, rel=1e-6)


def test_safety_stock_extends_protection_period_by_review_period() -> None:
    continuous = inventory.safety_stock(10.0, 2.0, 7, 0.95, review_period_days=0)
    periodic = inventory.safety_stock(10.0, 2.0, 7, 0.95, review_period_days=7)
    # protection period doubles (7 -> 14), so SS grows by sqrt(2)
    assert periodic == pytest.approx(continuous * math.sqrt(2.0), rel=1e-9)


def test_safety_stock_scales_with_sqrt_of_lead_time() -> None:
    short = inventory.safety_stock(10.0, 2.0, 7, 0.95, review_period_days=0)
    long = inventory.safety_stock(10.0, 2.0, 28, 0.95, review_period_days=0)
    assert long == pytest.approx(2.0 * short, rel=1e-9)


def test_safety_stock_zero_for_flat_demand_and_deterministic_lead_time() -> None:
    assert inventory.safety_stock(10.0, 0.0, 7, 0.95, review_period_days=0) == 0.0


def test_safety_stock_zero_when_there_is_no_demand() -> None:
    assert inventory.safety_stock(0.0, 0.0, 7, 0.95) == 0.0


def test_safety_stock_accounts_for_lead_time_variability() -> None:
    # sigma_d = 0, so only the d^2 * sigma_L^2 term survives:
    # sqrt(10^2 * 2^2) = 20 -> SS = Z * 20
    value = inventory.safety_stock(
        10.0, 0.0, 7, 0.95, review_period_days=0, lead_time_std_days=2.0
    )
    assert value == pytest.approx(1.6448536 * 20.0, rel=1e-6)


def test_safety_stock_absorbs_forecast_error_in_quadrature() -> None:
    inflated = inventory.safety_stock(
        10.0, 2.0, 7, 0.95, review_period_days=0, forecast_error_std=2.0
    )
    equivalent = inventory.safety_stock(10.0, 2.0 * math.sqrt(2.0), 7, 0.95, review_period_days=0)
    assert inflated == pytest.approx(equivalent, rel=1e-9)


def test_safety_stock_clips_negative_inputs() -> None:
    assert inventory.safety_stock(-10.0, -2.0, -7, 0.95) == 0.0


def test_safety_stock_uses_config_defaults() -> None:
    value = inventory.safety_stock(10.0, 2.0, config.LEAD_TIME_DAYS)
    expected = inventory.z_score(config.SERVICE_LEVEL) * 2.0 * math.sqrt(
        config.LEAD_TIME_DAYS + config.REVIEW_PERIOD_DAYS
    )
    assert value == pytest.approx(expected, rel=1e-9)


# ===========================================================================
# inventory — reorder point & order-up-to level
# ===========================================================================
def test_reorder_point_is_lead_time_demand_plus_safety_stock() -> None:
    assert inventory.reorder_point(10.0, 7, 5.0) == pytest.approx(75.0)


def test_reorder_point_zero_when_no_demand_and_no_buffer() -> None:
    assert inventory.reorder_point(0.0, 7, 0.0) == 0.0


def test_order_up_to_level_covers_lead_time_plus_review() -> None:
    assert inventory.order_up_to_level(10.0, 7, 5.0, 7.0) == pytest.approx(145.0)


def test_order_up_to_level_equals_reorder_point_without_review_period() -> None:
    assert inventory.order_up_to_level(10.0, 7, 5.0, 0.0) == pytest.approx(
        inventory.reorder_point(10.0, 7, 5.0)
    )


def test_expected_lead_time_demand() -> None:
    assert inventory.expected_lead_time_demand(10.0, 7) == pytest.approx(70.0)
    assert inventory.expected_lead_time_demand(10.0, 0) == 0.0


# ===========================================================================
# inventory — days of supply
# ===========================================================================
def test_days_of_supply_basic() -> None:
    assert inventory.days_of_supply(50.0, 10.0) == pytest.approx(5.0)
    assert inventory.days_of_supply(0.0, 10.0) == 0.0


def test_days_of_supply_is_infinite_without_demand() -> None:
    assert inventory.days_of_supply(50.0, 0.0) == math.inf
    assert inventory.days_of_supply(0.0, 0.0) == math.inf


# ===========================================================================
# inventory — recommended order quantity
# ===========================================================================
def test_order_quantity_fills_the_gap_to_target() -> None:
    assert inventory.recommended_order_quantity(50.0, 0.0, 152.309) == 103.0


def test_order_quantity_is_zero_when_target_is_met() -> None:
    assert inventory.recommended_order_quantity(200.0, 0.0, 152.309) == 0.0
    assert inventory.recommended_order_quantity(100.0, 60.0, 152.309) == 0.0


def test_order_quantity_nets_off_incoming_stock() -> None:
    assert inventory.recommended_order_quantity(0.0, 100.0, 152.309) == 53.0


def test_order_quantity_respects_minimum_order_quantity() -> None:
    # gap of 3 is lifted to the MOQ of 10
    assert inventory.recommended_order_quantity(20.0, 0.0, 23.0, min_order_qty=10) == 10.0


def test_order_quantity_rounds_up_to_case_multiple() -> None:
    # gap 102.309 -> 9 cases of 12 = 108
    assert inventory.recommended_order_quantity(50.0, 0.0, 152.309, order_multiple=12) == 108.0


def test_order_quantity_rounds_up_whole_units_by_default() -> None:
    assert inventory.recommended_order_quantity(10.2, 0.0, 20.0) == 10.0


# ===========================================================================
# inventory — full policy
# ===========================================================================
def test_inventory_policy_returns_the_documented_schema() -> None:
    policy = inventory.inventory_policy(10.0, 2.0, 50.0)
    assert list(policy.keys()) == list(inventory.POLICY_COLUMNS)


def test_inventory_policy_full_calculation() -> None:
    policy = inventory.inventory_policy(avg_daily_demand=10.0, demand_std=2.0, on_hand=50.0)
    assert policy["safety_stock"] == pytest.approx(12.3094, abs=1e-3)
    assert policy["reorder_point"] == pytest.approx(82.3094, abs=1e-3)
    assert policy["order_up_to_level"] == pytest.approx(152.3094, abs=1e-3)
    assert policy["reorder_quantity"] == 103.0
    assert policy["reorder_value"] == pytest.approx(515.0)
    assert policy["days_of_supply"] == pytest.approx(5.0)
    assert policy["z_score"] == pytest.approx(1.6449, abs=1e-3)
    assert policy["stockout_risk"] is True
    assert policy["needs_reorder"] is True
    assert policy["below_safety_stock"] is False
    assert policy["overstock_risk"] is False
    assert policy["stockout_risk_level"] == "medium"


def test_inventory_policy_zero_demand_is_completely_inert() -> None:
    policy = inventory.inventory_policy(avg_daily_demand=0.0, demand_std=0.0, on_hand=5.0)
    assert policy["safety_stock"] == 0.0
    assert policy["reorder_point"] == 0.0
    assert policy["order_up_to_level"] == 0.0
    assert policy["reorder_quantity"] == 0.0
    assert policy["days_of_supply"] == math.inf
    assert policy["cover_days"] == math.inf
    assert policy["stockout_risk"] is False
    assert policy["needs_reorder"] is False
    assert policy["below_safety_stock"] is False
    assert policy["overstock_risk"] is False
    assert policy["stockout_risk_level"] == "none"


def test_inventory_policy_exact_reorder_point_is_not_a_risk() -> None:
    safety = inventory.safety_stock(10.0, 2.0, DEFAULT_LEAD_TIME)
    rop = inventory.reorder_point(10.0, DEFAULT_LEAD_TIME, safety)
    policy = inventory.inventory_policy(avg_daily_demand=10.0, demand_std=2.0, on_hand=rop)
    assert policy["on_hand"] == pytest.approx(rop, abs=1e-3)   # policy values are rounded to 4 dp
    assert policy["stockout_risk"] is False        # strictly below triggers
    assert policy["needs_reorder"] is False
    assert policy["stockout_risk_level"] == "none"


def test_inventory_policy_exact_safety_stock_boundary() -> None:
    safety = inventory.safety_stock(10.0, 2.0, DEFAULT_LEAD_TIME)
    policy = inventory.inventory_policy(avg_daily_demand=10.0, demand_std=2.0, on_hand=safety)
    assert policy["below_safety_stock"] is False   # strictly below triggers
    assert policy["stockout_risk"] is True         # SS is under the ROP
    assert policy["stockout_risk_level"] == "medium"


def test_inventory_policy_escalates_to_critical_when_out_of_stock() -> None:
    policy = inventory.inventory_policy(avg_daily_demand=10.0, demand_std=2.0, on_hand=0.0)
    assert policy["is_out_of_stock"] is True
    assert policy["stockout_risk"] is True
    assert policy["stockout_risk_level"] == "critical"


def test_inventory_policy_escalates_to_high_below_safety_stock() -> None:
    safety = inventory.safety_stock(10.0, 2.0, DEFAULT_LEAD_TIME)
    policy = inventory.inventory_policy(avg_daily_demand=10.0, demand_std=2.0, on_hand=safety - 1.0)
    assert policy["below_safety_stock"] is True
    assert policy["stockout_risk_level"] == "high"


def test_inventory_policy_flags_overstock() -> None:
    policy = inventory.inventory_policy(avg_daily_demand=10.0, demand_std=2.0, on_hand=1000.0)
    assert policy["days_of_supply"] == pytest.approx(100.0)
    assert policy["overstock_risk"] is True
    assert policy["reorder_quantity"] == 0.0


def test_inventory_policy_clips_negative_on_hand() -> None:
    policy = inventory.inventory_policy(avg_daily_demand=10.0, demand_std=2.0, on_hand=-50.0)
    assert policy["on_hand"] == 0.0
    assert policy["is_out_of_stock"] is True


def test_inventory_policy_rejects_invalid_service_level() -> None:
    with pytest.raises(ValueError):
        inventory.inventory_policy(avg_daily_demand=10.0, demand_std=2.0, on_hand=50.0, service_level=1.5)


# ===========================================================================
# inventory — forecast -> demand summary -> policy table
# ===========================================================================
def _flat_forecasts(days: int = 30, daily: float = 10.0, stores: tuple[int, ...] = (1,), family: str = "A") -> pd.DataFrame:
    dates = pd.date_range("2017-08-01", periods=days, freq="D")
    rows = []
    for store in stores:
        for date in dates:
            rows.append({"date": date, "store_nbr": store, "family": family, "prediction": daily})
    return pd.DataFrame(rows)


def test_demand_summary_from_forecast_schema_and_values() -> None:
    summary = inventory.demand_summary_from_forecast(_flat_forecasts())
    _assert_columns(summary, inventory.DEMAND_SUMMARY_COLUMNS)
    row = summary.iloc[0]
    assert row["n_days"] == 30
    assert row["avg_daily_demand"] == pytest.approx(10.0)
    assert row["demand_std"] == pytest.approx(0.0)
    assert row["total_demand"] == pytest.approx(300.0)
    assert row["min_daily_demand"] == pytest.approx(10.0)
    assert row["max_daily_demand"] == pytest.approx(10.0)
    assert row["last_forecast_date"] == pd.Timestamp("2017-08-30")


def test_demand_summary_horizon_cap() -> None:
    summary = inventory.demand_summary_from_forecast(_flat_forecasts(), horizon=14)
    assert summary.iloc[0]["n_days"] == 14
    assert summary.iloc[0]["total_demand"] == pytest.approx(140.0)


def test_demand_summary_multi_series() -> None:
    forecasts = pd.concat(
        [_flat_forecasts(daily=10.0, stores=(1,)), _flat_forecasts(daily=4.0, stores=(2,))],
        ignore_index=True,
    )
    summary = inventory.demand_summary_from_forecast(forecasts)
    assert len(summary) == 2
    assert set(summary["store_nbr"]) == {1, 2}
    assert summary.loc[summary["store_nbr"] == 2, "avg_daily_demand"].iloc[0] == pytest.approx(4.0)


def test_demand_summary_ignores_negative_predictions() -> None:
    forecasts = _flat_forecasts()
    forecasts.loc[forecasts.index[:5], "prediction"] = -3.0
    summary = inventory.demand_summary_from_forecast(forecasts)
    assert summary.iloc[0]["avg_daily_demand"] == pytest.approx(250.0 / 30.0)  # -3 -> 0, 25*10/30


def test_demand_summary_rejects_non_positive_horizon() -> None:
    with pytest.raises(ValueError):
        inventory.demand_summary_from_forecast(_flat_forecasts(), horizon=0)


def test_demand_summary_empty_input_returns_schema() -> None:
    empty = pd.DataFrame(
        {
            "date": pd.Series([], dtype="datetime64[ns]"),
            "store_nbr": pd.Series([], dtype="int64"),
            "family": pd.Series([], dtype="object"),
            "prediction": pd.Series([], dtype="float64"),
        }
    )
    summary = inventory.demand_summary_from_forecast(empty)
    _assert_columns(summary, inventory.DEMAND_SUMMARY_COLUMNS)
    assert summary.empty


def test_demand_summary_requires_prediction_column() -> None:
    with pytest.raises(KeyError):
        inventory.demand_summary_from_forecast(pd.DataFrame({"date": [], "store_nbr": [], "family": []}))


def test_inventory_policy_table_schema_and_rows() -> None:
    summary = inventory.demand_summary_from_forecast(
        pd.concat([_flat_forecasts(stores=(1,)), _flat_forecasts(daily=2.0, stores=(2,))], ignore_index=True)
    )
    stock = pd.DataFrame({"store_nbr": [1, 2], "family": ["A", "A"], "on_hand": [400.0, 5.0]})
    table = inventory.inventory_policy_table(summary, stock)
    _assert_columns(table, inventory.POLICY_COLUMNS)
    assert len(table) == 2


def test_inventory_policy_table_matches_the_single_series_policy() -> None:
    summary = inventory.demand_summary_from_forecast(_flat_forecasts())
    stock = pd.DataFrame({"store_nbr": [1], "family": ["A"], "on_hand": [50.0]})
    table = inventory.inventory_policy_table(summary, stock)
    direct = inventory.inventory_policy(
        avg_daily_demand=10.0, demand_std=0.0, on_hand=50.0, store_nbr=1, family="A"
    )
    for column in inventory.POLICY_COLUMNS:
        assert table.iloc[0][column] == direct[column]


def test_inventory_policy_table_uses_supplied_unit_cost() -> None:
    summary = inventory.demand_summary_from_forecast(_flat_forecasts())
    stock = pd.DataFrame({"store_nbr": [1], "family": ["A"], "on_hand": [0.0], "unit_cost": [2.0]})
    table = inventory.inventory_policy_table(summary, stock)
    assert table.iloc[0]["unit_cost"] == pytest.approx(2.0)
    assert table.iloc[0]["reorder_value"] == pytest.approx(table.iloc[0]["reorder_quantity"] * 2.0)


def test_inventory_policy_table_empty_demand_returns_schema() -> None:
    demand = pd.DataFrame({"store_nbr": [], "family": [], "avg_daily_demand": []})
    stock = pd.DataFrame({"store_nbr": [1], "family": ["A"], "on_hand": [10.0]})
    table = inventory.inventory_policy_table(demand, stock)
    _assert_columns(table, inventory.POLICY_COLUMNS)
    assert table.empty


def test_inventory_policy_table_requires_on_hand() -> None:
    summary = inventory.demand_summary_from_forecast(_flat_forecasts())
    with pytest.raises(KeyError):
        inventory.inventory_policy_table(summary, pd.DataFrame({"store_nbr": [1], "family": ["A"]}))


# ===========================================================================
# transfers — classification
# ===========================================================================
def test_classify_statuses(positions: pd.DataFrame) -> None:
    classified = transfers.classify_stock_positions(positions)
    statuses = dict(zip(classified["store_nbr"], classified["status"]))
    assert statuses == {1: "overstocked", 2: "overstocked", 3: "understocked", 4: "understocked"}


def test_classify_surplus_and_deficit_math(positions: pd.DataFrame) -> None:
    classified = transfers.classify_stock_positions(positions).set_index("store_nbr")
    assert classified.loc[1, "surplus_units"] == pytest.approx(600.0)   # 900 - max(300, 30)
    assert classified.loc[2, "surplus_units"] == pytest.approx(700.0)
    assert classified.loc[3, "deficit_units"] == pytest.approx(240.0)   # 300 - 60
    assert classified.loc[4, "deficit_units"] == pytest.approx(260.0)
    assert classified.loc[1, "deficit_units"] == 0.0
    assert classified.loc[3, "surplus_units"] == 0.0
    assert classified.loc[1, "reserve_units"] == pytest.approx(300.0)


def test_classify_healthy_store_is_neutral(positions: pd.DataFrame) -> None:
    healthy = positions.assign(on_hand=[300.0, 300.0, 300.0, 300.0])
    classified = transfers.classify_stock_positions(healthy)
    assert set(classified["status"]) == {"healthy"}
    assert classified["surplus_units"].sum() == 0.0
    assert classified["deficit_units"].sum() == 0.0


def test_classify_no_demand_is_neutral() -> None:
    positions = pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "family": ["X", "X"],
            "on_hand": [900.0, 10.0],
            "avg_daily_demand": [0.0, 10.0],
        }
    )
    classified = transfers.classify_stock_positions(positions).set_index("store_nbr")
    assert classified.loc[1, "status"] == "no_demand"
    assert classified.loc[1, "surplus_units"] == 0.0
    assert classified.loc[1, "deficit_units"] == 0.0
    assert classified.loc[1, "days_of_supply"] == math.inf


def test_classify_defaults_missing_safety_stock_to_zero(positions: pd.DataFrame) -> None:
    classified = transfers.classify_stock_positions(positions.drop(columns=["safety_stock"]))
    row = classified.set_index("store_nbr").loc[1]
    assert row["safety_stock"] == 0.0
    assert row["reserve_units"] == pytest.approx(300.0)   # target units only
    assert row["surplus_units"] == pytest.approx(600.0)


def test_classify_defaults_missing_unit_cost(positions: pd.DataFrame) -> None:
    classified = transfers.classify_stock_positions(positions.drop(columns=["unit_cost"]))
    assert (classified["unit_cost"] == config.DEFAULT_UNIT_COST).all()


def test_classify_protect_donor_stock_can_be_disabled(positions: pd.DataFrame) -> None:
    policy = transfers.TransferPolicy(protect_donor_stock=False)
    classified = transfers.classify_stock_positions(positions, policy=policy).set_index("store_nbr")
    assert classified.loc[1, "reserve_units"] == 0.0
    assert classified.loc[1, "surplus_units"] == pytest.approx(900.0)


def test_classify_preserves_extra_columns(positions: pd.DataFrame) -> None:
    classified = transfers.classify_stock_positions(positions.assign(reorder_point=42.0))
    assert "reorder_point" in classified.columns


def test_classify_accepts_category_keys_without_mismatch() -> None:
    frame = pd.DataFrame(
        {
            "store_nbr": pd.Series([1, 2], dtype="int16"),
            "family": pd.Series(["X", "X"], dtype="category"),
            "on_hand": [900.0, 10.0],
            "avg_daily_demand": [10.0, 10.0],
        }
    )
    classified = transfers.classify_stock_positions(frame)
    assert set(classified["status"]) == {"overstocked", "understocked"}


def test_detect_imbalances_filters_neutral_rows(positions: pd.DataFrame) -> None:
    mixed = pd.concat(
        [positions, positions.assign(store_nbr=[5, 6, 7, 8], on_hand=300.0)], ignore_index=True
    )
    imbalances = transfers.detect_imbalances(mixed)
    assert set(imbalances["status"]) == {"overstocked", "understocked"}
    assert len(imbalances) == 4


def test_classify_requires_on_hand_column(positions: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        transfers.classify_stock_positions(positions.drop(columns=["on_hand"]))


# ===========================================================================
# transfers — plan generation
# ===========================================================================
def test_generate_transfers_basic_plan(positions: pd.DataFrame) -> None:
    plan = transfers.generate_transfers(positions)
    _assert_columns(plan, transfers.TRANSFER_COLUMNS)
    assert len(plan) == 2
    assert set(plan["from_store"]) == {2}
    assert set(plan["to_store"]) == {3, 4}
    assert plan["quantity"].tolist() == [260, 240]
    assert plan["estimated_value"].sum() == pytest.approx(2500.0)
    assert plan["estimated_transfer_cost"].sum() == pytest.approx(290.0)
    assert plan["estimated_net_value"].sum() == pytest.approx(2210.0)


def test_generate_transfers_never_promises_a_unit_twice(positions: pd.DataFrame) -> None:
    plan = transfers.generate_transfers(positions)
    classified = transfers.classify_stock_positions(positions).set_index("store_nbr")
    moved = plan.groupby("from_store")["quantity"].sum()
    for store, quantity in moved.items():
        assert quantity <= classified.loc[store, "surplus_units"]
    filled = plan.groupby("to_store")["quantity"].sum()
    for store, quantity in filled.items():
        assert quantity <= classified.loc[store, "deficit_units"] + 1e-9


def test_generate_transfers_donor_stays_above_its_reserve(positions: pd.DataFrame) -> None:
    plan = transfers.generate_transfers(positions)
    classified = transfers.classify_stock_positions(positions).set_index("store_nbr")
    for _, leg in plan.iterrows():
        reserve = classified.loc[leg["from_store"], "reserve_units"]
        assert leg["donor_on_hand_after"] >= reserve - 1e-9


def test_generate_transfers_recipient_cover_improves(positions: pd.DataFrame) -> None:
    plan = transfers.generate_transfers(positions)
    assert (plan["recipient_dos_after"] > plan["recipient_dos_before"]).all()
    assert (plan["donor_dos_after"] < plan["donor_dos_before"]).all()


def test_generate_transfers_empty_when_no_surplus(positions: pd.DataFrame) -> None:
    plan = transfers.generate_transfers(positions.assign(on_hand=[300.0, 300.0, 60.0, 40.0]))
    _assert_columns(plan, transfers.TRANSFER_COLUMNS)
    assert plan.empty


def test_generate_transfers_empty_when_no_deficit(positions: pd.DataFrame) -> None:
    plan = transfers.generate_transfers(positions.assign(on_hand=[900.0, 1000.0, 300.0, 300.0]))
    assert plan.empty


def test_generate_transfers_empty_when_everything_is_healthy(positions: pd.DataFrame) -> None:
    plan = transfers.generate_transfers(positions.assign(on_hand=[100.0, 100.0, 100.0, 100.0]))
    assert plan.empty


def test_generate_transfers_respects_min_transfer_units() -> None:
    policy = transfers.TransferPolicy(target_dos=15.0, min_transfer_units=5)
    positions = pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "family": ["X", "X"],
            "on_hand": [900.0, 148.0],   # DOS 90 (donor) vs 14.8 (deficit of only 2)
            "avg_daily_demand": [10.0, 10.0],
            "safety_stock": [30.0, 30.0],
        }
    )
    assert transfers.generate_transfers(positions, policy=policy).empty


def test_generate_transfers_caps_each_leg_at_max_units(positions: pd.DataFrame) -> None:
    policy = transfers.TransferPolicy(max_transfer_units=100)
    plan = transfers.generate_transfers(positions, policy=policy)
    assert not plan.empty
    assert (plan["quantity"] <= 100).all()
    assert len(plan) == 4   # each of the two receivers draws from both donors


def test_generate_transfers_skips_legs_without_net_value(positions: pd.DataFrame) -> None:
    policy = transfers.TransferPolicy(min_net_value=5000.0)
    assert transfers.generate_transfers(positions, policy=policy).empty


def test_generate_transfers_uses_multiple_donors() -> None:
    policy = transfers.TransferPolicy(overstock_dos_threshold=20.0, understock_dos_threshold=15.0)
    positions = pd.DataFrame(
        {
            "store_nbr": [1, 2, 3],
            "family": ["X", "X", "X"],
            "on_hand": [400.0, 450.0, 40.0],   # surpluses 100 and 150 vs deficit 260
            "avg_daily_demand": [10.0, 10.0, 10.0],
            "safety_stock": [30.0, 30.0, 30.0],
        }
    )
    plan = transfers.generate_transfers(positions, policy=policy)
    assert len(plan) == 2
    assert plan["quantity"].sum() == pytest.approx(250.0)
    assert set(plan["from_store"]) == {1, 2}
    assert set(plan["to_store"]) == {3}


def test_generate_transfers_honours_max_donors_per_recipient() -> None:
    policy = transfers.TransferPolicy(
        overstock_dos_threshold=20.0, max_transfer_units=40, max_donors_per_recipient=1
    )
    positions = pd.DataFrame(
        {
            "store_nbr": [1, 2, 3, 4],
            "family": ["X"] * 4,
            "on_hand": [400.0, 450.0, 500.0, 40.0],
            "avg_daily_demand": [10.0] * 4,
            "safety_stock": [30.0] * 4,
        }
    )
    plan = transfers.generate_transfers(positions, policy=policy)
    assert len(plan) == 1
    assert plan.iloc[0]["quantity"] == 40


def test_generate_transfers_does_not_mix_families() -> None:
    positions = pd.DataFrame(
        {
            "store_nbr": [1, 2, 3, 4],
            "family": ["X", "X", "Y", "Y"],
            "on_hand": [900.0, 40.0, 900.0, 40.0],
            "avg_daily_demand": [10.0] * 4,
            "safety_stock": [30.0] * 4,
        }
    )
    plan = transfers.generate_transfers(positions)
    assert len(plan) == 2
    by_family = plan.set_index("family")
    assert by_family.loc["X", "from_store"] == 1
    assert by_family.loc["X", "to_store"] == 2
    assert by_family.loc["Y", "from_store"] == 3
    assert by_family.loc["Y", "to_store"] == 4


def test_generate_transfers_ignores_no_demand_stores() -> None:
    positions = pd.DataFrame(
        {
            "store_nbr": [1, 2, 3],
            "family": ["X", "X", "X"],
            "on_hand": [900.0, 40.0, 500.0],
            "avg_daily_demand": [10.0, 10.0, 0.0],
            "safety_stock": [30.0] * 3,
        }
    )
    plan = transfers.generate_transfers(positions)
    assert set(plan["to_store"]) == {2}
    assert set(plan["from_store"]) == {1}
    assert plan["recipient_dos_after"].notna().all()


def test_generate_transfers_accepts_a_unit_cost_mapping(positions: pd.DataFrame) -> None:
    plan = transfers.generate_transfers(positions, unit_costs={"X": 2.5})
    assert np.allclose(plan["unit_cost"].to_numpy(), 2.5)
    assert plan.iloc[0]["estimated_value"] == pytest.approx(plan.iloc[0]["quantity"] * 2.5)


def test_generate_transfers_requires_on_hand(positions: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        transfers.generate_transfers(positions.drop(columns=["on_hand"]))


def test_generate_transfers_does_not_mutate_input(positions: pd.DataFrame) -> None:
    original = positions.copy(deep=True)
    transfers.generate_transfers(positions)
    pd.testing.assert_frame_equal(positions, original)


# ===========================================================================
# transfers — summary
# ===========================================================================
def test_transfer_summary_populated(positions: pd.DataFrame) -> None:
    summary = transfers.transfer_summary(transfers.generate_transfers(positions))
    assert summary["n_transfers"] == 2
    assert summary["total_units"] == pytest.approx(500.0)
    assert summary["total_value"] == pytest.approx(2500.0)
    assert summary["total_transfer_cost"] == pytest.approx(290.0)
    assert summary["total_net_value"] == pytest.approx(2210.0)
    assert summary["n_items"] == 1
    assert summary["n_donor_stores"] == 1
    assert summary["n_recipient_stores"] == 2
    assert summary["mean_transfer_units"] == pytest.approx(250.0)


def test_transfer_summary_empty_is_all_zeros() -> None:
    summary = transfers.transfer_summary(pd.DataFrame(columns=list(transfers.TRANSFER_COLUMNS)))
    assert all(value == 0 for value in summary.values())


# ===========================================================================
# deadstock — detection
# ===========================================================================
def test_detect_deadstock_schema_and_ordering(detected: pd.DataFrame) -> None:
    _assert_columns(detected, deadstock.DEADSTOCK_COLUMNS)
    assert detected["severity_score"].is_monotonic_decreasing


def test_detect_deadstock_flags_never_sold_stock(detected: pd.DataFrame) -> None:
    row = detected.set_index("family").loc["DEAD"]
    assert bool(row["is_deadstock"]) is True
    assert row["avg_daily_velocity"] == pytest.approx(0.0)
    assert bool(row["never_sold"]) is True
    assert row["days_since_last_sale"] >= config.DEADSTOCK_MIN_DAYS_SINCE_LAST_SALE
    assert row["severity"] == "critical"
    assert row["baseline_days_to_clear"] == pytest.approx(float(config.DEADSTOCK_MAX_DAYS_TO_CLEAR))


def test_detect_deadstock_flags_stalled_stock(detected: pd.DataFrame) -> None:
    row = detected.set_index("family").loc["SLOW"]
    assert bool(row["is_deadstock"]) is True
    assert row["days_since_last_sale"] == 76          # last sale 2017-07-14, panel ends 2017-09-28
    assert bool(row["never_sold"]) is False
    assert row["avg_daily_velocity"] == pytest.approx(28.0 / 90.0)
    assert row["severity"] in {"medium", "high", "critical"}


def test_detect_deadstock_ignores_fast_movers(detected: pd.DataFrame) -> None:
    row = detected.set_index("family").loc["FRESH"]
    assert bool(row["is_deadstock"]) is False
    assert row["severity"] == "none"
    assert row["severity_score"] == pytest.approx(0.0)
    assert row["holding_cost_savings"] == pytest.approx(0.0)
    assert row["days_since_last_sale"] == 0


def test_detect_deadstock_holding_cost_savings(detected: pd.DataFrame) -> None:
    row = detected.set_index("family").loc["DEAD"]
    # 150 units * $6 = $900; 20%/yr; 700 days saved of 365
    expected = 900.0 * config.DEADSTOCK_HOLDING_COST_RATE * (730 - 30) / 365.0
    assert row["inventory_value"] == pytest.approx(900.0)
    assert row["annual_holding_cost"] == pytest.approx(180.0)
    assert row["holding_cost_savings"] == pytest.approx(expected, rel=1e-6)


def test_detect_deadstock_skips_zero_stock(sales_panel: pd.DataFrame) -> None:
    stock = pd.DataFrame(
        {
            "store_nbr": [1, 1],
            "family": ["DEAD", "NEVER"],
            "on_hand": [0.0, 0.0],
            "unit_cost": [6.0, 6.0],
        }
    )
    result = deadstock.detect_deadstock(sales_panel, stock)
    assert not result["is_deadstock"].any()


def test_detect_deadstock_includes_skus_absent_from_sales(sales_panel: pd.DataFrame) -> None:
    stock = pd.DataFrame({"store_nbr": [1], "family": ["NEW"], "on_hand": [50.0]})
    result = deadstock.detect_deadstock(sales_panel, stock)
    row = result.iloc[0]
    assert row["family"] == "NEW"
    assert bool(row["never_sold"]) is True
    assert bool(row["is_deadstock"]) is True


def test_detect_deadstock_requires_recent_purchase_age(sales_panel: pd.DataFrame) -> None:
    # Sells rarely (velocity under the threshold) but sold yesterday -> not deadstock.
    panel = sales_panel.copy()
    last = panel["date"].max()
    panel.loc[(panel["family"] == "SLOW") & (panel["date"] == last), "sales"] = 1.0
    stock = pd.DataFrame({"store_nbr": [1], "family": ["SLOW"], "on_hand": [300.0]})
    result = deadstock.detect_deadstock(panel, stock)
    assert bool(result.iloc[0]["is_deadstock"]) is False
    assert result.iloc[0]["days_since_last_sale"] == 0


def test_detect_deadstock_respects_velocity_threshold(sales_panel: pd.DataFrame, stock_frame: pd.DataFrame) -> None:
    policy = deadstock.DeadstockPolicy(velocity_threshold=10.0, min_days_since_last_sale=0)
    result = deadstock.detect_deadstock(sales_panel, stock_frame, policy=policy)
    # With a very loose threshold even the healthy mover qualifies (it has stock).
    assert bool(result.set_index("family").loc["FRESH", "is_deadstock"]) is True


def test_detect_deadstock_unit_cost_override(sales_panel: pd.DataFrame) -> None:
    stock = pd.DataFrame({"store_nbr": [1], "family": ["DEAD"], "on_hand": [100.0]})
    result = deadstock.detect_deadstock(sales_panel, stock, unit_costs={"DEAD": 12.0})
    assert result.iloc[0]["inventory_value"] == pytest.approx(1200.0)


def test_detect_deadstock_rejects_bad_window(sales_panel: pd.DataFrame, stock_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        deadstock.detect_deadstock(
            sales_panel, stock_frame, policy=deadstock.DeadstockPolicy(window_days=0)
        )


def test_detect_deadstock_requires_on_hand(sales_panel: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        deadstock.detect_deadstock(sales_panel, pd.DataFrame({"store_nbr": [1], "family": ["DEAD"]}))


def test_detect_deadstock_empty_sales_returns_schema() -> None:
    sales = pd.DataFrame(
        {
            "date": pd.Series([], dtype="datetime64[ns]"),
            "store_nbr": pd.Series([], dtype="int64"),
            "family": pd.Series([], dtype="object"),
            "sales": pd.Series([], dtype="float64"),
        }
    )
    stock = pd.DataFrame({"store_nbr": [1], "family": ["DEAD"], "on_hand": [10.0]})
    result = deadstock.detect_deadstock(sales, stock)
    _assert_columns(result, deadstock.DEADSTOCK_COLUMNS)
    assert result.empty


# ===========================================================================
# deadstock — affinity
# ===========================================================================
def test_affinity_matrix_is_square_with_unit_diagonal(affinity_matrix: pd.DataFrame) -> None:
    assert affinity_matrix.shape == (3, 3)
    assert list(affinity_matrix.index) == list(affinity_matrix.columns)
    assert np.allclose(np.diag(affinity_matrix.to_numpy()), 1.0)


def test_affinity_matrix_is_symmetric(affinity_matrix: pd.DataFrame) -> None:
    assert np.allclose(affinity_matrix.to_numpy(), affinity_matrix.to_numpy().T)


def test_residual_affinity_detects_true_complements(affinity_matrix: pd.DataFrame) -> None:
    assert affinity_matrix.loc["A", "B"] > 0.5
    assert affinity_matrix.loc["A", "B"] > affinity_matrix.loc["A", "C"]


def test_residual_method_removes_the_common_factor(affinity: pd.DataFrame) -> None:
    residual = deadstock.build_affinity_matrix(affinity, method="residual")
    raw = deadstock.build_affinity_matrix(affinity, method="correlation")
    # Raw correlation conflates A and C through shared seasonality...
    assert raw.loc["A", "C"] > residual.loc["A", "C"]
    # ...so removing the common factor separates the true complement (B) from
    # the unrelated family (C) far more sharply.
    assert (residual.loc["A", "B"] - residual.loc["A", "C"]) > (
        raw.loc["A", "B"] - raw.loc["A", "C"]
    )


def test_affinity_returns_zero_without_overlap(affinity: pd.DataFrame) -> None:
    panel = affinity.copy()
    early = panel["family"] == "A"
    panel.loc[early, "date"] = panel.loc[early, "date"] - pd.Timedelta(days=200)
    matrix = deadstock.build_affinity_matrix(panel, min_periods=5)
    assert matrix.loc["B", "A"] == pytest.approx(0.0)


def test_affinity_single_family_is_identity(affinity: pd.DataFrame) -> None:
    single = affinity[affinity["family"] == "A"]
    matrix = deadstock.build_affinity_matrix(single)
    assert matrix.shape == (1, 1)
    assert matrix.iloc[0, 0] == pytest.approx(1.0)


def test_affinity_empty_panel(affinity: pd.DataFrame) -> None:
    assert deadstock.build_affinity_matrix(affinity.iloc[0:0]).empty


def test_affinity_rejects_unknown_method(affinity: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        deadstock.build_affinity_matrix(affinity, method="magic")


# ===========================================================================
# deadstock — hero items
# ===========================================================================
def test_select_hero_items_ranks_by_velocity(heroes: pd.DataFrame) -> None:
    assert heroes["family"].tolist() == ["FRESH", "SLOW", "DEAD"]
    assert heroes["rank"].tolist() == [1, 2, 3]
    assert heroes["units_sold"].is_monotonic_decreasing


def test_select_hero_items_top_n(sales_panel: pd.DataFrame) -> None:
    heroes = deadstock.select_hero_items(sales_panel, top_n=1)
    assert len(heroes) == 1
    assert heroes.iloc[0]["family"] == "FRESH"


def test_select_hero_items_empty_panel(sales_panel: pd.DataFrame) -> None:
    heroes = deadstock.select_hero_items(sales_panel.iloc[0:0])
    assert heroes.empty
    assert {"family", "units_sold", "rank"}.issubset(heroes.columns)


# ===========================================================================
# deadstock — discount schedule
# ===========================================================================
def test_suggested_discount_follows_the_schedule() -> None:
    assert deadstock.suggested_discount_pct(0) == pytest.approx(0.15)
    assert deadstock.suggested_discount_pct(30) == pytest.approx(0.20)
    assert deadstock.suggested_discount_pct(90) == pytest.approx(0.30)
    assert deadstock.suggested_discount_pct(200) == pytest.approx(0.45)


def test_suggested_discount_is_monotonic_and_clipped() -> None:
    values = [deadstock.suggested_discount_pct(age) for age in (0, 30, 60, 120, 240, 480, 3000)]
    assert all(later >= earlier for earlier, later in zip(values, values[1:]))
    assert values[-1] == pytest.approx(config.DEADSTOCK_DISCOUNT_MAX)
    assert all(config.DEADSTOCK_DISCOUNT_MIN <= value <= config.DEADSTOCK_DISCOUNT_MAX for value in values)


def test_suggested_discount_respects_policy_bounds() -> None:
    policy = deadstock.DeadstockPolicy(discount_min=0.5, discount_max=0.6, discount_base=0.2)
    assert deadstock.suggested_discount_pct(0, policy) == pytest.approx(0.5)
    assert deadstock.suggested_discount_pct(10000, policy) == pytest.approx(0.6)


# ===========================================================================
# deadstock — bundling
# ===========================================================================
def test_generate_bundles_schema(detected: pd.DataFrame, sales_affinity: pd.DataFrame, heroes: pd.DataFrame) -> None:
    bundles = deadstock.generate_bundles(detected, sales_affinity, heroes)
    _assert_columns(bundles, deadstock.BUNDLE_COLUMNS)
    assert not bundles.empty


def test_generate_bundles_pairs_every_deadstock_item(
    detected: pd.DataFrame, sales_affinity: pd.DataFrame, heroes: pd.DataFrame
) -> None:
    bundles = deadstock.generate_bundles(detected, sales_affinity, heroes)
    deadstock_families = set(detected.loc[detected["is_deadstock"], "family"])
    assert set(bundles["deadstock_family"]) == deadstock_families
    assert bundles["bundle_id"].is_unique


def test_generate_bundles_never_pairs_deadstock_with_deadstock(
    detected: pd.DataFrame, sales_affinity: pd.DataFrame, heroes: pd.DataFrame
) -> None:
    bundles = deadstock.generate_bundles(detected, sales_affinity, heroes)
    deadstock_families = set(detected.loc[detected["is_deadstock"], "family"])
    assert not (set(bundles["complement_family"]) & deadstock_families)
    assert not (bundles["deadstock_family"] == bundles["complement_family"]).any()


def test_generate_bundles_labels_the_selection_basis(
    detected: pd.DataFrame, sales_affinity: pd.DataFrame, heroes: pd.DataFrame
) -> None:
    bundles = deadstock.generate_bundles(detected, sales_affinity, heroes)
    assert set(bundles["selection_basis"]).issubset({"affinity", "best_seller_fallback"})
    # In this synthetic panel no deadstock line has a positively-correlated
    # hero, so every bundle is an explicit best-seller fallback rather than
    # being silently dropped.
    assert (bundles["selection_basis"] == "best_seller_fallback").all()
    assert (bundles["complement_family"] == "FRESH").all()      # the top seller
    assert (bundles["complement_units_sold"] > 0.0).all()


def test_generate_bundles_prefers_the_positive_affinity_complement() -> None:
    """
    With a hand-built affinity matrix (the engine is agnostic to its source)
    the positively-correlated hero must win, and the basis must say so.
    """
    deadstock_rows = pd.DataFrame(
        {
            "store_nbr": [1],
            "family": ["DEADLINE"],
            "is_deadstock": [True],
            "days_since_last_sale": [120],
            "on_hand": [200.0],
            "unit_cost": [4.0],
            "severity": ["high"],
            "severity_score": [0.7],
            "holding_cost_savings": [50.0],
        }
    )
    hero_items = pd.DataFrame(
        {"family": ["HERO_A", "HERO_B"], "units_sold": [900.0, 800.0], "rank": [1, 2]}
    )
    matrix = pd.DataFrame(
        {
            "DEADLINE": [1.0, 0.55, -0.3],
            "HERO_A": [0.55, 1.0, 0.1],
            "HERO_B": [-0.3, 0.1, 1.0],
        },
        index=["DEADLINE", "HERO_A", "HERO_B"],
    )
    bundles = deadstock.generate_bundles(deadstock_rows, matrix, hero_items)
    _assert_columns(bundles, deadstock.BUNDLE_COLUMNS)
    assert len(bundles) == 1
    assert bundles.iloc[0]["complement_family"] == "HERO_A"
    assert bundles.iloc[0]["selection_basis"] == "affinity"
    assert bundles.iloc[0]["affinity"] == pytest.approx(0.55)


def test_generate_bundles_discount_within_bounds(
    detected: pd.DataFrame, sales_affinity: pd.DataFrame, heroes: pd.DataFrame
) -> None:
    bundles = deadstock.generate_bundles(detected, sales_affinity, heroes)
    assert bundles["suggested_discount_pct"].between(
        config.DEADSTOCK_DISCOUNT_MIN, config.DEADSTOCK_DISCOUNT_MAX
    ).all()
    unit_cost = bundles["deadstock_value"] / bundles["deadstock_units"]
    assert (bundles["discounted_unit_cost"] < unit_cost).all()   # a real discount


def test_generate_bundles_recovery_math(
    detected: pd.DataFrame, sales_affinity: pd.DataFrame, heroes: pd.DataFrame
) -> None:
    bundles = deadstock.generate_bundles(detected, sales_affinity, heroes)
    for _, row in bundles.iterrows():
        expected = row["deadstock_units"] * row["discounted_unit_cost"]
        assert row["recovery_at_discount"] == pytest.approx(expected, rel=1e-3)
        assert 0.0 < row["recovery_at_discount"] <= row["deadstock_value"]


def test_generate_bundles_top_n_limits_complements(
    detected: pd.DataFrame, sales_affinity: pd.DataFrame, heroes: pd.DataFrame
) -> None:
    bundles = deadstock.generate_bundles(detected, sales_affinity, heroes, top_n_complements=1)
    assert bundles.groupby("deadstock_family").size().max() == 1


def test_generate_bundles_without_affinity_falls_back_to_heroes(
    detected: pd.DataFrame, heroes: pd.DataFrame
) -> None:
    bundles = deadstock.generate_bundles(detected, None, heroes)
    assert not bundles.empty
    assert (bundles["selection_basis"] == "best_seller_fallback").all()
    assert (bundles["affinity"] == 0.0).all()


def test_generate_bundles_zero_deadstock_returns_empty(
    detected: pd.DataFrame, sales_affinity: pd.DataFrame, heroes: pd.DataFrame
) -> None:
    bundles = deadstock.generate_bundles(
        detected.assign(is_deadstock=False), sales_affinity, heroes
    )
    _assert_columns(bundles, deadstock.BUNDLE_COLUMNS)
    assert bundles.empty


def test_generate_bundles_empty_input_returns_schema(affinity_matrix: pd.DataFrame) -> None:
    empty = pd.DataFrame(columns=list(deadstock.DEADSTOCK_COLUMNS))
    bundles = deadstock.generate_bundles(empty, affinity_matrix)
    _assert_columns(bundles, deadstock.BUNDLE_COLUMNS)
    assert bundles.empty


def test_generate_bundles_requires_detection_columns(affinity_matrix: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        deadstock.generate_bundles(pd.DataFrame({"store_nbr": [1], "family": ["X"]}), affinity_matrix)


# ===========================================================================
# deadstock — clearance summary
# ===========================================================================
def test_clearance_summary_populated(
    detected: pd.DataFrame, sales_affinity: pd.DataFrame, heroes: pd.DataFrame
) -> None:
    bundles = deadstock.generate_bundles(detected, sales_affinity, heroes)
    summary = deadstock.clearance_summary(detected, bundles)
    assert summary["n_skus"] == 3
    assert summary["n_deadstock_skus"] == 2
    assert summary["n_critical_skus"] == 1
    assert summary["deadstock_units"] == pytest.approx(450.0)
    assert summary["deadstock_value"] == pytest.approx(2100.0)
    assert summary["potential_holding_savings"] > 0.0
    assert summary["n_bundles"] == len(bundles)
    assert 0.0 < summary["mean_discount_pct"] <= config.DEADSTOCK_DISCOUNT_MAX
    assert summary["recovery_at_discount"] > 0.0


def test_clearance_summary_without_deadstock(detected: pd.DataFrame) -> None:
    summary = deadstock.clearance_summary(detected.assign(is_deadstock=False))
    assert summary["n_deadstock_skus"] == 0
    assert summary["n_bundles"] == 0
    assert summary["deadstock_value"] == 0.0


def test_clearance_summary_empty_input() -> None:
    summary = deadstock.clearance_summary(pd.DataFrame(columns=list(deadstock.DEADSTOCK_COLUMNS)))
    assert all(value == 0 for value in summary.values())


# ===========================================================================
# Policy defaults mirror config
# ===========================================================================
def test_transfer_policy_defaults_track_config() -> None:
    policy = transfers.TransferPolicy()
    assert policy.overstock_dos_threshold == config.OVERSTOCK_DOS_THRESHOLD
    assert policy.understock_dos_threshold == config.UNDERSTOCK_DOS_THRESHOLD
    assert policy.target_dos == config.TRANSFER_TARGET_DOS
    assert policy.min_transfer_units == config.MIN_TRANSFER_UNITS
    assert policy.max_transfer_units == config.MAX_TRANSFER_UNITS
    assert policy.fixed_cost == config.TRANSFER_FIXED_COST
    assert policy.cost_per_unit == config.TRANSFER_COST_PER_UNIT
    assert policy.min_net_value == config.MIN_TRANSFER_NET_VALUE


def test_deadstock_policy_defaults_track_config() -> None:
    policy = deadstock.DeadstockPolicy()
    assert policy.window_days == config.DEADSTOCK_WINDOW_DAYS
    assert policy.velocity_threshold == config.DEADSTOCK_VELOCITY_THRESHOLD
    assert policy.min_days_since_last_sale == config.DEADSTOCK_MIN_DAYS_SINCE_LAST_SALE
    assert policy.holding_cost_rate == config.DEADSTOCK_HOLDING_COST_RATE
    assert policy.clearance_window_days == config.DEADSTOCK_CLEARANCE_WINDOW_DAYS
    assert policy.top_n_complements == config.BUNDLE_TOP_N_COMPLEMENTS


def test_engines_package_exposes_submodules() -> None:
    import src.engines as engines

    assert engines.inventory is inventory
    assert engines.transfers is transfers
    assert engines.deadstock is deadstock


# ===========================================================================
# End-to-end: forecast -> policy table -> transfer plan
# ===========================================================================
def test_forecast_to_transfer_plan_pipeline() -> None:
    forecasts = pd.concat(
        [
            _flat_forecasts(stores=(1,), family="GROCERY I"),
            _flat_forecasts(stores=(2,), family="GROCERY I"),
        ],
        ignore_index=True,
    )
    stock = pd.DataFrame(
        {"store_nbr": [1, 2], "family": ["GROCERY I", "GROCERY I"], "on_hand": [1000.0, 50.0]}
    )
    summary = inventory.demand_summary_from_forecast(forecasts)
    policy_table = inventory.inventory_policy_table(summary, stock)
    plan = transfers.generate_transfers(policy_table)

    assert len(plan) == 1
    leg = plan.iloc[0]
    assert leg["from_store"] == 1
    assert leg["to_store"] == 2
    assert leg["quantity"] == pytest.approx(250.0)   # 300 target - 50 on hand
    assert leg["family"] == "GROCERY I"
    assert leg["estimated_value"] > 0.0
    assert leg["recipient_dos_after"] > leg["recipient_dos_before"]


def test_deadstock_to_bundle_pipeline(
    detected: pd.DataFrame, sales_affinity: pd.DataFrame, heroes: pd.DataFrame
) -> None:
    bundles = deadstock.generate_bundles(detected, sales_affinity, heroes)
    summary = deadstock.clearance_summary(detected, bundles)
    assert summary["n_deadstock_skus"] == len(set(bundles["deadstock_family"]))
    assert summary["n_bundles"] > 0
    assert bundles["holding_cost_savings"].sum() > 0.0
