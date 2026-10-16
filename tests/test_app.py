"""
Phase 5/6 smoke tests for the Streamlit dashboard.

Two layers:

* **Fast, deterministic checks** on the pure helpers the pages depend on
  (:mod:`_data` demo corpus, :mod:`_ui` formatters) — these do not need a
  Streamlit runtime.
* **AppTest runs** of every page: each file is executed exactly as Streamlit
  executes it (``streamlit run``), and the test asserts that no exception
  surfaced and that the page's key elements actually rendered. Widget
  interactions (horizon toggle, service-level slider, transfer approval)
  are exercised too.

The data layer is process-cached (``st.cache_resource`` / ``st.cache_data``),
so the expensive first forecast is built once per pytest session and every
page afterwards paints from cache — the same behaviour the dashboard relies
on in production.

**Hermetic by construction.** ``tests/conftest.py`` exports
``SDP_FORCE_DEMO=1``, which pins ``app/_data.py`` to the synthetic corpus even
when the real Phase 1-3 artifacts are present on the machine. The suite must
not change meaning (or runtime) depending on whether a developer has run the
pipeline: asserting demo-mode numbers against a 20-store real scope is both
wrong and minutes slower. :func:`test_dashboard_suite_is_hermetic` fails loudly
if that pin is lost.
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

import pandas as pd
import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PROJECT_ROOT = APP_DIR.parent
for _path in (str(PROJECT_ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src import config  # noqa: E402
from src.engines import deadstock as engine_deadstock  # noqa: E402
from src.engines import transfers as engine_transfers  # noqa: E402

import _data  # noqa: E402
import _ui  # noqa: E402

pytest.importorskip("streamlit", reason="Streamlit is required for the dashboard")
pytest.importorskip("plotly", reason="Plotly is required for the dashboard")
pytest.importorskip("pyarrow", reason="pyarrow is required by the data layer")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_FILES: tuple[Path, ...] = (
    APP_DIR / "_data.py",
    APP_DIR / "_ui.py",
    APP_DIR / "main.py",
    APP_DIR / "pages" / "1_Demand_Forecasting.py",
    APP_DIR / "pages" / "2_Inventory_Optimization.py",
    APP_DIR / "pages" / "3_Stock_Transfers.py",
    APP_DIR / "pages" / "4_Deadstock_Clearance.py",
)

MAIN_PAGE = APP_DIR / "main.py"
PAGE_1 = APP_DIR / "pages" / "1_Demand_Forecasting.py"
PAGE_2 = APP_DIR / "pages" / "2_Inventory_Optimization.py"
PAGE_3 = APP_DIR / "pages" / "3_Stock_Transfers.py"
PAGE_4 = APP_DIR / "pages" / "4_Deadstock_Clearance.py"

# The first AppTest run builds the demo corpus and the 30-day recursion; every
# later run is served from the data layer's cache.
APPTEST_TIMEOUT = 240.0

_ACTIVE_FAMILIES = tuple(f for f in _data.DEMO_FAMILIES if f != "TOYS")


# ===========================================================================
# Fixture: pay the one-time forecast cost once per pytest session
# ===========================================================================
@pytest.fixture(scope="module", autouse=True)
def warmed_data_layer() -> None:
    """Build the bundle and the canonical forecast before the page tests."""
    bundle = _data.get_data()
    _data.get_forecast(bundle.stores, bundle.families, config.FORECAST_HORIZON)
    _data.get_inventory_policy(bundle.stores, bundle.families, config.FORECAST_HORIZON)
    _data.get_transfers(bundle.stores, bundle.families, config.FORECAST_HORIZON)
    _data.get_deadstock(bundle.stores, bundle.families, config.FORECAST_HORIZON)


def _run(path: Path, **kwargs) -> AppTest:
    """Execute a page file exactly the way ``streamlit run`` would."""
    test = AppTest.from_file(str(path), default_timeout=APPTEST_TIMEOUT, **kwargs)
    test.run(timeout=APPTEST_TIMEOUT)
    assert not test.exception, f"{path.name} raised: {test.exception}"
    return test


# ===========================================================================
# 1. Sources compile (syntax errors / missing imports)
# ===========================================================================
@pytest.mark.parametrize("path", APP_FILES, ids=lambda p: p.name)
def test_app_source_compiles(path: Path) -> None:
    """Every app module compiles — no syntax errors anywhere under app/."""
    assert path.exists(), f"missing dashboard file: {path}"
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("path", APP_FILES, ids=lambda p: p.name)
def test_app_source_declares_an_entrypoint(path: Path) -> None:
    """Each page defines ``main()`` and invokes it (Streamlit runs the file)."""
    source = path.read_text(encoding="utf-8")
    if path.name.startswith(("_data", "_ui")):
        assert "def " in source
        return
    assert "def main()" in source
    assert 'if __name__ == "__main__":' in source


# ===========================================================================
# 2. Data layer — demo corpus
# ===========================================================================
def test_demo_sales_matches_the_clean_schema() -> None:
    sales = _data.build_demo_sales()

    assert list(sales.columns) == list(config.CLEAN_COLUMN_ORDER)
    assert set(config.GROUP_KEYS).issubset(sales.columns)
    assert str(sales[config.TARGET_COL].dtype) == "float32"
    assert str(sales["family"].dtype) == "category"
    assert str(sales["id"].dtype) == "int64"
    assert (sales[config.TARGET_COL] >= 0).all()
    assert sales[config.DATE_COL].is_monotonic_increasing is False  # sorted by group
    assert sales.groupby(config.GROUP_KEYS, observed=True).ngroups == (
        len(_data.DEMO_STORES) * len(_data.DEMO_FAMILIES)
    )


def test_demo_sales_is_deterministic() -> None:
    first = _data.build_demo_sales()
    second = _data.build_demo_sales()
    pd.testing.assert_frame_equal(first, second)


def test_demo_sales_encodes_real_signal() -> None:
    sales = _data.build_demo_sales()
    by_weekday = sales.groupby(sales[config.DATE_COL].dt.dayofweek)[config.TARGET_COL].mean()

    # Weekend peak (the shape of the real Favorita panel)...
    weekend = float(by_weekday.loc[[5, 6]].mean())
    midweek = float(by_weekday.loc[[1, 2]].mean())
    assert weekend > midweek * 1.1

    # ...promotional lift...
    promo = float(sales.loc[sales["onpromotion"] == 1, config.TARGET_COL].mean())
    quiet = float(sales.loc[sales["onpromotion"] == 0, config.TARGET_COL].mean())
    assert promo > quiet

    # ...and a stalled slow-mover that never sells again (deadstock shape).
    toys = sales[sales["family"] == "TOYS"]
    stall = _data.DEMO_END_DATE - pd.Timedelta(days=_data.DEMO_HISTORY_DAYS - 1 - 80)
    assert float(toys[toys[config.DATE_COL] <= stall][config.TARGET_COL].sum()) > 0
    assert float(toys[toys[config.DATE_COL] > stall][config.TARGET_COL].sum()) == 0.0


def test_demo_holidays_use_the_favorita_schema() -> None:
    holidays = _data.build_demo_holidays()
    assert {"date", "type", "locale", "locale_name", "description", "transferred"} <= set(
        holidays.columns
    )
    assert holidays["date"].is_monotonic_increasing
    assert set(holidays["locale"].astype(str)) == {"National"}
    assert not holidays["transferred"].any()


def test_demo_stock_creates_donors_and_receivers() -> None:
    """Every active family must be both over- and under-stocked somewhere."""
    stock = _data.build_demo_stock(_data.build_demo_sales())
    assert {"store_nbr", "family", "on_hand", "on_order", "unit_cost"} <= set(stock.columns)
    assert (stock["unit_cost"] > 0).all()
    assert (stock["on_hand"] >= 0).all()

    donor_cut = config.OVERSTOCK_DOS_THRESHOLD / config.TRANSFER_TARGET_DOS
    receiver_cut = config.UNDERSTOCK_DOS_THRESHOLD / config.TRANSFER_TARGET_DOS

    for family in _data.DEMO_FAMILIES[:-1]:
        covers = [_data.cover_multiplier(family, store) for store in _data.DEMO_STORES]
        assert max(covers) >= donor_cut, f"{family} has no donor store"
        assert min(covers) <= receiver_cut, f"{family} has no receiver store"

    # The stalled line is stocked everywhere: it is a clearance case, not a
    # rebalancing case.
    stalled = [_data.cover_multiplier("TOYS", store) for store in _data.DEMO_STORES]
    assert all(cover > 1.5 for cover in stalled)

    # A genuine zero-stock position must exist so the risk badges have a
    # "critical" case to render.
    zero = stock[stock["on_hand"] == 0]
    assert not zero.empty, "expected a genuine stock-out position"
    assert set(zero["family"].astype(str)) <= set(_data.DEMO_FAMILIES[:-1])


def test_policy_params_unpacks_into_the_engine_kwargs() -> None:
    params = _data.PolicyParams()
    flat = params.flat()

    for field in ("service_level", "lead_time_days", "min_transfer_units", "target_dos"):
        assert field in flat
    assert flat["service_level"] == config.SERVICE_LEVEL
    assert flat["target_dos"] == config.TRANSFER_TARGET_DOS

    policy = engine_transfers.TransferPolicy(
        min_transfer_units=flat["min_transfer_units"],
        max_transfer_units=flat["max_transfer_units"],
        target_dos=flat["target_dos"],
    )
    assert policy.target_dos == config.TRANSFER_TARGET_DOS

    dead = engine_deadstock.DeadstockPolicy(
        window_days=flat["window_days"],
        velocity_threshold=flat["velocity_threshold"],
        min_days_since_last_sale=flat["min_days_since_last_sale"],
        discount_base=flat["discount_base"],
        top_n_complements=flat["top_n_complements"],
    )
    assert dead.window_days == config.DEADSTOCK_WINDOW_DAYS


# ===========================================================================
# 3. Data layer — cached accessors
# ===========================================================================
def test_bundle_reports_demo_provenance() -> None:
    bundle = _data.get_data()
    assert bundle.is_demo is True
    assert bundle.stores == _data.DEMO_STORES
    assert bundle.families == _data.DEMO_FAMILIES
    assert "demo" in _data.model_source().lower()
    assert _data.scope_options() == (bundle.stores, bundle.families)
    assert _data.forecast_hygiene([1], ["DAIRY"]) is True
    assert _data.forecast_hygiene([], ["DAIRY"]) is False
    assert _data.forecast_hygiene([1], []) is False


def test_dashboard_suite_is_hermetic() -> None:
    """
    Guard: the suite must run against the synthetic corpus.

    Failing here beats failing obscurely 20 minutes later, which is what
    happens if the real 20-store artifacts get loaded (see the module
    docstring). ``tests/conftest.py`` sets the pin; if it is ever dropped,
    this test says so plainly.
    """
    assert _data.is_demo() is True, (
        "dashboard tests require SDP_FORCE_DEMO=1 (set by tests/conftest.py); "
        "otherwise they load the real artifacts and assert demo numbers"
    )


def test_empty_scope_returns_typed_empty_frames() -> None:
    forecast = _data.get_forecast((), (), config.FORECAST_HORIZON)
    history = _data.get_history((), (), 30)
    policy = _data.get_inventory_policy((), (), config.FORECAST_HORIZON)
    plan = _data.get_transfers((), (), config.FORECAST_HORIZON)

    for frame in (forecast, history, policy, plan):
        assert isinstance(frame, pd.DataFrame) and frame.empty

    assert {"prediction", "step", "sigma_1", "lower", "upper"} <= set(forecast.columns)
    assert set(engine_transfers.TRANSFER_COLUMNS) <= set(plan.columns)


def test_forecast_carries_a_non_negative_band() -> None:
    bundle = _data.get_data()
    forecast = _data.get_forecast(bundle.stores, bundle.families, config.FORECAST_HORIZON)

    assert len(forecast) == len(bundle.stores) * len(bundle.families) * config.FORECAST_HORIZON
    assert forecast["prediction"].notna().all()
    assert (forecast["prediction"] >= 0).all()
    assert (forecast["lower"] >= 0).all()
    assert (forecast["upper"] >= forecast["lower"]).all()
    widths = forecast["upper"] - forecast["lower"]
    assert (widths > 0).any(), "no series produced an uncertainty band"
    # The stalled family has no day-to-day volatility left, so its band
    # collapses onto the point estimate rather than faking width.
    assert (widths == 0).any()
    assert forecast.groupby(config.GROUP_KEYS, observed=True)["step"].max().eq(
        config.FORECAST_HORIZON
    ).all()


def test_shorter_horizons_are_prefixes_of_the_longest_forecast() -> None:
    bundle = _data.get_data()
    long = _data.get_forecast(bundle.stores, bundle.families, 30)
    for horizon in config.FORECAST_HORIZONS:
        short = _data.get_forecast(bundle.stores, bundle.families, horizon)
        expected = long.groupby(config.GROUP_KEYS, observed=True, group_keys=False).head(horizon)
        pd.testing.assert_frame_equal(
            short.reset_index(drop=True),
            expected.reset_index(drop=True)[short.columns],
            check_dtype=False,
        )


def test_promotion_scenario_changes_the_forecast() -> None:
    bundle = _data.get_data()
    scope = (bundle.stores, bundle.families)
    base = _data.get_forecast(*scope, 7, future_onpromotion=None)
    promo = _data.get_forecast(*scope, 7, future_onpromotion=1)
    assert float(promo["prediction"].sum()) > float(base["prediction"].sum())


def test_policy_transfers_and_deadstock_chain_together() -> None:
    bundle = _data.get_data()
    scope = (bundle.stores, bundle.families)

    policy = _data.get_inventory_policy(*scope, config.FORECAST_HORIZON)
    assert len(policy) == len(bundle.stores) * len(bundle.families)
    assert set(policy.columns) >= set(
        ["safety_stock", "reorder_point", "on_hand", "stockout_risk", "stockout_risk_level"]
    )
    assert policy["stockout_risk"].dtype == bool
    assert set(policy["stockout_risk_level"]) <= {"critical", "high", "medium", "none"}
    # The demo corpus must exercise every outcome the dashboard renders.
    assert policy["stockout_risk"].any()
    assert policy["overstock_risk"].any()

    plan = _data.get_transfers(*scope, config.FORECAST_HORIZON)
    assert not plan.empty, "demo corpus should produce a rebalancing plan"
    assert (plan["quantity"] >= config.MIN_TRANSFER_UNITS).all()
    assert (plan["from_store"] != plan["to_store"]).all()
    assert (plan["estimated_net_value"] >= config.MIN_TRANSFER_NET_VALUE).all()
    assert set(plan["family"]) <= set(bundle.families)
    # Each leg must come from a donor and go to a receiver of the same item.
    merged = plan.merge(
        policy[["store_nbr", "family", "overstock_risk"]].rename(
            columns={"store_nbr": "from_store", "overstock_risk": "donor_surplus"}
        ),
        on=["from_store", "family"],
        how="left",
    )
    assert merged["donor_surplus"].fillna(False).all()

    deadstock = _data.get_deadstock(*scope, config.FORECAST_HORIZON)
    assert set(deadstock) == {"detection", "affinity", "heroes", "bundles", "summary", "policy"}
    assert set(engine_deadstock.DEADSTOCK_COLUMNS) <= set(deadstock["detection"].columns)
    assert set(engine_deadstock.BUNDLE_COLUMNS) <= set(deadstock["bundles"].columns)
    summary = deadstock["summary"]
    for key in ("n_skus", "deadstock_value", "n_bundles", "potential_holding_savings"):
        assert key in summary
    assert summary["n_deadstock_skus"] > 0
    assert summary["deadstock_value"] > 0
    assert not deadstock["bundles"].empty
    assert set(deadstock["bundles"]["selection_basis"]) <= {"affinity", "best_seller_fallback"}


def test_scoped_requests_only_return_requested_series() -> None:
    bundle = _data.get_data()
    scope = ((bundle.stores[0],), (_ACTIVE_FAMILIES[0],))
    forecast = _data.get_forecast(*scope, 7)
    policy = _data.get_inventory_policy(*scope, 7)

    assert set(forecast["store_nbr"]) == {bundle.stores[0]}
    assert set(forecast["family"].astype(str)) == {_ACTIVE_FAMILIES[0]}
    assert len(forecast) == 7
    assert len(policy) == 1


# ===========================================================================
# 4. UI kit
# ===========================================================================
def test_ui_formatters_are_total() -> None:
    # NB: Python rounds .5 to even, so avoid half-ties in the expectations.
    assert _ui.money(1234.56) == "$1,235"
    assert _ui.money(1234.5, 2) == "$1,234.50"
    assert _ui.money(None) == "$0"
    assert _ui.units(1234.56) == "1,235"
    assert _ui.units(1234.5, 2) == "1,234.50"
    assert _ui.units(float("inf")) == "∞"
    assert _ui.units(None) == "—"
    assert _ui.pct(0.95, 0) == "95%"
    assert _ui.pct(0.951) == "95.1%"
    assert _ui.pct(None) == "—"


def test_ui_badges_use_the_expected_css_classes() -> None:
    for kind in ("critical", "high", "medium", "low", "none", "info", "affinity", "fallback"):
        assert f'badge {kind}' in _ui.badge("X", kind)


def test_plotly_config_hides_the_streamlit_logo() -> None:
    assert _ui.PLOTLY_CONFIG["displaylogo"] is False


# ===========================================================================
# 5. Pages render (AppTest)
# ===========================================================================
def test_landing_page_renders_kpis_and_charts() -> None:
    test = _run(MAIN_PAGE)

    labels = [metric.label for metric in test.metric]
    for expected in (
        "Total expected demand (30d)",
        "Items at stockout risk",
        "Estimated transfer savings",
        "Deadstock tied capital",
    ):
        assert any(expected in label for label in labels), f"missing KPI: {expected} / {labels}"

    assert len(test.metric) >= 4
    assert len(test.get("plotly_chart")) >= 3
    assert len(test.dataframe) >= 1


def test_landing_page_responds_to_scope_filters() -> None:
    test = AppTest.from_file(str(MAIN_PAGE), default_timeout=APPTEST_TIMEOUT)
    test.run(timeout=APPTEST_TIMEOUT)
    assert not test.exception

    # Empty scope renders the empty state instead of KPI cards.
    test.sidebar.multiselect[0].set_value([]).run(timeout=APPTEST_TIMEOUT)
    assert not test.exception
    assert len(test.metric) == 0


def test_demand_forecasting_page_renders_and_toggles_horizon() -> None:
    test = _run(PAGE_1)

    assert len(test.get("plotly_chart")) >= 2
    assert len(test.dataframe) >= 1
    assert len(test.sidebar.radio) == 1
    assert test.sidebar.radio[0].options == ["7 days", "14 days", "30 days"]
    assert test.sidebar.radio[0].value == "30 days"

    # Toggle the horizon and the promotion scenario: the page must re-render
    # cleanly both times.
    test.sidebar.radio[0].set_value("7 days")
    test.sidebar.selectbox[0].set_value("Promote every item")
    test.run(timeout=APPTEST_TIMEOUT)
    assert not test.exception
    assert test.sidebar.radio[0].value == "7 days"
    assert len(test.dataframe) >= 1


def test_inventory_page_applies_the_service_level_slider() -> None:
    test = _run(PAGE_2)

    labels = [metric.label for metric in test.metric]
    assert any("Service level" in label for label in labels)
    sliders = [slider for slider in test.sidebar.slider if slider.label == "Service level"]
    assert len(sliders) == 1
    assert (sliders[0].min, sliders[0].max) == (0.90, 0.99)

    baseline = [metric for metric in test.metric if "Safety-stock investment" in metric.label][0]
    baseline_value = float(baseline.value.replace("$", "").replace(",", ""))

    sliders[0].set_value(0.99)
    test.run(timeout=APPTEST_TIMEOUT)
    assert not test.exception

    stricter = [
        metric for metric in test.metric if "Safety-stock investment" in metric.label
    ][0]
    stricter_value = float(stricter.value.replace("$", "").replace(",", ""))
    assert stricter_value > baseline_value, "raising the service level must raise the buffer"
    assert len(test.dataframe) >= 1, "policy table did not render"


def test_transfers_page_approval_updates_session_state() -> None:
    test = _run(PAGE_3)

    assert len(test.dataframe) >= 1
    checkboxes = list(test.checkbox)
    assert len(checkboxes) >= 1, "no transfer legs to approve"

    approvals = test.session_state["approved_transfers"]
    assert approvals and not any(approvals.values())

    first_leg = list(approvals)[0]

    checkboxes[0].set_value(True)
    test.run(timeout=APPTEST_TIMEOUT)
    assert not test.exception

    updated = test.session_state["approved_transfers"]
    assert updated[first_leg] is True
    assert sum(updated.values()) == 1

    # Cumulative savings KPI must now reflect the approved leg.
    cumulative = [
        metric for metric in test.metric if "Cumulative net savings" in metric.label
    ]
    assert cumulative, "cumulative savings metric missing"
    assert cumulative[0].value.startswith("$")

    # Reset clears the approval again.
    buttons = [b for b in test.button if b.label == "Reset"]
    assert len(buttons) == 1
    buttons[0].set_value(True)
    test.run(timeout=APPTEST_TIMEOUT)
    assert not test.exception
    assert not any(test.session_state["approved_transfers"].values())


def test_transfers_page_survives_policy_changes() -> None:
    test = _run(PAGE_3)
    sliders = [s for s in test.sidebar.slider if s.label == "Target days of supply"]
    assert len(sliders) == 1

    signature_before = test.session_state["transfer::plan_signature"]
    sliders[0].set_value(15)
    test.run(timeout=APPTEST_TIMEOUT)
    assert not test.exception
    assert test.session_state["transfer::plan_signature"] != signature_before


def test_deadstock_page_renders_bundle_cards() -> None:
    test = _run(PAGE_4)

    assert len(test.metric) >= 4
    assert len(test.dataframe) >= 1
    markup = "".join(str(element.value) for element in test.markdown)
    assert "bundle-card" in markup, "no clearance bundle cards rendered"
    assert "recommended discount" in markup
    assert "affinity" in markup or "fallback" in markup


def test_deadstock_page_detects_the_stalled_line() -> None:
    test = _run(PAGE_4)
    markup = "".join(str(element.value) for element in test.markdown)
    assert _data.DEMO_FAMILIES[-1] in markup, "stalled family missing from the clearance plan"


def test_every_page_shows_the_data_source_banner() -> None:
    for path in (MAIN_PAGE, PAGE_1, PAGE_2, PAGE_3, PAGE_4):
        test = _run(path)
        markup = "".join(str(element.value) for element in test.markdown)
        assert "Demo mode" in markup or "Production data" in markup, (
            f"{path.name} does not disclose its data source"
        )


def test_pages_render_with_a_narrow_scope() -> None:
    """A single store / single family must not crash any page."""
    for path in (MAIN_PAGE, PAGE_1, PAGE_2, PAGE_3, PAGE_4):
        test = AppTest.from_file(str(path), default_timeout=APPTEST_TIMEOUT)
        test.run(timeout=APPTEST_TIMEOUT)
        assert not test.exception
        test.sidebar.multiselect[0].set_value([_data.DEMO_STORES[0]])
        test.sidebar.multiselect[1].set_value([_ACTIVE_FAMILIES[0]])
        test.run(timeout=APPTEST_TIMEOUT)
        assert not test.exception, f"{path.name} failed on a narrow scope"
