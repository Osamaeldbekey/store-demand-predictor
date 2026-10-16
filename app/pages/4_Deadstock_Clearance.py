"""
Page 4 — Deadstock clearance.

Slow-mover detection, severity ranking and the clearance plan: each deadstock
item paired with a hero product (by demand affinity where the data supports
it, best-seller rank otherwise) with a recommended discount and the holding
cost it frees up.
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (parent for parent in APP_DIR.parents if (parent / "src" / "config.py").exists()),
    APP_DIR.parent,
)
for _path in (str(PROJECT_ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
import _data
import _ui

_SEVERITY_COLOUR = {
    "critical": "#dc2626",
    "high": "#ea580c",
    "medium": "#ca8a04",
    "low": "#16a34a",
}
_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

_BASIS_LABEL = {
    "affinity": ("Demand affinity", "affinity"),
    "best_seller_fallback": ("Best-seller fallback", "fallback"),
}


def _bundle_card(row: pd.Series) -> None:
    """One clearance recommendation: deadstock SKU + hero product + price."""
    severity = str(row["severity"])
    accent = _SEVERITY_COLOUR.get(severity, "#2563eb")
    basis_text, basis_kind = _BASIS_LABEL.get(
        str(row["selection_basis"]), ("Best-seller fallback", "fallback")
    )
    affinity = float(row["affinity"]) if pd.notna(row["affinity"]) else 0.0
    discount = float(row["suggested_discount_pct"])
    threshold = "∞"

    markup = f"""
    <div class="bundle-card {severity}">
      <div class="card-title">Store {int(row['store_nbr'])} · {row['deadstock_family']}</div>
      <div class="card-sub">
        {row['deadstock_units']:,.0f} units on hand ·
        {row['days_since_last_sale']} days since last sale ·
        {_ui.badge(severity, severity)}
        {_ui.badge(basis_text, basis_kind)}
      </div>
      <div class="card-pair">
        Bundle with <b>{row['complement_family']}</b>
        <span style="color:#64748b;font-size:0.85rem;">
          ({row['complement_units_sold']:,.0f} units sold in the window,
          affinity {affinity:+.2f})
        </span>
      </div>
      <div class="card-stats">
        <b>{_ui.pct(discount, 0)}</b> recommended discount
        &nbsp;·&nbsp; unit price <b>{_ui.money(row['discounted_unit_cost'], 2)}</b>
        (from {_ui.money(row['deadstock_value'] / max(row['deadstock_units'], 1e-9), 2)})
        &nbsp;·&nbsp; recovery <b>{_ui.money(row['recovery_at_discount'])}</b><br>
        <b>{_ui.money(row['holding_cost_savings'])}</b> holding cost avoided
        &nbsp;·&nbsp; tied capital {_ui.money(row['deadstock_value'])}
        <br>
        <span style="color:#64748b;font-size:0.84rem;">
          {row['discount_rationale']}
        </span>
      </div>
    </div>
    """
    st.markdown(markup, unsafe_allow_html=True)


def main() -> None:
    _ui.bootstrap(
        title="Deadstock Clearance — Store Demand Predictor",
        icon="🧹",
        eyebrow="Phase 4 · clearance engine",
        subtitle=(
            "Find the stock that is not moving, work out how much capital it is "
            "tying up, and pair it with a hero product so it actually sells."
        ),
    )

    # ---------- sidebar ----------------------------------------------------
    stores, families = _ui.sidebar_scope()
    horizon = _ui.horizon_radio(key="horizon_deadstock")

    st.sidebar.markdown("### Detection")
    window_days = st.sidebar.slider(
        "Velocity lookback (days)", 14, 120, int(config.DEADSTOCK_WINDOW_DAYS), step=7,
    )
    velocity_threshold = st.sidebar.number_input(
        "Max velocity (units/day)",
        min_value=0.0,
        max_value=50.0,
        value=float(config.DEADSTOCK_VELOCITY_THRESHOLD),
        step=0.1,
        help="At or below this average daily sell-through an item counts as "
             "slow-moving. It must ALSO have gone this many days without a sale.",
    )
    min_days = st.sidebar.number_input(
        "Min days since last sale",
        min_value=0,
        max_value=730,
        value=int(config.DEADSTOCK_MIN_DAYS_SINCE_LAST_SALE),
        step=5,
    )
    discount_base = st.sidebar.slider(
        "Base clearance discount",
        0.0, 1.0, config.DEADSTOCK_DISCOUNT_BASE, step=0.05, format="%.0f%%",
        help="Increased by 5 points for every full 30 days since the last sale, "
             "clipped to 10–70%.",
    )
    top_n = st.sidebar.slider(
        "Complements per item", 1, 5, int(config.BUNDLE_TOP_N_COMPLEMENTS)
    )

    _ui.data_banner()

    if not _data.forecast_hygiene(stores, families):
        _ui.empty_state("No stores or families selected.", hint="Choose some in the sidebar.")
        _ui.footer()
        return

    with st.spinner("Scanning for slow movers…"):
        bundle = _data.get_deadstock(
            stores,
            families,
            horizon,
            window_days=window_days,
            velocity_threshold=velocity_threshold,
            min_days_since_last_sale=int(min_days),
            discount_base=discount_base,
            top_n_complements=int(top_n),
        )

    detection: pd.DataFrame = bundle["detection"]
    plans: pd.DataFrame = bundle["bundles"]
    summary: dict = bundle["summary"]
    flagged = (
        detection[detection["is_deadstock"].astype(bool)].copy()
        if not detection.empty
        else detection
    )

    # ---------- KPIs -------------------------------------------------------
    _ui.kpi_row(
        [
            {
                "label": "Deadstock tied capital",
                "value": _ui.money(summary["deadstock_value"]),
                "delta": f"{_ui.units(summary['deadstock_units'])} units across {summary['n_deadstock_skus']} SKUs",
            },
            {
                "label": "Holding cost avoided",
                "value": _ui.money(summary["potential_holding_savings"]),
                "delta": f"if cleared within {config.DEADSTOCK_CLEARANCE_WINDOW_DAYS} days",
                "help": "Capital, space, insurance and obsolescence at "
                        f"{_ui.pct(config.DEADSTOCK_HOLDING_COST_RATE, 0)} of value per year.",
            },
            {
                "label": "Clearance bundles",
                "value": _ui.units(summary["n_bundles"]),
                "delta": f"mean discount {_ui.pct(summary['mean_discount_pct'], 0)}",
            },
            {
                "label": "Recovery at discount",
                "value": _ui.money(summary["recovery_at_discount"]),
                "delta": "expected revenue from the clearance plan",
            },
        ]
    )

    st.markdown("")

    if not flagged.empty:
        left, right = st.columns([1.3, 1])
        with left:
            chart = flagged.copy()
            chart["label"] = (
                "S" + chart["store_nbr"].astype(str) + " · " + chart["family"].astype(str)
            )
            fig = px.bar(
                chart.sort_values("inventory_value", ascending=True),
                x="inventory_value",
                y="label",
                orientation="h",
                color="severity",
                color_discrete_map=_SEVERITY_COLOUR,
                labels={"inventory_value": "Tied capital ($)", "label": "", "severity": "Severity"},
                text_auto="$,.0f",
            )
            fig.update_layout(
                height=max(240, 34 * len(chart) + 80),
                margin=dict(l=8, r=8, t=30, b=8),
                legend=dict(orientation="h", y=1.12, x=0),
                yaxis=dict(categoryorder="total ascending"),
            )
            st.plotly_chart(fig, use_container_width=True, config=_ui.PLOTLY_CONFIG)
        with right:
            if detection.empty:
                _ui.empty_state("No SKUs in scope.")
            else:
                counts = (
                    detection.assign(
                        status=detection["is_deadstock"].map(
                            {True: "Deadstock", False: "Moving"}
                        )
                    )
                    .groupby(["severity", "status"], observed=True)
                    .size()
                    .rename("SKUs")
                    .reset_index()
                )
                fig = px.bar(
                    counts,
                    x="severity",
                    y="SKUs",
                    color="status",
                    barmode="group",
                    labels={"severity": "Severity band", "status": ""},
                    category_orders={
                        "severity": ["critical", "high", "medium", "low"],
                    },
                    color_discrete_map={"Deadstock": "#dc2626", "Moving": "#94a3b8"},
                )
                fig.update_layout(height=300, margin=dict(l=8, r=8, t=30, b=8),
                                  legend=dict(orientation="h", y=1.14, x=0),
                                  yaxis=dict(dtick=1))
                st.plotly_chart(fig, use_container_width=True, config=_ui.PLOTLY_CONFIG)

    # ---------- bundle cards ----------------------------------------------
    st.markdown("#### Recommended clearance bundles")
    if plans.empty:
        _ui.empty_state(
            "No deadstock items to bundle.",
            hint="Loosen the velocity threshold or the days-since-sale floor "
                 "in the sidebar.",
        )
    else:
        _ui.legend(
            [
                ("Critical", _SEVERITY_COLOUR["critical"]),
                ("High", _SEVERITY_COLOUR["high"]),
                ("Medium", _SEVERITY_COLOUR["medium"]),
                ("Low", _SEVERITY_COLOUR["low"]),
            ]
        )
        st.markdown("")
        for _, row in plans.iterrows():
            _bundle_card(row)

    _ui.notice(
        "Favorita ships <b>no basket data</b>, so complementarity is estimated "
        "from the correlation of daily family demand <b>after removing the "
        "shared daily factor</b> (raw correlation would just rediscover shared "
        "seasonality). Each bundle records how it was chosen: "
        "<b>affinity</b> when the co-movement supports it, "
        "<b>best-seller fallback</b> when it does not. Supply a real "
        "co-occurrence matrix to <code>generate_bundles()</code> when baskets "
        "become available — the algorithm does not care where it comes from.",
        kind="info",
        icon="ℹ️",
    )

    # ---------- detection table -------------------------------------------
    with st.expander("Full SKU table", expanded=False):
        if detection.empty:
            _ui.empty_state("No SKUs in scope.")
        else:
            table = detection[
                [
                    "store_nbr", "family", "avg_daily_velocity",
                    "days_since_last_sale", "on_hand", "unit_cost",
                    "inventory_value", "severity", "is_deadstock",
                    "baseline_days_to_clear", "holding_cost_savings",
                ]
            ].rename(
                columns={
                    "store_nbr": "Store",
                    "family": "Family",
                    "avg_daily_velocity": "Velocity/day",
                    "days_since_last_sale": "Days since sale",
                    "on_hand": "On hand",
                    "unit_cost": "Unit cost",
                    "inventory_value": "Tied capital",
                    "severity": "Severity",
                    "is_deadstock": "Deadstock?",
                    "baseline_days_to_clear": "Days to clear (baseline)",
                    "holding_cost_savings": "Holding saved",
                }
            )
            table = table.sort_values(
                ["Deadstock?", "Tied capital"],
                ascending=[False, False],
                kind="stable",
            )
            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Velocity/day": st.column_config.NumberColumn(format="%.3f"),
                    "Days since sale": st.column_config.NumberColumn(format="%d"),
                    "On hand": st.column_config.NumberColumn(format="%.1f"),
                    "Unit cost": st.column_config.NumberColumn(format="$,.2f"),
                    "Tied capital": st.column_config.NumberColumn(format="$,.2f"),
                    "Days to clear (baseline)": st.column_config.NumberColumn(format="%.0f"),
                    "Holding saved": st.column_config.NumberColumn(format="$,.2f"),
                    "Deadstock?": st.column_config.CheckboxColumn(),
                },
            )

    _ui.footer()


if __name__ == "__main__":
    main()
