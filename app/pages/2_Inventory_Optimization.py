"""
Page 2 — Inventory optimization.

Turns the forecast into a per-SKU replenishment policy: safety stock, reorder
point, order-up-to level and an automated stockout-risk verdict. The service
level slider re-derives every number live (higher service level → higher Z →
more safety stock).
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

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
import _data
import _ui

_RISK_FILL = {
    "critical": "#ffe4e6",
    "high": "#ffedd5",
    "medium": "#fef9c3",
    "none": "#ffffff",
}
_RISK_ORDER = {"critical": 0, "high": 1, "medium": 2, "none": 3}

_DISPLAY_COLUMNS = {
    "store_nbr": "Store",
    "family": "Family",
    "avg_daily_demand": "Demand/day",
    "service_level": "Service",
    "z_score": "Z",
    "safety_stock": "Safety stock",
    "reorder_point": "ROP",
    "order_up_to_level": "Order-up-to",
    "on_hand": "On hand",
    "days_of_supply": "DOS",
    "reorder_quantity": "Order qty",
    "reorder_value": "Order $",
    "stockout_risk_level": "Risk",
}
_NUMERIC_FORMAT = {
    "Demand/day": "%.1f",
    "Z": "%.3f",
    "Safety stock": "%.1f",
    "ROP": "%.1f",
    "Order-up-to": "%.1f",
    "On hand": "%.1f",
    "DOS": "%.1f",
    "Order qty": "%.1f",
    "Order $": "$,.0f",          # d3 format: thousands-separated dollars
}
# Columns rendered as text rather than numbers.
_TEXT_DISPLAY = ("Store", "Family", "Service", "Risk")


def _paint_rows(display: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Row colours driven by the automated risk verdict."""

    def paint(row: pd.Series) -> list[str]:
        fill = _RISK_FILL.get(str(row["Risk"]), "#ffffff")
        return [f"background-color: {fill}"] * len(row)

    return display.style.apply(paint, axis=1)


def main() -> None:
    _ui.bootstrap(
        title="Inventory Optimization — Store Demand Predictor",
        icon="📦",
        eyebrow="Phase 4 · safety stock engine",
        subtitle=(
            "Set the service level you want to protect and the engine sizes "
            "safety stock and reorder points for every store / family pair."
        ),
    )

    # ---------- sidebar ----------------------------------------------------
    stores, families = _ui.sidebar_scope()
    horizon = _ui.horizon_radio(key="horizon_inventory")

    st.sidebar.markdown("### Policy")
    service_level = st.sidebar.slider(
        "Service level",
        min_value=0.90,
        max_value=0.99,
        value=config.SERVICE_LEVEL,
        step=0.01,
        format="%.0f%%",
        help="Probability of not stocking out during the protection period "
             "(lead time + review period). Higher → more safety stock.",
    )
    lead_time_days = st.sidebar.slider(
        "Lead time (days)", 1, 45, int(config.LEAD_TIME_DAYS), step=1,
        help="Days between placing a replenishment order and receiving it.",
    )
    review_period_days = st.sidebar.slider(
        "Review period (days)", 0, 30, int(config.REVIEW_PERIOD_DAYS), step=1,
        help="How often stock is reviewed. 0 = continuous review.",
    )
    lead_time_std_days = st.sidebar.number_input(
        "Lead-time variability (σ days)",
        min_value=0.0,
        max_value=30.0,
        value=float(config.LEAD_TIME_STD_DAYS),
        step=0.5,
        help="0 assumes a constant lead time; a positive value widens safety "
             "stock to cover delivery-date uncertainty.",
    )

    _ui.data_banner()

    if not _data.forecast_hygiene(stores, families):
        _ui.empty_state("No stores or families selected.", hint="Choose some in the sidebar.")
        _ui.footer()
        return

    with st.spinner("Sizing safety stock across the portfolio…"):
        policy = _data.get_inventory_policy(
            stores,
            families,
            horizon,
            service_level=service_level,
            lead_time_days=lead_time_days,
            review_period_days=review_period_days,
            lead_time_std_days=lead_time_std_days,
        )

    if policy.empty:
        _ui.empty_state("No policy rows for this scope.")
        _ui.footer()
        return

    # ---------- KPIs -------------------------------------------------------
    risk = policy["stockout_risk"].astype(bool)
    critical = policy["stockout_risk_level"] == "critical"
    safety_value = float((policy["safety_stock"] * policy["unit_cost"]).sum())
    reorder_value = float(policy["reorder_value"].sum())
    z_value = float(policy["z_score"].iloc[0])

    _ui.kpi_row(
        [
            {
                "label": "Service level",
                "value": _ui.pct(service_level, 0),
                "delta": f"Z = {z_value:.3f}",
                "help": "Standard-normal quantile protecting the protection period.",
            },
            {
                "label": "Safety-stock investment",
                "value": _ui.money(safety_value),
                "delta": f"{_ui.units(policy['safety_stock'].sum())} units held as buffer",
            },
            {
                "label": "Items at stockout risk",
                "value": _ui.units(int(risk.sum())),
                "delta": f"{int(critical.sum())} critical · of {len(policy)} SKUs",
            },
            {
                "label": "Replenishment value",
                "value": _ui.money(reorder_value),
                "delta": f"{int(policy['needs_reorder'].sum())} SKUs below ROP",
            },
        ]
    )

    st.markdown("")

    # ---------- alerts -----------------------------------------------------
    flagged = policy[risk].copy()
    if flagged.empty:
        _ui.notice(
            "Every selected position sits above its reorder point — no "
            "replenishment is required at this service level.",
            kind="ok",
            icon="✅",
        )
    else:
        ordered = flagged.sort_values(
            ["stockout_risk_level", "days_of_supply"],
            key=lambda column: (
                column.map(_RISK_ORDER) if column.name == "stockout_risk_level" else column
            ),
            ascending=[True, True],
            kind="stable",
        ).head(6)

        for row in ordered.itertuples(index=False):
            level = str(row.stockout_risk_level)
            headline = (
                f"**Store {row.store_nbr} · {row.family}** — {level} risk: "
                f"{row.on_hand:,.1f} on hand vs. ROP {row.reorder_point:,.1f} "
                f"({row.days_of_supply:,.1f} days of cover, "
                f"order {row.reorder_quantity:,.0f} units ≈ "
                f"{_ui.money(row.reorder_value)})"
            )
            if level == "critical":
                st.error(headline)
            elif level == "high":
                st.warning(headline)
            else:
                st.info(headline)

    # ---------- policy table ----------------------------------------------
    st.markdown("#### Replenishment policy")
    display = policy[list(_DISPLAY_COLUMNS)].rename(columns=_DISPLAY_COLUMNS)
    display["Service"] = display["Service"].map(lambda v: _ui.pct(v, 0))
    display = display.sort_values(
        ["Risk", "DOS"],
        key=lambda column: (
            column.map(_RISK_ORDER) if column.name == "Risk" else column
        ),
        ascending=[True, True],
        kind="stable",
    ).reset_index(drop=True)

    st.dataframe(
        _paint_rows(display),
        use_container_width=True,
        hide_index=True,
        height=min(46 + 35 * len(display), 470),
        column_config={
            name: st.column_config.NumberColumn(name, format=fmt)
            for name, fmt in _NUMERIC_FORMAT.items()
            if name in display.columns
        },
    )
    st.caption(
        "Row colour = automated stockout-risk verdict (red critical · orange "
        "high · yellow medium). Safety stock uses "
        "Z·√(P·σ_d² + d̄²·σ_L²) over the protection period "
        "P = lead time + review period."
    )

    # ---------- trade-off chart -------------------------------------------
    with st.expander("Service-level trade-off", expanded=False):
        levels = [round(float(v), 2) for v in np.linspace(0.90, 0.99, 10)]
        rows = []
        for level in levels:
            table = _data.get_inventory_policy(
                stores,
                families,
                horizon,
                service_level=level,
                lead_time_days=lead_time_days,
                review_period_days=review_period_days,
                lead_time_std_days=lead_time_std_days,
            )
            rows.append(
                {
                    "Service level": level,
                    "Safety-stock $": float((table["safety_stock"] * table["unit_cost"]).sum()),
                    "Items at risk": int(table["stockout_risk"].sum()),
                }
            )
        trade = pd.DataFrame(rows)
        left, right = st.columns(2)
        with left:
            fig = px.line(
                trade,
                x="Service level",
                y="Safety-stock $",
                markers=True,
                labels={"Safety-stock $": "Safety-stock investment ($)"},
            )
            fig.add_vline(x=service_level, line_dash="dash", line_color="#dc2626")
            fig.update_layout(height=300, margin=dict(l=8, r=8, t=30, b=8))
            st.plotly_chart(fig, use_container_width=True, config=_ui.PLOTLY_CONFIG)
        with right:
            fig = px.line(
                trade,
                x="Service level",
                y="Items at risk",
                markers=True,
                color_discrete_sequence=["#dc2626"],
                labels={"Items at risk": "SKUs below ROP"},
            )
            fig.add_vline(x=service_level, line_dash="dash", line_color="#2563eb")
            fig.update_layout(height=300, margin=dict(l=8, r=8, t=30, b=8),
                              yaxis=dict(dtick=1))
            st.plotly_chart(fig, use_container_width=True, config=_ui.PLOTLY_CONFIG)
        st.caption(
            "The classic inventory trade-off: buying a higher service level "
            "costs safety-stock capital and reduces exposure."
        )

    _ui.footer()


if __name__ == "__main__":
    main()
