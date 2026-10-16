"""
Executive overview — the dashboard's landing page.

Shows the four headline KPIs (demand, stockout exposure, transfer savings,
deadstock capital) plus aggregate multi-store demand charts. Everything comes
from one cached :func:`executive_overview` call, so the page paints in a single
pass after the first run.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
import plotly.graph_objects as go
import streamlit as st

from src import config
import _data
import _ui

_CHART_COLOURS = ["#2563eb", "#0ea5e9", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2"]


# ===========================================================================
# Cached executive roll-up
# ===========================================================================
@st.cache_data(show_spinner=True, max_entries=4)
def executive_overview(
    stores: tuple[int, ...],
    families: tuple[str, ...],
    horizon: int,
) -> dict[str, Any]:
    """
    One cached pass over forecast + inventory + transfers + deadstock.

    Pages share :mod:`_data`'s caches, so opening a detail page afterwards
    re-uses the work done here instead of recomputing it.
    """
    if not stores or not families:
        return {"empty": True}

    forecast = _data.get_forecast(stores, families, horizon)
    history = _data.get_history(stores, families, days=max(horizon * 2, 60))
    policy = _data.get_inventory_policy(stores, families, horizon)
    plan = _data.get_transfers(stores, families, horizon)
    deadstock = _data.get_deadstock(stores, families, horizon)

    detection: pd.DataFrame = deadstock["detection"]
    flagged = (
        detection[detection["is_deadstock"].astype(bool)]
        if not detection.empty
        else detection
    )

    actual_recent = (
        float(history[config.TARGET_COL].sum()) if not history.empty else 0.0
    )
    # Comparable window: same number of observed days as the forecast window.
    comparable = (
        float(history[config.TARGET_COL].tail(max(horizon, 1)).sum())
        if not history.empty
        else 0.0
    )
    forecast_total = float(forecast["prediction"].sum()) if not forecast.empty else 0.0
    growth = (
        (forecast_total - comparable) / comparable * 100.0 if comparable > 0 else 0.0
    )

    daily_history = (
        history.groupby(config.DATE_COL, sort=True)[config.TARGET_COL]
        .sum()
        .rename("actual")
        .reset_index()
        if not history.empty
        else pd.DataFrame(columns=[config.DATE_COL, "actual"])
    )
    daily_forecast = (
        forecast.groupby(config.DATE_COL, sort=True)["prediction"]
        .sum()
        .rename("forecast")
        .reset_index()
        if not forecast.empty
        else pd.DataFrame(columns=[config.DATE_COL, "forecast"])
    )

    by_store = (
        forecast.groupby("store_nbr", sort=True)["prediction"]
        .sum()
        .rename("demand")
        .reset_index()
        if not forecast.empty
        else pd.DataFrame(columns=["store_nbr", "demand"])
    )
    by_family = (
        forecast.groupby("family", sort=True)["prediction"]
        .sum()
        .rename("demand")
        .reset_index()
        if not forecast.empty
        else pd.DataFrame(columns=["family", "demand"])
    )

    risk_order = {"critical": 0, "high": 1, "medium": 2, "none": 3}
    top_risks = (
        policy.assign(_rank=policy["stockout_risk_level"].map(risk_order).fillna(4))
        .sort_values(["_rank", "days_of_supply"], kind="stable")
        .head(10)
        if not policy.empty
        else policy
    )

    return {
        "empty": False,
        "forecast_total": forecast_total,
        "actual_recent": actual_recent,
        "growth_pct": growth,
        "n_series": int(len(policy)) if not policy.empty else 0,
        "n_stockout_risk": int(policy["stockout_risk"].sum()) if not policy.empty else 0,
        "n_critical": int((policy["stockout_risk_level"] == "critical").sum())
        if not policy.empty
        else 0,
        "n_overstock": int(policy["overstock_risk"].sum()) if not policy.empty else 0,
        "reorder_value": float(policy["reorder_value"].sum()) if not policy.empty else 0.0,
        "transfer_net_value": float(plan["estimated_net_value"].sum())
        if not plan.empty
        else 0.0,
        "transfer_units": float(plan["quantity"].sum()) if not plan.empty else 0.0,
        "n_transfers": int(len(plan)),
        "deadstock_value": float(flagged["inventory_value"].sum())
        if not flagged.empty
        else 0.0,
        "deadstock_units": float(flagged["on_hand"].sum()) if not flagged.empty else 0.0,
        "holding_savings": float(flagged["holding_cost_savings"].sum())
        if not flagged.empty
        else 0.0,
        "n_deadstock": int(len(flagged)),
        "daily_history": daily_history,
        "daily_forecast": daily_forecast,
        "by_store": by_store,
        "by_family": by_family,
        "top_risks": top_risks,
        "forecast": forecast,
        "history": history,
    }


# ===========================================================================
# Charts
# ===========================================================================
def _aggregate_demand_chart(daily_history: pd.DataFrame, daily_forecast: pd.DataFrame) -> go.Figure:
    """Actual vs. predicted chain-wide daily demand, with the forecast shaded."""
    fig = go.Figure()
    if not daily_history.empty:
        fig.add_trace(
            go.Scatter(
                x=daily_history[config.DATE_COL],
                y=daily_history["actual"],
                name="Historical sales",
                mode="lines",
                line=dict(color="#2563eb", width=2),
                hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} units<extra>Actual</extra>",
            )
        )
    if not daily_forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=daily_forecast[config.DATE_COL],
                y=daily_forecast["forecast"],
                name="LightGBM forecast",
                mode="lines",
                line=dict(color="#dc2626", width=2, dash="dash"),
                hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} units<extra>Forecast</extra>",
            )
        )
        first, last = daily_forecast[config.DATE_COL].min(), daily_forecast[config.DATE_COL].max()
        fig.add_vrect(
            x0=first,
            x1=last,
            fillcolor="rgba(220, 38, 38, 0.07)",
            layer="below",
            line_width=0,
        )

    fig.update_layout(
        title=dict(text="Aggregate multi-store daily demand", font=dict(size=15)),
        xaxis_title=None,
        yaxis_title="Units sold",
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(l=8, r=8, t=52, b=8),
        height=380,
        hovermode="x unified",
        yaxis=dict(gridcolor="rgba(148,163,184,0.25)"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
    )
    return fig


def _store_bar(by_store: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        by_store,
        x="store_nbr",
        y="demand",
        color="store_nbr",
        color_continuous_scale=["#2563eb", "#0ea5e9", "#16a34a"],
        labels={"store_nbr": "Store", "demand": "Forecast units"},
        text_auto=".2s",
    )
    fig.update_layout(
        title=dict(text="Forecast demand by store", font=dict(size=15)),
        showlegend=False,
        margin=dict(l=8, r=8, t=52, b=8),
        height=330,
        coloraxis_showscale=False,
        yaxis=dict(gridcolor="rgba(148,163,184,0.25)"),
    )
    fig.update_xaxes(type="category")
    return fig


def _family_donut(by_family: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        by_family,
        names="family",
        values="demand",
        hole=0.55,
        color="family",
        color_discrete_sequence=_CHART_COLOURS,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label", pull=[0.03] * len(by_family))
    fig.update_layout(
        title=dict(text="Demand mix by family", font=dict(size=15)),
        margin=dict(l=8, r=8, t=52, b=8),
        height=330,
        legend=dict(orientation="v", y=1, x=1.02),
        showlegend=False,
    )
    return fig


# ===========================================================================
# Page
# ===========================================================================
def main() -> None:
    _ui.bootstrap(
        title="Store Demand Predictor — Executive Overview",
        icon="🏬",
        eyebrow="Executive summary",
        subtitle=(
            "Chain-wide demand, inventory exposure, rebalancing savings and "
            "clearable deadstock — the four numbers that drive tomorrow's "
            "purchasing decisions."
        ),
    )

    stores, families = _ui.sidebar_scope()
    horizon = _ui.horizon_radio(key="horizon_home")

    _ui.data_banner()

    with st.spinner("Crunching the executive roll-up…"):
        data = executive_overview(stores, families, horizon)

    if data.get("empty"):
        _ui.empty_state(
            "No stores or families selected.",
            hint="Pick at least one of each in the sidebar.",
        )
        _ui.footer()
        return

    # ---------- KPI row -----------------------------------------------------
    stockout_help = (
        "Series whose on-hand stock is below the reorder point at the current "
        "service level."
    )
    _ui.kpi_row(
        [
            {
                "label": f"Total expected demand ({horizon}d)",
                "value": _ui.units(data["forecast_total"]),
                "delta": f"{data['growth_pct']:+.1f}% vs. trailing {horizon}d",
                "help": "Sum of LightGBM predictions across every selected store "
                        "and family for the chosen horizon.",
            },
            {
                "label": "Items at stockout risk",
                "value": _ui.units(data["n_stockout_risk"]),
                "delta": f"{data['n_critical']} critical · {_ui.pct(data['n_stockout_risk'] / max(data['n_series'], 1), 0)} of SKUs",
                "help": stockout_help,
            },
            {
                "label": "Estimated transfer savings",
                "value": _ui.money(data["transfer_net_value"]),
                "delta": f"{_ui.units(data['transfer_units'])} units across {data['n_transfers']} legs",
                "help": "Value of stock moved between stores, net of the cost "
                        "of moving it — inventory bought before it is needed.",
            },
            {
                "label": "Deadstock tied capital",
                "value": _ui.money(data["deadstock_value"]),
                "delta": f"{_ui.units(data['deadstock_units'])} units across {data['n_deadstock']} SKUs",
                "help": "Cost of slow-moving stock sitting on the shelf; "
                        "see the Clearance page for the recovery plan.",
            },
        ]
    )

    st.markdown("")

    # ---------- Aggregate demand -------------------------------------------
    left, right = st.columns([2.05, 1])
    with left:
        st.plotly_chart(
            _aggregate_demand_chart(data["daily_history"], data["daily_forecast"]),
            use_container_width=True,
            config=_ui.PLOTLY_CONFIG,
        )
    with right:
        if data["by_family"].empty:
            _ui.empty_state("No demand in scope.")
        else:
            st.plotly_chart(
                _family_donut(data["by_family"]),
                use_container_width=True,
                config=_ui.PLOTLY_CONFIG,
            )

    st.markdown("")

    # ---------- Store view + risk list --------------------------------------
    left, right = st.columns([1.35, 1])
    with left:
        if data["by_store"].empty:
            _ui.empty_state("No demand in scope.")
        else:
            st.plotly_chart(
                _store_bar(data["by_store"]),
                use_container_width=True,
                config=_ui.PLOTLY_CONFIG,
            )

    with right:
        st.markdown("#### ⚠️ Most urgent inventory positions")
        risks: pd.DataFrame = data["top_risks"]
        if risks.empty:
            _ui.empty_state("Nothing to flag — every position is covered.")
        else:
            display = risks[
                [
                    "store_nbr",
                    "family",
                    "on_hand",
                    "reorder_point",
                    "days_of_supply",
                    "stockout_risk_level",
                ]
            ].rename(
                columns={
                    "store_nbr": "Store",
                    "family": "Family",
                    "on_hand": "On hand",
                    "reorder_point": "ROP",
                    "days_of_supply": "DOS",
                    "stockout_risk_level": "Risk",
                }
            )
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                height=330,
                column_config={
                    "On hand": st.column_config.NumberColumn(format="%.1f"),
                    "ROP": st.column_config.NumberColumn(format="%.1f"),
                    "DOS": st.column_config.NumberColumn(format="%.1f"),
                },
            )
            st.caption(
                f"{data['n_stockout_risk']} of {data['n_series']} positions are below "
                f"their reorder point; {data['n_overstock']} are overstocked. "
                f"Reorder exposure: {_ui.money(data['reorder_value'])}."
            )

    # ---------- Situation strip --------------------------------------------
    _ui.divider()
    _ui.legend(
        [
            ("Forecast window shaded red", "#dc2626"),
            ("Historical sales", "#2563eb"),
            ("Rebalancing opportunity", "#16a34a"),
        ]
    )
    _ui.notice(
        f"<b>Next step:</b> {data['n_transfers']} internal transfer legs worth "
        f"<b>{_ui.money(data['transfer_net_value'])}</b> are recommended, and "
        f"<b>{_ui.money(data['holding_savings'])}</b> of annual holding cost can "
        "be avoided by clearing deadstock. Open the Transfer and Clearance pages "
        "to approve the plan.",
        kind="info",
        icon="➡️",
    )

    _ui.footer()


if __name__ == "__main__":
    main()
