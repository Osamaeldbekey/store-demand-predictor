"""
Page 1 — Demand forecasting.

Historical sales vs. the global LightGBM's recursive forecast, with an
uncertainty envelope, for any store / family combination and any of the
7 / 14 / 30-day horizons.
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
import plotly.graph_objects as go
import streamlit as st

from src import config
import _data
import _ui

_ACTUAL = "#2563eb"
_FORECAST = "#dc2626"
_BAND = "rgba(220, 38, 38, 0.13)"


def _series_label(frame: pd.DataFrame) -> pd.Series:
    """``'Store 1 · BEVERAGES'`` — readable axis labels for small multiples."""
    return (
        "Store " + frame["store_nbr"].astype(str) + " · " + frame["family"].astype(str)
    )


def _aggregate_chart(
    history: pd.DataFrame, forecast: pd.DataFrame, show_band: bool
) -> go.Figure:
    """Chain-wide history vs. forecast, with the ±1σ envelope."""
    agg_history = (
        history.groupby(config.DATE_COL, sort=True)[config.TARGET_COL]
        .sum()
        .rename("actual")
        .reset_index()
    )
    agg_forecast = (
        forecast.groupby(config.DATE_COL, sort=True)[["prediction", "lower", "upper"]]
        .sum()
        .reset_index()
    )

    fig = go.Figure()
    if show_band and not agg_forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=agg_forecast[config.DATE_COL],
                y=agg_forecast["upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=agg_forecast[config.DATE_COL],
                y=agg_forecast["lower"],
                mode="lines",
                line=dict(width=0),
                fillcolor=_BAND,
                fill="tonexty",
                showlegend=False,
                name="±1σ band",
                hovertemplate="%{x|%d %b}<br>band: %{y:,.0f}<extra>±1σ</extra>",
            )
        )
    if not agg_history.empty:
        fig.add_trace(
            go.Scatter(
                x=agg_history[config.DATE_COL],
                y=agg_history["actual"],
                name="Historical sales",
                mode="lines",
                line=dict(color=_ACTUAL, width=2),
                hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} units<extra>Actual</extra>",
            )
        )
    if not agg_forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=agg_forecast[config.DATE_COL],
                y=agg_forecast["prediction"],
                name="LightGBM forecast",
                mode="lines",
                line=dict(color=_FORECAST, width=2, dash="dash"),
                hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} units<extra>Forecast</extra>",
            )
        )

    fig.update_layout(
        title=dict(text="Aggregate demand — historical vs. forecast", font=dict(size=15)),
        xaxis_title=None,
        yaxis_title="Units",
        height=400,
        legend=dict(orientation="h", y=1.14, x=0),
        margin=dict(l=8, r=8, t=56, b=8),
        hovermode="x unified",
        yaxis=dict(gridcolor="rgba(148,163,184,0.25)"),
    )
    return fig


def _faceted_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """Small multiples: one panel per store, one line per family."""
    actual = history.rename(columns={config.TARGET_COL: "value"})[
        [config.DATE_COL, "store_nbr", "family", "value"]
    ].copy()
    actual["kind"] = "Historical sales"

    predicted = forecast.rename(columns={"prediction": "value"})[
        [config.DATE_COL, "store_nbr", "family", "value"]
    ].copy()
    predicted["kind"] = "Forecast"

    frame = pd.concat([actual, predicted], ignore_index=True)
    frame["series"] = _series_label(frame)
    frame[config.DATE_COL] = pd.to_datetime(frame[config.DATE_COL])
    frame["store"] = "Store " + frame["store_nbr"].astype(str)

    fig = px.line(
        frame,
        x=config.DATE_COL,
        y="value",
        color="kind",
        line_dash="kind",
        facet_col="store",
        # NB: every label value must be a string — plotly's hover builder does
        # ``key + "=" + value``, so a ``None`` label raises at build time.
        labels={"value": "Units", "kind": "", "family": "Family"},
        color_discrete_map={"Historical sales": _ACTUAL, "Forecast": _FORECAST},
        category_orders={"kind": ["Historical sales", "Forecast"]},
    )
    fig.update_traces(line=dict(width=1.9), selector=lambda t: t.mode == "lines")
    fig.update_layout(
        height=330,
        legend=dict(orientation="h", y=1.16, x=0),
        margin=dict(l=8, r=8, t=64, b=8),
        hovermode="x unified",
        yaxis=dict(gridcolor="rgba(148,163,184,0.2)"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", title_text="")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


def main() -> None:
    _ui.bootstrap(
        title="Demand Forecasting — Store Demand Predictor",
        icon="📈",
        eyebrow="Phase 3 · global LightGBM",
        subtitle=(
            "One model trained across every store and family. Select any scope, "
            "pick a horizon, and compare what actually sold against what the "
            "model predicts will sell."
        ),
    )

    # ---------- sidebar ----------------------------------------------------
    stores, families = _ui.sidebar_scope()
    horizon = _ui.horizon_radio(key="horizon_forecast")
    history_days = st.sidebar.slider(
        "History window (days)", 30, 180, 90, step=10,
        help="How much observed sales to plot behind the forecast.",
    )
    show_band = st.sidebar.checkbox(
        "Show prediction band (±1σ)",
        value=True,
        help="Regression gives point estimates, so the band is the series' "
             "observed day-to-day volatility growing with √step "
             "(random-walk approximation). It is not a calibrated quantile.",
    )
    scenario = st.sidebar.selectbox(
        "Promotion scenario",
        options=["Model default", "Promote every item"],
        index=0,
        help="Phase 3 what-if: forces the onpromotion flag for all forecast "
             "days. Decision engines (inventory/transfers) always use the "
             "base forecast.",
    )
    future_onpromotion = None if scenario == "Model default" else 1

    _ui.data_banner()

    if not _data.forecast_hygiene(stores, families):
        _ui.empty_state("No stores or families selected.", hint="Choose some in the sidebar.")
        _ui.footer()
        return

    # ---------- data -------------------------------------------------------
    with st.spinner(
        f"Forecasting {len(stores) * len(families)} series × {horizon} days…"
        " (first call builds the recursion, then it is cached)"
    ):
        forecast = _data.get_forecast(
            stores, families, horizon, future_onpromotion=future_onpromotion
        )
    history = _data.get_history(stores, families, days=history_days)

    if forecast.empty:
        _ui.empty_state("The model produced no rows for this scope.")
        _ui.footer()
        return

    # ---------- KPIs -------------------------------------------------------
    total = float(forecast["prediction"].sum())
    n_series = forecast.groupby(config.GROUP_KEYS, observed=True).ngroups
    peak_row = forecast.groupby(config.DATE_COL, sort=True)["prediction"].sum()
    observed_days = history[config.DATE_COL].nunique() if not history.empty else 0
    observed_total = (
        float(history[config.TARGET_COL].sum()) if not history.empty else 0.0
    )

    _ui.kpi_row(
        [
            {
                "label": f"Forecast demand ({horizon}d)",
                "value": _ui.units(total),
                "delta": f"across {n_series} series",
            },
            {
                "label": "Average daily demand",
                "value": _ui.units(total / max(horizon, 1), 1),
                "delta": "units / day",
            },
            {
                "label": "Peak forecast day",
                "value": _ui.units(peak_row.max()),
                "delta": f"{peak_row.idxmax():%d %b %Y}",
            },
            {
                "label": "Observed in window",
                "value": _ui.units(observed_total),
                "delta": f"last {observed_days} days",
            },
        ]
    )

    st.markdown("")

    # ---------- charts -----------------------------------------------------
    st.plotly_chart(
        _aggregate_chart(history, forecast, show_band),
        use_container_width=True,
        config=_ui.PLOTLY_CONFIG,
    )

    st.markdown("")
    _ui.divider()

    st.markdown("#### Detail by store")
    st.plotly_chart(
        _faceted_chart(history, forecast),
        use_container_width=True,
        config=_ui.PLOTLY_CONFIG,
    )

    # ---------- per-series table ------------------------------------------
    st.markdown("#### Forecast by series")
    table = (
        forecast.groupby(config.GROUP_KEYS, observed=True)
        .agg(
            **{
                "Forecast units": ("prediction", "sum"),
                "Avg / day": ("prediction", "mean"),
                "Min day": ("prediction", "min"),
                "Max day": ("prediction", "max"),
                "First date": (config.DATE_COL, "min"),
                "Last date": (config.DATE_COL, "max"),
            }
        )
        .reset_index()
        .sort_values("Forecast units", ascending=False, kind="stable")
    )
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=min(40 + 35 * len(table), 420),
        column_config={
            "Forecast units": st.column_config.NumberColumn(format="%.1f"),
            "Avg / day": st.column_config.NumberColumn(format="%.2f"),
            "Min day": st.column_config.NumberColumn(format="%.1f"),
            "Max day": st.column_config.NumberColumn(format="%.1f"),
            "First date": st.column_config.DateColumn(format="DD MMM YYYY"),
            "Last date": st.column_config.DateColumn(format="DD MMM YYYY"),
        },
    )

    if future_onpromotion is not None:
        base = _data.get_forecast(stores, families, horizon, future_onpromotion=None)
        base_total = float(base["prediction"].sum())
        delta = total - base_total
        lift = (delta / base_total * 100.0) if base_total else 0.0
        _ui.notice(
            f"<b>Promotion scenario:</b> forcing promotions on every forecast day "
            f"moves {horizon}-day demand from <b>{_ui.units(base_total)}</b> to "
            f"<b>{_ui.units(total)}</b> units "
            f"(<b>{lift:+.1f}%</b>, {delta:+,.0f} units).",
            kind="info",
            icon="🎯",
        )

    _ui.notice(
        f"Model: <b>{_data.model_source()}</b>. The band is a "
        "<b>±1σ random-walk approximation</b> built from each series' observed "
        "day-to-day volatility — LightGBM returns point estimates, so this is an "
        "uncertainty indicator rather than a calibrated prediction interval.",
        kind="info",
        icon="ℹ️",
    )

    _ui.footer()


if __name__ == "__main__":
    main()
