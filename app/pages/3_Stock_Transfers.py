"""
Page 3 — Inter-store transfers.

Shows the rebalancing plan produced by :mod:`src.engines.transfers` (deficit
stores matched against surplus stores inside the same item family) and lets
the planner approve legs one by one; approved savings accumulate live in
``st.session_state``.
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
import streamlit as st

from src import config
from src.engines import transfers as engine_transfers
import _data
import _ui

_APPROVAL_PREFIX = "transfer::approve::"
_SIGNATURE_KEY = "transfer::plan_signature"

_URGENCY_COLOUR = {"critical": "#dc2626", "high": "#ea580c", "medium": "#ca8a04"}


def _urgency(recipient_dos: float) -> str:
    """Urgency band from how little cover the receiving store has left."""
    try:
        dos = float(recipient_dos)
    except (TypeError, ValueError):
        return "medium"
    if dos != dos or dos == float("inf"):     # NaN / zero-demand receiver
        return "medium"
    if dos < 7:
        return "critical"
    if dos < config.UNDERSTOCK_DOS_THRESHOLD:
        return "high"
    return "medium"


def _leg_id(row: Any) -> str:
    """Stable, unique key for a ``(family, from, to)`` leg."""
    return f"{row.family}|{int(row.from_store)}->{int(row.to_store)}"


def _plan_signature(plan: pd.DataFrame, min_units: int, target_dos: float) -> str:
    """Fingerprint of the current plan — changes invalidate old approvals."""
    if plan.empty:
        return f"empty|{min_units}|{target_dos}"
    return (
        f"{len(plan)}|{plan['quantity'].sum():.0f}|"
        f"{plan['estimated_net_value'].sum():.2f}|{min_units}|{target_dos}"
    )


def _reset_stale_approvals(signature: str) -> None:
    """Drop approvals that belong to a different plan or policy."""
    if st.session_state.get(_SIGNATURE_KEY) == signature:
        return
    for key in [k for k in list(st.session_state) if str(k).startswith(_APPROVAL_PREFIX)]:
        del st.session_state[key]
    st.session_state[_SIGNATURE_KEY] = signature
    st.session_state["approved_transfers"] = {}


def _matrix_table(plan: pd.DataFrame) -> pd.DataFrame:
    """Units on the From → To route matrix (the interactive transfer matrix)."""
    if plan.empty:
        return pd.DataFrame()
    matrix = plan.pivot_table(
        index="from_store", columns="to_store", values="quantity",
        aggfunc="sum", fill_value=0, observed=True,
    )
    matrix.columns = [f"To {c}" for c in matrix.columns]
    matrix.index = [f"From {i}" for i in matrix.index]
    return matrix.reset_index().rename(columns={"from_store": "Route"})


def main() -> None:
    _ui.bootstrap(
        title="Stock Transfers — Store Demand Predictor",
        icon="🔁",
        eyebrow="Phase 4 · rebalancing engine",
        subtitle=(
            "Move stock from stores that have too much to stores that are about "
            "to run out, before buying anything new. Approve the legs you want "
            "and watch the savings accumulate."
        ),
    )

    # ---------- sidebar ----------------------------------------------------
    stores, families = _ui.sidebar_scope()
    horizon = _ui.horizon_radio(key="horizon_transfers")

    st.sidebar.markdown("### Guardrails")
    min_transfer_units = st.sidebar.number_input(
        "Minimum transfer units", 1, 500, int(config.MIN_TRANSFER_UNITS), step=1,
        help="Legs smaller than this are not worth the handling effort.",
    )
    max_transfer_units = st.sidebar.number_input(
        "Maximum units per leg", 10, 100000, int(config.MAX_TRANSFER_UNITS), step=10,
    )
    target_dos = st.sidebar.slider(
        "Target days of supply", 5, 120, int(config.TRANSFER_TARGET_DOS), step=5,
        help="Cover a receiver is topped up to, and the level a donor keeps.",
    )
    service_level = st.sidebar.slider(
        "Service level (drives safety stock)", 0.90, 0.99, config.SERVICE_LEVEL, step=0.01,
        format="%.0f%%",
    )

    _ui.data_banner()

    if not _data.forecast_hygiene(stores, families):
        _ui.empty_state("No stores or families selected.", hint="Choose some in the sidebar.")
        _ui.footer()
        return

    with st.spinner("Matching deficits against surplus…"):
        plan = _data.get_transfers(
            stores,
            families,
            horizon,
            service_level=service_level,
            min_transfer_units=int(min_transfer_units),
            target_dos=float(target_dos),
            max_transfer_units=int(max_transfer_units),
        )

    summary = engine_transfers.transfer_summary(plan)
    signature = _plan_signature(plan, int(min_transfer_units), float(target_dos))
    _reset_stale_approvals(signature)

    # ---------- KPIs -------------------------------------------------------
    _ui.kpi_row(
        [
            {
                "label": "Recommended legs",
                "value": _ui.units(summary["n_transfers"]),
                "delta": f"{summary['n_items']} items · {summary['n_donor_stores']} donor → {summary['n_recipient_stores']} receiver stores",
            },
            {
                "label": "Units relocated",
                "value": _ui.units(summary["total_units"]),
                "delta": f"avg {_ui.units(summary['mean_transfer_units'])} / leg",
            },
            {
                "label": "Net savings (all legs)",
                "value": _ui.money(summary["total_net_value"]),
                "delta": f"value {_ui.money(summary['total_value'])} − cost {_ui.money(summary['total_transfer_cost'])}",
                "help": "Value of goods moved minus the cost of moving them. "
                        "Legs that cost more than they are worth are refused.",
            },
            {
                "label": "Purchasing avoided",
                "value": _ui.money(summary["total_value"]),
                "delta": "internal stock used instead of new POs",
            },
        ]
    )

    st.markdown("")

    if plan.empty:
        _ui.empty_state(
            "No economically viable transfers for this scope and policy.",
            hint="Try lowering the minimum transfer units, changing the target "
                 "days of supply, or widening the store/family selection.",
        )
        _ui.notice(
            "The engine only moves stock when three tests all pass: the leg is "
            "at least the minimum size, the donor stays above "
            "max(target cover, safety stock), and the goods are worth more "
            "than the cost of moving them.",
            kind="info",
            icon="ℹ️",
        )
        _ui.footer()
        return

    # ---------- matrix -----------------------------------------------------
    st.markdown("#### Transfer matrix — units by route")
    matrix = _matrix_table(plan)
    st.dataframe(matrix, use_container_width=True, hide_index=True,
                 column_config={c: st.column_config.NumberColumn(c) for c in matrix.columns[1:]})

    st.markdown("#### Recommended legs")
    legs = plan[
        [
            "family", "from_store", "to_store", "quantity",
            "recipient_dos_before", "estimated_value",
            "estimated_transfer_cost", "estimated_net_value", "rationale",
        ]
    ].copy()
    legs["urgency"] = legs["recipient_dos_before"].map(_urgency)
    legs = legs.sort_values(
        ["urgency", "estimated_net_value"],
        key=lambda column: (
            column.map({"critical": 0, "high": 1, "medium": 2})
            if column.name == "urgency"
            else column
        ),
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)

    st.dataframe(
        legs.rename(
            columns={
                "family": "Family",
                "from_store": "From",
                "to_store": "To",
                "quantity": "Units",
                "recipient_dos_before": "Receiver cover",
                "estimated_value": "Value",
                "estimated_transfer_cost": "Move cost",
                "estimated_net_value": "Net savings",
                "urgency": "Urgency",
                "rationale": "Why",
            }
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Units": st.column_config.NumberColumn(),
            "Receiver cover": st.column_config.NumberColumn(format="%.1f days"),
            "Value": st.column_config.NumberColumn(format="$,.0f"),
            "Move cost": st.column_config.NumberColumn(format="$,.0f"),
            "Net savings": st.column_config.NumberColumn(format="$,.0f"),
        },
    )

    # ---------- approvals --------------------------------------------------
    _ui.divider()
    header, actions = st.columns([2, 1])
    with header:
        st.markdown("#### ✅ Approve transfer legs")
    with actions:
        left, right = st.columns(2)
        approve_all = left.button("Approve all", use_container_width=True)
        clear_all = right.button("Reset", use_container_width=True)

    if approve_all:
        for row in plan.itertuples(index=False):
            st.session_state[_APPROVAL_PREFIX + _leg_id(row)] = True
    if clear_all:
        for key in [k for k in list(st.session_state) if str(k).startswith(_APPROVAL_PREFIX)]:
            st.session_state[key] = False

    approved_mask: list[bool] = []
    approved_value = 0.0
    approved_units = 0.0
    approved_cost = 0.0

    for row in plan.itertuples(index=False):
        key = _APPROVAL_PREFIX + _leg_id(row)
        urgency = _urgency(row.recipient_dos_before)
        border = _URGENCY_COLOUR.get(urgency, "#2563eb")

        with st.container(border=True):
            tick, body, money_col = st.columns([0.6, 5.4, 2.0])
            with tick:
                checked = st.checkbox("Approve", key=key, label_visibility="collapsed")
            with body:
                st.markdown(
                    f'<div style="border-left:4px solid {border}; padding-left:0.7rem;">'
                    f"<b>{_ui.badge(urgency, urgency)}</b>"
                    f"<b>{row.family}</b> &nbsp; Store&nbsp;<b>{int(row.from_store)}</b> "
                    f"&nbsp;→&nbsp; Store&nbsp;<b>{int(row.to_store)}</b>"
                    f" &nbsp;·&nbsp; <b>{row.quantity:,.0f} units</b>"
                    f"<br><span style='color:#64748b;font-size:0.85rem;'>"
                    f"receiver cover {row.recipient_dos_before:,.1f}d → "
                    f"{row.recipient_dos_after:,.1f}d · donor "
                    f"{row.donor_dos_before:,.1f}d → {row.donor_dos_after:,.1f}d"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )
            with money_col:
                st.markdown(
                    f"<div style='text-align:right;'><span style='font-size:0.78rem;"
                    f"color:#64748b;'>NET SAVINGS</span><br>"
                    f"<b style='font-size:1.15rem;color:#16a34a;'>"
                    f"{_ui.money(row.estimated_net_value)}</b></div>",
                    unsafe_allow_html=True,
                )

            approved_mask.append(bool(checked))
            if checked:
                approved_value += float(row.estimated_value)
                approved_units += float(row.quantity)
                approved_cost += float(row.estimated_transfer_cost)

    # Live approval state, exposed for other pages / downstream tooling.
    st.session_state["approved_transfers"] = {
        _leg_id(row): bool(mask)
        for row, mask in zip(plan.itertuples(index=False), approved_mask)
    }

    approved_net = approved_value - approved_cost
    n_approved = sum(approved_mask)
    total_net = float(plan["estimated_net_value"].sum())

    _ui.divider()
    st.markdown("#### Cumulative position")
    _ui.kpi_row(
        [
            {
                "label": "Approved legs",
                "value": f"{n_approved} / {len(plan)}",
                "delta": f"{approved_units:,.0f} units committed",
            },
            {
                "label": "Cumulative net savings",
                "value": _ui.money(approved_net),
                "delta": f"{(approved_net / total_net * 100.0) if total_net else 0.0:.0f}% of {_ui.money(total_net)} available",
            },
            {
                "label": "Moving cost incurred",
                "value": _ui.money(approved_cost),
                "delta": f"goods moved: {_ui.money(approved_value)}",
            },
        ]
    )

    if n_approved:
        _ui.notice(
            f"<b>{n_approved} leg(s) approved.</b> Moving <b>{approved_units:,.0f}</b> "
            f"units for a net saving of <b>{_ui.money(approved_net)}</b> — "
            f"stock bought internally instead of {_ui.money(approved_value)} of "
            "new purchase orders.",
            kind="ok",
            icon="✅",
        )
    else:
        _ui.notice(
            "Tick a leg to approve it. Approvals are stored in "
            "<code>st.session_state['approved_transfers']</code> and reset "
            "automatically whenever the plan or policy changes.",
            kind="info",
            icon="☑️",
        )

    # ---------- flows ------------------------------------------------------
    with st.expander("Where the stock is going", expanded=False):
        chart = plan.copy()
        chart["route"] = (
            "S" + chart["from_store"].astype(str) + " → S" + chart["to_store"].astype(str)
        )
        fig = px.bar(
            chart,
            x="route",
            y="quantity",
            color="family",
            labels={"route": "Route", "quantity": "Units", "family": "Family"},
            text_auto=".0f",
        )
        fig.update_layout(
            barmode="stack",
            height=340,
            margin=dict(l=8, r=8, t=30, b=8),
            legend=dict(orientation="h", y=1.14, x=0),
        )
        st.plotly_chart(fig, use_container_width=True, config=_ui.PLOTLY_CONFIG)

    _ui.footer()


if __name__ == "__main__":
    main()
