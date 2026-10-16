"""
Shared presentation kit for the Store Demand Predictor dashboard (Phase 5).

Two responsibilities:

* **Theming** — one stylesheet injected once per script run so every page has
  the same cards, badges and spacing without repeating CSS.
* **Components** — thin wrappers over Streamlit primitives (KPI cards, badges,
  empty states, footer) that enforce a consistent layout and
  ``use_container_width=True`` everywhere.

Page scripts call :func:`bootstrap` first; it configures the page, injects the
stylesheet and imports the data layer.

Usage
-----
    from _ui import bootstrap
    bootstrap(title="Store Demand Predictor", icon="🏬")
    ...
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (parent for parent in APP_DIR.parents if (parent / "src" / "config.py").exists()),
    APP_DIR.parent,
)
for _path in (str(PROJECT_ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import streamlit as st  # noqa: E402

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
_BASE_CSS = """
<style>
/* ---------- layout ---------- */
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }

[data-testid="stSidebar"] {
  border-right: 1px solid rgba(49, 51, 63, 0.16);
  background: linear-gradient(180deg, #fbfcfe 0%, #f4f7fb 100%);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { font-size: 0.92rem; }

/* ---------- headings ---------- */
.app-title { font-size: 1.95rem; font-weight: 750; letter-spacing: -0.02em;
             margin: 0 0 0.15rem 0; color: #0f172a; }
.app-subtitle { font-size: 1.02rem; color: #475569; margin: 0 0 1.1rem 0;
                line-height: 1.55; max-width: 78ch; }
.page-eyebrow { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.14em;
                text-transform: uppercase; color: #2563eb; margin-bottom: 0.2rem; }
section[data-testid="stSidebarNav"] ul { list-style: none; }

/* ---------- KPI cards ---------- */
div[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid rgba(148, 163, 184, 0.35);
  border-radius: 14px;
  padding: 1rem 1.1rem 0.9rem 1.1rem;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
  min-height: 118px;
}
div[data-testid="stMetric"] label { color: #64748b; font-size: 0.78rem;
  font-weight: 650; letter-spacing: 0.06em; text-transform: uppercase; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: #0f172a; font-weight: 700; font-size: 1.85rem; letter-spacing: -0.01em; }
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 0.86rem; }

/* ---------- panels & cards ---------- */
.panel {
  background: #ffffff;
  border: 1px solid rgba(148, 163, 184, 0.32);
  border-radius: 14px;
  padding: 1.1rem 1.2rem 1.15rem 1.2rem;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
  margin-bottom: 0.4rem;
}
.bundle-card {
  background: #ffffff;
  border: 1px solid rgba(148, 163, 184, 0.34);
  border-left: 5px solid #2563eb;
  border-radius: 14px;
  padding: 1.05rem 1.15rem 1.1rem 1.1rem;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.07);
  margin-bottom: 0.85rem;
}
.bundle-card.critical { border-left-color: #dc2626; }
.bundle-card.high     { border-left-color: #ea580c; }
.bundle-card.medium   { border-left-color: #ca8a04; }
.bundle-card.low      { border-left-color: #16a34a; }
.card-title { font-size: 1.06rem; font-weight: 700; color: #0f172a; margin-bottom: 0.1rem; }
.card-sub   { font-size: 0.86rem; color: #64748b; margin-bottom: 0.55rem; }
.card-pair  { font-size: 0.97rem; color: #1e293b; margin-bottom: 0.6rem; line-height: 1.5; }
.card-pair b { color: #0f172a; }
.card-stats { font-size: 0.9rem; color: #334155; line-height: 1.65; }
.card-stats b { color: #0f172a; font-weight: 700; }

/* ---------- badges ---------- */
.badge { display: inline-block; padding: 0.16rem 0.6rem; border-radius: 999px;
         font-size: 0.73rem; font-weight: 700; letter-spacing: 0.04em;
         text-transform: uppercase; margin-right: 0.4rem; margin-top: 0.15rem; }
.badge.critical { background: #fee2e2; color: #b91c1c; }
.badge.high     { background: #ffedd5; color: #c2410c; }
.badge.medium   { background: #fef9c3; color: #a16207; }
.badge.low      { background: #dcfce7; color: #15803d; }
.badge.none     { background: #e2e8f0; color: #475569; }
.badge.info     { background: #dbeafe; color: #1d4ed8; }
.badge.neutral  { background: #f1f5f9; color: #475569; }
.badge.affinity     { background: #dbeafe; color: #1d4ed8; }
.badge.fallback     { background: #fef3c7; color: #92400e; }

/* ---------- notices ---------- */
.app-notice { border-radius: 12px; padding: 0.7rem 0.95rem; font-size: 0.9rem;
              line-height: 1.55; border: 1px solid; margin-bottom: 0.75rem; }
.app-notice.demo   { background: #fffbeb; border-color: #fcd34d; color: #92400e; }
.app-notice.info   { background: #eff6ff; border-color: #bfdbfe; color: #1e40af; }
.app-notice.ok     { background: #f0fdf4; border-color: #bbf7d0; color: #166534; }

/* ---------- misc ---------- */
.app-divider { border-top: 1px solid rgba(148, 163, 184, 0.35); margin: 0.9rem 0 1rem 0; }
.app-footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(148, 163, 184, 0.35);
              font-size: 0.82rem; color: #94a3b8; }
.stDataFrame { border-radius: 10px; }
div[data-testid="stExpander"] { border-radius: 10px; }
</style>
"""

PLOTLY_CONFIG: dict[str, Any] = {"displaylogo": False, "responsive": True}


def bootstrap(
    title: str = "Store Demand Predictor",
    icon: str = "🏬",
    subtitle: str | None = None,
    eyebrow: str | None = None,
) -> None:
    """
    Run once at the top of every page: page config, stylesheet, header.

    Must be called before any other Streamlit command (Streamlit requires
    ``set_page_config`` first).
    """
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_BASE_CSS, unsafe_allow_html=True)
    if eyebrow or subtitle:
        page_header(title, eyebrow=eyebrow, subtitle=subtitle)


def page_header(title: str, eyebrow: str | None = None, subtitle: str | None = None) -> None:
    """Responsive page heading (eyebrow + title + optional subtitle)."""
    if eyebrow:
        st.markdown(f'<div class="page-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="app-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def panel(markup: str) -> None:
    """Render raw HTML inside a white card."""
    st.markdown(f'<div class="panel">{markup}</div>', unsafe_allow_html=True)


# ===========================================================================
# Formatters
# ===========================================================================
def money(value: Any, decimals: int = 0) -> str:
    """$1,234,567 (0 decimals by default — money is a headline number)."""
    try:
        return f"${float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "$0"


def units(value: Any, decimals: int = 0) -> str:
    """1,234 units, with ``inf``/``NaN``/unparseable input made printable."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number == float("inf"):
        return "∞"
    if number != number:
        return "—"
    return f"{number:,.{decimals}f}"


def pct(value: Any, decimals: int = 1) -> str:
    """0.95 -> '95.0%'."""
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def badge(text: str, kind: str = "neutral") -> str:
    """Inline status pill (``critical``/``high``/``medium``/``low``/...)."""
    return f'<span class="badge {kind}">{text}</span>'


# ===========================================================================
# Components
# ===========================================================================
def kpi_row(cards: Sequence[dict[str, Any]]) -> None:
    """
    A responsive row of KPI metric cards.

    Each card: ``{"label", "value", "delta", "help"}``. Streamlit colours a
    delta up/down from its **sign** (a leading ``+``/``-``), so direction is
    encoded in the delta text rather than a separate argument. ``st.metric``
    already fills its column, so no width argument is needed (and
    ``use_container_width`` is not accepted by this API).
    """
    if not cards:
        return
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        with column:
            st.metric(
                label=str(card.get("label", "")),
                value=str(card.get("value", "")),
                delta=card.get("delta"),
                help=card.get("help"),
            )


def notice(message: str, kind: str = "info", icon: str = "ℹ️") -> None:
    """Styled inline banner (``demo`` / ``info`` / ``ok``)."""
    st.markdown(
        f'<div class="app-notice {kind}">{icon} {message}</div>',
        unsafe_allow_html=True,
    )


def data_banner() -> None:
    """Explain which corpus and model the dashboard is currently using."""
    from _data import is_demo, model_source, source_note  # local import

    if is_demo():
        notice(
            f"<b>Demo mode.</b> {source_note()} Model: {model_source()}.",
            kind="demo",
            icon="🧪",
        )
    else:
        notice(
            f"<b>Production data.</b> {source_note()} Model: {model_source()}.",
            kind="ok",
            icon="✅",
        )


def empty_state(message: str, hint: str | None = None, icon: str = "📭") -> None:
    """Friendly placeholder for zero-result screens."""
    body = f"{icon} {message}"
    if hint:
        body += f"  \n{hint}"
    st.info(body)


def footer(extra: str | None = None) -> None:
    """Common footer with the Phase/section credit."""
    line = "Store Demand Predictor · LightGBM global forecast · Phase 4 engines · Phase 5 dashboard"
    if extra:
        line = f"{extra} — {line}"
    st.markdown(f'<div class="app-footer">{line}</div>', unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<div class="app-divider"></div>', unsafe_allow_html=True)


def legend(items: Iterable[tuple[str, str]]) -> None:
    """Compact colour legend row: ``[("text", "css colour"), ...]``."""
    markup = "".join(
        f'<span style="display:inline-block;margin-right:1.1rem;font-size:0.82rem;'
        f'color:#475569;">'
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:3px;'
        f'background:{colour};margin-right:0.4rem;"></span>{text}</span>'
        for text, colour in items
    )
    st.markdown(markup, unsafe_allow_html=True)


def sidebar_scope(
    stores_label: str = "Stores",
    families_label: str = "Families (SKU groups)",
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """
    Shared store/family multi-selects in the sidebar.

    Returns the selection as hashable tuples so results can be cached.
    """
    from _data import scope_options

    all_stores, all_families = scope_options()
    stores = st.sidebar.multiselect(
        stores_label, list(all_stores), default=list(all_stores)
    )
    families = st.sidebar.multiselect(
        families_label, list(all_families), default=list(all_families)
    )
    return tuple(int(s) for s in stores), tuple(str(f) for f in families)


def horizon_radio(key: str = "horizon") -> int:
    """7 / 14 / 30-day forecast horizon toggle (sidebar)."""
    from src import config as src_config

    options = list(src_config.FORECAST_HORIZONS)
    default_index = (
        options.index(src_config.FORECAST_HORIZON)
        if src_config.FORECAST_HORIZON in options
        else len(options) - 1
    )
    labels = [f"{d} days" for d in options]
    choice = st.sidebar.radio(
        "Forecast horizon",
        options=labels,
        index=default_index,
        key=key,
        help="The 30-day recursion is computed once and truncated, so every "
        "horizon is a valid prefix of the same forecast.",
    )
    return int(choice.split()[0])
