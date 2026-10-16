"""
Phase 4 decision engines.

Three business-logic engines sit on top of the Phase 3 forecasts and turn
predictive numbers into operational decisions:

* :mod:`src.engines.inventory`  — safety stock, reorder point, order quantity
* :mod:`src.engines.transfers`  — inter-store stock rebalancing
* :mod:`src.engines.deadstock`  — slow-mover detection and bundle generation

Every engine is a pure function of tabular inputs (no model or I/O dependency),
so it can be unit-tested and reused by the Streamlit dashboard directly.

Submodules are loaded lazily (PEP 562) so importing the package is cheap and
free of import cycles::

    from src.engines import inventory, transfers, deadstock
"""
from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["inventory", "transfers", "deadstock"]

_SUBMODULES: frozenset[str] = frozenset({"inventory", "transfers", "deadstock"})


def __getattr__(name: str) -> ModuleType:
    """Lazily import a public submodule on first attribute access."""
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
