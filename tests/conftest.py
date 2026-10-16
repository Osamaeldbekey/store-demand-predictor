"""Make the project root importable for every test module.

Also pins the dashboard data layer to its synthetic corpus. Tests must be
hermetic: whether a developer happens to have run the Phase 1-3 pipeline must
not change what the suite asserts, and loading 37 MB of local parquet plus
forecasting a 20-store scope would turn a 30-second suite into a multi-minute
one. ``SDP_FORCE_DEMO`` is read by ``app/_data.py``; see its docstring.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("SDP_FORCE_DEMO", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
