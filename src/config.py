"""
Central configuration for the Store Demand Predictor project.

Every module imports paths and global parameters from here, so tuning the
project scope (stores, families, model params, business defaults) is a
single-file edit.

Usage:
    from src import config
    config.SALES_CLEAN_PARQUET          # PosixPath/WindowsPath to artifact
    config.SCOPE_STORE_NBR              # demo store scope
"""
from __future__ import annotations

from pathlib import Path

# ===========================================================================
# 1. PROJECT ROOT & DIRECTORY PATHS
# ===========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
APP_DIR = PROJECT_ROOT / "app"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# --- Raw inputs (Kaggle competition: store-sales-time-series-forecasting) --
TRAIN_CSV = RAW_DIR / "train.csv"
STORES_CSV = RAW_DIR / "stores.csv"
HOLIDAYS_CSV = RAW_DIR / "holidays_events.csv"
OIL_CSV = RAW_DIR / "oil.csv"
TEST_CSV = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DIR / "sample_submission.csv"

# --- Processed outputs & model artifacts -----------------------------------
SALES_CLEAN_PARQUET = PROCESSED_DIR / "sales_clean.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"
FORECASTS_PARQUET = PROCESSED_DIR / "forecasts.parquet"
MODEL_ARTIFACT = MODELS_DIR / "lgbm_v1.joblib"

# ===========================================================================
# 2. SCOPE FILTERS (demo-friendly subset for Streamlit Cloud)
# ===========================================================================
# 20 of the 54 Favorita stores → roughly ~3M rows after calendar reindexing,
# which fits comfortably under Streamlit Community Cloud resource limits.
SCOPE_STORE_NBR: list[int] = list(range(1, 21))

# Long-tail product families that are almost entirely zero-sales noise in
# this scope. Dropping them cuts row count without hurting forecast quality.
EXCLUDED_FAMILIES: tuple[str, ...] = (
    "BABY CARE",
    "BOOKS",
    "CELEBRATION",
    "GROCERY II",
    "HARDWARE",
    "LAWN AND GARDEN",
    "MAGAZINES",
)

# ===========================================================================
# 3. SCHEMA / COLUMN DEFINITIONS
# ===========================================================================
ID_COL = "id"
DATE_COL = "date"
TARGET_COL = "sales"
GROUP_KEYS: list[str] = ["store_nbr", "family"]   # one time series per group

# Native LightGBM categorical features (high-cardinality handled internally —
# this avoids target-encoding leakage entirely)
CATEGORICAL_FEATURES: list[str] = [
    "store_nbr",
    "family",
    "type",       # store format A–E
    "cluster",    # cluster of similar stores
    "city",
    "state",
]

# Columns that are never features (identifier / target / date)
NON_FEATURE_COLS: list[str] = [ID_COL, DATE_COL, TARGET_COL]

# Memory-efficient dtypes enforced at CSV load time (train.csv)
DTYPE_MAP: dict[str, str] = {
    "id": "int64",
    "store_nbr": "int16",
    "family": "category",
    "sales": "float32",
    "onpromotion": "int8",
}

# Memory-efficient dtypes for stores.csv
STORES_DTYPE_MAP: dict[str, object] = {
    "store_nbr": "int16",
    "cluster": "int8",
    "city": "category",
    "state": "category",
    "type": "category",
}

# Canonical column order of the cleaned parquet artifact
CLEAN_COLUMN_ORDER: list[str] = [
    ID_COL,
    DATE_COL,
    "store_nbr",
    "family",
    TARGET_COL,
    "onpromotion",
    "cluster",
    "city",
    "state",
    "type",
]

# ===========================================================================
# 4. BUSINESS / ENGINE DEFAULTS (Phase 4+)
# ===========================================================================
FORECAST_HORIZONS: tuple[int, ...] = (7, 14, 30)   # days
SERVICE_LEVEL = 0.95                               # z ≈ 1.645
LEAD_TIME_DAYS = 7
LEAD_TIME_STD_DAYS = 0.0                           # 0 → constant lead time
DEADSTOCK_WINDOW_DAYS = 90                         # slow-mover lookback
RANDOM_SEED = 42

# ===========================================================================
# 5. LIGHTGBM BASELINE PARAMETERS (Phase 3 will tune these)
# ===========================================================================
LGBM_PARAMS: dict[str, object] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 20,
    "verbose": -1,
    "seed": RANDOM_SEED,
    "n_jobs": -1,
    # Reproducibility: same data + params + threads -> identical model.
    "deterministic": True,
    "force_row_wise": True,
}

# ===========================================================================
# 6. MODEL TRAINING, TUNING & FORECASTING DEFAULTS (Phase 3)
# ===========================================================================
LGBM_NUM_BOOST_ROUND = 3000
LGBM_EARLY_STOPPING_ROUNDS = 50

# Walk-forward cross-validation used by src/models/build.py
WALK_FORWARD_SPLITS = 3
VALIDATION_DAYS = 16
TEST_DAYS = 16
MIN_TRAIN_DAYS = 90

# Optuna
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT_SECONDS: int | None = None
# Headline business metric by default; set to "rmse" for the training loss.
TUNE_OBJECTIVE_METRIC = "wape"

# Forecasting
FORECAST_HORIZON = 30
# Sales cannot be negative; clip regression outputs at zero.
CLIP_NON_NEGATIVE = True

# ===========================================================================
# 7. DECISION-ENGINE DEFAULTS (Phase 4)
# ===========================================================================
# --- Inventory: safety stock & reorder point -------------------------------
REVIEW_PERIOD_DAYS = 7.0            # 0 -> continuous review
MIN_ORDER_QUANTITY = 1.0            # MOQ in units
ORDER_MULTIPLE = 1.0                # case pack / pallet rounding
DEFAULT_UNIT_COST = 5.0             # fallback when no cost table is supplied
STOCKOUT_ALERT_DOS_DAYS = 3.0       # days-of-supply at/below which to escalate

# --- Inter-store transfers -------------------------------------------------
OVERSTOCK_DOS_THRESHOLD = 60.0      # above this many days of supply -> donor
UNDERSTOCK_DOS_THRESHOLD = 15.0     # below this many days of supply -> receiver
TRANSFER_TARGET_DOS = 30.0          # days of supply a receiver is topped up to
MIN_TRANSFER_UNITS = 5              # smallest economically useful transfer
MAX_TRANSFER_UNITS = 500            # cap per transfer leg
TRANSFER_FIXED_COST = 20.0          # per transfer (handling, admin)
TRANSFER_COST_PER_UNIT = 0.5        # per unit moved (labour, freight)
MIN_TRANSFER_NET_VALUE = 0.0        # skip legs whose value < transfer cost

# --- Deadstock clearance ---------------------------------------------------
DEADSTOCK_VELOCITY_THRESHOLD = 0.5      # avg units/day over the lookback
DEADSTOCK_MIN_DAYS_SINCE_LAST_SALE = 30
DEADSTOCK_HOLDING_COST_RATE = 0.20      # annual, as a fraction of unit value
DEADSTOCK_CLEARANCE_WINDOW_DAYS = 30    # target days to clear once promoted
DEADSTOCK_MAX_DAYS_TO_CLEAR = 730       # cap on the baseline sell-through time
DEADSTOCK_DISCOUNT_BASE = 0.15
DEADSTOCK_DISCOUNT_PER_30_DAYS = 0.05
DEADSTOCK_DISCOUNT_MIN = 0.10
DEADSTOCK_DISCOUNT_MAX = 0.70
BUNDLE_TOP_N_COMPLEMENTS = 3
