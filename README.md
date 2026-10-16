# Store Demand Predictor

Graduation project: a global LightGBM demand-forecasting system for multi-store
retail chains, delivered as an interactive Streamlit dashboard with inventory
optimization, inter-store transfer, and deadstock-clearance engines.

## Dataset

Kaggle competition **Store Sales - Time Series Forecasting** (Corporación
Favorita, Ecuador).

### 1. Configure the Kaggle API

```powershell
# Place your API token at C:\Users\<you>\.kaggle\kaggle.json
# (Kaggle -> Account -> Create New API Token)
python -c "import kaggle; print('kaggle API configured')"
```

### 2. Download and unzip into `data/raw/`

```powershell
kaggle competitions download -c store-sales-time-series-forecasting -p data/raw
Expand-Archive data/raw/store-sales-time-series-forecasting.zip -DestinationPath data/raw -Force
```

Required files in `data/raw/`: `train.csv`, `stores.csv`, `holidays_events.csv`,
`oil.csv`, `test.csv`, `sample_submission.csv`.

`data/raw/` stays gitignored, with one deliberate exception:
`holidays_events.csv` is committed, because the deployed model consumes three
holiday features and forecasting without them is measurably biased (see
*What is committed* below).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/init_repo.py          # idempotent scaffold
```

## Phase 1 — Data pipeline

```powershell
python -m src.data.clean
# -> data/processed/sales_clean.parquet
```

The pipeline: loads `train.csv` with memory-efficient dtypes → clips negative
sales → filters scope (20 stores, noise families dropped) → reindexes every
`(store, family)` series onto a contiguous daily calendar (missing days become
`sales = 0`) → merges store metadata → validates → writes compressed Parquet.

Regenerate artifacts any time; the parquet and model directories are gitignored.

## Phase 2 — Feature engineering

```powershell
python -m src.features.build
# -> data/processed/features.parquet
python -m src.features.build --drop-warmup   # trim rows lacking a full year of history
```

`src/features/` builds the modelling matrix:

* `calendar.py` — day/week/month/year flags, weekend, Ecuadorian payday flags
  (`is_payday`, `days_to/days_since_payday`), and observed-holiday features
  (`is_holiday`, `days_to/days_since_holiday`). `transferred == True` holiday
  rows (moved celebrations) are dropped; `Work Day`/`Event` rows are not
  treated as days off.
* `lags.py` — lags (`7,14,21,28,30,365`), rolling mean/std (`7,14,28`) and EMAs
  (`7,14`). **All rolling/EMA statistics are computed on the `shift(1)` series**
  so the window never includes the target day. Early rows are intentionally
  `NaN` (warm-up); LightGBM handles them natively.
* `split.py` — expanding-window walk-forward validation with the final
  `test_days` (default 16) sealed off entirely from cross-validation.

| Feature group | Columns |
|---|---|
| Calendar | 12 |
| Holiday | 3 |
| Lag / rolling / EMA | 14 |
| **Total engineered** | **29** |

## Phase 3 — Model training, tuning & evaluation

```powershell
python -m src.models.build                          # walk-forward + final model + sealed test
python -m src.models.build --tune --trials 50       # Optuna search before the final fit
python -m src.models.build --refit-all              # deploy on 100% of the data
python -m src.models.build --no-cv                  # skip cross-validation (faster)
# -> models/lgbm_v1.joblib  and  models/test_predictions.parquet

python -m src.models.forecast                       # pre-compute the offline forecast batch
# -> data/processed/forecasts.parquet   (tier 1 of the dashboard's data layer)
```

| Module | Responsibility |
|---|---|
| `train.py` | Global `LGBMRegressor` over all series; `family/city/state/type/store_nbr/cluster` declared categorical; early stopping on validation RMSE; Joblib save/load with metadata |
| `tune.py` | Optuna (seeded TPE) over `num_leaves`, `learning_rate`, `feature_fraction`, `min_data_in_leaf`, `lambda_l1`, `lambda_l2`, `bagging_fraction`. Default objective is **WAPE**, not RMSE |
| `evaluate.py` | RMSE / MAE / **WAPE** (headline) + bias, naive "last week repeat" baseline comparison, per-series diagnostics |
| `predict.py` | Recursive multi-horizon forecasting (day-by-day feedback), batch `forecast_all`, horizon truncation |
| `forecast.py` | CLI: batch every series for the dashboard's offline tier-1 artifact, with a contract check on the output |
| `build.py` | CLI: walk-forward validation → optional tuning → final fit → sealed-test scoring → persist |

Metrics are reported with **WAPE** first because it is scale-free and answers the
business question ("we are off by *X%* of total demand"). The naive baseline
reuses the Phase 2 `lag_7` column, so it is information-matched to the model.

### Recursive forecasting guarantees

A model that predicts `sales[T]` from lag/rolling features needs `sales[T]` to
forecast `T+1`. `predict.py` solves this by predicting one day at a time and
feeding each prediction back as pseudo-history — reusing the **Phase 2 feature
functions** rather than re-deriving the maths. Two invariants are under test:

1. **Exactness** — the features the model scores during recursion are identical
   to a one-shot batch computation over the same (predicted) history.
2. **Prefix consistency** — a 7-day forecast equals the first 7 days of a 30-day
   forecast. Forward calendar distances (`days_to_payday`) are derived from the
   frame's date axis, so the forecaster pads the horizon with a 45-day
   *calendar buffer* to keep them populated and horizon-independent.

Predictions are clipped at zero (sales cannot be negative).

### Known caveats

* `onpromotion` is known for historical dates and for Kaggle's test set (promo
  calendars are planned in advance). A **blind** forecast assumes no promotion
  unless you pass a `future_onpromotion` scenario (scalar, `dict` or date-indexed
  `Series`) — the dashboard's scenario planner uses this hook.
* `days_to_payday` is `NaN` for the final days of the *training* frame (no next
  payday yet in-scope); this is a warm-tail artefact, and forecasts use the
  buffered axis described above.

## Phase 4 — Business logic & decision engines

Phase 4 turns a forecast into an operational decision. Every engine is a **pure
function of tabular inputs** — no model, no I/O — so the dashboard can call it
directly and the test suite needs no Kaggle data.

```python
from src.engines import inventory, transfers, deadstock

summary = inventory.demand_summary_from_forecast(forecasts)      # Phase 3 output
policy  = inventory.inventory_policy_table(summary, stock)       # SS / ROP per SKU
plan    = transfers.generate_transfers(policy)                   # rebalance before buy
flagged = deadstock.detect_deadstock(sales, stock)               # slow movers
bundles = deadstock.generate_bundles(flagged, affinity, heroes)  # clearance bundles
```

| Module | Responsibility |
|---|---|
| `inventory.py` | Safety stock (`Z·√(P·σ_d² + d̄²·σ_L²)`, optional forecast-error inflation), reorder point, order-up-to level, order quantity with MOQ / case-pack rounding, days-of-supply cover and stockout-risk flags |
| `transfers.py` | Days-of-supply classification (donor / receiver / healthy / no-demand), greedy most-urgent-first matching, and guardrails: minimum transfer units, per-leg cap, donor reserve protection, and a net-value test (never spend more moving stock than it is worth) |
| `deadstock.py` | Slow-mover detection (velocity **and** days-since-last-sale), severity scoring, clearance discount schedule, holding-cost savings, and bundle generation against an item-affinity matrix |
| `_common.py` | Internal helpers: column validation, unit-cost resolution, mypy-friendly key canonicalisation |

### Design notes

* **Safety stock has to absorb two things.** Natural demand variance *and* model
  error, combined in quadrature, over the *protection period* `P = lead time +
  review period`. A deterministic lead time collapses the formula to the textbook
  `Z·σ_d·√P`.
* **Zero-demand series are first-class.** They yield `SS = ROP = 0`, infinite days
  of supply and no stockout risk, instead of dividing by zero. Boundaries are
  strict: on-hand exactly *at* the reorder point is **not** flagged.
* **Rebalancing beats purchasing.** `generate_transfers` only recommends internal
  moves; a donor is never pushed below `max(target DOS · demand, safety stock)`,
  and one unit can never be promised twice.
* **Deadstock needs both signals.** Low velocity alone is a slow seller, not
  deadstock; the item must also have gone `≥ 30` days without a sale. Only stock
  that exists is flagged.
* **Affinity is a documented proxy.** The Favorita dataset has no basket data, so
  complementarity is estimated from the correlation of daily family demand
  *after removing the common daily factor* (`method="residual"`) — raw sales
  correlation is dominated by shared seasonality and would call every family
  complementary to every other. Pass a real co-occurrence matrix to
  `generate_bundles(..., affinity=...)` when baskets become available; the
  algorithm is agnostic to its source. Each bundle records a `selection_basis`
  of `affinity` or `best_seller_fallback`, so a fallback is never silent.
* Search-space / policy numbers are not hard-coded: they live in
  `src/config.py` section 7.

## Phase 5 — Streamlit dashboard

```powershell
streamlit run app/main.py        # -> http://localhost:8501
```

| Page | What it answers |
|---|---|
| `app/main.py` | Executive overview — 30-day expected demand, items at stockout risk, transfer savings, deadstock capital |
| `app/pages/1_Demand_Forecasting.py` | History vs. LightGBM forecast with a ±1σ band, per store/family, 7/14/30-day horizons |
| `app/pages/2_Inventory_Optimization.py` | Service-level slider (90–99%) → live safety stock, reorder points and stockout alerts |
| `app/pages/3_Stock_Transfers.py` | From→To transfer matrix with urgency and net savings; approve individual legs and watch cumulative savings |
| `app/pages/4_Deadstock_Clearance.py` | Clearance bundle cards pairing slow movers with hero products, discount and holding cost recovered |

The pages are read-only views over the Phase 3/4 APIs: the dashboard holds no
business logic of its own, it calls the same engines the test suite does.

### Data layer (`app/_data.py`)

One cached access layer feeds every page:

* **Bundle** (`get_data`) — the `(store, family)` scope, the model and the
  on-hand stock, cached with `@st.cache_resource`.
* **Forecast** (`get_forecast`) — resolved in three tiers: a precomputed
  `forecasts.parquet`, else a **canonical recursion** cached once for the whole
  bundle scope and sliced per request, else a direct subset call. Recursing 30
  days × 18 demo series costs ~11 s **once**; every later scope/horizon answers
  from the cached slice in <0.05 s, which is what keeps interactions inside the
  ~2 s budget.
* **Derived tables** (`get_history`, `get_inventory_policy`, `get_transfers`,
  `get_deadstock`) — `@st.cache_data`, keyed on the resolved arguments.

`st.session_state` is never used here: it does not exist outside a script run,
and the tests call these helpers directly.

### Demo mode

Demo mode is a **fallback, not the default**. The deployable artifacts
(`features.parquet`, `lgbm_v1.joblib`, `forecasts.parquet`) and the holiday
calendar are committed — see *What is committed* below — so a fresh checkout
boots straight into real mode. Demo mode engages only when the two required
artifacts are absent, and `tests/conftest.py` pins it with `SDP_FORCE_DEMO=1`
so the suite stays hermetic regardless of what a developer has on disk.

It builds a **deterministic synthetic panel** (3 stores × 6 families, 200 days,
ending 2017-08-15, seed `20240815`) pushed through the *real* Phase 2 feature
builder and used to train a *real* (deliberately small) LightGBM. Every page
therefore exercises the production code path, and a banner plus `is_demo` state
plainly which path is live. The panel is seeded with a stalled family (TOYS),
per-family stock-cover trios giving every active family both a donor and a
receiver, and one genuine zero-stock position — so every page renders
non-trivial results rather than empty states.

Two disclosures are shown in the UI rather than buried here:

* **On-hand stock is synthetic.** Favorita ships no inventory table, so
  quantities are generated for the demo; the decision *logic* is real.
* **The prediction band is a random-walk approximation.** `±1σ·√step` from each
  series' trailing 90 days of first differences, clipped at zero — not a
  calibrated quantile. It is labelled as such where it is drawn.

### Scenario hook

Phase 3's `future_onpromotion` override is exposed as **"Promote every item"** on
the forecasting page. The decision engines deliberately ignore it: a what-if
forecast must not silently move safety stock or transfer plans.

### Transfer approvals

Approving a leg stores it in `st.session_state`, keyed per
`family|from->to`. The stored approvals are invalidated whenever the plan
signature (leg count, units, net value, policy thresholds) changes, so a stale
tick can never inflate the displayed savings.

## Phase 6 — Deployment readiness

```powershell
python -m pytest tests/test_app.py -v     # headless page smoke tests
streamlit run app/main.py --server.port 8501
```

`tests/test_app.py` drives every page through Streamlit's own `AppTest` harness
— the same execution path as `streamlit run` — so a missing dependency, an
invalid Streamlit API call or an empty-scope crash fails the build instead of
the demo.

### What is committed

The deployable artifacts ship with the repository, so a plain clone serves real
forecasts with no Kaggle credentials and no Phase 1–3 run:

| Path | Size | Why it ships |
|---|---|---|
| `models/lgbm_v1.joblib` | 8.1 MB | the trained global LightGBM (model, feature list, params, metrics) |
| `data/processed/features.parquet` | 35.8 MB | the history every forecast rebuilds features from |
| `data/processed/forecasts.parquet` | 142 KB | tier-1 batch: 520 series × 30 days |
| `data/raw/holidays_events.csv` | 21.8 KB | the model reads 3 holiday features; without this file they zero out |

Only the bulky raw Kaggle download (`train.csv`, `oil.csv`, `stores.csv`, …)
remains gitignored — it is needed to *retrain*, not to *serve*. The committed
`features.parquet` also carries the clean sales columns, so the loader falls
back to it when `sales_clean.parquet` is absent.

`_data.py` detects `features.parquet` together with `lgbm_v1.joblib` and
switches off demo mode automatically, so deploying is `git clone` →
`pip install -r requirements.txt` → `streamlit run app/main.py`.

The Phase 1–3 pipeline **has** been executed end-to-end on the real Favorita
data. Scores on the sealed test window (2017-07-31 → 2017-08-15, 8,320
series-days never seen in training; training window 2013-01-01 → 2017-07-14):

| Metric | LightGBM | Seasonal naive (last week) |
|---|---|---|
| **WAPE** (headline) | **10.5%** | 20.9% |
| RMSE | 155.6 | 335.5 |
| MAE | 54.1 | 108.0 |

The model roughly halves the naive baseline's error on held-out days.

Two deployment costs worth knowing:

* A holidayless clone is not merely slower, it is **wrong**: with
  `holidays_events.csv` absent the 30-day forecast moves ~16% on a per-series
  basis (up to 50% on individual days) and **−17% in total volume** — enough to
  corrupt every KPI. That is why the 21.8 KB file is committed.
* Retraining invalidates the tier-1 batch; refresh it with
  `python -m src.models.forecast` (~7 minutes for the full scope).

### Real-scope performance

The three-tier forecast resolution is what keeps the UI responsive, and the
tier that carries full scale is **tier 1**: the precomputed
`data/processed/forecasts.parquet` written by `python -m src.models.forecast`.
The canonical-recursion tier is capped at `CANONICAL_MAX_SERIES = 60` series,
which fits the 18-series demo corpus but not the real scope
(20 stores × 26 families = **520 series**), where requests fall through to
tier 3 (direct subset computation).

Measured on the real artifacts: the recursion costs **~0.72 s per series**, so a
full-scope request needs **~6.3 minutes** — long enough that a hosted landing
page would time out. With the committed tier-1 batch the same request is a mask
over ~15.6k rows. Measured end-to-end on the real scope: **cold first paint
3.05 s** (0.7 s to load the bundle, 2.4 s across forecast + history + inventory
+ transfers + deadstock), then **~0.004 s** for every subsequent render. Narrow
scopes (a handful of stores/families) were already fine without tier 1.

A promotion scenario (`future_onpromotion`) intentionally bypasses tier 1: a
precomputed batch cannot answer a what-if, so those requests pay the recursion
cost for the selected subset only.

## Tests

```powershell
python -m pytest tests/ -v
```

The feature suite includes explicit **leakage tests**: mutating `sales[T]` (or
everything from a cutoff onward) must leave every earlier row's engineered
features bit-for-bit identical.

The model suite (`tests/test_models.py`) covers metric correctness on
hand-computed values, training determinism, Optuna search-space bounds,
baseline comparison, the recursive-forecast invariants above, and Joblib
round-tripping. It runs against a synthetic panel pushed through the real
Phase 2 pipeline, so it needs no Kaggle data:

```powershell
python -m pytest tests/test_models.py -q
```

The Phase 4 suite (`tests/test_engines.py`) exercises safety stock at the exact
reorder-point boundary, zero-demand and zero-deadstock edge cases, transfer
guardrails (min units, per-leg caps, donor-reserve protection, net-value
filtering), affinity logic (residual correlation), and end-to-end pipeline
integration — all with fully synthetic data:

```powershell
python -m pytest tests/test_engines.py -q
```

The dashboard suite (`tests/test_app.py`) executes every page headlessly, checks
that the demo corpus really contains donors, receivers, a stalled line and a
stock-out, and exercises the interactions that matter (horizon toggle, promotion
scenario, service-level slider, transfer approval + reset, narrow scope):

```powershell
python -m pytest tests/test_app.py -v
```

Because the data layer is process-cached, the expensive forecast is built once
and the entire suite runs in ~45 s. Current totals: **269 tests** (8 cleaning,
31 features, 61 models, 128 engines, 41 dashboard).

The dashboard suite is **hermetic**: `tests/conftest.py` exports
`SDP_FORCE_DEMO=1`, which pins `app/_data.py` to the synthetic corpus even when
the real Phase 1–3 artifacts are present. `test_dashboard_suite_is_hermetic`
enforces that pin, because the alternative is a suite whose assertions (and
runtime) silently depend on whether you happen to have run the pipeline.

`SDP_FORCE_DEMO=1` is equally useful outside pytest — e.g. to demo the UI
without loading the real parquet:

```powershell
$env:SDP_FORCE_DEMO=1; streamlit run app/main.py
```

## Layout

```
data/raw/         Kaggle CSVs (gitignored)
data/processed/   sales_clean.parquet etc. (gitignored)
models/           trained artifacts (gitignored)
src/config.py     single source of truth for paths/params/scope
src/data/         ingestion & cleaning
src/features/     feature engineering (Phase 2)
src/models/       training, tuning, inference (Phase 3)
src/engines/      ROP/SS, transfers, deadstock (Phase 4)
app/_data.py      cached data access (bundle, forecasts, engines)
app/_ui.py        shared theme, KPI cards, formatters, sidebar widgets
app/main.py       executive overview page
app/pages/        1 forecast · 2 inventory · 3 transfers · 4 clearance
notebooks/        EDA only
tests/            pytest suites
scripts/          operational scripts
```

## Scope configuration

Tune the demo subset in `src/config.py`: `SCOPE_STORE_NBR` (stores),
`EXCLUDED_FAMILIES` (dropped families), and the business/engine defaults.
