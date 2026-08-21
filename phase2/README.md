# FMCG Sales Analytics — Phase 2

Turns the Phase 1 FMCG forecasting pipeline (cleaning → Snowflake schema →
SQL EDA → feature engineering → XGBoost model, Test RMSE 7.83 / MAE 5.41 /
R² 0.515) into a usable business application: a queryable sales warehouse
behind a REST API, with a Streamlit UI for exploring sales, comparing
brands/products, checking inventory & promotions, and generating forecasts.

Phase 1 artifacts are used **as-is** — no retraining, no changed features,
no changed metrics. This layer is purely: warehouse → API → UI on top of
what Phase 1 already produced.

```
                    FMCG DATA (Phase 1 fmcg_features.csv)
                       │
                       ↓
              DuckDB warehouse (Snowflake-schema-identical)
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
        SQL Analytics       ML Prediction
        (analytics.py)      (forecasting.py, Phase 1's XGBoost pipeline)
             │                   │
             └─────────┬─────────┘
                       ↓
                    FastAPI  (backend/main.py)
                       │
                       ↓
                  Streamlit UI (frontend/app.py)
```

## Why DuckDB instead of live Snowflake

This environment has no Snowflake credentials. `warehouse/build_warehouse.py`
builds a local DuckDB database using **the exact same star schema** as
Phase 1's `sql/01_snowflake_ddl.sql` (`DIM_DATE`, `DIM_PRODUCT`,
`DIM_REGION`, `DIM_CHANNEL`, `DIM_PACK_TYPE`, `FACT_SALES`, same grain and
column names). Every query in `backend/analytics.py` is standard SQL that
runs unchanged against real Snowflake — swap the connection object in
`backend/database.py` for `snowflake-connector-python` and nothing else
needs to change.

## Project layout

```
phase2/
├── data/
│   ├── fmcg_features.csv        # Phase 1 output, copied verbatim (source of truth)
│   └── fmcg_warehouse.duckdb    # built by warehouse/build_warehouse.py (gitignore this)
├── warehouse/
│   └── build_warehouse.py       # builds the DuckDB star schema from fmcg_features.csv
├── backend/
│   ├── main.py                  # FastAPI app — all HTTP endpoints
│   ├── analytics.py             # SQL analytics queries (trend, compare, inventory, promo)
│   ├── forecasting.py           # Phase 1's predict_final.py logic, wired into the API
│   ├── database.py              # DuckDB connection helper
│   ├── schemas.py               # Pydantic request/response models
│   ├── config.py                # paths
│   ├── models/                  # copied Phase 1 model + feature artifacts
│   │   ├── final_best_model_pipeline.joblib
│   │   ├── model_features.json
│   │   ├── sku_attributes.json
│   │   ├── fmcg_features.parquet   # per-series history, used to build lag/rolling features
│   │   └── final_model_selection.json
│   └── requirements.txt
├── frontend/
│   ├── app.py                   # Streamlit entry point + sidebar filters
│   ├── pages_impl.py            # page implementations (Overview, Trends, Compare, etc.)
│   ├── api_client.py            # thin HTTP client to the FastAPI backend
│   └── requirements.txt
├── tests/
│   ├── test_warehouse.py        # DuckDB build integrity (row counts, joins, totals)
│   ├── test_backend.py          # API correctness vs. independent pandas computation
│   └── test_frontend.py         # Streamlit AppTest — every page, filters, forecast button
└── README.md
```

## Setup

```bash
# 1. Backend
cd backend
pip install -r requirements.txt

# 2. Build the warehouse (run once, or whenever data/fmcg_features.csv changes)
cd ..
python warehouse/build_warehouse.py

# 3. Start the API
cd backend
uvicorn main:app --reload --port 8000
# → docs at http://localhost:8000/docs

# 4. Frontend (separate terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py
# → app at http://localhost:8501
```

The frontend looks for the API at `http://localhost:8000` by default;
override with `FMCG_API_BASE` if you deploy the backend elsewhere.

## What the app does

- **Overview** — headline KPIs (units sold, revenue, promo share, stockout-risk
  days), a daily trend chart, top categories/brands, and a collapsed Phase 1
  model performance card.
- **Sales Trends** — day/week/month time series for units sold, revenue, and
  promo vs non-promo split.
- **Compare** — rank SKUs/brands/categories/regions/channels/segments/pack
  types by units sold or revenue.
- **Inventory & Promotions** — average stock vs units sold, a stockout-risk
  proxy (`units_sold > stock_available`) by dimension, and promo uplift
  (% difference in average units sold on promo vs non-promo days).
- **Explore Data** — filtered raw rows from `FACT_SALES`, with CSV download.
- **Forecast** — pick a SKU × Region × Channel series, prefilled with that
  series' most recent actuals; enter tomorrow's planned price/promo/stock/
  delivery and get a predicted `units_sold` from the Phase 1 XGBoost model.
  Lag/rolling features (`lag_1/7/14/28`, `roll_mean_*`, etc.) are computed
  automatically from stored history — nothing to type in by hand.

All filters in the sidebar (date range, category, brand, region, channel,
SKU, segment, promo status) apply across every analytics page.

## Running the tests

```bash
pip install -r tests/requirements.txt   # on top of backend/requirements.txt

# Warehouse integrity (needs the warehouse already built)
pytest tests/test_warehouse.py -v

# Backend correctness — no server needed, uses FastAPI's TestClient
pytest tests/test_backend.py -v

# Frontend — needs a live backend (these are integration tests)
uvicorn backend.main:app --port 8000 &
FMCG_API_BASE=http://localhost:8000 pytest tests/test_frontend.py -v
```

`test_backend.py` cross-checks every analytics endpoint (`summary`, `trend`,
`compare`, `inventory`, `promotions`) against an independent pandas
computation over the same source CSV, and locks the forecast endpoint to
Phase 1's documented reference prediction so a future change to feature
construction would be caught immediately. `test_frontend.py` uses
Streamlit's `AppTest` harness to actually render every page (not just check
the HTTP status of the app), click the Forecast page's "Generate Forecast"
button, and verify a sidebar filter produces matching numbers on two
different pages.

## Notes on the forecasting endpoint

`backend/forecasting.py` is Phase 1's `predict_final.py` unchanged in logic
— same feature construction, same `np.clip(pred, 0, None)`, same model file.
It requires at least 28 days of prior history for the requested
(sku, region, channel) series (matching the model's `lag_28` feature); the
UI surfaces this as an error if you pick a date too close to a series'
launch date.
