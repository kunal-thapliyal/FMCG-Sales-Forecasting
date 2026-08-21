"""
Backend test suite.

Run from the phase2/ directory:
    pytest tests/test_backend.py -v

Requires the DuckDB warehouse to already be built:
    python warehouse/build_warehouse.py

Uses FastAPI's TestClient — no running server needed, and no network calls.
Correctness tests cross-check API output against independent pandas
computation over the same source CSV, so a regression in the SQL (wrong
join, wrong aggregation, wrong filter) would be caught here, not just a
"does it 200" check.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402

client = TestClient(app)


@pytest.fixture(scope="module")
def raw_df():
    df = pd.read_csv(DATA_DIR / "fmcg_features.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


# --------------------------------------------------------------------------
# Health / meta
# --------------------------------------------------------------------------

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_meta_filters_shape():
    r = client.get("/meta/filters")
    assert r.status_code == 200
    body = r.json()
    for key in ("skus", "regions", "channels", "categories", "brands", "segments", "pack_types", "min_date", "max_date"):
        assert key in body
    assert len(body["skus"]) == 30
    assert len(body["regions"]) == 3
    assert len(body["channels"]) == 3


def test_model_info_matches_phase1():
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["final_model"] == "XGBoost"
    # Locked to Phase 1's reported metrics — this test fails if the model
    # artifact or selection file is ever swapped for a retrained one.
    assert body["final_model_test_rmse"] == pytest.approx(7.833759639957998, abs=1e-6)
    assert body["final_model_test_mae"] == pytest.approx(5.41225363586398, abs=1e-6)
    assert body["final_model_test_r2"] == pytest.approx(0.5149556828372448, abs=1e-6)


# --------------------------------------------------------------------------
# /sales/summary — correctness against independent pandas computation
# --------------------------------------------------------------------------

def test_summary_no_filters_matches_pandas(raw_df):
    r = client.get("/sales/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["total_units_sold"] == pytest.approx(raw_df["units_sold"].sum(), rel=1e-6)
    assert body["total_revenue"] == pytest.approx((raw_df["units_sold"] * raw_df["price_unit"]).sum(), rel=1e-6)
    assert body["n_rows"] == len(raw_df)


def test_summary_category_filter_matches_pandas(raw_df):
    sub = raw_df[raw_df["category"] == "Milk"]
    r = client.get("/sales/summary", params={"categories": "Milk"})
    assert r.status_code == 200
    body = r.json()
    assert body["total_units_sold"] == pytest.approx(sub["units_sold"].sum(), rel=1e-6)
    assert body["n_rows"] == len(sub)


def test_summary_empty_selection_returns_zeroed_response():
    """Conflicting filters (a Juice SKU under the Milk category) match zero
    rows. This must return a well-formed zeroed response, not a 500."""
    r = client.get("/sales/summary", params={"skus": "MI-006", "categories": "Juice"})
    assert r.status_code == 200
    body = r.json()
    assert body["n_rows"] == 0
    assert body["total_units_sold"] == 0.0
    assert body["date_range"] == []


def test_summary_inverted_date_range_returns_zeroed_response():
    r = client.get("/sales/summary", params={"start_date": "2024-12-31", "end_date": "2022-01-01"})
    assert r.status_code == 200
    assert r.json()["n_rows"] == 0


def test_summary_malformed_date_returns_422():
    r = client.get("/sales/summary", params={"start_date": "not-a-date"})
    assert r.status_code == 422


# --------------------------------------------------------------------------
# /sales/trend
# --------------------------------------------------------------------------

def test_trend_monthly_matches_pandas(raw_df):
    expected = (
        raw_df.assign(month=raw_df["date"].values.astype("datetime64[M]"))
        .groupby("month")["units_sold"].sum()
        .sort_index()
    )
    r = client.get("/sales/trend", params={"granularity": "month"})
    assert r.status_code == 200
    df = pd.DataFrame(r.json())
    df["period"] = pd.to_datetime(df["period"])
    got = df.set_index("period")["units_sold"].sort_index()
    assert len(got) == len(expected)
    assert (got.values - expected.values == 0).all()


def test_trend_invalid_granularity_returns_422():
    r = client.get("/sales/trend", params={"granularity": "year"})
    assert r.status_code == 422


def test_trend_empty_selection_returns_empty_list():
    r = client.get("/sales/trend", params={"skus": "MI-006", "categories": "Juice"})
    assert r.status_code == 200
    assert r.json() == []


# --------------------------------------------------------------------------
# /sales/compare
# --------------------------------------------------------------------------

def test_compare_by_brand_matches_pandas(raw_df):
    expected = raw_df.groupby("brand")["units_sold"].sum().sort_values(ascending=False).head(5)
    r = client.get("/sales/compare", params={"dimension": "brand", "top_n": 5})
    assert r.status_code == 200
    got = {row["group"]: row["units_sold"] for row in r.json()}
    assert set(got) == set(expected.index)
    for brand, val in expected.items():
        assert got[brand] == pytest.approx(val, rel=1e-6)


def test_compare_invalid_dimension_returns_422():
    r = client.get("/sales/compare", params={"dimension": "not_a_real_dimension"})
    assert r.status_code == 422


# --------------------------------------------------------------------------
# /inventory/analysis and /promotions/analysis
# --------------------------------------------------------------------------

def test_inventory_stockout_risk_matches_pandas(raw_df):
    sub = raw_df.assign(stockout=raw_df["units_sold"] > raw_df["stock_available"])
    expected = 100 * sub.groupby("category")["stockout"].mean()
    r = client.get("/inventory/analysis", params={"dimension": "category"})
    assert r.status_code == 200
    got = {row["group"]: row["stockout_risk_pct"] for row in r.json()}
    for cat, val in expected.items():
        assert got[cat] == pytest.approx(val, rel=1e-6, abs=1e-9)


def test_promotion_uplift_matches_pandas(raw_df):
    promo_avg = raw_df[raw_df["promotion_flag"] == 1].groupby("category")["units_sold"].mean()
    r = client.get("/promotions/analysis", params={"dimension": "category"})
    assert r.status_code == 200
    got = {row["group"]: row["promo_avg_units"] for row in r.json()}
    for cat, val in promo_avg.items():
        assert got[cat] == pytest.approx(val, rel=1e-6)


# --------------------------------------------------------------------------
# /sales/table — pagination must be deterministic (no dupes/gaps across pages)
# --------------------------------------------------------------------------

def test_table_pagination_has_no_overlap():
    seen = set()
    for offset in range(0, 50, 5):
        r = client.get("/sales/table", params={"limit": 5, "offset": offset})
        assert r.status_code == 200
        rows = r.json()
        for row in rows:
            key = (row["date"], row["sku"], row["region"], row["channel"], row.get("pack_type"))
            assert key not in seen, f"Duplicate row across pages at offset {offset}: {key}"
            seen.add(key)
    assert len(seen) == 50


def test_table_limit_boundary():
    assert client.get("/sales/table", params={"limit": 5000}).status_code == 200
    assert client.get("/sales/table", params={"limit": 5001}).status_code == 422


# --------------------------------------------------------------------------
# /forecast/* — the Phase 1 model, unchanged
# --------------------------------------------------------------------------

def test_forecast_series_lists_all_270_combinations():
    r = client.get("/forecast/series")
    assert r.status_code == 200
    assert len(r.json()) == 270  # 30 SKUs x 3 regions x 3 channels


def test_forecast_last_known_valid_combo():
    r = client.get("/forecast/last-known", params={"sku": "MI-006", "region": "PL-Central", "channel": "Retail"})
    assert r.status_code == 200
    body = r.json()
    assert body["n_history_days"] > 0


def test_forecast_last_known_unknown_combo_returns_404():
    r = client.get("/forecast/last-known", params={"sku": "ZZ-999", "region": "PL-Central", "channel": "Retail"})
    assert r.status_code == 404


def test_forecast_predict_matches_phase1_reference_case():
    """This exact input/output pair is documented in Phase 1's
    predict_final.py docstring — locking it here guards against any future
    change to feature construction silently changing predictions."""
    payload = {
        "target_date": "2025-01-02", "sku": "MI-006", "region": "PL-Central", "channel": "Retail",
        "price_unit": 5.20, "promotion_flag": True, "delivery_days": 3,
        "stock_available": 160, "delivered_qty": 180,
    }
    r = client.post("/forecast/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_units_sold"] == pytest.approx(22.6, abs=0.05)


def test_forecast_predict_insufficient_history_returns_400():
    payload = {
        "target_date": "2022-03-01", "sku": "MI-006", "region": "PL-Central", "channel": "Retail",
        "price_unit": 5.20, "promotion_flag": True, "delivery_days": 3,
        "stock_available": 160, "delivered_qty": 180,
    }
    r = client.post("/forecast/predict", json=payload)
    assert r.status_code == 400
    assert "Not enough history" in r.json()["detail"]


def test_forecast_predict_unknown_series_returns_400():
    payload = {
        "target_date": "2024-12-15", "sku": "ZZ-999", "region": "PL-Central", "channel": "Retail",
        "price_unit": 5.20, "promotion_flag": True, "delivery_days": 3,
        "stock_available": 160, "delivered_qty": 180,
    }
    r = client.post("/forecast/predict", json=payload)
    assert r.status_code == 400


def test_forecast_predict_rejects_non_positive_price():
    payload = {
        "target_date": "2024-12-15", "sku": "MI-006", "region": "PL-Central", "channel": "Retail",
        "price_unit": -5.0, "promotion_flag": True, "delivery_days": 3,
        "stock_available": 160, "delivered_qty": 180,
    }
    r = client.post("/forecast/predict", json=payload)
    assert r.status_code == 422
