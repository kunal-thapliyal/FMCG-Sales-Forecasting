"""
FMCG Sales Analytics — FastAPI backend.

Run with:
    uvicorn main:app --reload --port 8000
(from inside the backend/ directory, so the relative imports resolve)

Endpoints are grouped into:
    /meta/*        static filter options for the UI
    /sales/*       SQL analytics over FACT_SALES (trend, comparison, table)
    /inventory/*   stock/delivery analytics
    /promotions/*  promo-lift analytics
    /forecast/*    wraps the Phase 1 XGBoost model
    /model/info    Phase 1 model card (metrics, unchanged from training)
"""
import json
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

import analytics
import forecasting
from config import MODEL_SELECTION_PATH
from database import get_connection
from schemas import (
    ChatRequest,
    ChatResponse,
    ComparisonRow,
    FilterOptions,
    ForecastRequest,
    ForecastResponse,
    InventoryRow,
    ModelInfo,
    PromotionRow,
    SalesFilters,
    SummaryKPIs,
    TrendPoint,
)

# `agent` (Phase 3 chatbot) is imported lazily inside the /chat endpoint,
# not at module load time. Its only hard dependency, GROQ_API_KEY, is an
# environment variable read at call time — importing agent.py itself never
# requires it — but importing lazily keeps the chatbot fully decoupled from
# app startup: if agent.py or its packages (groq/langgraph) ever fail to
# import for an unrelated reason, the rest of this API still starts and
# every existing Phase 1/Phase 2 endpoint keeps working.

app = FastAPI(
    title="FMCG Sales Analytics API",
    description="Phase 2 API — SQL analytics + ML forecasting over the Phase 1 FMCG warehouse.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _filters_from_query(
    skus: Optional[List[str]] = Query(None),
    regions: Optional[List[str]] = Query(None),
    channels: Optional[List[str]] = Query(None),
    categories: Optional[List[str]] = Query(None),
    brands: Optional[List[str]] = Query(None),
    segments: Optional[List[str]] = Query(None),
    pack_types: Optional[List[str]] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    promotion_only: Optional[bool] = Query(None),
) -> SalesFilters:
    try:
        return SalesFilters(
            skus=skus, regions=regions, channels=channels, categories=categories,
            brands=brands, segments=segments, pack_types=pack_types,
            start_date=start_date, end_date=end_date, promotion_only=promotion_only,
        )
    except ValidationError as e:
        # Surface bad query params (e.g. malformed dates) as a clean 422
        # instead of an uncaught 500.
        raise HTTPException(status_code=422, detail=e.errors())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta/filters", response_model=FilterOptions)
def meta_filters():
    con = get_connection()
    try:
        return analytics.get_filter_options(con)
    finally:
        con.close()


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    if not MODEL_SELECTION_PATH.exists():
        raise HTTPException(status_code=404, detail="Model selection file not found.")
    with open(MODEL_SELECTION_PATH) as f:
        return json.load(f)


@app.get("/sales/summary", response_model=SummaryKPIs)
def get_sales_summary(filters: SalesFilters = Depends(_filters_from_query)):
    con = get_connection()
    try:
        return analytics.get_summary_kpis(con, filters)
    finally:
        con.close()


@app.get("/sales/trend", response_model=List[TrendPoint])
def get_sales_trend(
    granularity: str = Query("day", pattern="^(day|week|month)$"),
    filters: SalesFilters = Depends(_filters_from_query),
):
    con = get_connection()
    try:
        df = analytics.get_sales_trend(con, filters, granularity)
        df["period"] = df["period"].astype(str)
        return df.to_dict(orient="records")
    finally:
        con.close()


@app.get("/sales/compare", response_model=List[ComparisonRow])
def get_sales_compare(
    dimension: str = Query("category", pattern="^(sku|region|channel|category|brand|segment|pack_type)$"),
    top_n: int = Query(15, ge=1, le=100),
    filters: SalesFilters = Depends(_filters_from_query),
):
    con = get_connection()
    try:
        df = analytics.get_comparison(con, filters, dimension, top_n)
        return df.to_dict(orient="records")
    finally:
        con.close()


@app.get("/sales/table")
def get_sales_table(
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    filters: SalesFilters = Depends(_filters_from_query),
):
    con = get_connection()
    try:
        df = analytics.get_sales_table(con, filters, limit, offset)
        df["date"] = df["date"].astype(str)
        return df.to_dict(orient="records")
    finally:
        con.close()


@app.get("/inventory/analysis", response_model=List[InventoryRow])
def get_inventory_analysis(
    dimension: str = Query("category", pattern="^(sku|region|channel|category|brand)$"),
    filters: SalesFilters = Depends(_filters_from_query),
):
    con = get_connection()
    try:
        df = analytics.get_inventory_analysis(con, filters, dimension)
        return df.to_dict(orient="records")
    finally:
        con.close()


@app.get("/promotions/analysis", response_model=List[PromotionRow])
def get_promotion_analysis(
    dimension: str = Query("category", pattern="^(sku|region|channel|category|brand)$"),
    filters: SalesFilters = Depends(_filters_from_query),
):
    con = get_connection()
    try:
        df = analytics.get_promotion_analysis(con, filters, dimension)
        return df.to_dict(orient="records")
    finally:
        con.close()


@app.get("/forecast/series")
def forecast_series():
    """All (sku, region, channel) combinations available for forecasting."""
    return forecasting.list_available_series()


@app.get("/forecast/last-known")
def forecast_last_known(sku: str, region: str, channel: str):
    result = forecasting.get_last_known_operational_values(sku, region, channel)
    if result is None:
        raise HTTPException(status_code=404, detail="No history found for that (sku, region, channel).")
    return result


@app.post("/forecast/predict", response_model=ForecastResponse)
def forecast_predict(req: ForecastRequest):
    try:
        return forecasting.predict_units_sold(
            target_date=req.target_date, sku=req.sku, region=req.region, channel=req.channel,
            price_unit=req.price_unit, promotion_flag=req.promotion_flag,
            delivery_days=req.delivery_days, stock_available=req.stock_available,
            delivered_qty=req.delivered_qty, pack_type=req.pack_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """AI FMCG Analyst — natural-language Q&A over the existing analytics and
    forecasting endpoints. Requires GROQ_API_KEY to be set on the server; if
    it isn't, returns a normal 200 response explaining that the chat isn't
    configured (never a 500 — this must not affect the rest of the API)."""
    import agent  # lazy import, see note above
    # return agent.run_chat(message=req.message, history=[h.model_dump() for h in req.history])
    return agent.run_chat(
        message=req.message,
        history=req.history,
    )