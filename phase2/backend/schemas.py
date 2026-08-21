from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field


class FilterOptions(BaseModel):
    skus: List[str]
    regions: List[str]
    channels: List[str]
    categories: List[str]
    brands: List[str]
    segments: List[str]
    pack_types: List[str]
    min_date: date
    max_date: date


class SalesFilters(BaseModel):
    """Shared query filters. All optional — an empty list/None means 'no filter'."""
    skus: Optional[List[str]] = None
    regions: Optional[List[str]] = None
    channels: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    brands: Optional[List[str]] = None
    segments: Optional[List[str]] = None
    pack_types: Optional[List[str]] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    promotion_only: Optional[bool] = None


class SummaryKPIs(BaseModel):
    total_units_sold: float
    total_revenue: float
    avg_price_unit: float
    avg_daily_units: float
    promo_share_pct: float
    stockout_risk_days_pct: float
    n_rows: int
    date_range: List[date]


class TrendPoint(BaseModel):
    period: str
    units_sold: float
    revenue: float
    promo_units_sold: Optional[float] = None
    non_promo_units_sold: Optional[float] = None


class ComparisonRow(BaseModel):
    group: str
    units_sold: float
    revenue: float
    avg_price_unit: float
    n_rows: int


class InventoryRow(BaseModel):
    group: str
    avg_stock_available: float
    avg_units_sold: float
    avg_delivered_qty: float
    avg_delivery_days: float
    stockout_risk_pct: float


class PromotionRow(BaseModel):
    group: str
    promo_avg_units: float
    non_promo_avg_units: float
    uplift_pct: float


class ForecastRequest(BaseModel):
    target_date: date
    sku: str
    region: str
    channel: str
    price_unit: float = Field(..., gt=0)
    promotion_flag: bool
    delivery_days: float = Field(..., ge=0)
    stock_available: float = Field(..., ge=0)
    delivered_qty: float = Field(..., ge=0)
    pack_type: Optional[str] = None


class ForecastResponse(BaseModel):
    date: str
    sku: str
    region: str
    channel: str
    predicted_units_sold: float


class ModelInfo(BaseModel):
    final_model: str
    final_model_test_rmse: float
    final_model_test_r2: float
    final_model_test_mae: float
    final_model_test_mape_pct: float
    best_baseline: str
    best_baseline_test_rmse: float
    ml_beats_baseline: bool

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: List[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    tool_calls: List[dict] = Field(default_factory=list)
