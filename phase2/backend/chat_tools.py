"""
Phase 3 — tool layer for the AI FMCG Analyst chatbot.

Every function in this module is a thin wrapper around the EXISTING
analytics.py / forecasting.py functions that already back the Phase 2
REST API. No SQL, no model logic, and no business rules are duplicated
here — each wrapper builds a `SalesFilters` object (the same one
`main.py` builds from query params) and calls straight into the
already-tested analytics/forecasting layer, then returns a plain dict
that's easy for an LLM to reason about and safe to json.dumps().

Design rules (per project instructions):
- Never raise past this layer. Every function returns either the
  requested data or {"error": "..."} so the LLM can explain a failure
  instead of the app crashing or the model inventing an answer.
- Categorical entities (sku/region/channel/category/brand/segment/
  pack_type) are validated against `/meta/filters`-equivalent metadata
  before hitting the database, so the LLM gets a clear "not a valid
  SKU" message instead of a silently-empty result.
- `target_date` for forecasting is validated only for parseability —
  it is explicitly allowed (and expected) to be beyond the historical
  max_date, since forecasting the future is the entire point of
  /forecast/predict. We do not reject future dates here.
"""
import json
from datetime import date, datetime
from typing import List, Optional

import pandas as pd

import analytics
import forecasting
from database import get_connection
from schemas import SalesFilters

DIMENSION_CHOICES = ["sku", "region", "channel", "category", "brand", "segment", "pack_type"]
INVENTORY_PROMO_DIMENSIONS = ["sku", "region", "channel", "category", "brand"]


def _get_meta() -> dict:
    con = get_connection()
    try:
        return analytics.get_filter_options(con)
    finally:
        con.close()


def _validate_values(field_name: str, values: Optional[List[str]], valid_set: set) -> Optional[str]:
    """Returns an error string if any value isn't in the known metadata set, else None."""
    if not values:
        return None
    bad = [v for v in values if v not in valid_set]
    if bad:
        return f"Unknown {field_name}: {bad}. Call get_metadata to see valid values."
    return None


def _validate_filters(meta: dict, skus=None, regions=None, channels=None, categories=None,
                       brands=None, segments=None, pack_types=None) -> Optional[str]:
    checks = [
        ("sku", skus, set(meta["skus"])),
        ("region", regions, set(meta["regions"])),
        ("channel", channels, set(meta["channels"])),
        ("category", categories, set(meta["categories"])),
        ("brand", brands, set(meta["brands"])),
        ("segment", segments, set(meta["segments"])),
        ("pack_type", pack_types, set(meta["pack_types"])),
    ]
    for name, values, valid_set in checks:
        err = _validate_values(name, values, valid_set)
        if err:
            return err
    return None


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid date '{s}'. Use YYYY-MM-DD.")


def _df_to_records(df: pd.DataFrame, date_cols=None) -> list:
    if df.empty:
        return []
    df = df.copy()
    for col in (date_cols or []):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def get_metadata() -> dict:
    """Returns the valid SKUs, regions, channels, categories, brands, segments,
    pack_types, and the historical date range covered by the dataset."""
    try:
        meta = _get_meta()
        return {
            "skus": meta["skus"], "regions": meta["regions"], "channels": meta["channels"],
            "categories": meta["categories"], "brands": meta["brands"], "segments": meta["segments"],
            "pack_types": meta["pack_types"],
            "min_date": str(meta["min_date"]), "max_date": str(meta["max_date"]),
            "note": "Historical data only covers min_date..max_date. Forecasts can target dates after max_date.",
        }
    except Exception as e:
        return {"error": f"Could not load metadata: {e}"}


def get_sales_summary(skus=None, regions=None, channels=None, categories=None, brands=None,
                       segments=None, pack_types=None, start_date=None, end_date=None,
                       promotion_only=None) -> dict:
    """Summary KPIs (total units sold, revenue, avg price, promo share, stockout risk)
    for the given filters. Any filter left as None means 'no restriction on that dimension'."""
    try:
        meta = _get_meta()
        err = _validate_filters(meta, skus, regions, channels, categories, brands, segments, pack_types)
        if err:
            return {"error": err}
        filters = SalesFilters(
            skus=skus, regions=regions, channels=channels, categories=categories,
            brands=brands, segments=segments, pack_types=pack_types,
            start_date=_parse_date(start_date), end_date=_parse_date(end_date),
            promotion_only=promotion_only,
        )
        con = get_connection()
        try:
            result = analytics.get_summary_kpis(con, filters)
        finally:
            con.close()
        result["date_range"] = [str(d) for d in result["date_range"]]
        return result
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Sales summary lookup failed: {e}"}


def get_sales_trend(granularity="day", skus=None, regions=None, channels=None, categories=None,
                     brands=None, segments=None, pack_types=None, start_date=None, end_date=None,
                     promotion_only=None) -> dict:
    """Sales trend over time (units_sold, revenue per period) at day/week/month granularity."""
    try:
        if granularity not in ("day", "week", "month"):
            return {"error": "granularity must be one of: day, week, month"}
        meta = _get_meta()
        err = _validate_filters(meta, skus, regions, channels, categories, brands, segments, pack_types)
        if err:
            return {"error": err}
        filters = SalesFilters(
            skus=skus, regions=regions, channels=channels, categories=categories,
            brands=brands, segments=segments, pack_types=pack_types,
            start_date=_parse_date(start_date), end_date=_parse_date(end_date),
            promotion_only=promotion_only,
        )
        con = get_connection()
        try:
            df = analytics.get_sales_trend(con, filters, granularity)
        finally:
            con.close()
        return {"granularity": granularity, "points": _df_to_records(df, date_cols=["period"])}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Sales trend lookup failed: {e}"}


def compare_sales(dimension="category", top_n=10, skus=None, regions=None, channels=None,
                   categories=None, brands=None, segments=None, pack_types=None,
                   start_date=None, end_date=None, promotion_only=None) -> dict:
    """Compares units sold / revenue across a dimension (sku, region, channel, category,
    brand, segment, or pack_type), ranked highest to lowest. Use this for questions like
    'which category sold the most' or 'compare Retail and E-commerce'."""
    try:
        if dimension not in DIMENSION_CHOICES:
            return {"error": f"dimension must be one of {DIMENSION_CHOICES}"}
        meta = _get_meta()
        err = _validate_filters(meta, skus, regions, channels, categories, brands, segments, pack_types)
        if err:
            return {"error": err}
        filters = SalesFilters(
            skus=skus, regions=regions, channels=channels, categories=categories,
            brands=brands, segments=segments, pack_types=pack_types,
            start_date=_parse_date(start_date), end_date=_parse_date(end_date),
            promotion_only=promotion_only,
        )
        con = get_connection()
        try:
            df = analytics.get_comparison(con, filters, dimension, min(max(int(top_n), 1), 100))
        finally:
            con.close()
        return {"dimension": dimension, "ranked": _df_to_records(df)}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Sales comparison failed: {e}"}


def get_sales_table(skus=None, regions=None, channels=None, categories=None, brands=None,
                     segments=None, pack_types=None, start_date=None, end_date=None,
                     promotion_only=None, limit=50) -> dict:
    """Returns raw row-level sales records matching the filters (capped at 200 rows —
    use narrower filters or get_sales_summary/compare_sales for aggregate questions)."""
    try:
        meta = _get_meta()
        err = _validate_filters(meta, skus, regions, channels, categories, brands, segments, pack_types)
        if err:
            return {"error": err}
        filters = SalesFilters(
            skus=skus, regions=regions, channels=channels, categories=categories,
            brands=brands, segments=segments, pack_types=pack_types,
            start_date=_parse_date(start_date), end_date=_parse_date(end_date),
            promotion_only=promotion_only,
        )
        con = get_connection()
        try:
            df = analytics.get_sales_table(con, filters, limit=min(max(int(limit), 1), 200), offset=0)
        finally:
            con.close()
        return {"rows": _df_to_records(df, date_cols=["date"])}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Sales table lookup failed: {e}"}


def get_inventory_analysis(dimension="category", skus=None, regions=None, channels=None,
                            categories=None, brands=None, segments=None, pack_types=None,
                            start_date=None, end_date=None, promotion_only=None) -> dict:
    """Inventory analytics (avg stock available, avg delivered qty, avg delivery days,
    stockout risk %) grouped by a dimension (sku, region, channel, category, or brand)."""
    try:
        if dimension not in INVENTORY_PROMO_DIMENSIONS:
            return {"error": f"dimension must be one of {INVENTORY_PROMO_DIMENSIONS}"}
        meta = _get_meta()
        err = _validate_filters(meta, skus, regions, channels, categories, brands, segments, pack_types)
        if err:
            return {"error": err}
        filters = SalesFilters(
            skus=skus, regions=regions, channels=channels, categories=categories,
            brands=brands, segments=segments, pack_types=pack_types,
            start_date=_parse_date(start_date), end_date=_parse_date(end_date),
            promotion_only=promotion_only,
        )
        con = get_connection()
        try:
            df = analytics.get_inventory_analysis(con, filters, dimension)
        finally:
            con.close()
        return {"dimension": dimension, "groups": _df_to_records(df)}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Inventory analysis failed: {e}"}


def get_promotion_analysis(dimension="category", skus=None, regions=None, channels=None,
                            categories=None, brands=None, segments=None, pack_types=None,
                            start_date=None, end_date=None) -> dict:
    """Promotion-lift analytics (avg units sold on promo vs non-promo days, uplift %)
    grouped by a dimension (sku, region, channel, category, or brand)."""
    try:
        if dimension not in INVENTORY_PROMO_DIMENSIONS:
            return {"error": f"dimension must be one of {INVENTORY_PROMO_DIMENSIONS}"}
        meta = _get_meta()
        err = _validate_filters(meta, skus, regions, channels, categories, brands, segments, pack_types)
        if err:
            return {"error": err}
        filters = SalesFilters(
            skus=skus, regions=regions, channels=channels, categories=categories,
            brands=brands, segments=segments, pack_types=pack_types,
            start_date=_parse_date(start_date), end_date=_parse_date(end_date),
        )
        con = get_connection()
        try:
            df = analytics.get_promotion_analysis(con, filters, dimension)
        finally:
            con.close()
        return {"dimension": dimension, "groups": _df_to_records(df)}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Promotion analysis failed: {e}"}


def get_forecast_series() -> dict:
    """Lists every (sku, region, channel) combination that has enough history to be
    forecastable. Use this if a forecast request fails due to an unclear series."""
    try:
        series = forecasting.list_available_series()
        return {"series": series, "count": len(series)}
    except Exception as e:
        return {"error": f"Could not list forecastable series: {e}"}


def get_forecast_last_known(sku: str, region: str, channel: str) -> dict:
    """Returns the most recent actual operational values (price, promo flag, delivery
    days, stock, delivered qty, pack type) for a (sku, region, channel) series. Use this
    to fill in reasonable defaults before calling forecast_predict if the user hasn't
    specified price/stock/etc."""
    try:
        meta = _get_meta()
        err = _validate_filters(meta, skus=[sku] if sku else None,
                                 regions=[region] if region else None,
                                 channels=[channel] if channel else None)
        if err:
            return {"error": err}
        result = forecasting.get_last_known_operational_values(sku, region, channel)
        if result is None:
            return {"error": f"No history found for ({sku}, {region}, {channel})."}
        return result
    except Exception as e:
        return {"error": f"Last-known lookup failed: {e}"}


def forecast_predict(target_date: str, sku: str, region: str, channel: str,
                      price_unit: Optional[float] = None, promotion_flag: Optional[bool] = None,
                      delivery_days: Optional[float] = None, stock_available: Optional[float] = None,
                      delivered_qty: Optional[float] = None, pack_type: Optional[str] = None) -> dict:
    """Predicts units sold for a future (or any) date for a given SKU/region/channel,
    using the Phase 1 XGBoost model via the existing forecasting pipeline. target_date
    is REQUIRED and is very often after the historical max_date — that is expected and
    fine, since this tool exists specifically to forecast future sales. If any of
    price_unit/promotion_flag/delivery_days/stock_available/delivered_qty/pack_type are
    not given, call get_forecast_last_known first and use those values as sensible
    defaults rather than guessing numbers yourself."""
    try:
        meta = _get_meta()
        err = _validate_filters(meta, skus=[sku] if sku else None,
                                 regions=[region] if region else None,
                                 channels=[channel] if channel else None,
                                 pack_types=[pack_type] if pack_type else None)
        if err:
            return {"error": err}
        try:
            parsed_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return {"error": f"Invalid target_date '{target_date}'. Use YYYY-MM-DD."}

        missing = [
            name for name, val in [
                ("price_unit", price_unit), ("promotion_flag", promotion_flag),
                ("delivery_days", delivery_days), ("stock_available", stock_available),
                ("delivered_qty", delivered_qty),
            ] if val is None
        ]
        if missing:
            return {
                "error": (
                    f"Missing required forecast inputs: {missing}. Call get_forecast_last_known "
                    f"with sku='{sku}', region='{region}', channel='{channel}' to get reasonable "
                    f"defaults, then call forecast_predict again with all values filled in."
                )
            }

        result = forecasting.predict_units_sold(
            target_date=parsed_date, sku=sku, region=region, channel=channel,
            price_unit=float(price_unit), promotion_flag=bool(promotion_flag),
            delivery_days=float(delivery_days), stock_available=float(stock_available),
            delivered_qty=float(delivered_qty), pack_type=pack_type,
        )
        result["note"] = "This is a model-generated forecast, not an actual recorded sale."
        return result
    except ValueError as e:
        # Raised by forecasting.py e.g. for insufficient history — not a bug, a real
        # domain-level constraint the LLM should explain rather than paper over.
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Forecast prediction failed: {e}"}


# ---------------------------------------------------------------------------
# OpenAI/Groq-format tool schemas (used by agent.py)
# ---------------------------------------------------------------------------

def _dim_filter_props(extra: dict = None) -> dict:
    props = {
        "skus": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Filter to these SKUs. Omit when no SKU filter is requested."
        },
        "regions": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Filter to these regions. Omit when no region filter is requested."
        },
        "channels": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Filter to these channels. Omit when no channel filter is requested."
        },
        "categories": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Filter to these categories. Omit when no category filter is requested."
        },
        "brands": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Filter to these brands. Omit when no brand filter is requested."
        },
        "segments": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Filter to these segments. Omit when no segment filter is requested."
        },
        "pack_types": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Filter to these pack types. Omit when no pack type filter is requested."
        },
        "start_date": {
            "type": ["string", "null"],
            "description": "YYYY-MM-DD, inclusive lower bound on the sale date. Omit when no start date is requested."
        },
        "end_date": {
            "type": ["string", "null"],
            "description": "YYYY-MM-DD, inclusive upper bound on the sale date. Omit when no end date is requested."
        },
    }

    if extra:
        props.update(extra)

    return props


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_metadata",
            "description": get_metadata.__doc__.strip(),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sales_summary",
            "description": get_sales_summary.__doc__.strip(),
            "parameters": {
                "type": "object",
                "properties": _dim_filter_props({
                    "promotion_only": {"type": "boolean", "description": "True = only promo days, False = only non-promo days."},
                }),
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sales_trend",
            "description": get_sales_trend.__doc__.strip(),
            "parameters": {
                "type": "object",
                "properties": _dim_filter_props({
                    "granularity": {"type": "string", "enum": ["day", "week", "month"]},
                    "promotion_only": {"type": "boolean"},
                }),
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_sales",
            "description": compare_sales.__doc__.strip(),
            "parameters": {
                "type": "object",
                "properties": _dim_filter_props({
                    "dimension": {"type": "string", "enum": DIMENSION_CHOICES},
                    "top_n": {"type": "integer", "description": "How many ranked rows to return, default 10."},
                    "promotion_only": {"type": "boolean"},
                }),
                "required": ["dimension"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sales_table",
            "description": get_sales_table.__doc__.strip(),
            "parameters": {
                "type": "object",
                "properties": _dim_filter_props({
                    "promotion_only": {"type": "boolean"},
                    "limit": {"type": "integer", "description": "Max rows to return, default 50, hard cap 200."},
                }),
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_inventory_analysis",
            "description": get_inventory_analysis.__doc__.strip(),
            "parameters": {
                "type": "object",
                "properties": _dim_filter_props({
                    "dimension": {"type": "string", "enum": INVENTORY_PROMO_DIMENSIONS},
                    "promotion_only": {"type": "boolean"},
                }),
                "required": ["dimension"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_promotion_analysis",
            "description": get_promotion_analysis.__doc__.strip(),
            "parameters": {
                "type": "object",
                "properties": _dim_filter_props({
                    "dimension": {"type": "string", "enum": INVENTORY_PROMO_DIMENSIONS},
                }),
                "required": ["dimension"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast_series",
            "description": get_forecast_series.__doc__.strip(),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast_last_known",
            "description": get_forecast_last_known.__doc__.strip(),
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string"}, "region": {"type": "string"}, "channel": {"type": "string"},
                },
                "required": ["sku", "region", "channel"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_predict",
            "description": forecast_predict.__doc__.strip(),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_date": {"type": "string", "description": "YYYY-MM-DD. Commonly a future date beyond the historical max_date — that's expected."},
                    "sku": {"type": "string"}, "region": {"type": "string"}, "channel": {"type": "string"},
                    "price_unit": {"type": "number"},
                    "promotion_flag": {"type": "boolean"},
                    "delivery_days": {"type": "number"},
                    "stock_available": {"type": "number"},
                    "delivered_qty": {"type": "number"},
                    "pack_type": {"type": "string"},
                },
                "required": ["target_date", "sku", "region", "channel"],
            },
        },
    },
]

FUNCTIONS = {
    "get_metadata": get_metadata,
    "get_sales_summary": get_sales_summary,
    "get_sales_trend": get_sales_trend,
    "compare_sales": compare_sales,
    "get_sales_table": get_sales_table,
    "get_inventory_analysis": get_inventory_analysis,
    "get_promotion_analysis": get_promotion_analysis,
    "get_forecast_series": get_forecast_series,
    "get_forecast_last_known": get_forecast_last_known,
    "forecast_predict": forecast_predict,
}


def call_function(name: str, arguments: dict) -> str:
    """Executes a tool by name with the given (already-parsed) arguments and
    returns a JSON string — never raises."""
    fn = FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool '{name}'."})
    try:
        result = fn(**arguments)
    except TypeError as e:
        result = {"error": f"Invalid arguments for {name}: {e}"}
    except Exception as e:
        result = {"error": f"Tool '{name}' failed unexpectedly: {e}"}
    return json.dumps(result, default=str)
