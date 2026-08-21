"""
SQL analytics layer.

Every function here runs a single SQL statement against the star schema
(FACT_SALES joined to DIM_PRODUCT / DIM_REGION / DIM_CHANNEL as needed) and
returns a plain pandas DataFrame or dict. Filters are applied via a shared
WHERE-clause builder (`_build_where`) so every endpoint respects the same
set of business-dimension filters.

This module is intentionally warehouse-agnostic SQL (standard ANSI +
DuckDB/Snowflake-compatible functions like DATE_TRUNC), so it can be
pointed at Snowflake by swapping the connection in database.py.
"""
from typing import Optional

import pandas as pd

from schemas import SalesFilters

BASE_QUERY = """
SELECT
    f.DATE_KEY AS date, f.SKU AS sku, f.REGION AS region, f.CHANNEL AS channel,
    f.PACK_TYPE AS pack_type, f.BRAND AS brand, f.CATEGORY AS category, f.SEGMENT AS segment,
    f.PRICE_UNIT AS price_unit, f.PROMOTION_FLAG AS promotion_flag,
    f.DELIVERY_DAYS AS delivery_days, f.STOCK_AVAILABLE AS stock_available,
    f.DELIVERED_QTY AS delivered_qty, f.UNITS_SOLD AS units_sold,
    f.IS_ANOMALOUS_ROW AS is_anomalous_row
FROM FACT_SALES f
"""


def _build_where(filters: Optional[SalesFilters]):
    """Returns (where_sql, params) for the given filters."""
    clauses = []
    params = []

    if filters is None:
        return "", params

    def add_in(col, values):
        if values:
            placeholders = ",".join(["?"] * len(values))
            clauses.append(f"{col} IN ({placeholders})")
            params.extend(values)

    add_in("f.SKU", filters.skus)
    add_in("f.REGION", filters.regions)
    add_in("f.CHANNEL", filters.channels)
    add_in("f.CATEGORY", filters.categories)
    add_in("f.BRAND", filters.brands)
    add_in("f.SEGMENT", filters.segments)
    add_in("f.PACK_TYPE", filters.pack_types)

    if filters.start_date:
        clauses.append("f.DATE_KEY >= ?")
        params.append(filters.start_date)
    if filters.end_date:
        clauses.append("f.DATE_KEY <= ?")
        params.append(filters.end_date)
    if filters.promotion_only is True:
        clauses.append("f.PROMOTION_FLAG = TRUE")
    elif filters.promotion_only is False:
        clauses.append("f.PROMOTION_FLAG = FALSE")

    where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where_sql, params


def get_filter_options(con) -> dict:
    skus = con.execute("SELECT SKU FROM DIM_PRODUCT ORDER BY SKU").fetchdf()["SKU"].tolist()
    categories = con.execute("SELECT DISTINCT CATEGORY FROM DIM_PRODUCT ORDER BY CATEGORY").fetchdf()["CATEGORY"].tolist()
    brands = con.execute("SELECT DISTINCT BRAND FROM DIM_PRODUCT ORDER BY BRAND").fetchdf()["BRAND"].tolist()
    segments = con.execute("SELECT DISTINCT SEGMENT FROM DIM_PRODUCT ORDER BY SEGMENT").fetchdf()["SEGMENT"].tolist()
    regions = con.execute("SELECT REGION FROM DIM_REGION ORDER BY REGION").fetchdf()["REGION"].tolist()
    channels = con.execute("SELECT CHANNEL FROM DIM_CHANNEL ORDER BY CHANNEL").fetchdf()["CHANNEL"].tolist()
    pack_types = con.execute("SELECT PACK_TYPE FROM DIM_PACK_TYPE ORDER BY PACK_TYPE").fetchdf()["PACK_TYPE"].tolist()
    date_range = con.execute("SELECT MIN(DATE_KEY), MAX(DATE_KEY) FROM FACT_SALES").fetchone()
    return {
        "skus": skus, "regions": regions, "channels": channels,
        "categories": categories, "brands": brands, "segments": segments,
        "pack_types": pack_types, "min_date": date_range[0], "max_date": date_range[1],
    }


def get_summary_kpis(con, filters: SalesFilters) -> dict:
    where_sql, params = _build_where(filters)
    sql = f"""
    SELECT
        SUM(f.UNITS_SOLD) AS total_units_sold,
        SUM(f.UNITS_SOLD * f.PRICE_UNIT) AS total_revenue,
        AVG(f.PRICE_UNIT) AS avg_price_unit,
        AVG(f.UNITS_SOLD) AS avg_daily_units,
        100.0 * AVG(CASE WHEN f.PROMOTION_FLAG THEN 1 ELSE 0 END) AS promo_share_pct,
        100.0 * AVG(CASE WHEN f.STOCK_AVAILABLE < f.UNITS_SOLD THEN 1 ELSE 0 END) AS stockout_risk_days_pct,
        COUNT(*) AS n_rows,
        MIN(f.DATE_KEY) AS min_date,
        MAX(f.DATE_KEY) AS max_date
    FROM FACT_SALES f
    {where_sql}
    """
    row = con.execute(sql, params).fetchdf().iloc[0]

    def _f(val):
        """NaN-safe float coercion — `val or 0` doesn't catch NaN since NaN
        is truthy in Python, which crashed empty-selection responses."""
        return 0.0 if pd.isna(val) else float(val)

    n_rows = int(row["n_rows"] or 0)
    if n_rows == 0 or pd.isna(row["min_date"]) or pd.isna(row["max_date"]):
        # No rows matched the filters — return a well-formed, empty summary
        # instead of NaT dates that fail response-model validation.
        date_range = [filters.start_date, filters.end_date] if (filters and filters.start_date and filters.end_date) else []
    else:
        date_range = [pd.Timestamp(row["min_date"]).date(), pd.Timestamp(row["max_date"]).date()]

    return {
        "total_units_sold": _f(row["total_units_sold"]),
        "total_revenue": _f(row["total_revenue"]),
        "avg_price_unit": _f(row["avg_price_unit"]),
        "avg_daily_units": _f(row["avg_daily_units"]),
        "promo_share_pct": _f(row["promo_share_pct"]),
        "stockout_risk_days_pct": _f(row["stockout_risk_days_pct"]),
        "n_rows": n_rows,
        "date_range": date_range,
    }


def get_sales_trend(con, filters: SalesFilters, granularity: str = "day") -> pd.DataFrame:
    granularity = {"day": "day", "week": "week", "month": "month"}.get(granularity, "day")
    where_sql, params = _build_where(filters)
    sql = f"""
    SELECT
        CAST(DATE_TRUNC('{granularity}', f.DATE_KEY) AS DATE) AS period,
        SUM(f.UNITS_SOLD) AS units_sold,
        SUM(f.UNITS_SOLD * f.PRICE_UNIT) AS revenue,
        SUM(CASE WHEN f.PROMOTION_FLAG THEN f.UNITS_SOLD ELSE 0 END) AS promo_units_sold,
        SUM(CASE WHEN NOT f.PROMOTION_FLAG THEN f.UNITS_SOLD ELSE 0 END) AS non_promo_units_sold
    FROM FACT_SALES f
    {where_sql}
    GROUP BY 1
    ORDER BY 1
    """
    return con.execute(sql, params).fetchdf()


def get_comparison(con, filters: SalesFilters, dimension: str, top_n: int = 15) -> pd.DataFrame:
    dim_map = {
        "sku": "f.SKU", "region": "f.REGION", "channel": "f.CHANNEL",
        "category": "f.CATEGORY", "brand": "f.BRAND", "segment": "f.SEGMENT",
        "pack_type": "f.PACK_TYPE",
    }
    col = dim_map.get(dimension, "f.CATEGORY")
    where_sql, params = _build_where(filters)
    sql = f"""
    SELECT
        {col} AS "group",
        SUM(f.UNITS_SOLD) AS units_sold,
        SUM(f.UNITS_SOLD * f.PRICE_UNIT) AS revenue,
        AVG(f.PRICE_UNIT) AS avg_price_unit,
        COUNT(*) AS n_rows
    FROM FACT_SALES f
    {where_sql}
    GROUP BY 1
    ORDER BY units_sold DESC
    LIMIT {int(top_n)}
    """
    return con.execute(sql, params).fetchdf()


def get_inventory_analysis(con, filters: SalesFilters, dimension: str = "category") -> pd.DataFrame:
    dim_map = {
        "sku": "f.SKU", "region": "f.REGION", "channel": "f.CHANNEL",
        "category": "f.CATEGORY", "brand": "f.BRAND",
    }
    col = dim_map.get(dimension, "f.CATEGORY")
    where_sql, params = _build_where(filters)
    sql = f"""
    SELECT
        {col} AS "group",
        AVG(f.STOCK_AVAILABLE) AS avg_stock_available,
        AVG(f.UNITS_SOLD) AS avg_units_sold,
        AVG(f.DELIVERED_QTY) AS avg_delivered_qty,
        AVG(f.DELIVERY_DAYS) AS avg_delivery_days,
        100.0 * AVG(CASE WHEN f.STOCK_AVAILABLE < f.UNITS_SOLD THEN 1 ELSE 0 END) AS stockout_risk_pct
    FROM FACT_SALES f
    {where_sql}
    GROUP BY 1
    ORDER BY stockout_risk_pct DESC
    """
    return con.execute(sql, params).fetchdf()


def get_promotion_analysis(con, filters: SalesFilters, dimension: str = "category") -> pd.DataFrame:
    dim_map = {
        "sku": "f.SKU", "region": "f.REGION", "channel": "f.CHANNEL",
        "category": "f.CATEGORY", "brand": "f.BRAND",
    }
    col = dim_map.get(dimension, "f.CATEGORY")
    where_sql, params = _build_where(filters)
    sql = f"""
    SELECT
        {col} AS "group",
        AVG(CASE WHEN f.PROMOTION_FLAG THEN f.UNITS_SOLD END) AS promo_avg_units,
        AVG(CASE WHEN NOT f.PROMOTION_FLAG THEN f.UNITS_SOLD END) AS non_promo_avg_units
    FROM FACT_SALES f
    {where_sql}
    GROUP BY 1
    """
    df = con.execute(sql, params).fetchdf()
    df["promo_avg_units"] = df["promo_avg_units"].fillna(0)
    df["non_promo_avg_units"] = df["non_promo_avg_units"].fillna(0)
    df["uplift_pct"] = df.apply(
        lambda r: (100.0 * (r["promo_avg_units"] - r["non_promo_avg_units"]) / r["non_promo_avg_units"])
        if r["non_promo_avg_units"] else 0.0,
        axis=1,
    )
    return df.sort_values("uplift_pct", ascending=False)


def get_sales_table(con, filters: SalesFilters, limit: int = 500, offset: int = 0) -> pd.DataFrame:
    where_sql, params = _build_where(filters)
    sql = f"""
    {BASE_QUERY}
    {where_sql}
    ORDER BY f.DATE_KEY DESC, f.SKU, f.REGION, f.CHANNEL, f.ROW_ID
    LIMIT {int(limit)} OFFSET {int(offset)}
    """
    # NOTE: DATE_KEY alone has many ties (thousands of rows share a date),
    # so ORDER BY DATE_KEY DESC on its own makes LIMIT/OFFSET pagination
    # non-deterministic — different pages can return overlapping or
    # duplicate rows. SKU/REGION/CHANNEL/ROW_ID break ties deterministically.
    return con.execute(sql, params).fetchdf()


def get_series_for_forecast(con, sku: str, region: str, channel: str) -> pd.DataFrame:
    """Full history for one (sku, region, channel) series — used to build
    lag/rolling features for a forecast."""
    sql = """
    SELECT DATE_KEY AS date, SKU AS sku, REGION AS region, CHANNEL AS channel,
           PACK_TYPE AS pack_type, CATEGORY AS category, UNITS_SOLD AS units_sold
    FROM FACT_SALES
    WHERE SKU = ? AND REGION = ? AND CHANNEL = ?
    ORDER BY DATE_KEY
    """
    return con.execute(sql, [sku, region, channel]).fetchdf()
