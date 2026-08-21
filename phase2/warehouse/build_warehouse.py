"""
Phase 2 — Warehouse Builder
============================
Builds a local DuckDB warehouse that mirrors the Snowflake star schema
defined in Phase 1 (sql/01_snowflake_ddl.sql):

    DIM_DATE, DIM_PRODUCT, DIM_REGION, DIM_CHANNEL, DIM_PACK_TYPE, FACT_SALES

DuckDB is used as a drop-in local stand-in for Snowflake so the app runs
without live warehouse credentials. The schema, grain (date x sku x region
x channel), and column names are kept identical to the Snowflake DDL, so
every query in backend/analytics.py is a straight lift-and-shift to
Snowflake — swap the DuckDB connection in backend/database.py for a
snowflake-connector-python connection and the SQL does not change.

Source: data/fmcg_features.csv (Phase 1's cleaned + feature-engineered
output, copied verbatim from the Phase 1 deliverable).

Usage:
    python warehouse/build_warehouse.py
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
SOURCE_CSV = ROOT / "data" / "fmcg_features.csv"
DB_PATH = ROOT / "data" / "fmcg_warehouse.duckdb"


def build():
    DB_PATH.unlink(missing_ok=True)
    con = duckdb.connect(str(DB_PATH))

    print(f"Loading source data from {SOURCE_CSV} ...")
    con.execute(f"""
        CREATE TABLE STG AS
        SELECT *, CAST(date AS DATE) AS date_parsed
        FROM read_csv_auto('{SOURCE_CSV}', header=True)
    """)

    con.execute("""
        CREATE TABLE DIM_DATE AS
        SELECT DISTINCT
            date_parsed AS DATE_KEY,
            year AS YEAR,
            quarter AS QUARTER,
            month AS MONTH,
            day_of_week AS DAY_OF_WEEK,
            CAST(is_weekend AS BOOLEAN) AS IS_WEEKEND,
            week_of_year AS WEEK_OF_YEAR
        FROM STG
    """)

    con.execute("""
        CREATE TABLE DIM_PRODUCT AS
        SELECT DISTINCT
            sku AS SKU, brand AS BRAND, category AS CATEGORY, segment AS SEGMENT
        FROM STG
    """)

    con.execute("""
        CREATE TABLE DIM_REGION AS
        SELECT ROW_NUMBER() OVER () AS REGION_KEY, REGION
        FROM (SELECT DISTINCT region AS REGION FROM STG)
    """)

    con.execute("""
        CREATE TABLE DIM_CHANNEL AS
        SELECT ROW_NUMBER() OVER () AS CHANNEL_KEY, CHANNEL
        FROM (SELECT DISTINCT channel AS CHANNEL FROM STG)
    """)

    con.execute("""
        CREATE TABLE DIM_PACK_TYPE AS
        SELECT ROW_NUMBER() OVER () AS PACK_TYPE_KEY, PACK_TYPE
        FROM (SELECT DISTINCT pack_type AS PACK_TYPE FROM STG)
    """)

    con.execute("""
        CREATE TABLE FACT_SALES AS
        SELECT
            ROW_NUMBER() OVER () AS ROW_ID,
            s.date_parsed AS DATE_KEY,
            s.sku AS SKU,
            r.REGION_KEY, c.CHANNEL_KEY, p.PACK_TYPE_KEY,
            s.region AS REGION, s.channel AS CHANNEL, s.pack_type AS PACK_TYPE,
            s.price_unit AS PRICE_UNIT,
            CAST(s.promotion_flag AS BOOLEAN) AS PROMOTION_FLAG,
            s.delivery_days AS DELIVERY_DAYS,
            s.stock_available AS STOCK_AVAILABLE,
            s.delivered_qty AS DELIVERED_QTY,
            s.units_sold AS UNITS_SOLD,
            CAST(s.is_anomalous_row AS BOOLEAN) AS IS_ANOMALOUS_ROW,
            s.brand AS BRAND, s.category AS CATEGORY, s.segment AS SEGMENT
        FROM STG s
        JOIN DIM_REGION r ON s.region = r.REGION
        JOIN DIM_CHANNEL c ON s.channel = c.CHANNEL
        JOIN DIM_PACK_TYPE p ON s.pack_type = p.PACK_TYPE
    """)

    con.execute("CREATE INDEX idx_fact_date ON FACT_SALES(DATE_KEY)")
    con.execute("CREATE INDEX idx_fact_sku ON FACT_SALES(SKU)")

    n_stg = con.execute("SELECT COUNT(*) FROM STG").fetchone()[0]
    n_fact = con.execute("SELECT COUNT(*) FROM FACT_SALES").fetchone()[0]
    print(f"Staging rows: {n_stg} | Fact rows: {n_fact} | Match: {n_stg == n_fact}")
    for t in ["DIM_DATE", "DIM_PRODUCT", "DIM_REGION", "DIM_CHANNEL", "DIM_PACK_TYPE", "FACT_SALES"]:
        print(f"  {t}: {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:,} rows")

    con.execute("DROP TABLE STG")
    con.close()
    print(f"\nWarehouse built: {DB_PATH}")


if __name__ == "__main__":
    build()
