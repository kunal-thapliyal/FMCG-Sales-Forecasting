"""
Warehouse integrity test.

Run from the phase2/ directory, after building the warehouse:
    python warehouse/build_warehouse.py
    pytest tests/test_warehouse.py -v
"""
from pathlib import Path

import duckdb
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "fmcg_warehouse.duckdb"
CSV_PATH = ROOT / "data" / "fmcg_features.csv"

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="Warehouse not built — run `python warehouse/build_warehouse.py` first.",
)


@pytest.fixture(scope="module")
def con():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def raw_df():
    return pd.read_csv(CSV_PATH)


def test_fact_row_count_matches_source(con, raw_df):
    n_fact = con.execute("SELECT COUNT(*) FROM FACT_SALES").fetchone()[0]
    assert n_fact == len(raw_df)


def test_no_rows_lost_in_dimension_joins(con, raw_df):
    """A FACT_SALES row is only produced when its region/channel/pack_type
    successfully joins to a dimension table — if any source value were
    missing from a dimension, that row would silently vanish. Row-count
    parity between source and fact confirms no join drops occurred."""
    assert con.execute("SELECT COUNT(*) FROM FACT_SALES").fetchone()[0] == len(raw_df)


def test_dimension_cardinalities(con):
    assert con.execute("SELECT COUNT(*) FROM DIM_PRODUCT").fetchone()[0] == 30
    assert con.execute("SELECT COUNT(*) FROM DIM_REGION").fetchone()[0] == 3
    assert con.execute("SELECT COUNT(*) FROM DIM_CHANNEL").fetchone()[0] == 3


def test_total_units_sold_matches_source(con, raw_df):
    warehouse_total = con.execute("SELECT SUM(UNITS_SOLD) FROM FACT_SALES").fetchone()[0]
    assert warehouse_total == pytest.approx(raw_df["units_sold"].sum(), rel=1e-9)


def test_date_range_matches_source(con, raw_df):
    min_d, max_d = con.execute("SELECT MIN(DATE_KEY), MAX(DATE_KEY) FROM FACT_SALES").fetchone()
    assert str(min_d) == raw_df["date"].min()
    assert str(max_d) == raw_df["date"].max()
