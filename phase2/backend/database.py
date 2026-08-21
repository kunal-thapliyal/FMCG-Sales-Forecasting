"""
DuckDB connection helper.

Kept deliberately thin: every function in analytics.py takes a `con` and
runs plain SQL against the star schema built by warehouse/build_warehouse.py.
To point this app at real Snowflake instead of local DuckDB, swap
`get_connection()` for a snowflake-connector-python connection object that
exposes the same `.execute(sql).fetchdf()` / `.execute(sql, params)` shape
(duckdb and the Snowflake connector are close enough that analytics.py's
SQL does not need to change).
"""
import duckdb

from config import DB_PATH


def get_connection():
    """Open a fresh read-only connection. DuckDB allows multiple concurrent
    read-only connections against the same file, which fits FastAPI's
    request-per-call model without needing a connection pool."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Warehouse not found at {DB_PATH}. Run "
            f"`python warehouse/build_warehouse.py` first."
        )
    return duckdb.connect(str(DB_PATH), read_only=True)
