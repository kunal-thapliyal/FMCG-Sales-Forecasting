"""
Frontend test suite (Streamlit AppTest).

Run from the phase2/ directory with the backend already running:
    uvicorn backend.main:app --port 8000 &   # or point FMCG_API_BASE elsewhere
    FMCG_API_BASE=http://localhost:8000 pytest tests/test_frontend.py -v

Requires a live backend (these are integration tests, not unit tests) —
AppTest actually runs app.py's Python and every page pulls data from the API.
"""
import os
import sys
from pathlib import Path

import pytest

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
sys.path.insert(0, str(FRONTEND_DIR))
os.chdir(FRONTEND_DIR)  # app.py uses relative imports (api_client, pages_impl)

from streamlit.testing.v1 import AppTest  # noqa: E402

API_BASE = os.environ.get("FMCG_API_BASE", "http://localhost:8000")

PAGES = ["Overview", "Sales Trends", "Compare", "Inventory & Promotions", "Explore Data", "Forecast"]


def _backend_reachable():
    import requests
    try:
        requests.get(f"{API_BASE}/health", timeout=3).raise_for_status()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _backend_reachable(),
    reason=f"Backend not reachable at {API_BASE} — start it before running these tests.",
)


def _fresh_app():
    os.environ["FMCG_API_BASE"] = API_BASE
    at = AppTest.from_file(str(FRONTEND_DIR / "app.py"))
    at.run(timeout=30)
    return at


def test_app_loads_without_exception():
    at = _fresh_app()
    assert not list(at.exception)


@pytest.mark.parametrize("page", PAGES)
def test_each_page_renders_without_exception(page):
    at = _fresh_app()
    page_radio = [r for r in at.sidebar.radio if r.label == "Go to"][0]
    page_radio.set_value(page).run(timeout=45)
    assert not list(at.exception), f"Page '{page}' raised: {list(at.exception)}"


def test_forecast_generate_button_produces_a_prediction():
    at = _fresh_app()
    page_radio = [r for r in at.sidebar.radio if r.label == "Go to"][0]
    page_radio.set_value("Forecast").run(timeout=30)
    assert not list(at.exception)

    gen_btn = [b for b in at.button if b.label == "Generate Forecast"][0]
    gen_btn.click().run(timeout=30)
    assert not list(at.exception)

    metrics = {m.label: m.value for m in at.metric}
    assert "Predicted Units Sold" in metrics
    # sanity: should parse as a non-negative float
    assert float(metrics["Predicted Units Sold"]) >= 0


def test_category_filter_applies_consistently_across_pages():
    """Regression guard for filter propagation: Overview and Compare must
    both reflect the same Category=Milk restriction, and their numbers
    must agree with each other (not just individually look plausible)."""
    at = _fresh_app()
    cat_ms = [ms for ms in at.sidebar.multiselect if ms.label == "Category"][0]
    cat_ms.set_value(["Milk"]).run(timeout=30)
    assert not list(at.exception)

    page_radio = [r for r in at.sidebar.radio if r.label == "Go to"][0]

    page_radio.set_value("Overview").run(timeout=30)
    overview_metrics = {m.label: m.value for m in at.metric}
    overview_units = float(overview_metrics["Total Units Sold"].replace(",", ""))

    page_radio.set_value("Compare").run(timeout=30)
    compare_df = at.dataframe[0].value
    assert list(compare_df["Category"].unique()) == ["Milk"]
    compare_units = float(compare_df["Units Sold"].iloc[0])

    assert overview_units == pytest.approx(compare_units, rel=1e-6)


def test_explore_data_respects_category_filter():
    at = _fresh_app()
    cat_ms = [ms for ms in at.sidebar.multiselect if ms.label == "Category"][0]
    cat_ms.set_value(["Yogurt"]).run(timeout=30)

    page_radio = [r for r in at.sidebar.radio if r.label == "Go to"][0]
    page_radio.set_value("Explore Data").run(timeout=30)
    assert not list(at.exception)

    df = at.dataframe[0].value
    assert set(df["category"].unique()) == {"Yogurt"}
