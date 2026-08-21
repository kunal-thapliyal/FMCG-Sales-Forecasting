"""
Thin client for the FastAPI backend. All Streamlit pages go through here so
there's exactly one place that knows the API base URL and handles errors.
"""
import os

import pandas as pd
import requests
import streamlit as st

API_BASE = os.environ.get("FMCG_API_BASE", "http://localhost:8000")


def _get(path, params=None):
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Can't reach the API at {API_BASE}. Start it with "
            f"`uvicorn main:app --port 8000` from the backend/ directory."
        )
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")
        st.stop()


def _post(path, json_body):
    try:
        resp = requests.post(f"{API_BASE}{path}", json=json_body, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Can't reach the API at {API_BASE}. Start it with "
            f"`uvicorn main:app --port 8000` from the backend/ directory."
        )
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")
        return None


@st.cache_data(ttl=3600)
def get_filter_options():
    return _get("/meta/filters")


@st.cache_data(ttl=3600)
def get_model_info():
    return _get("/model/info")


def _filter_params(f: dict):
    """Drop empty values so we don't send empty-list filters over the wire."""
    return {k: v for k, v in f.items() if v not in (None, [], "")}


@st.cache_data(ttl=300)
def get_summary(filters: dict):
    return _get("/sales/summary", _filter_params(filters))


@st.cache_data(ttl=300)
def get_trend(filters: dict, granularity: str):
    data = _get("/sales/trend", {**_filter_params(filters), "granularity": granularity})
    return pd.DataFrame(data)


@st.cache_data(ttl=300)
def get_compare(filters: dict, dimension: str, top_n: int = 15):
    data = _get("/sales/compare", {**_filter_params(filters), "dimension": dimension, "top_n": top_n})
    return pd.DataFrame(data)


@st.cache_data(ttl=300)
def get_table(filters: dict, limit: int = 500, offset: int = 0):
    data = _get("/sales/table", {**_filter_params(filters), "limit": limit, "offset": offset})
    return pd.DataFrame(data)


@st.cache_data(ttl=300)
def get_inventory(filters: dict, dimension: str = "category"):
    data = _get("/inventory/analysis", {**_filter_params(filters), "dimension": dimension})
    return pd.DataFrame(data)


@st.cache_data(ttl=300)
def get_promotions(filters: dict, dimension: str = "category"):
    data = _get("/promotions/analysis", {**_filter_params(filters), "dimension": dimension})
    return pd.DataFrame(data)


@st.cache_data(ttl=3600)
def get_forecast_series():
    return pd.DataFrame(_get("/forecast/series"))


def get_last_known(sku, region, channel):
    return _get("/forecast/last-known", {"sku": sku, "region": region, "channel": channel})


def predict(payload: dict):
    return _post("/forecast/predict", payload)


def chat(message: str, history: list):
    """Calls the Phase 3 /chat endpoint. `history` is a list of
    {"role": "user"|"assistant", "content": str} dicts. Returns
    {"answer": str, "tool_calls": [...]}, or None on a request-level failure
    (a configuration issue like a missing GROQ_API_KEY comes back as a normal
    200 response with an explanatory `answer`, not an error)."""
    return _post("/chat", {"message": message, "history": history})
