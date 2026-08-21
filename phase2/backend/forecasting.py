"""
Forecasting layer.

This is Phase 1's `predict_final.py` (models/predict_final.py) wired into
the app, unchanged in logic — same feature construction, same
`np.clip(pred, 0, None)`, same model artifact. Only the plumbing differs:
paths come from config.py and history is loaded once at import time from
the Phase 1 parquet snapshot rather than a CLI argument.

DO NOT alter `_build_features` or the feature list without re-validating
against Phase 1's reported test RMSE/MAE/R^2 (7.83 / 5.41 / 0.515) — the
model was trained on features built exactly this way.
"""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from config import (
    FEATURES_PATH,
    HISTORY_PATH,
    MIN_HISTORY_DAYS,
    MODEL_PATH,
    SKU_ATTRS_PATH,
)


@lru_cache(maxsize=1)
def _load_model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_feature_config():
    with open(FEATURES_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_sku_attrs():
    if SKU_ATTRS_PATH.exists():
        with open(SKU_ATTRS_PATH) as f:
            return json.load(f)
    return {}


@lru_cache(maxsize=1)
def _load_history() -> pd.DataFrame:
    df = pd.read_parquet(HISTORY_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _build_features(history_df, target_date, sku, region, channel,
                     price_unit, promotion_flag, delivery_days,
                     stock_available, delivered_qty, pack_type, sku_attrs):
    target_date = pd.Timestamp(target_date)
    series = history_df[
        (history_df["sku"] == sku) & (history_df["region"] == region)
        & (history_df["channel"] == channel) & (history_df["date"] < target_date)
    ].sort_values("date")

    if len(series) < MIN_HISTORY_DAYS:
        raise ValueError(
            f"Not enough history for ({sku}, {region}, {channel}) before "
            f"{target_date.date()}: found {len(series)} prior days, need >= {MIN_HISTORY_DAYS}."
        )

    units = series["units_sold"].values
    attrs = sku_attrs.get(sku, {})

    row = {
        "sku": sku, "region": region, "channel": channel,
        "pack_type": pack_type or series.iloc[-1]["pack_type"],
        "category": attrs.get("category", series.iloc[-1]["category"]),
        "price_unit": price_unit,
        "promotion_flag": int(promotion_flag),
        "delivery_days": delivery_days,
        "stock_available": stock_available,
        "delivered_qty": delivered_qty,
        "year": target_date.year, "month": target_date.month, "quarter": target_date.quarter,
        "day_of_week": target_date.dayofweek,
        "is_weekend": int(target_date.dayofweek in (5, 6)),
        "day_of_year": target_date.dayofyear,
        "week_of_year": int(target_date.isocalendar().week),
        "month_sin": np.sin(2 * np.pi * target_date.month / 12),
        "month_cos": np.cos(2 * np.pi * target_date.month / 12),
        "dow_sin": np.sin(2 * np.pi * target_date.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * target_date.dayofweek / 7),
        "days_since_launch": (target_date - series["date"].min()).days,
        "lag_1": units[-1],
        "lag_7": units[-7] if len(units) >= 7 else np.nan,
        "lag_14": units[-14] if len(units) >= 14 else np.nan,
        "lag_28": units[-28] if len(units) >= 28 else np.nan,
        "roll_mean_7": units[-7:].mean(),
        "roll_mean_14": units[-14:].mean(),
        "roll_mean_28": units[-28:].mean(),
        "roll_std_7": units[-7:].std(ddof=1) if len(units) >= 7 else np.nan,
        "expanding_mean": units.mean(),
    }
    return row


def predict_units_sold(target_date, sku, region, channel,
                        price_unit, promotion_flag, delivery_days,
                        stock_available, delivered_qty, pack_type=None):
    pipe = _load_model()
    fconf = _load_feature_config()
    all_features = fconf["categorical_features"] + fconf["numeric_features"]
    history_df = _load_history()
    sku_attrs = _load_sku_attrs()

    row = _build_features(history_df, target_date, sku, region, channel,
                           price_unit, promotion_flag, delivery_days,
                           stock_available, delivered_qty, pack_type, sku_attrs)
    X = pd.DataFrame([row])[all_features]
    pred = float(np.clip(pipe.predict(X), 0, None)[0])
    return {
        "date": str(pd.Timestamp(target_date).date()), "sku": sku, "region": region,
        "channel": channel, "predicted_units_sold": round(pred, 1),
    }


def get_last_known_operational_values(sku: str, region: str, channel: str):
    """Convenience lookup so the UI can prefill price/stock/delivery inputs
    with the most recent actuals for a series."""
    history_df = _load_history()
    series = history_df[
        (history_df["sku"] == sku) & (history_df["region"] == region) & (history_df["channel"] == channel)
    ].sort_values("date")
    if series.empty:
        return None
    last = series.iloc[-1]
    return {
        "last_date": str(last["date"].date()),
        "price_unit": float(last["price_unit"]),
        "promotion_flag": bool(int(last["promotion_flag"])),
        "delivery_days": float(last["delivery_days"]),
        "stock_available": float(last["stock_available"]),
        "delivered_qty": float(last["delivered_qty"]),
        "pack_type": str(last["pack_type"]),
        "n_history_days": int(len(series)),
    }


def list_available_series():
    history_df = _load_history()
    return (
        history_df[["sku", "region", "channel"]]
        .drop_duplicates()
        .sort_values(["sku", "region", "channel"])
        .to_dict(orient="records")
    )
