from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent

DB_PATH = ROOT / "data" / "fmcg_warehouse.duckdb"

MODEL_PATH = BACKEND_DIR / "models" / "final_best_model_pipeline.joblib"
FEATURES_PATH = BACKEND_DIR / "models" / "model_features.json"
HISTORY_PATH = BACKEND_DIR / "models" / "fmcg_features.parquet"
SKU_ATTRS_PATH = BACKEND_DIR / "models" / "sku_attributes.json"
MODEL_SELECTION_PATH = BACKEND_DIR / "models" / "final_model_selection.json"

MIN_HISTORY_DAYS = 28
