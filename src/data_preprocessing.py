"""
src/data_preprocessing.py
═════════════════════════
Data-Preprocessing Component.

Reads `train.csv` and `test.csv`, applies feature engineering and
outlier handling, and outputs `train_processed.csv` and `test_processed.csv`.
"""

import sys
from pathlib import Path
import pandas as pd
import yaml
import json

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.logger import get_logger

logger = get_logger("data_preprocessing")

def run_data_preprocessing(config_path: str = "config/config.yaml") -> dict:
    """Run data preprocessing stage."""
    logger.info("━" * 60)
    logger.info("  DATA PREPROCESSING – Started")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open("config/params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
        
    proc_dir = Path(cfg["directories"]["processed_data"])
    train_path = proc_dir / cfg["data_ingestion"]["train_filename"]
    test_path = proc_dir / cfg["data_ingestion"]["test_filename"]
    
    outer_method = params["preprocessing"].get("outlier_method", "iqr")
    threshold = params["preprocessing"].get("outlier_threshold", 1.5)
    target = params["preprocessing"].get("target_col", "salary_in_usd")

    # Load data
    train_df = pd.pd.read_csv(train_path) if hasattr(pd, "pd") else pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

    # Outlier removal only on train!
    initial_train_len = len(train_df)
    if outer_method == "iqr" and target in train_df.columns:
        Q1 = train_df[target].quantile(0.25)
        Q3 = train_df[target].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        train_df = train_df[(train_df[target] >= lower) & (train_df[target] <= upper)]
        logger.info(f"Dropped {initial_train_len - len(train_df)} rows due to IQR outliers.")
        
    # Feature engineering: Example drop redundant columns
    cols_to_drop = ["salary", "salary_currency", "Unnamed: 0", "id"] # salary is redundant with salary_in_usd, ID cols must be dropped
    for c in cols_to_drop:
        if c in train_df.columns:
            train_df = train_df.drop(columns=[c])
        if c in test_df.columns:
            test_df = test_df.drop(columns=[c])

    # Save
    train_out = proc_dir / "train_processed.csv"
    test_out = proc_dir / "test_processed.csv"
    
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    
    logger.info(f"Saved train_processed: {train_df.shape} -> {train_out}")
    logger.info(f"Saved test_processed:  {test_df.shape} -> {test_out}")
    
    return {"train_shape": list(train_df.shape), "test_shape": list(test_df.shape)}

if __name__ == "__main__":
    run_data_preprocessing()
