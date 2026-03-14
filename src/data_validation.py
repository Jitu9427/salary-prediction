"""
src/data_validation.py
══════════════════════
Data-Validation Component.

Ensures the preprocessed datasets adhere to the expected schema and constraints.
Outputs a JSON report to `reports/validation_report.json`.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.logger import get_logger

logger = get_logger("data_validation")

def run_data_validation(config_path: str = "config/config.yaml") -> dict:
    logger.info("━" * 60)
    logger.info("  DATA VALIDATION – Started")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    proc_dir = Path(cfg["directories"]["processed_data"])
    train_path = proc_dir / "train_processed.csv"
    test_path = proc_dir / "test_processed.csv"
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    report = {"status": "SUCCESS", "errors": []}
    
    # Validation Rules
    # 1. Same columns in both
    if set(train_df.columns) != set(test_df.columns):
        msg = f"Columns mismatch: Train {train_df.columns} vs Test {test_df.columns}"
        logger.error(msg)
        report["status"] = "FAILURE"
        report["errors"].append(msg)
        
    # 2. Target presence
    target = "salary_in_usd"
    if target not in train_df.columns:
        msg = f"Target column '{target}' missing."
        logger.error(msg)
        report["status"] = "FAILURE"
        report["errors"].append(msg)
    
    # 3. Null values
    train_nulls = train_df.isnull().sum().sum()
    if train_nulls > 0:
        logger.warning(f"Train has {train_nulls} null values.")
        
    report["train_nulls"] = int(train_nulls)
    report["test_nulls"] = int(test_df.isnull().sum().sum())
    
    # Save Report
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "validation_report.json"
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    if report["status"] == "FAILURE":
        raise ValueError(f"Validation failed: {report['errors']}")
        
    logger.info(f"Validation successful. Report -> {report_path}")
    return report

if __name__ == "__main__":
    run_data_validation()
