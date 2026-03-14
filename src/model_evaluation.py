"""
src/model_evaluation.py
═══════════════════════
Model-Evaluation Component.

Loads the preprocessor and model, evaluates on the test set,
computes metrics, logs them to Weights & Biases, and saves them
to `reports/evaluation_metrics.json`.
"""

import sys
import json
import os
import joblib
import pandas as pd
import yaml
from pathlib import Path
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.logger import get_logger
import wandb

logger = get_logger("model_evaluation")

def run_model_evaluation(config_path: str = "config/config.yaml") -> dict:
    logger.info("━" * 60)
    logger.info("  MODEL EVALUATION – Started")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open("config/params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
        
    proc_dir = Path(cfg["directories"]["processed_data"])
    test_path = proc_dir / "test_processed.csv"
    
    df = pd.read_csv(test_path)
    target = params["preprocessing"].get("target_col", "salary_in_usd")
    
    X = df.drop(columns=[target])
    y_true = df[target]
    
    # Load Models
    models_dir = Path("models")
    preprocessor = joblib.load(models_dir / "preprocessor.pkl")
    model = joblib.load(models_dir / "stacking_model.pkl")
    
    # Predict
    X_trans = preprocessor.transform(X)
    y_pred = model.predict(X_trans)
    
    # Metrics
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    
    logger.info(f"Test RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
    
    # ── Log to W&B ───────────────────────────────────────────────────────────
    use_wandb = bool(os.getenv("WANDB_API_KEY"))
    if use_wandb:
        try:
            run = wandb.init(
                project="salary-prediction",
                job_type="evaluate",
                name="model_evaluation",
                config=params
            )
            # Log all evaluation metrics to W&B
            wandb.log({
                "eval/rmse": rmse,
                "eval/mae": mae,
                "eval/r2": r2
            })
            logger.info("✅ Metrics logged to Weights & Biases (eval/rmse, eval/mae, eval/r2)")
            wandb.finish()
        except Exception as e:
            logger.error(f"W&B logging failed: {e}")
    else:
        logger.warning("WANDB_API_KEY not set – metrics saved locally only.")
    
    # ── Save Metrics Locally ─────────────────────────────────────────────────
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "evaluation_metrics.json"
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Saved metrics -> {report_path}")
    return metrics

if __name__ == "__main__":
    run_model_evaluation()
