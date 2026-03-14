"""
src/model_registration.py
═════════════════════════
Model-Registration Component.

Reads `reports/evaluation_metrics.json`. If thresholds meet the
ones in `config/params.yaml`, logs the model as "production" to W&B.
"""

import sys
import json
import os
import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.logger import get_logger
import wandb

logger = get_logger("model_registration")

def run_model_registration(config_path: str = "config/config.yaml") -> dict:
    logger.info("━" * 60)
    logger.info("  MODEL REGISTRATION – Started")
    
    with open("config/params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
        
    metrics_path = Path("reports/evaluation_metrics.json")
    if not metrics_path.exists():
        logger.error(f"Metrics file not found: {metrics_path}")
        raise FileNotFoundError(f"{metrics_path} missing.")
        
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    eval_cfg = params.get("evaluation", {})
    min_r2 = eval_cfg.get("min_r2", 0.0)
    max_rmse = eval_cfg.get("max_rmse", 999999.0)
    
    r2 = metrics.get("r2", 0)
    rmse = metrics.get("rmse", float('inf'))
    
    logger.info(f"Target: R2 >= {min_r2}, RMSE <= {max_rmse}")
    logger.info(f"Actual: R2 = {r2:.4f}, RMSE = {rmse:.2f}")
    
    status = "REJECTED"
    
    if r2 >= min_r2 and rmse <= max_rmse:
        logger.info("✅ Model passed evaluation thresholds! Promoting to Production.")
        status = "ACCEPTED"
        
        # Log to W&B Model Registry if active
        use_wandb = bool(os.getenv("WANDB_API_KEY"))
        if use_wandb:
            try:
                # We initialize a light run just to log the model
                run = wandb.init(project="salary-prediction", job_type="register")
                
                # Create artifact
                model_artifact = wandb.Artifact(
                    "salary_stacking_model", 
                    type="model",
                    metadata={"rmse": rmse, "r2": r2}
                )
                model_artifact.add_dir("models")
                
                # Log to run and link to registry
                run.log_artifact(model_artifact, aliases=["latest", "production"])
                logger.info("Model registered in W&B Registry.")
                wandb.finish()
            except Exception as e:
                logger.error(f"Failed to register model to Weights & Biases: {e}")
    else:
        logger.warning("❌ Model failed to meet thresholds. Registration skipped.")
        
    return {"status": status, "metrics": metrics}

if __name__ == "__main__":
    run_model_registration()
