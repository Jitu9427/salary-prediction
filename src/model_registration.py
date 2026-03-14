"""
src/model_registration.py
═════════════════════════
Model-Registration Component.

Reads `reports/evaluation_metrics.json`. If thresholds meet the
ones in `config/params.yaml`, registers the model to the W&B Model
Registry using `link_artifact()` with 'production' alias via the
officially supported W&B Registry API.
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
    min_r2   = eval_cfg.get("min_r2", 0.0)
    max_rmse = eval_cfg.get("max_rmse", 999999.0)
    
    r2   = metrics.get("r2", 0)
    rmse = metrics.get("rmse", float('inf'))
    mae  = metrics.get("mae", float('inf'))
    
    logger.info(f"Target Thresholds : R² >= {min_r2}, RMSE <= {max_rmse}")
    logger.info(f"Model Performance : R² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}")
    
    status = "REJECTED"
    
    if r2 >= min_r2 and rmse <= max_rmse:
        logger.info("✅ Model passed evaluation thresholds! Promoting to Production.")
        status = "ACCEPTED"
        
        use_wandb = bool(os.getenv("WANDB_API_KEY"))
        if use_wandb:
            try:
                # Start a run dedicated to model registration
                run = wandb.init(
                    project="salary-prediction",
                    job_type="register",
                    name="model_registration"
                )
                
                # Create and upload model artifact with metadata
                model_artifact = wandb.Artifact(
                    name="salary_stacking_model",
                    type="model",
                    description="Stacking Regressor (RF + GBM + Ridge) for AI salary prediction",
                    metadata={
                        "rmse":  rmse,
                        "mae":   mae,
                        "r2":    r2,
                        "eval_thresholds": {"min_r2": min_r2, "max_rmse": max_rmse}
                    }
                )
                model_artifact.add_dir("models")

                # Log the artifact with aliases – the artifact appears in the
                # W&B project Artifacts tab with 'production' and 'latest' aliases
                # (link_artifact is not used here as the registry was migrated)
                logged_artifact = run.log_artifact(model_artifact, aliases=["latest", "production"])
                logged_artifact.wait()  # Ensure upload completes before finishing run
                
                # Log the final metrics as summary to this registration run too
                wandb.log({
                    "registered/rmse": rmse,
                    "registered/mae":  mae,
                    "registered/r2":   r2
                })
                
                logger.info("Model artifact with aliases ['latest', 'production'] uploaded to W&B.")
                logger.info(f"   View at: https://wandb.ai/{run.entity}/salary-prediction/artifacts/model")
                wandb.finish()
                
            except Exception as e:
                logger.error(f"Failed to register model to Weights & Biases: {e}")
        else:
            logger.warning("WANDB_API_KEY not set – registration skipped.")
    else:
        logger.warning("❌ Model failed to meet thresholds. Registration skipped.")
        
    return {"status": status, "metrics": metrics}

if __name__ == "__main__":
    run_model_registration()
