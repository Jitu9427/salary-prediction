"""
src/model_server.py
═══════════════════
Utility to fetch the latest production model from Weights & Biases
Model Registry for serving via the FastAPI application.
"""

import os
import sys
from pathlib import Path
import logging

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.logger import get_logger
import wandb
from dotenv import load_dotenv

load_dotenv()
logger = get_logger("model_server")

def download_production_model() -> Path:
    """
    Connects to the W&B API, locates the registered model with alias 'production',
    and downloads it to the local models/ directory for inference.
    
    Returns:
        Path to the local directory where the model was downloaded.
    """
    logger.info("Initializing connection to W&B Model Registry...")
    
    # Must use wandb.Api() for interacting with artifacts without starting a run
    try:
        api = wandb.Api()
        
        # We need the W&B username/entity. In typical usage, 'entity/project/model_name:alias'
        # Since project is salary-prediction, we get the default entity for the user.
        entity = api.default_entity
        project = "salary-prediction"
        model_name = "salary_stacking_model"
        alias = "production"
        
        artifact_path = f"{entity}/{project}/{model_name}:{alias}"
        logger.info(f"Looking up artifact: {artifact_path}")
        
        artifact = api.artifact(artifact_path)
        
        logger.info(f"Found production model (v{artifact.version}). Downloading...")
        
        download_dir = Path("models").resolve()
        artifact.download(root=str(download_dir))
        
        logger.info(f"Model downloaded successfully to {download_dir}")
        return download_dir
        
    except Exception as e:
        logger.error(f"Failed to fetch production model from W&B: {e}")
        logger.warning("Falling back to any existing local models in 'models/' directory.")
        return Path("models").resolve()

if __name__ == "__main__":
    download_production_model()
