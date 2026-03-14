"""
src/model_training.py
═════════════════════
Model-Training Component.

Trains a pipeline composed of preprocessing (StandardScaler + OneHotEncoder)
and a StackingRegressor on the train_processed.csv data.
Logs experiments to Weights & Biases if configured.
"""

import sys
import os
import joblib
import pandas as pd
import yaml
from pathlib import Path

# scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.logger import get_logger
import wandb

logger = get_logger("model_training")

def run_model_training(config_path: str = "config/config.yaml") -> dict:
    logger.info("━" * 60)
    logger.info("  MODEL TRAINING – Started")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open("config/params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
        
    proc_dir = Path(cfg["directories"]["processed_data"])
    train_path = proc_dir / "train_processed.csv"
    
    df = pd.read_csv(train_path)
    target = params["preprocessing"].get("target_col", "salary_in_usd")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    
    logger.info(f"Found {len(num_cols)} numerical and {len(cat_cols)} categorical features.")
    
    # ── Preprocessor ───────────────────────────────────────────────
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    # ── Models ─────────────────────────────────────────────────────
    bm = params["base_models"]
    ml = params["meta_learner"]
    
    estimators = [
        ('rf', RandomForestRegressor(**bm["random_forest"])),
        ('gbm', GradientBoostingRegressor(**bm["gradient_boosting"])),
        ('ridge', Ridge(**bm["ridge"]))
    ]
    
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=ml["alpha"]),
        cv=ml["cv_folds"],
        n_jobs=-1
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', stacking)
    ])
    
    # ── Tracking ───────────────────────────────────────────────────
    use_wandb = bool(os.getenv("WANDB_API_KEY"))
    if use_wandb:
        wandb.init(project="salary-prediction", job_type="train", config=params)
    
    logger.info("Fitting model pipeline ... this may take a moment.")
    pipeline.fit(X, y)
    logger.info("Model fitted successfully.")
    
    # ── Saving ─────────────────────────────────────────────────────
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Dump separating prep and model to match DVC spec
    # It's better to dump both if someone needs them separately, or just dump the whole object.
    # We will dump the separated components for modularity, and also the pipeline if needed.
    # Actually, DVC expects models/stacking_model.pkl and models/preprocessor.pkl
    
    # Extract fitted parts from the pipeline
    fitted_preprocessor = pipeline.named_steps['preprocessor']
    stack_model = pipeline.named_steps['model']
    
    joblib.dump(fitted_preprocessor, models_dir / "preprocessor.pkl")
    joblib.dump(stack_model, models_dir / "stacking_model.pkl")
    
    logger.info("Saved models/preprocessor.pkl and models/stacking_model.pkl")
    
    if use_wandb:
        # Save models as artifact
        artifact = wandb.Artifact("model_artifacts", type="model")
        artifact.add_dir("models")
        wandb.log_artifact(artifact)
        wandb.finish()
        
    return {"status": "SUCCESS", "models_saved": True}

if __name__ == "__main__":
    run_model_training()
