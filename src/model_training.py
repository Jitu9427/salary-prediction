"""
src/model_training.py
═════════════════════
Model-Training Component.

Trains a pipeline composed of preprocessing (StandardScaler + OneHotEncoder)
and a StackingRegressor on the train_processed.csv data.
Logs hyperparameters, training metrics, and the model artifact to W&B.
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
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

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
    
    logger.info(f"Features - Num: {num_cols} | Cat: {cat_cols}")
    
    # ── Mimicking Notebook Preprocessing exactly ───────────────────
    # In the notebook, the target 'salary_usd' is scaled with StandardScaler
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            # OrdinalEncoder acts essentially like LabelEncoder for multiple columns
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
        ]
    )
    
    # ── Models ─────────────────────────────────────────────────────
    bm = params["base_models"]
    ml = params["meta_learner"]
    
    estimators = [
        ('lgb', LGBMRegressor(**bm["lightgbm"])),
        ('xgb', XGBRegressor(**bm["xgboost"]))
    ]
    
    cv_kfold = KFold(
        n_splits=ml["cv_folds"], 
        shuffle=ml.get("shuffle", True), 
        random_state=ml.get("random_state", 23)
    )
    
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=ml["alpha"]),
        cv=cv_kfold,
        n_jobs=-1
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', stacking)
    ])
    
    # ── W&B Run ────────────────────────────────────────────────────
    use_wandb = bool(os.getenv("WANDB_API_KEY"))
    run = None
    if use_wandb:
        run = wandb.init(
            project="salary-prediction",
            job_type="train",
            name="stacking_regressor_train",
            config=params   # Logs ALL hyperparams from params.yaml
        )
        logger.info("W&B run initialized – hyperparameters logged.")
    
    # ── Training ────────────────────────────────────────────────────
    logger.info("Fitting model pipeline ... this may take a moment.")
    pipeline.fit(X, y_scaled) # Fitting on scaled target to match notebook MAE
    logger.info("Model fitted successfully.")

    # ── Training Metrics ──────────────────────────────────────────
    y_train_pred = pipeline.predict(X)
    train_rmse = float(root_mean_squared_error(y_scaled, y_train_pred))
    train_r2   = float(r2_score(y_scaled, y_train_pred))
    logger.info(f"Train RMSE: {train_rmse:.4f} | Train R²: {train_r2:.4f}")
    
    # ── Saving ─────────────────────────────────────────────────────
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    fitted_preprocessor = pipeline.named_steps['preprocessor']
    stack_model = pipeline.named_steps['model']
    
    joblib.dump(fitted_preprocessor, models_dir / "preprocessor.pkl")
    joblib.dump(stack_model, models_dir / "stacking_model.pkl")
    joblib.dump(scaler_y, models_dir / "target_scaler.pkl") # Save to inverse transform later!
    
    logger.info("Saved models/preprocessor.pkl, stacking_model.pkl, and target_scaler.pkl")
    
    # ── Log Metrics + Artifact to W&B ──────────────────────────────
    if use_wandb and run:
        # Log training metrics
        wandb.log({
            "train/rmse": train_rmse,
            "train/r2":   train_r2,
        })
        logger.info("Training metrics logged to W&B (train/rmse, train/r2).")
        
        # Log model as a W&B Artifact
        artifact = wandb.Artifact(
            "model_artifacts",
            type="model",
            metadata={"train_rmse": train_rmse, "train_r2": train_r2}
        )
        artifact.add_dir(str(models_dir))
        run.log_artifact(artifact)
        logger.info("Model artifact uploaded to W&B.")
        
        wandb.finish()
        
    return {"status": "SUCCESS", "train_rmse": train_rmse, "train_r2": train_r2}

if __name__ == "__main__":
    run_model_training()
