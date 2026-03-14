"""
app.py
══════
Main FastAPI application entry point.
Serves a sleek dynamic HTML frontend and exposes a `/predict` JSON endpoint
using the production models pulled from Weights & Biases.
"""

import sys
import os
import joblib
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# Ensure local imports work
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.logger import get_logger
from src.model_server import download_production_model

logger = get_logger("fastapi_app")

# ── Global Model State ────────────────────────────────────────────────────────
models_cache = {
    "preprocessor": None,
    "model": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup tasks:
    1. Fetch the latest production model from W&B Model Registry to local /models.
    2. Load preprocessor and model into memory.
    """
    logger.info("Starting up FastAPI application...")
    
    # 1. Download production artifacts
    model_dir = download_production_model()
    prep_path = model_dir / "preprocessor.pkl"
    model_path = model_dir / "stacking_model.pkl"
    
    # 2. Load into memory
    if prep_path.exists() and model_path.exists():
        logger.info("Loading preprocessor and model into memory...")
        models_cache["preprocessor"] = joblib.load(prep_path)
        models_cache["model"] = joblib.load(model_path)
        logger.info("✅ Models loaded successfully. Ready for inference!")
    else:
        logger.error(f"Models not found at {model_dir}! Inference will fail.")
        
    yield
    
    # Teardown logic here if necessary
    logger.info("Shutting down application...")

# ── FastAPI Initialization ────────────────────────────────────────────────────
app = FastAPI(
    title="AI Salary Predictor",
    description="End-to-End MLOps FastAPI Serving Layer",
    version="1.0.0",
    lifespan=lifespan
)

# ── Static & Templates ────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ── Pydantic Request Model ────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    work_year: int
    experience_level: str
    employment_type: str
    job_title: str
    employee_residence: str
    remote_ratio: int
    company_location: str
    company_size: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serve the premium dynamic HTML frontend."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_salary(req: PredictionRequest):
    """
    Inference endpoint:
    Takes structured JSON → converts to DataFrame → standardizes/encodes → stacking prediction → returns salary
    """
    if models_cache["preprocessor"] is None or models_cache["model"] is None:
        raise HTTPException(
            status_code=503, 
            detail="Models are not loaded on server. Please ensure they were downloaded correctly."
        )
        
    try:
        # 1. Convert payload to DataFrame exactly matching the training data schema
        df = pd.DataFrame([req.model_dump()])
        df["Unnamed: 0"] = 0  # Add missing index column expected by preprocessor
        
        # 2. Transform numericals & categoricals
        X_trans = models_cache["preprocessor"].transform(df)
        
        # 3. Predict via Stacking Regressor
        pred = models_cache["model"].predict(X_trans)[0]
        
        # 4. Return formatted response
        logger.info(f"Made prediction for {req.job_title} ({req.experience_level}) -> ${pred:,.2f}")
        return {
            "predicted_salary_usd": round(pred, 2),
            "formatted_salary": f"${pred:,.0f} USD"
        }
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    logger.info("Run the server using: uvicorn app:app --reload")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
