# ════════════════════════════════════════════════════════════════
# Dockerfile — Lean FastAPI App Image
# ════════════════════════════════════════════════════════════════
# Builds a minimal image (~550MB) that:
#   1. Installs only app dependencies (no DVC/training tools)
#   2. Downloads the production model from W&B on container startup
#   3. Serves the FastAPI prediction endpoint via uvicorn
# ════════════════════════════════════════════════════════════════

# ── Base: slim Python 3.11 for smallest footprint ───────────────
FROM python:3.11-slim

# ── Labels ──────────────────────────────────────────────────────
LABEL maintainer="salary-prediction-app"
LABEL description="AI Salary Prediction FastAPI App"
LABEL version="1.0"

# ── System dependencies (only what's needed for numpy/pandas) ───
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies first (leverages Docker cache) ──
COPY requirements_app.txt .
RUN pip install --no-cache-dir -r requirements_app.txt

# ── Copy only the files needed for the app ──────────────────────
# We deliberately DO NOT copy data/, models/, logs/, etc.
# The model is downloaded from W&B at container startup.
COPY app.py .
COPY src/ ./src/
COPY utils/ ./utils/
COPY config/ ./config/
COPY static/ ./static/
COPY templates/ ./templates/

# ── Create necessary directories ─────────────────────────────────
RUN mkdir -p models reports logs

# ── Expose port ──────────────────────────────────────────────────
EXPOSE 8000

# ── Environment variables (override at runtime via -e or .env) ──
# WANDB_API_KEY must be passed at runtime:  -e WANDB_API_KEY=xxxx
# KAGGLE credentials are NOT needed in the app container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# ── Health check ─────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# ── Startup command ───────────────────────────────────────────────
# On startup the lifespan event in app.py will automatically
# connect to W&B and download the 'production' model artifact.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
