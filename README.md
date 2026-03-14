# AI Salary Prediction – End-to-End MLOps Pipeline

This repository contains a full MLOps pipeline for predicting data science and AI job salaries based on features like experience level, employment type, remotely ratio, and company location. 

The project enforces strict reproducibility using **DVC** (Data Version Control) and experiment tracking via **Weights & Biases** (W&B).

## 🚀 Pipeline Architecture

The machine learning lifecycle is split into 6 modular stages defined in `dvc.yaml`:

1. **`data_ingestion`**: Downloads the `ruchi798/data-science-job-salaries` dataset directly via the Kaggle API, validates the schema, and creates stratified 80/20 train/test splits.
2. **`data_preprocessing`**: Removes outliers on the training set using IQR methodology, and drops redundant features (e.g., local currency/salary amounts).
3. **`data_validation`**: Validates the processed splits to ensure schema targets, null checks, and shape consistency.
4. **`model_training`**: Fits a highly robust Scikit-Learn pipeline (StandardScaler + OneHotEncoder) wrapped around a **Stacking Regressor** (Random Forest + Gradient Boosting + Ridge Meta-Learner). Logs all metrics, hyperparameters, and the serialized model artifacts (`preprocessor.pkl`, `stacking_model.pkl`) to Weights & Biases.
5. **`model_evaluation`**: Evaluates the model strictly on the held-out test split, calculating RMSE, MAE, and R². Evaluated metrics are saved as JSON artifacts for DVC tracking.
6. **`model_registration`**: If the model evaluation gracefully beats the predefined baseline thresholds (`min_r2` and `max_rmse` set in `params.yaml`), the model is automatically registered into the W&B Model Registry with a `production` alias.

## 🛠️ Tech Stack
* **Language & Modelling:** Python 3.9+, pandas, scikit-learn
* **Pipeline Orchestration:** DVC (Data Version Control)
* **Tracking & Registry:** Weights & Biases
* **Config Management:** YAML (`config.yaml` and `params.yaml`)
* **Logging:** Centralised, rotating colour-formatted log files

## 📦 Setup & Installation

**1. Clone the repository and install dependencies:**
```bash
git clone https://github.com/your-username/salary-prediction.git
cd salary_prediction
pip install -r requirements.txt
```

**2. Configure Environment Variables:**
Create a `.env` file at the root of the project with your API keys:
```env
# Required for downloading the dataset
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# Required for experiment tracking and model registry
WANDB_API_KEY=your_wandb_api_key
```

**3. Run the Pipeline:**
You can orchestrate the entire end-to-end pipeline efficiently using DVC:
```bash
dvc repro
```
*DVC intelligently skips stages that haven't changed (e.g. it won't repeatedly download the dataset from Kaggle if `data/raw/ds_salaries.csv` is already cached locally).*

To run a single stage in isolation (for debugging):
```bash
python main.py --stage model_training
```

## ⚙️ Configuration & Hyperparameters
All paths, pipeline metadata, and ML hyperparameters are strictly decoupled from the codebase:
- `config/config.yaml`: Defines directory paths, dataset names, ingestion split sizes.
- `config/params.yaml`: Centralised hyperparameter configs for models, outlier thresholds, and evaluation targets. Editing this file automatically invalidates the relevant DVC stages.