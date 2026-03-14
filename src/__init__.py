"""src – pipeline components for the Salary-Prediction MLOps project."""

from src.data_ingestion import DataIngestionConfig, DataIngestionPipeline, run_data_ingestion
from src.data_preprocessing import run_data_preprocessing
from src.data_validation import run_data_validation
from src.model_training import run_model_training
from src.model_evaluation import run_model_evaluation
from src.model_registration import run_model_registration

__all__ = [
    "DataIngestionConfig",
    "DataIngestionPipeline",
    "run_data_ingestion",
    "run_data_preprocessing",
    "run_data_validation",
    "run_model_training",
    "run_model_evaluation",
    "run_model_registration",
]
