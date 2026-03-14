"""
src/data_ingestion.py
═════════════════════
Data-Ingestion Component for the Salary-Prediction MLOps pipeline.

Responsibilities
────────────────
1. Read project configuration  (config/config.yaml)
2. Download the dataset from Kaggle using the official kaggle-python client
3. Validate the raw CSV (schema, null-check, dtype checks)
4. Stratified train / test split
5. Persist artefacts to  data/raw/  and  data/processed/
6. Write a JSON ingestion report  (reports/data_ingestion_report.json)
7. Emit structured logs at every step

Usage (standalone)
──────────────────
    python -m src.data_ingestion          # uses default config path
    python -m src.data_ingestion --config config/config.yaml

Design
──────
* DataIngestionConfig  – frozen dataclass built from config.yaml
* DataIngestionPipeline – orchestrator class; call .run()
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# ── Local utilities ────────────────────────────────────────────────────────────
# Ensure project root is on sys.path when run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger  # noqa: E402

load_dotenv()  # Picks up KAGGLE_USERNAME, KAGGLE_KEY, WANDB_API_KEY from .env

logger = get_logger("data_ingestion")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Configuration dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DataIngestionConfig:
    """Immutable configuration object for the data-ingestion stage."""

    # Kaggle
    kaggle_dataset: str
    kaggle_filename: str

    # Directories
    raw_data_dir: Path
    processed_data_dir: Path
    logs_dir: Path

    # Split
    test_size: float
    random_state: int
    stratify_col: Optional[str]

    # Output filenames
    train_filename: str
    test_filename: str

    # Report
    ingestion_report_path: Path

    # ── schema that the raw CSV MUST satisfy ──────────────────────────────────
    # We define the minimum required columns and their expected dtypes.
    required_columns: tuple[str, ...] = field(
        default=(
            "work_year",
            "experience_level",
            "employment_type",
            "job_title",
            "salary",
            "salary_currency",
            "salary_in_usd",
            "employee_residence",
            "remote_ratio",
            "company_location",
            "company_size",
        )
    )

    @property
    def raw_csv_path(self) -> Path:
        return self.raw_data_dir / self.kaggle_filename

    @property
    def train_path(self) -> Path:
        return self.processed_data_dir / self.train_filename

    @property
    def test_path(self) -> Path:
        return self.processed_data_dir / self.test_filename


def load_config(config_path: str | Path = "config/config.yaml") -> DataIngestionConfig:
    """Parse *config.yaml* and return a :class:`DataIngestionConfig`."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    di = cfg["data_ingestion"]
    dirs = cfg["directories"]
    kag = cfg["kaggle"]
    art = cfg["artifacts"]

    return DataIngestionConfig(
        kaggle_dataset=kag["dataset"],
        kaggle_filename=kag["filename"],
        raw_data_dir=Path(dirs["raw_data"]),
        processed_data_dir=Path(dirs["processed_data"]),
        logs_dir=Path(dirs["logs"]),
        test_size=float(di["test_size"]),
        random_state=int(di["random_state"]),
        stratify_col=di.get("stratify_col") or None,
        train_filename=di["train_filename"],
        test_filename=di["test_filename"],
        ingestion_report_path=Path(art["ingestion_report"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Pipeline class
# ══════════════════════════════════════════════════════════════════════════════

class DataIngestionPipeline:
    """
    End-to-end data-ingestion pipeline.

    Call :meth:`run` to execute all stages sequentially.

    Stages
    ------
    * ``_setup_directories``    – create folder hierarchy
    * ``_download_dataset``     – Kaggle → data/raw/
    * ``_validate_raw_data``    – schema + quality checks
    * ``_split_and_save``       – stratified train/test split → data/processed/
    * ``_write_report``         – JSON report → reports/
    """

    def __init__(self, config: DataIngestionConfig) -> None:
        self.cfg = config
        self._report: dict = {}

    # ── Public entry-point ────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute the full data-ingestion pipeline.

        Returns
        -------
        dict
            Ingestion report dictionary (also persisted to disk).
        """
        stage_start = time.perf_counter()
        logger.info("═" * 60)
        logger.info("  DATA INGESTION  –  Starting pipeline")
        logger.info("═" * 60)

        try:
            self._setup_directories()
            self._download_dataset()
            df = self._load_raw_data()
            df = self._validate_raw_data(df)
            self._split_and_save(df)
            self._write_report(elapsed=time.perf_counter() - stage_start)
        except Exception as exc:
            logger.exception("💥 Pipeline failed: %s", exc)
            raise

        logger.info("═" * 60)
        logger.info("  DATA INGESTION  –  Completed in %.2fs", time.perf_counter() - stage_start)
        logger.info("═" * 60)
        return self._report

    # ── Stage 1: Directory setup ───────────────────────────────────────────────

    def _setup_directories(self) -> None:
        logger.info("[Stage 1/5] Setting up directory structure …")
        for directory in (
            self.cfg.raw_data_dir,
            self.cfg.processed_data_dir,
            self.cfg.logs_dir,
            self.cfg.ingestion_report_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug("  ✔  %s", directory)
        logger.info("[Stage 1/5] ✅  Directories ready.")

    # ── Stage 2: Download ──────────────────────────────────────────────────────

    def _download_dataset(self) -> None:
        logger.info("[Stage 2/5] Downloading dataset from Kaggle …")

        # Validate credentials exist
        kaggle_user = os.getenv("KAGGLE_USERNAME")
        kaggle_key  = os.getenv("KAGGLE_KEY")
        if not kaggle_user or not kaggle_key:
            raise EnvironmentError(
                "KAGGLE_USERNAME and/or KAGGLE_KEY not set. "
                "Add them to your .env file."
            )

        # If raw CSV already exists, skip download
        if self.cfg.raw_csv_path.exists():
            logger.info(
                "  ⏭  Raw CSV already present at %s – skipping download.",
                self.cfg.raw_csv_path,
            )
            return

        # Use native Kaggle API instead of CLI for better robustness
        from kaggle.api.kaggle_api_extended import KaggleApi

        try:
            # Ensure the API picks up the vars from the current process env
            os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME', '')
            os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY', '')

            api = KaggleApi()
            api.authenticate()
            logger.info("  ▶  Downloading dataset via Kaggle API...")
            api.dataset_download_files(
                self.cfg.kaggle_dataset, 
                path=str(self.cfg.raw_data_dir), 
                unzip=False,
                force=True
            )
            logger.info("  ✔  Download complete.")
        except Exception as e:
            logger.error(f"Kaggle API Error: {e}")
            raise RuntimeError(f"Kaggle API download failed. Reason: {e}")

        # Handle case where download produces a zip instead of direct extraction
        zip_candidates = list(self.cfg.raw_data_dir.glob("*.zip"))
        for zf in zip_candidates:
            logger.info("  📦  Extracting %s …", zf.name)
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(self.cfg.raw_data_dir)
            zf.unlink()
            logger.debug("  🗑  Removed zip: %s", zf.name)

        if not self.cfg.raw_csv_path.exists():
            # Search one level deep for the CSV
            found = list(self.cfg.raw_data_dir.rglob(self.cfg.kaggle_filename))
            if found:
                shutil.move(str(found[0]), str(self.cfg.raw_csv_path))
                logger.info("  ✔  Moved %s → %s", found[0], self.cfg.raw_csv_path)
            else:
                raise FileNotFoundError(
                    f"Expected file '{self.cfg.kaggle_filename}' not found in "
                    f"{self.cfg.raw_data_dir} after download."
                )

        logger.info("[Stage 2/5] ✅  Dataset stored at %s", self.cfg.raw_csv_path)

    # ── Stage 3a: Load ────────────────────────────────────────────────────────

    def _load_raw_data(self) -> pd.DataFrame:
        logger.info("[Stage 3/5] Loading raw CSV …")
        df = pd.read_csv(self.cfg.raw_csv_path)
        logger.info(
            "  ✔  Loaded  shape=%s  columns=%s",
            df.shape,
            list(df.columns),
        )
        return df

    # ── Stage 3b: Validation ──────────────────────────────────────────────────

    def _validate_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("[Stage 3/5] Validating raw data …")
        issues: list[str] = []

        # 3b-i  Column presence
        missing_cols = [c for c in self.cfg.required_columns if c not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")

        if issues:
            for issue in issues:
                logger.error("  ✗  %s", issue)
            raise ValueError("Raw-data validation failed:\n" + "\n".join(issues))

        # 3b-ii  Null summary
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if not null_cols.empty:
            logger.warning(
                "  ⚠  Null values detected:\n%s",
                null_cols.to_string(),
            )
        else:
            logger.info("  ✔  No null values detected.")

        # 3b-iii  Duplicate rows
        n_dupes = df.duplicated().sum()
        if n_dupes > 0:
            logger.warning("  ⚠  %d duplicate rows detected – dropping.", n_dupes)
            df = df.drop_duplicates().reset_index(drop=True)
        else:
            logger.info("  ✔  No duplicate rows.")

        # 3b-iv  Target column sanity
        target = "salary_in_usd"
        if target in df.columns:
            neg_count = (df[target] <= 0).sum()
            if neg_count > 0:
                logger.warning(
                    "  ⚠  %d rows with non-positive %s – these may be errors.",
                    neg_count,
                    target,
                )
            logger.info(
                "  ✔  Target '%s'  min=%.0f  median=%.0f  max=%.0f",
                target,
                df[target].min(),
                df[target].median(),
                df[target].max(),
            )

        # 3b-v  Store basic stats for report
        self._report["raw_shape"] = list(df.shape)
        self._report["null_counts"] = null_cols.to_dict()
        self._report["duplicates_dropped"] = int(n_dupes)
        self._report["columns"] = list(df.columns)

        logger.info(
            "[Stage 3/5] ✅  Validation passed. Final shape: %s",
            df.shape,
        )
        return df

    # ── Stage 4: Split & save ─────────────────────────────────────────────────

    def _split_and_save(self, df: pd.DataFrame) -> None:
        logger.info("[Stage 4/5] Performing stratified train/test split …")

        stratify_arr = None
        if self.cfg.stratify_col and self.cfg.stratify_col in df.columns:
            stratify_arr = df[self.cfg.stratify_col]
            logger.info(
                "  ℹ  Stratifying on '%s'  (classes=%d)",
                self.cfg.stratify_col,
                stratify_arr.nunique(),
            )
        elif self.cfg.stratify_col:
            logger.warning(
                "  ⚠  Stratify column '%s' not in DataFrame – splitting without stratification.",
                self.cfg.stratify_col,
            )

        train_df, test_df = train_test_split(
            df,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=stratify_arr,
        )

        train_df = train_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)

        train_df.to_csv(self.cfg.train_path, index=False)
        test_df.to_csv(self.cfg.test_path, index=False)

        self._report["train_shape"] = list(train_df.shape)
        self._report["test_shape"]  = list(test_df.shape)
        self._report["train_path"]  = str(self.cfg.train_path)
        self._report["test_path"]   = str(self.cfg.test_path)

        logger.info(
            "  ✔  Train set: %s  →  %s",
            train_df.shape,
            self.cfg.train_path,
        )
        logger.info(
            "  ✔  Test set:  %s  →  %s",
            test_df.shape,
            self.cfg.test_path,
        )
        logger.info("[Stage 4/5] ✅  Splits saved successfully.")

    # ── Stage 5: Persist report ───────────────────────────────────────────────

    def _write_report(self, elapsed: float) -> None:
        logger.info("[Stage 5/5] Writing ingestion report …")

        self._report["status"]        = "SUCCESS"
        self._report["elapsed_secs"]  = round(elapsed, 3)
        self._report["kaggle_dataset"]= self.cfg.kaggle_dataset
        self._report["raw_csv"]       = str(self.cfg.raw_csv_path)

        self.cfg.ingestion_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.ingestion_report_path, "w", encoding="utf-8") as fh:
            json.dump(self._report, fh, indent=4)

        logger.info(
            "[Stage 5/5] ✅  Report saved → %s",
            self.cfg.ingestion_report_path,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Module-level helper (used by main.py / DVC stage)
# ══════════════════════════════════════════════════════════════════════════════

def run_data_ingestion(config_path: str | Path = "config/config.yaml") -> dict:
    """
    Convenience wrapper – build config, run pipeline, return report.

    Parameters
    ----------
    config_path : path to ``config.yaml``

    Returns
    -------
    dict  –  ingestion report
    """
    cfg      = load_config(config_path)
    pipeline = DataIngestionPipeline(cfg)
    return pipeline.run()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CLI entry-point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Data Ingestion stage of the Salary-Prediction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = run_data_ingestion(config_path=args.config)
    print("\n📋  Ingestion report summary:")
    print(json.dumps(report, indent=4))
