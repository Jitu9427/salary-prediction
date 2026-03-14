"""
main.py  –  Pipeline Runner
════════════════════════════
Runs one or more DVC pipeline stages directly (useful for development
and debugging without invoking the full DVC pipeline).

Usage
─────
    # Run only data-ingestion
    python main.py --stage data_ingestion

    # Run all stages in sequence (future-proof)
    python main.py --stage all

Environment
───────────
Requires KAGGLE_USERNAME, KAGGLE_KEY (and optionally WANDB_API_KEY)
to be set in the .env file at the project root.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Ensure project root is importable ─────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.logger import get_logger  # noqa: E402

logger = get_logger("main")

# ── Available stages ──────────────────────────────────────────────────────────
STAGES: dict[str, str] = {
    "data_ingestion":    "src.data_ingestion",
    "data_preprocessing": "src.data_preprocessing",
    "data_validation":   "src.data_validation",
    "model_training":    "src.model_training",
    "model_evaluation":  "src.model_evaluation",
    "model_registration":"src.model_registration",
}


def run_stage(stage: str, config_path: str = "config/config.yaml") -> None:
    """Dynamically import and execute a pipeline stage."""
    if stage not in STAGES:
        raise ValueError(
            f"Unknown stage '{stage}'. Available: {list(STAGES.keys())}"
        )

    logger.info("━" * 60)
    logger.info("  STAGE : %s", stage.upper())
    logger.info("━" * 60)

    module_name = STAGES[stage]
    import importlib

    t0 = time.perf_counter()
    mod = importlib.import_module(module_name)

    # Each module exposes a canonical  run_<stage_name>()  function
    runner_fn_name = f"run_{stage}"
    runner_fn = getattr(mod, runner_fn_name, None)

    if runner_fn is None:
        raise AttributeError(
            f"Module '{module_name}' has no function '{runner_fn_name}'."
        )

    result = runner_fn(config_path=config_path)
    elapsed = time.perf_counter() - t0
    logger.info("  Stage '%s' completed in %.2fs.", stage, elapsed)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Salary-Prediction MLOps Pipeline Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()) + ["all"],
        default="data_ingestion",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    stages_to_run = list(STAGES.keys()) if args.stage == "all" else [args.stage]

    pipeline_start = time.perf_counter()
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  SALARY PREDICTION  –  MLOps Pipeline Runner" + " " * 13 + "║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("  Running stage(s): %s", stages_to_run)
    logger.info("  Config:           %s", args.config)

    for stage in stages_to_run:
        run_stage(stage, config_path=args.config)

    total = time.perf_counter() - pipeline_start
    logger.info("✅  All stages finished in %.2fs.", total)


if __name__ == "__main__":
    main()
