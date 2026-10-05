"""Tracking wrapper for MLflow runs."""

import logging
from contextlib import contextmanager

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

logger = logging.getLogger(__name__)

@contextmanager
def start_run(run_name: str | None = None, experiment_name: str = "ReadmitIQ"):
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_name) as run:
                yield run
        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {e}. Continuing without tracking.")
            yield None
    else:
        yield None

def log_params(params: dict) -> None:
    if HAS_MLFLOW:
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.debug(f"Failed to log params: {e}")

def log_metrics(metrics: dict, step: int | None = None) -> None:
    if HAS_MLFLOW:
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.debug(f"Failed to log metrics: {e}")
