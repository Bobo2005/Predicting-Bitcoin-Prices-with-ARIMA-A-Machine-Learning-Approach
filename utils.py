from __future__ import annotations

import functools
import logging
import random
import time
from typing import Callable

import joblib
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for numpy and random to improve reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def save_model(obj, path: str) -> None:
    """Persist a fitted model or object using joblib."""
    joblib.dump(obj, path)


def load_model(path: str):
    """Load a joblib-saved object."""
    return joblib.load(path)


def log_run(output_path: str, metadata: dict) -> None:
    """Log run metadata and metrics with MLflow.

    The run will be logged to a local `mlruns` folder next to the provided output path.
    """
    from pathlib import Path

    try:
        import mlflow
    except ImportError:
        import datetime
        import json

        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            **metadata,
        }
        with open(output_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return

    tracking_dir = Path(output_path).resolve().parent / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(str(tracking_dir))
    mlflow.set_experiment("arima_forecast_experiment")

    with mlflow.start_run():
        for param_name, param_value in metadata.get("config", {}).items():
            mlflow.log_param(str(param_name), str(param_value))

        metrics = metadata.get("metrics", {})
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, dict):
                for sub_metric, sub_value in metric_value.items():
                    mlflow.log_metric(f"{metric_name}_{sub_metric}", float(sub_value))
            else:
                mlflow.log_metric(str(metric_name), float(metric_value))

        for artifact_name, artifact_path in metadata.get("artifacts", {}).items():
            try:
                mlflow.log_artifact(
                    str(artifact_path), artifact_path=str(artifact_name)
                )
            except Exception:
                continue


def retry(
    times: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type, ...] = (Exception,),
) -> Callable:
    """Decorator to retry a function call with exponential backoff.

    Usage:
        @retry(times=3, delay=1.0)
        def fetch(...):
            ...
    """

    def deco(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _delay = delay
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    logging.warning(f"Attempt {attempt} failed with {exc}")
                    if attempt == times:
                        raise
                    time.sleep(_delay)
                    _delay *= backoff

        return wrapper

    return deco
