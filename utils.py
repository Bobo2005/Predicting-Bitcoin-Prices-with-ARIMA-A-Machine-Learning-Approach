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
    """Write run metadata (config and metrics) to a JSON file.

    The file will be appended as a new JSON object per line for easy ingestion.
    """
    import datetime
    import json

    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        **metadata,
    }
    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


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
