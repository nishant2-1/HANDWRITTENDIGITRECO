"""Model evaluation utilities for handwritten digit recognition."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.constants import (
    CONFUSION_MATRIX_PATH,
    METRICS_PATH,
    METRICS_SCHEMA_VERSION,
    MODEL_NAME,
    MODEL_PATH,
    MODEL_VERSION,
    RESULTS_DIR,
    TARGET_LATENCY_MS,
    TRAINING_METRICS_PATH,
)
from .data_loader import load_mnist_data

NUM_CLASSES: int = 10


def _load_training_metrics() -> Dict[str, Any]:
    """Load training metrics if available.

    Returns:
        Dict[str, Any]: Training metrics dictionary or empty dictionary.
    """
    if not TRAINING_METRICS_PATH.exists():
        return {}

    with TRAINING_METRICS_PATH.open("r", encoding="utf-8") as file:
        return cast(Dict[str, Any], json.load(file))


def evaluate_model() -> Dict[str, Any]:
    """Evaluate the trained model and persist metrics artifacts.

    Returns:
        Dict[str, Any]: Dictionary containing accuracy, latency, report, and confusion matrix.

    Raises:
        FileNotFoundError: If trained model file does not exist.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    (_, _), (x_test, y_test) = load_mnist_data()

    start_time: float = time.perf_counter()
    y_pred: np.ndarray = model.predict(x_test)
    elapsed_seconds: float = time.perf_counter() - start_time

    latency_ms_per_image: float = (elapsed_seconds / max(x_test.shape[0], 1)) * 1000.0
    accuracy: float = float(accuracy_score(y_test, y_pred))

    report: Dict[str, Any] = cast(
        Dict[str, Any],
        classification_report(y_test, y_pred, output_dict=True),
    )
    conf_matrix: np.ndarray = confusion_matrix(y_test, y_pred)

    save_confusion_matrix(conf_matrix)

    training_metrics: Dict[str, Any] = _load_training_metrics()

    metrics: Dict[str, Any] = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **training_metrics,
        "accuracy": accuracy,
        "latency_ms_per_image": latency_ms_per_image,
        "target_latency_ms": TARGET_LATENCY_MS,
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


def save_confusion_matrix(conf_matrix: np.ndarray) -> None:
    """Plot and save confusion matrix heatmap.

    Args:
        conf_matrix: Confusion matrix array with shape (10, 10).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("MNIST Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()


if __name__ == "__main__":
    evaluate_model()
