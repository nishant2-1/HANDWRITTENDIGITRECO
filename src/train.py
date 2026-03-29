"""Training pipeline for handwritten digit recognition."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.constants import (
    METRICS_SCHEMA_VERSION,
    MODEL_NAME,
    MODEL_PATH,
    MODEL_VERSION,
    RESULTS_DIR,
    TARGET_ACCURACY,
    TRAINING_METRICS_PATH,
)
from .data_loader import load_mnist_data
from .model import DEFAULT_CV_SPLITS, TrainingResult, tune_and_train_model

DEFAULT_TRAIN_SAMPLES: int = 12000
DEFAULT_TEST_SAMPLES: int = 2000


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def _save_training_metrics(
    training_result: TrainingResult,
    test_accuracy: float,
    elapsed_seconds: float,
    full_mode: bool,
) -> None:
    """Save training metrics to JSON for downstream reporting.

    Args:
        training_result: Training result object from model tuning.
        test_accuracy: Accuracy measured on the test split used during training.
        elapsed_seconds: End-to-end training time in seconds.
        full_mode: Whether full training mode was used.
    """
    cv_mean: float = float(np.mean(training_result.cv_scores))
    cv_std: float = float(np.std(training_result.cv_scores))

    metrics_payload: Dict[str, Any] = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_mode": "full" if full_mode else "fast",
        "training_seconds": elapsed_seconds,
        "test_accuracy": test_accuracy,
        "best_params": training_result.best_params,
        "best_cv_score": training_result.best_score,
        "cv_scores": training_result.cv_scores.tolist(),
        "cv_mean_accuracy": cv_mean,
        "cv_std_accuracy": cv_std,
        "target_accuracy": TARGET_ACCURACY,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with TRAINING_METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training configuration.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train KNN model for handwritten digit recognition."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full dataset training and full parameter grid (slower).",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES,
        help="Number of training samples to use in fast mode.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=DEFAULT_TEST_SAMPLES,
        help="Number of test samples to use in fast mode.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=DEFAULT_CV_SPLITS,
        help="Number of CV splits for K-Fold validation.",
    )
    return parser.parse_args()


def _sample_dataset(
    x_data: np.ndarray,
    y_data: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice arrays to a deterministic subset size.

    Args:
        x_data: Feature array.
        y_data: Label array.
        sample_count: Number of samples to keep.

    Returns:
        tuple[np.ndarray, np.ndarray]: Sliced features and labels.
    """
    capped_count: int = min(sample_count, x_data.shape[0])
    return x_data[:capped_count], y_data[:capped_count]


def train_pipeline(
    full_mode: bool = False,
    train_samples: int = DEFAULT_TRAIN_SAMPLES,
    test_samples: int = DEFAULT_TEST_SAMPLES,
    cv_splits: int = DEFAULT_CV_SPLITS,
) -> tuple[float, float, np.ndarray]:
    """Run end-to-end training and persist the best model.

    Args:
        full_mode: Whether to train on full datasets and full parameter grid.
        train_samples: Sample size used in fast mode for training split.
        test_samples: Sample size used in fast mode for test split.
        cv_splits: Number of folds for cross-validation.

    Returns:
        tuple[float, float, np.ndarray]: Test accuracy, training time seconds, and CV scores.
    """
    progress_steps: list[str] = [
        "Loading and preprocessing MNIST data",
        "Running hyperparameter tuning with GridSearchCV",
        "Evaluating best model on test set",
        "Saving trained model",
    ]

    start_time: float = time.perf_counter()

    for _ in tqdm(progress_steps, desc="Training Pipeline", unit="step"):
        if _ == progress_steps[0]:
            (x_train, y_train), (x_test, y_test) = load_mnist_data()
            if not full_mode:
                x_train, y_train = _sample_dataset(x_train, y_train, train_samples)
                x_test, y_test = _sample_dataset(x_test, y_test, test_samples)
            LOGGER.info("Data loaded: train=%s, test=%s", x_train.shape, x_test.shape)

        if _ == progress_steps[1]:
            training_result = tune_and_train_model(
                x_train,
                y_train,
                cv_splits=cv_splits,
                fast_mode=not full_mode,
            )
            LOGGER.info("Best params: %s", training_result.best_params)
            LOGGER.info("Best CV score: %.4f", training_result.best_score)

        if _ == progress_steps[2]:
            y_pred: np.ndarray = training_result.best_model.predict(x_test)
            test_accuracy: float = float(accuracy_score(y_test, y_pred))
            LOGGER.info("Test accuracy: %.4f", test_accuracy)

        if _ == progress_steps[3]:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(training_result.best_model, MODEL_PATH)
            LOGGER.info("Model saved to %s", MODEL_PATH)

    elapsed: float = time.perf_counter() - start_time
    LOGGER.info("Training completed in %.2f seconds", elapsed)

    if test_accuracy < TARGET_ACCURACY:
        LOGGER.warning(
            "Target accuracy %.2f not reached (current: %.4f)",
            TARGET_ACCURACY,
            test_accuracy,
        )

    _save_training_metrics(
        training_result=training_result,
        test_accuracy=test_accuracy,
        elapsed_seconds=elapsed,
        full_mode=full_mode,
    )
    LOGGER.info("Training metrics saved to %s", TRAINING_METRICS_PATH)

    return test_accuracy, elapsed, training_result.cv_scores


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(
        full_mode=args.full,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        cv_splits=args.cv_splits,
    )
