"""Shared constants and paths for the handwritten digit recognition project."""

from __future__ import annotations

from pathlib import Path

# Image preprocessing constants.
IMAGE_HEIGHT: int = 28
IMAGE_WIDTH: int = 28
NUM_PIXELS: int = IMAGE_HEIGHT * IMAGE_WIDTH
PIXEL_SCALE: float = 255.0
BINARIZATION_THRESHOLD: int = 127

# Model and artifact metadata.
MODEL_NAME: str = "knn_mnist"
MODEL_VERSION: str = "1.1.0"
METRICS_SCHEMA_VERSION: str = "1.1"

# Training and evaluation targets.
TARGET_ACCURACY: float = 0.95
TARGET_LATENCY_MS: float = 100.0

# Standard project paths.
MODEL_DIR: Path = Path("models")
MODEL_PATH: Path = MODEL_DIR / "knn_best.joblib"
RESULTS_DIR: Path = Path("results")
METRICS_PATH: Path = RESULTS_DIR / "metrics.json"
TRAINING_METRICS_PATH: Path = RESULTS_DIR / "training_metrics.json"
CONFUSION_MATRIX_PATH: Path = RESULTS_DIR / "confusion_matrix.png"
