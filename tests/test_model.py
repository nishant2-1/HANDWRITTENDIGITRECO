"""Unit tests for handwritten digit recognition pipeline."""

from __future__ import annotations

import numpy as np

from src.data_loader import NUM_PIXELS, preprocess_images
from src.model import create_knn_model

NUM_SAMPLES: int = 4
IMAGE_HEIGHT: int = 28
IMAGE_WIDTH: int = 28
NUM_CLASSES: int = 10


def test_preprocess_images_output_shape_and_range() -> None:
    """Ensure preprocessing returns expected shape and normalized pixel range."""
    images: np.ndarray = np.random.randint(
        low=0,
        high=256,
        size=(NUM_SAMPLES, IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.uint8,
    )

    processed: np.ndarray = preprocess_images(images)

    assert processed.shape == (NUM_SAMPLES, NUM_PIXELS)
    assert np.all(processed >= 0.0)
    assert np.all(processed <= 1.0)


def test_model_prediction_shape() -> None:
    """Ensure fitted model returns predictions with expected shape."""
    x_train: np.ndarray = np.random.rand(NUM_SAMPLES * 3, NUM_PIXELS).astype(np.float32)
    y_train: np.ndarray = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES * 3,))
    x_test: np.ndarray = np.random.rand(NUM_SAMPLES, NUM_PIXELS).astype(np.float32)

    model = create_knn_model(n_neighbors=3)
    model.fit(x_train, y_train)
    predictions: np.ndarray = model.predict(x_test)

    assert predictions.shape == (NUM_SAMPLES,)
