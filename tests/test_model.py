"""Unit tests for handwritten digit recognition pipeline."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.data_loader import NUM_PIXELS, load_mnist_data, preprocess_images
from src.model import create_knn_model

NUM_SAMPLES: int = 4
IMAGE_HEIGHT: int = 28
IMAGE_WIDTH: int = 28
NUM_CLASSES: int = 10
TRAIN_SAMPLES: int = 8
TEST_SAMPLES: int = 4


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


def test_load_mnist_data_shapes_with_mocked_tf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure MNIST loader returns normalized, flattened train/test arrays."""
    x_train_raw: np.ndarray = np.random.randint(
        low=0,
        high=256,
        size=(TRAIN_SAMPLES, IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.uint8,
    )
    y_train_raw: np.ndarray = np.random.randint(0, NUM_CLASSES, size=(TRAIN_SAMPLES,))
    x_test_raw: np.ndarray = np.random.randint(
        low=0,
        high=256,
        size=(TEST_SAMPLES, IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.uint8,
    )
    y_test_raw: np.ndarray = np.random.randint(0, NUM_CLASSES, size=(TEST_SAMPLES,))

    mock_mnist_module = types.SimpleNamespace(
        load_data=lambda: ((x_train_raw, y_train_raw), (x_test_raw, y_test_raw))
    )
    mock_datasets_module = types.SimpleNamespace(mnist=mock_mnist_module)
    mock_keras_module = types.SimpleNamespace(datasets=mock_datasets_module)
    mock_tf_module = types.SimpleNamespace(keras=mock_keras_module)

    monkeypatch.setitem(sys.modules, "tensorflow", mock_tf_module)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", mock_keras_module)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.datasets", mock_datasets_module)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.datasets.mnist", mock_mnist_module)

    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    assert x_train.shape == (TRAIN_SAMPLES, NUM_PIXELS)
    assert x_test.shape == (TEST_SAMPLES, NUM_PIXELS)
    assert y_train.shape == (TRAIN_SAMPLES,)
    assert y_test.shape == (TEST_SAMPLES,)
    assert np.all(x_train >= 0.0)
    assert np.all(x_train <= 1.0)


def test_model_prediction_shape() -> None:
    """Ensure fitted model returns predictions with expected shape."""
    x_train: np.ndarray = np.random.rand(NUM_SAMPLES * 3, NUM_PIXELS).astype(np.float32)
    y_train: np.ndarray = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES * 3,))
    x_test: np.ndarray = np.random.rand(NUM_SAMPLES, NUM_PIXELS).astype(np.float32)

    model = create_knn_model(n_neighbors=3)
    model.fit(x_train, y_train)
    predictions: np.ndarray = model.predict(x_test)

    assert predictions.shape == (NUM_SAMPLES,)
