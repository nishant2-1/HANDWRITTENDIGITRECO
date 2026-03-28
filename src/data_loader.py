"""Data loading and preprocessing utilities for handwritten digit recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

IMAGE_HEIGHT: int = 28
IMAGE_WIDTH: int = 28
NUM_PIXELS: int = IMAGE_HEIGHT * IMAGE_WIDTH
PIXEL_SCALE: float = 255.0


ArrayPair = Tuple[np.ndarray, np.ndarray]
DatasetSplit = Tuple[ArrayPair, ArrayPair]


def load_mnist_data() -> DatasetSplit:
    """Load and preprocess MNIST dataset.

    Returns:
        DatasetSplit: Tuple of train and test splits where images are normalized
            to [0, 1] and flattened to shape (n_samples, 784).
    """
    try:
        from tensorflow.keras.datasets import mnist
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required to load MNIST. Install a TensorFlow build compatible with your Python version."
        ) from exc

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_processed: np.ndarray = preprocess_images(x_train)
    x_test_processed: np.ndarray = preprocess_images(x_test)

    return (x_train_processed, y_train), (x_test_processed, y_test)


def preprocess_images(images: np.ndarray) -> np.ndarray:
    """Normalize and flatten image arrays using vectorized NumPy operations.

    Args:
        images: Array of shape (n_samples, 28, 28) or (28, 28).

    Returns:
        np.ndarray: Processed array with shape (n_samples, 784) and dtype float32.
    """
    image_array: np.ndarray = np.asarray(images, dtype=np.float32)

    if image_array.ndim == 2:
        image_array = np.expand_dims(image_array, axis=0)

    normalized: np.ndarray = image_array / PIXEL_SCALE
    flattened: np.ndarray = normalized.reshape(normalized.shape[0], NUM_PIXELS)
    return flattened


def load_custom_image(image_path: str) -> np.ndarray:
    """Load and preprocess a custom handwritten image from disk.

    Supports PIL loading with OpenCV enhancement path for robust grayscale handling.

    Args:
        image_path: Path to an input image.

    Returns:
        np.ndarray: Processed image with shape (1, 784).

    Raises:
        FileNotFoundError: If image does not exist.
        ValueError: If image cannot be loaded or processed.
    """
    path: Path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    pil_image: Image.Image = Image.open(path).convert("L").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    pil_array: np.ndarray = np.asarray(pil_image, dtype=np.uint8)

    cv_array: np.ndarray | None = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if cv_array is not None:
        resized_cv: np.ndarray = cv2.resize(
            cv_array,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        # Combine PIL and OpenCV grayscale interpretations for stable preprocessing.
        merged: np.ndarray = (
            (pil_array.astype(np.float32) + resized_cv.astype(np.float32)) / 2.0
        ).astype(np.uint8)
    else:
        merged = pil_array

    inverted: np.ndarray = np.where(merged > 127, 0, 255).astype(np.uint8)
    return preprocess_images(inverted)
