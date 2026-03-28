"""CLI prediction utility for handwritten digit recognition."""

from __future__ import annotations

import argparse

import joblib
import numpy as np

from src.constants import MODEL_PATH
from .data_loader import load_custom_image

NUM_CLASSES: int = 10


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for prediction.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Predict handwritten digit from image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file")
    return parser.parse_args()


def predict_digit(image_path: str) -> tuple[int, np.ndarray]:
    """Predict digit class and confidence scores.

    Args:
        image_path: Path to input image.

    Returns:
        tuple[int, np.ndarray]: Predicted class and confidence scores for classes 0-9.

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    image_features: np.ndarray = load_custom_image(image_path)

    predicted_digit: int = int(model.predict(image_features)[0])

    confidences: np.ndarray = model.predict_proba(image_features)[0]

    return predicted_digit, confidences


def main() -> None:
    """Run the prediction CLI entry point."""
    args = parse_args()
    predicted_digit, confidences = predict_digit(args.image)

    print(f"Predicted digit: {predicted_digit}")
    print("Confidence scores:")
    for class_idx, score in enumerate(confidences):
        print(f"  Class {class_idx}: {score:.4f}")


if __name__ == "__main__":
    main()
