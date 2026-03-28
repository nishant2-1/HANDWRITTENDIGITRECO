"""CLI prediction utility for handwritten digit recognition."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import argparse
from typing import Any

import joblib
import numpy as np

from src.constants import MODEL_PATH
from .data_loader import load_custom_image, load_custom_image_with_preview

NUM_CLASSES: int = 10
DEFAULT_TOP_K: int = 3
DEFAULT_UNCERTAINTY_THRESHOLD: float = 0.60


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
    return predict_from_features(model=model, image_features=image_features)


def predict_from_features(model: Any, image_features: np.ndarray) -> tuple[int, np.ndarray]:
    """Predict digit and confidence from preprocessed feature vectors.

    Args:
        model: Trained classifier with predict and predict_proba methods.
        image_features: Feature vector shape (1, 784).

    Returns:
        tuple[int, np.ndarray]: Predicted digit and confidence array.
    """
    predicted_digit: int = int(model.predict(image_features)[0])
    confidences: np.ndarray = model.predict_proba(image_features)[0]
    return predicted_digit, confidences


def get_top_k_predictions(confidences: np.ndarray, top_k: int = DEFAULT_TOP_K) -> list[dict[str, float]]:
    """Return top-k prediction classes with confidence scores.

    Args:
        confidences: Confidence array for all classes.
        top_k: Number of top classes to return.

    Returns:
        list[dict[str, float]]: Sorted class-confidence records.
    """
    capped_k: int = min(top_k, confidences.shape[0])
    top_indices: np.ndarray = np.argsort(confidences)[::-1][:capped_k]
    return [
        {
            "class": int(class_idx),
            "confidence": float(confidences[class_idx]),
        }
        for class_idx in top_indices
    ]


def predict_digit_with_details(
    image_path: str,
    top_k: int = DEFAULT_TOP_K,
    uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
) -> dict[str, Any]:
    """Predict with extended details for UI/API consumers.

    Args:
        image_path: Path to image.
        top_k: Number of top classes to include.
        uncertainty_threshold: Confidence threshold to mark uncertain predictions.

    Returns:
        dict[str, Any]: Prediction payload with confidence and top-k classes.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    image_features, preview = load_custom_image_with_preview(image_path)
    predicted_digit, confidences = predict_from_features(model=model, image_features=image_features)
    max_confidence: float = float(np.max(confidences))

    return {
        "predicted_digit": predicted_digit,
        "confidences": confidences,
        "max_confidence": max_confidence,
        "is_uncertain": max_confidence < uncertainty_threshold,
        "uncertainty_threshold": float(uncertainty_threshold),
        "top_predictions": get_top_k_predictions(confidences, top_k=top_k),
        "processed_preview": preview,
    }


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
