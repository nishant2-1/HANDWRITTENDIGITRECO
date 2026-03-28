"""FastAPI app exposing prediction endpoints for handwritten digit recognition."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.predict import DEFAULT_TOP_K, DEFAULT_UNCERTAINTY_THRESHOLD, predict_digit_with_details

ALLOWED_SUFFIXES: set[str] = {".png", ".jpg", ".jpeg", ".bmp"}

app = FastAPI(
    title="Handwritten Digit Recognition API",
    version="1.0.0",
    description="Upload an image and receive the predicted digit with confidence scores.",
)


def _validate_filename(filename: str) -> str:
    """Validate uploaded filename extension.

    Args:
        filename: Original uploaded filename.

    Returns:
        str: Normalized file suffix.

    Raises:
        HTTPException: If file extension is unsupported.
    """
    suffix: str = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=("Unsupported file type. Use one of: " + ", ".join(sorted(ALLOWED_SUFFIXES))),
        )
    return suffix


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health check endpoint.

    Returns:
        dict[str, str]: Service status payload.
    """
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    """Predict digit from an uploaded image.

    Args:
        file: Uploaded image file.

    Returns:
        dict[str, object]: Predicted digit and class confidence scores.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    suffix: str = _validate_filename(file.filename)
    file_bytes: bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(file_bytes)

    try:
        prediction: dict[str, Any] = predict_digit_with_details(
            image_path=str(temp_path),
            top_k=DEFAULT_TOP_K,
            uncertainty_threshold=DEFAULT_UNCERTAINTY_THRESHOLD,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive path for invalid images
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        "predicted_digit": int(prediction["predicted_digit"]),
        "confidences": [float(score) for score in prediction["confidences"]],
        "top_predictions": prediction["top_predictions"],
        "max_confidence": float(prediction["max_confidence"]),
        "is_uncertain": bool(prediction["is_uncertain"]),
        "uncertainty_threshold": float(prediction["uncertainty_threshold"]),
    }
