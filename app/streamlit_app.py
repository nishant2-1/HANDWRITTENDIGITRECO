"""Streamlit app for interactive handwritten digit prediction.

Supports two modes controlled by the ``API_URL`` environment variable:

- **API mode** (cloud): Set ``API_URL`` to your deployed Vercel API base URL
  (e.g. ``https://your-project.vercel.app``). Predictions are sent as HTTP
  requests to the ``/predict`` endpoint. Use this when deploying on Streamlit
  Community Cloud pointing at the Vercel backend.

- **Local mode** (development): Leave ``API_URL`` unset. Predictions run
  directly via the local ``src.predict`` module using the saved model file.
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import os
from typing import Any

import httpx
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit.errors import StreamlitSecretNotFoundError

# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

CLASS_LABELS: list[str] = [str(index) for index in range(10)]
DEFAULT_TOP_K: int = 3
DEFAULT_UNCERTAINTY_THRESHOLD: float = 0.60

# Read the optional Vercel API base URL from the environment.
# Set via Streamlit secrets (TOML) or a plain environment variable.


def _get_api_url() -> str:
    """Read API URL from env or Streamlit secrets without crashing if secrets are missing."""
    env_url = os.environ.get("API_URL", "").strip()
    if env_url:
        return env_url

    try:
        return str(st.secrets.get("API_URL", "")).strip()  # type: ignore[attr-defined]
    except StreamlitSecretNotFoundError:
        return ""


_API_URL: str = _get_api_url()
_USE_API: bool = bool(_API_URL)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def _predict_via_api(image_bytes: bytes, filename: str) -> dict[str, Any]:
    """Call the deployed Vercel /predict endpoint and return a normalised payload."""
    url = _API_URL.rstrip("/") + "/predict"
    with httpx.Client(timeout=30) as client:
        response = client.post(url, files={"file": (filename, image_bytes, "image/png")})
    response.raise_for_status()
    data: dict[str, Any] = response.json()

    # Build a confidences array so the rest of the UI code works unchanged.
    data["confidences"] = np.array(data["confidences"])
    # API does not return a preprocessed preview; supply a blank placeholder.
    data["processed_preview"] = np.zeros((28, 28), dtype=np.float32)
    return data


def _predict_locally(image_bytes: bytes) -> dict[str, Any]:
    """Run prediction in-process using the saved local model."""
    import tempfile
    from pathlib import Path
    from src.predict import predict_digit_with_details

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp_path.write_bytes(image_bytes)

    try:
        return predict_digit_with_details(
            image_path=str(tmp_path),
            top_k=DEFAULT_TOP_K,
            uncertainty_threshold=DEFAULT_UNCERTAINTY_THRESHOLD,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Digit Recognition Demo", page_icon="🔢", layout="centered")
st.title("Handwritten Digit Recognition")
st.caption("Upload a handwritten digit image to get a prediction and confidence chart.")

if _USE_API:
    st.sidebar.info(f"**Mode:** Cloud API\n\n`{_API_URL}`")
else:
    st.sidebar.info("**Mode:** Local model")

top_k: int = st.sidebar.slider("Top-K Predictions", min_value=1, max_value=5, value=DEFAULT_TOP_K)
uncertainty_threshold: float = st.sidebar.slider(
    "Uncertainty Threshold",
    min_value=0.40,
    max_value=0.95,
    value=DEFAULT_UNCERTAINTY_THRESHOLD,
    step=0.01,
)

uploaded_file = st.file_uploader(
    "Upload image",
    type=["png", "jpg", "jpeg", "bmp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded image", width=220)

    image_bytes: bytes = uploaded_file.getvalue()

    try:
        if _USE_API:
            prediction = _predict_via_api(image_bytes, uploaded_file.name)
        else:
            prediction = _predict_locally(image_bytes)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    else:
        predicted_digit: int = int(prediction["predicted_digit"])
        confidences: np.ndarray = np.array(prediction["confidences"])
        max_confidence: float = float(prediction["max_confidence"])
        processed_preview: np.ndarray = np.array(prediction["processed_preview"])

        if bool(prediction["is_uncertain"]):
            st.warning(
                "Low confidence prediction. The image may not contain a clear handwritten digit."
            )
        else:
            st.success(f"Predicted digit: {predicted_digit}")

        st.metric(label="Top Confidence", value=f"{max_confidence * 100:.2f}%")

        confidence_map = {
            CLASS_LABELS[index]: float(value) for index, value in enumerate(confidences)
        }
        st.bar_chart(confidence_map)

        top_df = pd.DataFrame(prediction["top_predictions"])
        top_df["confidence"] = top_df["confidence"].map(lambda score: f"{score * 100:.2f}%")
        top_df.rename(columns={"class": "digit"}, inplace=True)
        st.subheader("Top Predictions")
        st.table(top_df)

        if not _USE_API:
            st.subheader("Preprocessing Preview (28x28)")
            st.image(
                processed_preview,
                clamp=True,
                caption="Normalized image fed to the model",
                width=220,
            )
else:
    st.info("Upload an image to start inference.")
