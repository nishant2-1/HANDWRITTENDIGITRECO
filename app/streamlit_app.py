"""Streamlit app for interactive handwritten digit prediction."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.predict import DEFAULT_TOP_K, DEFAULT_UNCERTAINTY_THRESHOLD, predict_digit_with_details

CLASS_LABELS: list[str] = [str(index) for index in range(10)]


st.set_page_config(page_title="Digit Recognition Demo", page_icon="🔢", layout="centered")
st.title("Handwritten Digit Recognition")
st.caption("Upload a handwritten digit image to get a prediction and confidence chart.")

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

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        image.save(temp_path)

    try:
        prediction = predict_digit_with_details(
            image_path=str(temp_path),
            top_k=top_k,
            uncertainty_threshold=uncertainty_threshold,
        )
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(f"Prediction failed: {exc}")
    else:
        predicted_digit: int = int(prediction["predicted_digit"])
        confidences: np.ndarray = prediction["confidences"]
        max_confidence: float = float(prediction["max_confidence"])
        processed_preview: np.ndarray = prediction["processed_preview"]

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

        st.subheader("Preprocessing Preview (28x28)")
        st.image(processed_preview, clamp=True, caption="Normalized image fed to the model", width=220)
    finally:
        temp_path.unlink(missing_ok=True)
else:
    st.info("Upload an image to start inference.")
