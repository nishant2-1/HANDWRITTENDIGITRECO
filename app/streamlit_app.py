"""Streamlit app for interactive handwritten digit prediction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.predict import predict_digit

CLASS_LABELS: list[str] = [str(index) for index in range(10)]


st.set_page_config(page_title="Digit Recognition Demo", page_icon="🔢", layout="centered")
st.title("Handwritten Digit Recognition")
st.caption("Upload a handwritten digit image to get a prediction and confidence chart.")

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
        predicted_digit, confidences = predict_digit(str(temp_path))
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(f"Prediction failed: {exc}")
    else:
        st.success(f"Predicted digit: {predicted_digit}")
        confidence_map = {
            CLASS_LABELS[index]: float(value) for index, value in enumerate(confidences)
        }
        st.bar_chart(confidence_map)
        st.write("Top confidence:", float(np.max(confidences)))
    finally:
        temp_path.unlink(missing_ok=True)
else:
    st.info("Upload an image to start inference.")
