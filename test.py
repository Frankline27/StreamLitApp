"""
Streamlit App — Demented vs Non-Demented MRI Classifier
============================================================
EfficientNetB0 binary classification, single run.

Loads "efficientnetb0 deploy.keras" if present, otherwise falls
back to "efficientnetb0 deploy.h5". Both are expected to sit in
the same folder as this app.py file.

Class mapping (confirmed at training time via flow_from_directory):
    demented      -> 0
    non demented  -> 1
Sigmoid output close to 0 = demented, close to 1 = non demented.
"""

import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ── Config ──────────────────────────────────────────────────────────────

APP_DIR = Path(__file__).parent

MODEL_NAME_KERAS = "efficientnetb0 deploy.keras"
MODEL_NAME_H5    = "efficientnetb0 deploy.h5"

INPUT_SIZE = (224, 224)  # (width, height) for PIL resize
CLASS_NAMES = {0: "Demented", 1: "Non Demented"}
THRESHOLD = 0.5

st.set_page_config(page_title="Dementia MRI Classifier", layout="centered")

# ── Model loading (cached so it only loads once per session) ────────────

@st.cache_resource
def load_classifier():
    keras_path = APP_DIR / MODEL_NAME_KERAS
    h5_path    = APP_DIR / MODEL_NAME_H5

    if keras_path.exists():
        model = load_model(str(keras_path))
        return model, str(keras_path)
    elif h5_path.exists():
        model = load_model(str(h5_path))
        return model, str(h5_path)
    else:
        return None, None


model, loaded_from = load_classifier()

# ── UI ──────────────────────────────────────────────────────────────────

st.title("Dementia MRI Classifier")
st.write("Upload a brain MRI image to classify it as **Demented** or **Non Demented**.")

if model is None:
    st.error(
        f"Could not find a model file. Expected either "
        f"`{MODEL_NAME_KERAS}` or `{MODEL_NAME_H5}` in the same folder as app.py."
    )
    st.stop()

st.caption(f"Model loaded from: `{Path(loaded_from).name}`")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Force RGB — handles grayscale source images, same as training pipeline
    image_rgb = image.convert("RGB")

    st.image(image_rgb, caption="Uploaded image", use_container_width=True)

    # Preprocess: resize to model input size, apply EfficientNet preprocessing
    resized = image_rgb.resize(INPUT_SIZE)
    img_array = np.array(resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    img_array = preprocess_input(img_array)

    with st.spinner("Classifying..."):
        prediction = model.predict(img_array, verbose=0)
        prob = float(prediction[0][0])

    predicted_class = 1 if prob > THRESHOLD else 0
    label = CLASS_NAMES[predicted_class]

    # Confidence: how far the sigmoid output is from the decision boundary,
    # expressed as confidence in the predicted class
    confidence = prob if predicted_class == 1 else (1 - prob)

    st.subheader("Result")
    if predicted_class == 0:
        st.error(f"**{label}**")
    else:
        st.success(f"**{label}**")

    st.write(f"Confidence: {confidence * 100:.2f}%")
    st.write(f"Raw sigmoid output: {prob:.4f}")

    with st.expander("How to read this"):
        st.write(
            "The model outputs a single probability between 0 and 1. "
            "Values close to 0 indicate **Demented**, values close to 1 "
            "indicate **Non Demented**. The decision threshold is 0.5."
        )
