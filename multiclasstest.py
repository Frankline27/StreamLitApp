"""
Streamlit App — Multiclass Dementia Stage Classifier
============================================================
EfficientNetB0, 4-class: non demented / very mild demented /
mild demented / moderate demented.

Model is hosted on Hugging Face (public repo) and downloaded at
runtime rather than committed to the GitHub repo directly — this
avoids the large-binary git issues encountered with the mould
project's earlier deployment attempt.

Class mapping (confirmed at training time, explicit class order
via flow_from_directory(classes=CLASSES)):
    non demented        -> 0
    very mild demented  -> 1
    mild demented       -> 2
    moderate demented   -> 3
"""

import streamlit as st
import numpy as np
import requests
from pathlib import Path
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ── Config ──────────────────────────────────────────────────────────────

HF_MODEL_URL = (
    "https://huggingface.co/NdahTah/MRIVsNonMRI/resolve/main/"
    "efficientnetb0_multiclass_best.h5"
)
LOCAL_MODEL_PATH = Path("efficientnetb0_multiclass_best.h5")

INPUT_SIZE = (224, 224)
CLASS_NAMES = ["Non Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]

st.set_page_config(page_title="Dementia Stage Classifier", layout="centered")

# ── Model download + loading (cached so it only happens once per session) ──

@st.cache_resource
def download_and_load_model():
    if not LOCAL_MODEL_PATH.exists():
        with st.spinner("Downloading model from Hugging Face (first run only)..."):
            response = requests.get(HF_MODEL_URL, stream=True)
            response.raise_for_status()
            with open(LOCAL_MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    model = load_model(str(LOCAL_MODEL_PATH))
    return model


try:
    model = download_and_load_model()
    model_load_error = None
except Exception as e:
    model = None
    model_load_error = str(e)

# ── UI ──────────────────────────────────────────────────────────────────

st.title("Dementia Stage Classifier")
st.write(
    "Upload a brain MRI image to classify dementia stage: "
    "**Non Demented**, **Very Mild**, **Mild**, or **Moderate**."
)

if model is None:
    st.error(f"Could not load model from Hugging Face.\n\nDetails: {model_load_error}")
    st.stop()

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
        prediction = model.predict(img_array, verbose=0)[0]  # shape: (4,)

    predicted_idx = int(np.argmax(prediction))
    predicted_label = CLASS_NAMES[predicted_idx]
    confidence = float(prediction[predicted_idx])

    st.subheader("Result")

    if predicted_idx == 0:
        st.success(f"**{predicted_label}**")
    elif predicted_idx == 1:
        st.warning(f"**{predicted_label}**")
    else:
        st.error(f"**{predicted_label}**")

    st.write(f"Confidence: {confidence * 100:.2f}%")

    with st.expander("Full probability breakdown"):
        for class_name, prob in zip(CLASS_NAMES, prediction):
            st.write(f"{class_name}: {prob * 100:.2f}%")
            st.progress(float(prob))

    with st.expander("How to read this"):
        st.write(
            "The model outputs a probability for each of the four dementia "
            "stages, and the highest-probability class is shown as the "
            "result. Confidence reflects how strongly the model favors "
            "that prediction over the other three stages."
        )
