import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time

# Constants
IMG_SIZE = (224, 224)
AD_CLASSES = ["Non Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]

# Show TensorFlow version
st.write(f"TensorFlow version: {tf.__version__}")

# Load binary MRI classifier model
@st.cache_resource
def load_mri_classifier():
    return tf.keras.models.load_model("efficientnet_model.keras")

# Load Alzheimer’s multiclass classifier
@st.cache_resource
def load_ad_classifier():
    return tf.keras.models.load_model("My4wayADefficientnet.keras")

# Preprocess image
def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(IMG_SIZE)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Load models
try:
    mri_classifier = load_mri_classifier()
    ad_classifier = load_ad_classifier()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# UI
st.title("🧠 MRI and Alzheimer's Stage Classifier")
st.write("Upload a brain scan image to determine if it's an MRI and if so, detect Alzheimer's stage.")

uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    preprocessed = preprocess_image(image)

    st.write(f"Preprocessed image shape: {preprocessed.shape}")
    st.write(f"Pixel range after rescaling: min={preprocessed.min()}, max={preprocessed.max()}")

    with st.spinner("🧪 Classifying..."):
        time.sleep(1)

        try:
            # Stage 1: MRI Classification
            mri_pred = mri_classifier.predict(preprocessed)
            mri_confidence = float(mri_pred[0][0])  # Ensure scalar

            if mri_confidence < 0.5:
                st.success("✅ This is likely an **MRI image**.")
                st.write(f"🔍 Confidence score (MRI probability): {1 - mri_confidence:.4f}")

                # Stage 2: Alzheimer's Classification
                with st.spinner("🧠 Detecting Alzheimer’s stage..."):
                    ad_pred = ad_classifier.predict(preprocessed)
                    ad_class_index = int(np.argmax(ad_pred))
                    ad_confidence = float(np.max(ad_pred))

                    st.info(f"🧬 Predicted Stage: **{AD_CLASSES[ad_class_index]}**")
                    st.write(f"🔎 Confidence: {ad_confidence:.4f}")
            else:
                st.warning("❌ This is likely **not** an MRI image.")
                st.write("Alzheimer’s stage classification: **Not Applicable** ❌")

        except Exception as e:
            st.error(f"Prediction error: {e}")
