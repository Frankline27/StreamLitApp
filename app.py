import streamlit as st
from PIL import Image
import numpy as np
import time
import tensorflow as tf  # Or torch, depending on your models

# ---- Load your models (replace with actual model loading) ----
@st.cache_resource
def load_models():
    mri_classifier = tf.keras.models.load_model("efficientnet_model.h5")  # Binary classifier
    detailed_classifier = tf.keras.models.load_model("My4wayADefficientnet.keras")  # 4-class classifier
    return mri_classifier, detailed_classifier

mri_classifier, detailed_classifier = load_models()

# ---- Helper: Preprocess image ----
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))  # Change size if required by model
    image_array = np.array(image) / 255.0
    if image_array.ndim == 2:  # Grayscale
        image_array = np.stack([image_array]*3, axis=-1)
    elif image_array.shape[2] == 4:  # RGBA
        image_array = image_array[..., :3]
    image_array = np.expand_dims(image_array, axis=0)
    return image_array.astype(np.float32)

# ---- UI Layout ----
st.title("MRI Image Classifier")
st.write("Upload an image to check if it is an MRI. If so, it will be classified into one of four categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    preprocessed = preprocess_image(image)

    with st.spinner("Checking if this is an MRI image..."):
        time.sleep(1)  # Simulate delay
        mri_pred = mri_classifier.predict(preprocessed)[0][0]
        is_mri = mri_pred > 0.5

    if is_mri:
        st.success("MRI image detected ✅")
        with st.spinner("Classifying MRI image into one of the 4 categories..."):
            time.sleep(1)
            class_probs = detailed_classifier.predict(preprocessed)[0]
            class_names = ["non demented" , "very mild demented" , "mild demented", "moderate demented"]  # Replace with actual names
            predicted_class = class_names[np.argmax(class_probs)]
            st.success(f"Prediction: **{predicted_class}**")
    else:
        st.warning("The image is not recognized as an MRI ❌")
