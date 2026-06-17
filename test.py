import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from huggingface_hub import hf_hub_download
import os

# Page configuration
st.set_page_config(
    page_title="MRI vs Non-MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 MRI vs Non-MRI Image Classifier")
st.write("Upload a medical image to classify it as MRI or Non-MRI")

# Model loading with caching
@st.cache_resource
def load_model_from_hf():
    """Download and load the model from Hugging Face"""
    with st.spinner("📥 Downloading model from Hugging Face... This may take a moment."):
        try:
            # Download the model file
            model_path = hf_hub_download(
                repo_id="NdahTah/MRIVsNonMRI",
                filename="efficientnetb0 mri deploy.h5",
                cache_dir="./model_cache"  # Cache to avoid re-downloading
            )
            
            # Load the model
            model = load_model(model_path, compile=False)
            return model
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.stop()

# Load the model
try:
    model = load_model_from_hf()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Image upload section
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "dcm"],
    help="Upload a medical image (JPG, JPEG, PNG, or DICOM)"
)

if uploaded_file is not None:
    # Display the uploaded image
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"❌ Error opening image: {e}")
        st.stop()
    
    # Prediction button
    if st.button("🔍 Classify Image", type="primary"):
        with st.spinner("🔍 Analyzing image..."):
            try:
                # Preprocess for EfficientNetB0
                img = image.resize((224, 224))
                img_array = np.array(img)
                
                # Ensure RGB format
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array]*3, axis=-1)
                elif img_array.shape[-1] == 4:
                    img_array = img_array[:, :, :3]
                
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                
                # Make prediction
                predictions = model.predict(img_array, verbose=0)
                
                # --- UPDATED: STANDARD PREDICTION LOGIC ---
                # The new model works correctly - no inversion needed
                if predictions.shape[-1] == 1:
                    # Single neuron output - probability for class 1 (MRI)
                    probability_mri = float(predictions[0][0])
                    probability_non_mri = 1 - probability_mri
                    
                    # Determine class based on MRI probability
                    if probability_mri > 0.5:
                        predicted_class = "MRI"
                        confidence = probability_mri
                    else:
                        predicted_class = "Non-MRI"
                        confidence = probability_non_mri
                else:
                    # For 2-class softmax, index 0 is Non-MRI, index 1 is MRI
                    probability_non_mri = float(predictions[0][0])
                    probability_mri = float(predictions[0][1])
                    
                    if probability_mri > 0.5:
                        predicted_class = "MRI"
                        confidence = probability_mri
                    else:
                        predicted_class = "Non-MRI"
                        confidence = probability_non_mri
                
                # Display results
                st.subheader("📊 Prediction Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", predicted_class)
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Progress bar for visual representation
                st.write("### Probability Distribution")
                st.progress(float(probability_mri))
                st.caption(f"MRI probability: {probability_mri:.2%} | Non-MRI probability: {probability_non_mri:.2%}")
                
                # Additional info based on prediction
                if predicted_class == "MRI":
                    st.info("🧠 This image appears to be an MRI scan.")
                else:
                    st.info("📷 This image does not appear to be an MRI scan.")
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

# Instructions and info
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    **Instructions:**
    1. Click 'Browse files' to upload a medical image
    2. Supported formats: JPG, JPEG, PNG, DICOM
    3. Click 'Classify Image' to analyze
    4. The model will classify it as MRI or Non-MRI
    
    **About:**
    - This app uses an EfficientNetB0 model trained on medical images
    - The model is hosted on Hugging Face and downloaded on-demand
    - Results include prediction and confidence score
    """)

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Model hosted on Hugging Face")
