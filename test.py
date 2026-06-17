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
            model_path = hf_hub_download(
                repo_id="NdahTah/MRIVsNonMRI",
                filename="efficientnetb0 mri deploy.h5",
                cache_dir="./model_cache"
            )
            
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
    
    # Automatic classification
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
            
            # --- DIAGNOSTIC: Show raw prediction ---
            with st.expander("🔬 Debug Info (Raw Prediction)"):
                st.write(f"Raw model output: {predictions[0]}")
                st.write(f"Shape: {predictions.shape}")
                if predictions.shape[-1] == 1:
                    st.write(f"Single neuron output: {float(predictions[0][0]):.4f}")
                    st.write("If this is > 0.5, model thinks it's Class 1")
                    st.write("If this is < 0.5, model thinks it's Class 0")
                else:
                    st.write(f"Class 0 probability: {float(predictions[0][0]):.4f}")
                    st.write(f"Class 1 probability: {float(predictions[0][1]):.4f}")
            
            # --- PREDICTION LOGIC ---
            # Option A: If your training had Class 0 = Non-MRI, Class 1 = MRI
            # Option B: If your training had Class 0 = MRI, Class 1 = Non-MRI
            
            # CHOOSE ONE BASED ON YOUR TRAINING:
            
            # OPTION 1: Class 0 = Non-MRI, Class 1 = MRI (most common)
            if predictions.shape[-1] == 1:
                # Single neuron - probability for Class 1 (MRI)
                probability_mri = float(predictions[0][0])
                probability_non_mri = 1 - probability_mri
                
                if probability_mri > 0.5:
                    predicted_class = "MRI"
                    confidence = probability_mri
                else:
                    predicted_class = "Non-MRI"
                    confidence = probability_non_mri
            else:
                # Two neurons - index 0 = Non-MRI, index 1 = MRI
                probability_non_mri = float(predictions[0][0])
                probability_mri = float(predictions[0][1])
                
                if probability_mri > 0.5:
                    predicted_class = "MRI"
                    confidence = probability_mri
                else:
                    predicted_class = "Non-MRI"
                    confidence = probability_non_mri
            
            # --- OPTION 2: If your training had Class 0 = MRI, Class 1 = Non-MRI
            # Uncomment this block and comment OPTION 1 if needed:
            """
            if predictions.shape[-1] == 1:
                probability_non_mri = float(predictions[0][0])  # Actually MRI probability
                probability_mri = 1 - probability_non_mri
                
                if probability_mri > 0.5:
                    predicted_class = "MRI"
                    confidence = probability_mri
                else:
                    predicted_class = "Non-MRI"
                    confidence = probability_non_mri
            else:
                probability_mri = float(predictions[0][0])  # Index 0 = MRI
                probability_non_mri = float(predictions[0][1])  # Index 1 = Non-MRI
                
                if probability_mri > 0.5:
                    predicted_class = "MRI"
                    confidence = probability_mri
                else:
                    predicted_class = "Non-MRI"
                    confidence = probability_non_mri
            """
            
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
    3. Classification happens automatically!
    4. Check the Debug Info expander to see raw predictions
    
    **Troubleshooting:**
    - If predictions are inverted, check the Debug Info
    - See which class has higher probability
    - The app uses Option 1 (Class 0 = Non-MRI, Class 1 = MRI)
    - If your training used different class ordering, switch to Option 2
    """)

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Model hosted on Hugging Face")
