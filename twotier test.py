import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from huggingface_hub import hf_hub_download
import os
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="MRI Dementia Staging Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 MRI Dementia Staging Classifier")
st.write("Upload a medical image to classify it as MRI vs Non-MRI, then stage dementia if MRI")

# Model loading with caching
@st.cache_resource
def load_models():
    """Download and load both models from Hugging Face"""
    models = {}
    
    # Load first model: MRI vs Non-MRI
    with st.spinner("📥 Downloading MRI vs Non-MRI model..."):
        try:
            model1_path = hf_hub_download(
                repo_id="NdahTah/MRIVsNonMRI",
                filename="efficientnetb0 mri deploy.h5",
                cache_dir="./model_cache"
            )
            models["mri_vs_non"] = load_model(model1_path, compile=False)
        except Exception as e:
            st.error(f"❌ Error loading MRI vs Non-MRI model: {e}")
            st.stop()
    
    # Load second model: Dementia staging (4-way classification)
    with st.spinner("📥 Downloading Dementia staging model..."):
        try:
            model2_path = hf_hub_download(
                repo_id="NdahTah/MRIVsNonMRI",
                filename="My4wayADefficientnet.keras",
                cache_dir="./model_cache"
            )
            models["dementia_staging"] = load_model(model2_path, compile=False)
        except Exception as e:
            st.error(f"❌ Error loading Dementia staging model: {e}")
            st.stop()
    
    return models

# Load models
try:
    models = load_models()
    mri_model = models["mri_vs_non"]
    dementia_model = models["dementia_staging"]
    st.success("✅ Both models loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
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
                
                # === STAGE 1: MRI vs Non-MRI Classification ===
                predictions_mri = mri_model.predict(img_array, verbose=0)
                
                # Interpret Stage 1 results (using your fixed logic)
                if predictions_mri.shape[-1] == 1:
                    probability_non_mri = float(predictions_mri[0][0])
                    probability_mri = 1 - probability_non_mri
                    
                    if probability_mri > 0.5:
                        stage1_class = "MRI"
                        stage1_confidence = probability_mri
                    else:
                        stage1_class = "Non-MRI"
                        stage1_confidence = probability_non_mri
                else:
                    probability_mri = float(predictions_mri[0][0])
                    probability_non_mri = float(predictions_mri[0][1])
                    
                    if probability_mri > 0.5:
                        stage1_class = "MRI"
                        stage1_confidence = probability_mri
                    else:
                        stage1_class = "Non-MRI"
                        stage1_confidence = probability_non_mri
                
                # === STAGE 2: Dementia Staging (only if MRI) ===
                stage2_result = None
                stage2_probs = None
                dementia_classes = [
                    "Non-Demented",
                    "Mild Demented",
                    "Moderate Demented",
                    "Severe Demented"
                ]
                
                if stage1_class == "MRI":
                    # Use the dementia staging model
                    predictions_dementia = dementia_model.predict(img_array, verbose=0)
                    
                    # Handle different output shapes
                    if predictions_dementia.shape[-1] == 1:
                        # Binary case (unlikely for 4-way, but safe)
                        stage2_index = 0 if float(predictions_dementia[0][0]) > 0.5 else 1
                        stage2_confidence = float(predictions_dementia[0][0])
                        stage2_probs = np.array([1 - stage2_confidence, stage2_confidence, 0, 0])
                    else:
                        # Multi-class classification (4 classes)
                        stage2_index = np.argmax(predictions_dementia[0])
                        stage2_confidence = float(predictions_dementia[0][stage2_index])
                        stage2_probs = predictions_dementia[0]
                    
                    stage2_class = dementia_classes[stage2_index]
                
                # === DISPLAY RESULTS ===
                st.subheader("📊 Prediction Results")
                
                # Stage 1: MRI vs Non-MRI
                st.write("### Stage 1: MRI vs Non-MRI")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", stage1_class)
                with col2:
                    st.metric("Confidence", f"{stage1_confidence:.2%}")
                
                # Progress bar for MRI probability
                st.write("#### Probability Distribution")
                st.progress(float(probability_mri))
                st.caption(f"MRI probability: {probability_mri:.2%} | Non-MRI probability: {probability_non_mri:.2%}")
                
                # Stage 2: Dementia Staging (only if MRI)
                if stage1_class == "MRI":
                    st.write("---")
                    st.write("### Stage 2: Dementia Staging")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Dementia Stage", stage2_class)
                    with col2:
                        st.metric("Confidence", f"{stage2_confidence:.2%}")
                    
                    # Display all class probabilities for dementia staging
                    if stage2_probs is not None:
                        st.write("#### Class Probabilities")
                        
                        # Create a bar chart for probabilities
                        fig, ax = plt.subplots()
                        colors = ['#2ecc71' if i == stage2_index else '#e74c3c' for i in range(len(dementia_classes))]
                        ax.barh(dementia_classes, stage2_probs, color=colors)
                        ax.set_xlim(0, 1)
                        ax.set_xlabel("Probability")
                        ax.set_title("Dementia Stage Probabilities")
                        st.pyplot(fig)
                    
                    st.success(f"🧠 The image is classified as an MRI scan with **{stage1_confidence:.2%}** confidence, and the dementia stage is **{stage2_class}**.")
                else:
                    st.info(f"📷 The image is classified as a non-MRI scan with **{stage1_confidence:.2%}** confidence. No dementia staging performed.")
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
                st.write("Full error:", str(e))

# Instructions and info
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    **How it works:**
    1. Upload a medical image (JPG, JPEG, PNG, or DICOM)
    2. **Stage 1**: The first model checks if it's an MRI or Non-MRI scan
    3. **Stage 2**: If it's an MRI, the second model stages dementia severity
    4. Results include both classifications with confidence scores
    
    **Dementia Stages:**
    - **Non-Demented**: No signs of dementia
    - **Mild Demented**: Early stage dementia
    - **Moderate Demented**: Progressive dementia
    - **Severe Demented**: Advanced dementia
    
    **About:**
    - This app uses two EfficientNetB0 models trained on medical images
    - Both models are hosted on Hugging Face and downloaded on-demand
    - The two-stage approach ensures accurate dementia staging only for MRI scans
    """)

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Models hosted on Hugging Face")
