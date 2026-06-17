import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from huggingface_hub import hf_hub_download
import os
import matplotlib.pyplot as plt
import json
import tempfile

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Page configuration
st.set_page_config(
    page_title="MRI Dementia Staging Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 MRI Dementia Staging Classifier")
st.write("Upload a medical image to classify it as MRI vs Non-MRI, then stage dementia if MRI")

def load_model_safe(model_path):
    """
    Custom function to load Keras models while handling compatibility issues.
    Tries multiple strategies to load the model.
    """
    try:
        # Strategy 1: Try loading with custom_objects and compile=False
        return load_model(model_path, compile=False, custom_objects={})
    except Exception as e1:
        st.warning(f"⚠️ Direct load failed: {str(e1)[:100]}...")
        
        try:
            # Strategy 2: Try loading with legacy format
            with tf.keras.utils.custom_object_scope({}):
                return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e2:
            st.warning(f"⚠️ Legacy load failed: {str(e2)[:100]}...")
            
            try:
                # Strategy 3: Reconstruct model from config
                return reconstruct_model_from_weights(model_path)
            except Exception as e3:
                st.error(f"❌ All loading strategies failed: {str(e3)}")
                raise

def reconstruct_model_from_weights(model_path):
    """
    Reconstruct the model architecture and load weights from the saved file.
    This bypasses the InputLayer configuration issues.
    """
    try:
        # Create a new model with the same architecture
        base_model = EfficientNetB0(
            include_top=False,
            weights=None,  # We'll load weights from file
            input_shape=(224, 224, 3)
        )
        
        # Add custom classification head (match your training architecture)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(4, activation='softmax')(x)  # 4 classes for dementia staging
        
        # Build the model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Try to load weights from the saved file
        try:
            # Load the original model to get weights
            original_model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={}
            )
            
            # Copy weights layer by layer (if architectures match)
            for layer in model.layers:
                try:
                    original_layer = original_model.get_layer(layer.name)
                    layer.set_weights(original_layer.get_weights())
                except:
                    st.warning(f"⚠️ Could not load weights for layer: {layer.name}")
                    continue
                    
            return model
            
        except Exception as e:
            st.warning(f"⚠️ Weight transfer failed: {e}. Trying alternative method...")
            
            # Alternative: Use tf.saved_model to load
            try:
                saved_model = tf.saved_model.load(model_path)
                # Extract and rebuild weights
                # This is a simplified version - may need adjustment
                return saved_model
            except:
                # Final fallback: Try with older Keras format
                try:
                    with tf.compat.v1.Session().as_default():
                        return tf.keras.models.load_model(model_path, compile=False)
                except:
                    raise Exception("Unable to load the model file. Please check if it's a valid TensorFlow model.")
                    
    except Exception as e:
        raise Exception(f"Model reconstruction failed: {e}")

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
            
            # Use the custom safe loader
            models["dementia_staging"] = load_model_safe(model2_path)
            
        except Exception as e:
            st.error(f"❌ Error loading Dementia staging model: {e}")
            st.error("Please check if the model file is accessible and not corrupted.")
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
                
                # Interpret Stage 1 results
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
                    try:
                        # Use the dementia staging model
                        predictions_dementia = dementia_model.predict(img_array, verbose=0)
                        
                        # Handle different output shapes
                        if len(predictions_dementia.shape) == 2:
                            # Multi-class classification
                            stage2_index = np.argmax(predictions_dementia[0])
                            stage2_confidence = float(predictions_dementia[0][stage2_index])
                            stage2_probs = predictions_dementia[0]
                            stage2_class = dementia_classes[stage2_index]
                        else:
                            # Fallback for unexpected output
                            st.warning("⚠️ Unexpected model output shape, using fallback interpretation")
                            stage2_class = "Staging Unavailable"
                            stage2_confidence = 0.0
                            stage2_probs = np.array([0.25, 0.25, 0.25, 0.25])
                    except Exception as e:
                        st.warning(f"⚠️ Dementia staging failed: {e}")
                        stage2_class = "Staging Error"
                        stage2_confidence = 0.0
                        stage2_probs = None
                
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
                    
                    if stage2_class and stage2_class not in ["Staging Unavailable", "Staging Error"]:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Dementia Stage", stage2_class)
                        with col2:
                            st.metric("Confidence", f"{stage2_confidence:.2%}")
                        
                        # Display all class probabilities for dementia staging
                        if stage2_probs is not None and len(stage2_probs) >= 4:
                            st.write("#### Class Probabilities")
                            
                            # Create a bar chart for probabilities
                            fig, ax = plt.subplots()
                            colors = ['#2ecc71' if i == stage2_index else '#e74c3c' for i in range(len(dementia_classes))]
                            ax.barh(dementia_classes, stage2_probs[:4], color=colors)
                            ax.set_xlim(0, 1)
                            ax.set_xlabel("Probability")
                            ax.set_title("Dementia Stage Probabilities")
                            st.pyplot(fig)
                        
                        st.success(f"🧠 The image is classified as an MRI scan with **{stage1_confidence:.2%}** confidence, and the dementia stage is **{stage2_class}**.")
                    else:
                        st.warning(f"⚠️ Dementia staging could not be completed. The image is an MRI, but staging failed.")
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
