import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys
from tensorflow.keras.applications.efficientnet import preprocess_input

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EfficientNet Binary Test",
    page_icon="🧪",
    layout="centered"
)

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)

# ── Load model with version handling ────────────────────────────────────────
@st.cache_resource
def load_binary_model():
    """Load the binary EfficientNet model with version compatibility"""
    try:
        model_path = "efficientnet_binary_classification.h5"
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            st.stop()
        
        # Show TensorFlow version
        st.sidebar.write(f"TensorFlow version: {tf.__version__}")
        
        # Try loading with different methods
        try:
            # Method 1: Standard load with compile=False
            model = tf.keras.models.load_model(model_path, compile=False)
            st.sidebar.success("✅ Loaded with standard method")
            return model
            
        except Exception as e1:
            st.sidebar.warning(f"Standard load failed: {str(e1)[:100]}...")
            
            try:
                # Method 2: Load with legacy format
                import h5py
                with h5py.File(model_path, 'r') as f:
                    st.sidebar.write(f"File keys: {list(f.keys())}")
                
                # Try loading with custom_objects
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False,
                    custom_objects={
                        'InputLayer': tf.keras.layers.InputLayer,
                        'Functional': tf.keras.models.Model
                    }
                )
                st.sidebar.success("✅ Loaded with custom_objects")
                return model
                
            except Exception as e2:
                # Method 3: Try with Keras 3 compatibility
                try:
                    # Set Keras backend to TensorFlow (if using Keras 3)
                    if hasattr(tf.keras, 'backend'):
                        st.sidebar.write(f"Keras backend: {tf.keras.backend.backend()}")
                    
                    # Try loading with legacy format flag
                    model = tf.keras.models.load_model(
                        model_path,
                        compile=False,
                        safe_mode=False  # Disable safe mode for compatibility
                    )
                    st.sidebar.success("✅ Loaded with safe_mode=False")
                    return model
                    
                except Exception as e3:
                    st.sidebar.error(f"❌ All loading methods failed")
                    st.sidebar.error(f"Final error: {e3}")
                    raise e3
                    
    except Exception as e:
        st.error(f"❌ Model loading error: {e}")
        st.error("💡 Try converting your model to .h5 format locally with TensorFlow 2.15")
        st.stop()

# ── Preprocess ──────────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for EfficientNet"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(IMG_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array

# ── UI ──────────────────────────────────────────────────────────────────────
st.title("🧪 EfficientNet Binary Classification Test")

# Show system info
col1, col2 = st.columns(2)
with col1:
    st.metric("TensorFlow", tf.__version__)
with col2:
    st.metric("Python", sys.version.split()[0])

# Load model
model = load_binary_model()

# Show model info if loaded
if model is not None:
    with st.expander("📊 Model Info"):
        st.write(f"Input shape: {model.input_shape}")
        st.write(f"Output shape: {model.output_shape}")
        st.write(f"Total parameters: {model.count_params():,}")

st.divider()

uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("🧪 Classifying..."):
            try:
                preprocessed = preprocess_image(image)
                prediction = model.predict(preprocessed, verbose=0)
                
                # Handle different output shapes
                if len(prediction.shape) == 2 and prediction.shape[1] == 1:
                    prob = float(prediction[0][0])
                elif len(prediction.shape) == 1:
                    prob = float(prediction[0])
                elif len(prediction.shape) == 2 and prediction.shape[1] == 2:
                    prob = float(prediction[0][1])
                else:
                    st.error(f"Unexpected prediction shape: {prediction.shape}")
                    st.stop()
                
                # Interpret as binary
                if prob < 0.5:
                    label = "Mold"
                    confidence = (1 - prob) * 100
                else:
                    label = "No Mold"
                    confidence = prob * 100
                
                # Display result
                st.markdown(f"""
                <div style="padding: 1.5rem; border-radius: 15px; background: {'#ff6b6b' if label == 'Mold' else '#51cf66'}; text-align: center;">
                    <h2 style="color: white; margin: 0;">{label}</h2>
                    <p style="color: white; margin: 0.5rem 0;">Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📊 Details"):
                    st.write(f"Raw prediction: {prediction}")
                    st.write(f"Probability: {prob:.6f}")
                    st.write(f"Threshold: 0.5")
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

else:
    st.info("👆 Please upload an image to begin.")
