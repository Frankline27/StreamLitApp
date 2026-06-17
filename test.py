import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EfficientNet Binary Test",
    page_icon="🧪",
    layout="centered"
)

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)

# ── Load model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_binary_model():
    """Load the binary EfficientNet model"""
    try:
        # Check if file exists
        if not os.path.exists("efficientnet_binary_classification.h5"):
            st.error("❌ Model file not found: efficientnet_binary_classification.h5")
            st.stop()
        
        # Load model
        model = tf.keras.models.load_model("efficientnet_binary_classification.h5", compile=False)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading error: {e}")
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
st.write(f"TensorFlow version: {tf.__version__}")

# Load model
model = load_binary_model()

st.divider()

uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("🧪 Classifying..."):
            preprocessed = preprocess_image(image)
            prediction = model.predict(preprocessed, verbose=0)
            
            st.write(f"Raw prediction: {prediction}")
            st.write(f"Prediction shape: {prediction.shape}")
            
            # Binary classification
            prob = float(prediction[0][0])
            st.write(f"Probability: {prob:.4f}")
            
            # Interpret as binary
            if prob < 0.5:
                label = "Mold"
                confidence = (1 - prob) * 100
            else:
                label = "No Mold"
                confidence = prob * 100
            
            st.markdown(f"""
            <div style="padding: 1.5rem; border-radius: 15px; background: {'#ff6b6b' if label == 'Mold' else '#51cf66'}; text-align: center;">
                <h2 style="color: white; margin: 0;">{label}</h2>
                <p style="color: white; margin: 0.5rem 0;">Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
