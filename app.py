"""
Streamlit App — Sequential Dementia Diagnosis Pipeline
============================================================
Stage 1: MRI vs Non-MRI gate (EfficientNetB0, binary)
Stage 2: Dementia stage classification (EfficientNetB0, 4-class)
          — only runs if Stage 1 passes (image is an MRI)

Both models hosted on Hugging Face (NdahTah/MRIVsNonMRI, public
repo) and downloaded at runtime via huggingface_hub, which handles
caching/redirects more robustly than a raw HTTP GET.

Class mappings (confirmed at training time):
  Stage 1 — MRI gate:
    mri      -> 0
    non mri  -> 1
    (sigmoid output = P(non mri); P(mri) = 1 - sigmoid output)

  Stage 2 — Dementia staging:
    non demented        -> 0
    very mild demented  -> 1
    mild demented       -> 2
    moderate demented   -> 3
"""

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from huggingface_hub import hf_hub_download

# ── Config ──────────────────────────────────────────────────────────────

HF_REPO_ID = "NdahTah/MRIVsNonMRI"
MRI_GATE_FILENAME = "efficientnetb0 mri deploy.h5"
MULTICLASS_FILENAME = "efficientnetb0_multiclass_best.h5"

INPUT_SIZE = (224, 224)
MRI_GATE_THRESHOLD = 0.5   # P(mri) above this = passes gate

STAGE_CLASS_NAMES = [
    "Non Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented",
]

st.set_page_config(page_title="Dementia Diagnosis Pipeline", page_icon="🧠", layout="centered")

# ── Visual styling (additive only — no logic below is affected) ─────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Title block */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        border-bottom: 1px solid rgba(217, 160, 91, 0.25);
        padding-bottom: 0.6rem;
        margin-bottom: 0.4rem !important;
    }

    /* Section headers act like scan-slice markers */
    h3 {
        font-weight: 600 !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-size: 0.95rem !important;
        color: #D9A05B !important;
        margin-top: 2rem !important;
        position: relative;
        padding-left: 0.9rem;
    }
    h3::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0.15em;
        bottom: 0.15em;
        width: 3px;
        background: #D9A05B;
        border-radius: 2px;
    }

    /* Metric values in the data face */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
    }
    [data-testid="stMetricLabel"] {
        text-transform: uppercase;
        font-size: 0.7rem !important;
        letter-spacing: 0.05em;
        opacity: 0.7;
    }

    /* Confidence / probability percentages */
    .stMarkdown p code,
    .stMarkdown p strong {
        font-family: 'JetBrains Mono', monospace;
    }

    /* File uploader: quieter, bordered like a viewer frame */
    [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed rgba(217, 160, 91, 0.35) !important;
        border-radius: 6px;
        background: rgba(217, 160, 91, 0.03);
    }

    /* Uploaded image: subtle frame, like a scan viewer */
    [data-testid="stImage"] img {
        border-radius: 4px;
        border: 1px solid rgba(232, 230, 225, 0.12);
    }

    /* Progress bars (probability breakdown) */
    [data-testid="stProgress"] > div > div {
        background-color: #D9A05B !important;
    }

    /* Expander headers */
    [data-testid="stExpander"] summary {
        font-size: 0.85rem;
        opacity: 0.85;
    }

    /* Caption / footer */
    .stCaption {
        opacity: 0.5;
        letter-spacing: 0.03em;
    }

    /* Horizontal rule between sections — scan-line motif */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(217, 160, 91, 0.4), transparent);
        margin: 2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Model loading (cached so downloads/loads happen once per session) ───

@st.cache_resource
def load_mri_gate_model():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MRI_GATE_FILENAME,
        cache_dir="./model_cache",
    )
    return load_model(model_path, compile=False)


@st.cache_resource
def load_multiclass_model():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MULTICLASS_FILENAME,
        cache_dir="./model_cache",
    )
    return load_model(model_path, compile=False)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Shared preprocessing for both models — same training-time pipeline."""
    image_rgb = image.convert("RGB")
    resized = image_rgb.resize(INPUT_SIZE)
    img_array = np.array(resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array, image_rgb


# ── UI ──────────────────────────────────────────────────────────────────

st.title("🧠 Dementia Diagnosis Pipeline")
st.write(
    "Upload a brain scan. The image is first checked to confirm it's an "
    "MRI — if it passes, dementia stage is classified as **Non Demented**, "
    "**Very Mild**, **Mild**, or **Moderate**."
)

try:
    with st.spinner("Loading models from Hugging Face (first run only)..."):
        mri_gate_model = load_mri_gate_model()
        multiclass_model = load_multiclass_model()
except Exception as e:
    st.error(f"Could not load models from Hugging Face.\n\nDetails: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array, image_rgb = preprocess_image(image)

    st.image(image_rgb, caption="Uploaded image", use_container_width=True)

    # ── Stage 1: MRI Gate ────────────────────────────────────────────

    st.subheader("Stage 1 — MRI Check")

    with st.spinner("Checking image type..."):
        gate_prediction = mri_gate_model.predict(img_array, verbose=0)
        probability_non_mri = float(gate_prediction[0][0])
        probability_mri = 1 - probability_non_mri

    passes_gate = probability_mri > MRI_GATE_THRESHOLD

    col1, col2 = st.columns(2)
    with col1:
        st.metric("MRI probability", f"{probability_mri:.2%}")
    with col2:
        st.metric("Non-MRI probability", f"{probability_non_mri:.2%}")

    if passes_gate:
        st.success("✓ Image identified as MRI — proceeding to dementia staging.")
    else:
        st.error(
            "✗ This image does not appear to be an MRI scan. "
            "Dementia staging requires an MRI image — please upload one."
        )
        st.stop()

    # ── Stage 2: Dementia Staging (only reached if gate passed) ──────

    st.subheader("Stage 2 — Dementia Stage Classification")

    with st.spinner("Classifying dementia stage..."):
        stage_prediction = multiclass_model.predict(img_array, verbose=0)[0]  # shape: (4,)

    predicted_idx = int(np.argmax(stage_prediction))
    predicted_label = STAGE_CLASS_NAMES[predicted_idx]
    confidence = float(stage_prediction[predicted_idx])

    if predicted_idx == 0:
        st.success(f"**{predicted_label}**")
    elif predicted_idx == 1:
        st.warning(f"**{predicted_label}**")
    else:
        st.error(f"**{predicted_label}**")

    st.write(f"Confidence: {confidence:.2%}")

    with st.expander("Full probability breakdown"):
        for class_name, prob in zip(STAGE_CLASS_NAMES, stage_prediction):
            st.write(f"{class_name}: {prob:.2%}")
            st.progress(float(prob))

with st.expander("ℹ️ How this pipeline works"):
    st.markdown(
        """
        **Stage 1 — MRI Gate:** confirms the uploaded image is actually
        an MRI scan before attempting any diagnosis. This avoids
        presenting a confident-looking but meaningless result if a
        non-MRI image (e.g. a CT scan, X-ray, or unrelated photo) is
        uploaded by mistake.

        **Stage 2 — Dementia Staging:** only runs if Stage 1 passes.
        Classifies the MRI into one of four dementia stages.

        Both models are EfficientNetB0, hosted on Hugging Face and
        downloaded automatically on first use.
        """
    )

st.markdown("---")
st.caption("Built with Streamlit • Models hosted on Hugging Face")
