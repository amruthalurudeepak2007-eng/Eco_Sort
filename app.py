import json
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="EcoSort | Smart Waste AI", page_icon="♻️", layout="wide")

BG = "#101713"
GREEN = "#35D07F"
MODEL_PATH = Path("models/waste_classifier.keras")
LABELS_PATH = Path("models/labels.json")
TIPS = {
    "cardboard": ("📦", "Cardboard", "Flatten the box and keep it dry before placing it in paper/cardboard recycling."),
    "glass": ("🍾", "Glass", "Empty and rinse the container. Handle broken glass carefully and follow local collection rules."),
    "metal": ("🥫", "Metal", "Empty and rinse cans or tins before placing them in the metal recycling bin."),
    "paper": ("📄", "Paper", "Keep paper clean and dry. Greasy or heavily soiled paper may not be recyclable."),
    "plastic": ("🧴", "Plastic", "Empty and rinse the item. Check local recycling rules because not all plastic types are accepted."),
    "trash": ("🗑️", "General waste", "This item may not fit the recyclable classes. Check your local waste collection guidance."),
}
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
<style>
.stApp { background: #101713; color: #F2F7F3; }
[data-testid="stHeader"] { background: rgba(16,23,19,0); }
.block-container { padding-top: 2rem; max-width: 1120px; }
.hero { background: linear-gradient(135deg,#1b3325,#17241c); border:1px solid #2a4935;
        border-radius:22px; padding:26px 30px; margin-bottom:22px; }
.brand { color:#35D07F; font-size:2.15rem; font-weight:800; margin:0; }
.sub { color:#A9B8AE; letter-spacing:2px; font-size:.76rem; }
.panel { background:#1B2820; border:1px solid #2b4032; border-radius:18px; padding:20px; }
.muted { color:#A9B8AE; }
.result { font-size:1.8rem; font-weight:800; color:#F2F7F3; }
.tip { background:#203629; border-left:4px solid #35D07F; border-radius:10px; padding:14px; }
div.stButton > button { background:#35D07F; color:#102016; border:0; border-radius:12px; font-weight:700; }
div.stButton > button:hover { background:#65E69E; color:#102016; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="brand">♻️ EcoSort</div>
  <div class="sub">SMART WASTE CLASSIFICATION • MACHINE LEARNING</div>
  <h2 style="margin:18px 0 6px 0;">Give your waste a second chance.</h2>
  <p style="color:#C1CEC5;margin:0;">Upload a photo and let the AI identify its waste category with a disposal tip.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path_string):
    return tf.keras.models.load_model(path_string)


def classify(image):
    model = load_model(str(MODEL_PATH))

    image = image.convert("RGB").resize((224, 224))
    arr = np.asarray(image, dtype=np.float32)

    # Match the preprocessing used during model training
    arr = arr / 127.5 - 1.0

    batch = np.expand_dims(arr, axis=0)
    probs = model.predict(batch, verbose=0)[0]

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))

    order = np.argsort(probs)[::-1]

    return (
        labels[int(order[0])],
        float(probs[order[0]]),
        [
            (labels[int(i)], float(probs[int(i)]))
            for i in order[:3]
        ]
    )

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.markdown("### 📸 Upload waste image")
    uploaded = st.file_uploader("Choose a JPG, JPEG or PNG image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Selected image", use_container_width=True)
    else:
        st.markdown('<div class="panel"><h3>Start with an image</h3><p class="muted">Take a clear photo of one waste item against a simple background, or choose an image from your device.</p></div>', unsafe_allow_html=True)
    run = st.button("🔍 Classify waste", use_container_width=True, disabled=uploaded is None)

with right:
    st.markdown("### 🧠 Prediction")
    model_ready = MODEL_PATH.exists() and LABELS_PATH.exists()
    if not model_ready:
        st.markdown('<div class="panel"><h3 style="color:#35D07F;">Model setup required</h3><p class="muted">The interface is ready, but the trained classifier has not been added yet. Follow the README instructions to train the model with a labelled waste image dataset.</p></div>', unsafe_allow_html=True)
        st.info("No fake prediction is shown. Train the model first to enable real ML classification.")
    elif run and uploaded:
        try:
            label, confidence, top3 = classify(image)
            emoji, display_name, tip = TIPS.get(label, ("♻️", label.title(), "Check local waste guidance."))
            st.markdown(f'<div class="panel"><div class="muted">PREDICTED CATEGORY</div><div class="result">{emoji} {display_name}</div><h3 style="color:#35D07F;">{confidence*100:.1f}% confidence</h3></div>', unsafe_allow_html=True)
            st.progress(confidence)
            st.markdown("**Disposal recommendation**")
            st.markdown(f'<div class="tip">{tip}</div>', unsafe_allow_html=True)
            st.markdown("**Top predictions**")
            for name, score in top3:
                st.write(f"{name.title()} — {score*100:.1f}%")
                st.progress(score)
            st.session_state.history.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "category": display_name,
                "confidence": f"{confidence*100:.1f}%"
            })
        except Exception as exc:
            st.error(f"Could not classify this image: {exc}")
    else:
        st.markdown('<div class="panel"><h3>Waiting for image</h3><p class="muted">Your category, confidence score and disposal recommendation will appear here.</p></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("### ♻️ Categories supported")
cols = st.columns(6)
for col, (key, value) in zip(cols, TIPS.items()):
    emoji, name, _ = value
    with col:
        st.markdown(f'<div class="panel" style="text-align:center;padding:12px 5px;"><div style="font-size:1.6rem">{emoji}</div><b>{name}</b></div>', unsafe_allow_html=True)

with st.expander("🕘 Session prediction history"):
    if st.session_state.history:
        st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("Your predictions from this session will appear here.")

st.caption("EcoSort is an educational ML project. Predictions are estimates; always follow local waste-sorting rules.")
