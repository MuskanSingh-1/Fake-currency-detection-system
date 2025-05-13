import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from PIL import Image, ImageStat
import numpy as np
import pandas as pd
import time

# Page config
st.set_page_config(page_title="Fake Currency Detector", layout="centered", page_icon="🞾")

# Load models
@st.cache_resource
def load_models():
    return {
        '₹50': load_model('model_50rs.keras'),
        '₹100': load_model('model_100rs.keras'),
        '₹500': load_model('model_500rs.keras'),
        '₹2000': load_model('model_2000rs.keras')
    }

models = load_models()
IMAGE_SIZE = (300, 300)

# Preprocess and predict
def predict_note(model, uploaded_img):
    img = Image.open(uploaded_img).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)[0][0]
    label = "Real" if prediction > 0.8 else "Fake"
    confidence = round(float(prediction if prediction > 0.5 else 1 - prediction) * 100, 2)
    return label, confidence, prediction

# Apply dark mode styles
def apply_styles(dark_mode):
    if dark_mode:
        st.markdown("""
        <style>
        .stApp {
            background-color: #000000;
            color: white;
        }
        .stButton > button, .stSelectbox, .stTextInput, .stTextArea {
            background-color: #333 !important;
            color: white !important;
            font-weight: bold;
            border: 2px solid #ddd !important;
            box-shadow: none !important;
        }
        .stButton > button:hover, .stSelectbox:hover {
            background-color: #4CAF50 !important;
        }

        /* Style the dropdown menu */
        div[data-baseweb="select"] {
            background-color: #333 !important;
            color: white !important;
        }

        /* Style the dropdown control area (button) */
        div[data-baseweb="select"] > div {
            background-color: #333 !important;
            color: white !important;
        }

        h1, h2, h3, h4, h5, h6, p, small, div {
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""<style>
            .stApp {
                background-color: #f5f5dc;
                color: black;
            }
            .stButton > button, .stSelectbox, .stTextInput, .stTextArea, .stFileUploader {
                background-color: #4CAF50;
                color: black;
                font-weight: bold;
                border: 2px solid #ddd;
                box-shadow: none;
            }
            .stButton > button:hover, .stSelectbox:hover, .stFileUploader:hover {
                background-color: #388E3C;
            }
            h1, h2, h3, h4, h5, h6, p, small, div {
                color: black !important;
            }
        </style>""", unsafe_allow_html=True)

# HOMEPAGE
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    dark_mode = st.checkbox("🌗 Dark Mode", value=st.session_state.dark_mode)
    st.session_state.dark_mode = dark_mode

    apply_styles(dark_mode)

    st.markdown(f"<h1 style='text-align:center; padding-top: 60px; font-size: 3em;'>{'Fake Currency Detector System'}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; font-size: 1.2em;'>AI-powered detection of Indian currency notes</p>", unsafe_allow_html=True)

    if st.button("🔍 Detect Currency"):
        st.session_state.page = "detect"
        st.experimental_rerun()

    if st.button("📘 About"):
        st.session_state.page = "about"
        st.experimental_rerun()

    st.markdown("<div style='height: 200px;'></div>", unsafe_allow_html=True)

    st.markdown(f"""
        <div style='position: fixed; bottom: 30px; width: 100%; text-align: center;'>
            <small style="font-size: 1.2em;">
                ⚠️ This tool is for educational use only. For official confirmation, consult a bank or RBI.
            </small>
        </div>
    """, unsafe_allow_html=True)

# DETECT CURRENCY
elif st.session_state.page == "detect":
    apply_styles(st.session_state.dark_mode)
    st.title("💵 Fake Currency Note Detector")
    st.markdown("Upload a photo of an Indian currency note to check its authenticity.")

    if "history" not in st.session_state:
        st.session_state.history = []

    denomination = st.selectbox("🪙 Select Denomination", ["₹50", "₹100", "₹500", "₹2000"])
    uploaded_file = st.file_uploader("📄 Upload Currency Note Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption='🖼️ Uploaded Image', use_column_width=True)

        stat = ImageStat.Stat(img)
        brightness = sum(stat.mean) / 3
        if brightness < 50:
            st.warning("⚠️ Image may be too dark for accurate analysis.")

        if st.button("🔍 Run Detection"):
            with st.spinner("Analyzing currency note..."):
                time.sleep(1)
                try:
                    model = models[denomination]
                    label, confidence, raw_score = predict_note(model, uploaded_file)

                    result_msg = f"🟢 <span style='color:green'><b>Real</b></span>" if label == "Real" else f"🔴 <span style='color:red'><b>Fake</b></span>"
                    st.markdown(f"<h4>Prediction: {result_msg}</h4>", unsafe_allow_html=True)

                    st.session_state.history.append({
                        'Denomination': denomination,
                        'Label': label
                    })

                    if label == "Fake":
                        st.error("⚠️ This note might be counterfeit. Please verify manually.")
                        st.markdown("""### 🔎 Suggested Next Steps:
                        - Do **not** circulate the note.
                        - Visit the nearest **bank branch** and report it.
                        - Keep a copy (photo) of the note for reference.
                        - Try to **remember when and where** you received it.
                        - If found in bulk, contact **local authorities or RBI**.
                        """)
                    else:
                        st.success("✅ This note appears genuine.")
                except Exception as e:
                    st.error("🚫 Prediction failed.")
                    st.exception(e)

    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 Detection History")
        df = pd.DataFrame(st.session_state.history[::-1])
        st.dataframe(df)

    if st.button("🔄 Reset Session"):
        st.session_state.history = []
        st.experimental_rerun()

    if st.button("🏠 Back to Home"):
        st.session_state.page = "home"
        st.experimental_rerun()

# ABOUT PAGE
elif st.session_state.page == "about":
    apply_styles(st.session_state.dark_mode)
    st.title("ℹ️ About This App")
    st.markdown("""
    This app detects **fake Indian currency notes** using a deep learning model (ResNet50).

    **Workflow:**
    - Upload a note
    - Select its denomination
    - The app predicts if it's *Real* or *Fake*

    **Tech Stack:**
    - TensorFlow / Keras
    - Streamlit
    - Transfer Learning

    👩‍💻 Built by **Jagriti, Muskan and Hrushikesh** to make currency detection easy and accessible.
    """)

    if st.button("🏠 Back to Home"):
        st.session_state.page = "home"
        st.experimental_rerun()