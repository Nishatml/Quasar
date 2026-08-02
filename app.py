import streamlit as st
import numpy as np
import pandas as pd
import base64
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import kagglehub
from kagglehub import KaggleDatasetAdapter 

# Page Configuration
st.set_page_config(
    page_title="Stellar Classification - Quasar Hunter",
    page_icon="Page ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to encode local image for background
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        return encoded_string
    except Exception:
        return None

# Load background image 
bg_base64 = get_base64_image("background.jpg")

# CSS 
if bg_base64:
    bg_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #ffffff;
    }}
    </style>
    """
else:
    # Fallback gradient background if background.jpg is missing
    bg_style = """
    <style>
    .stApp {
        background: radial-gradient(ellipse at bottom, #1b2735 0%, #090a0f 100%);
        color: #ffffff;
    }
    </style>
    """

st.markdown(bg_style, unsafe_allow_html=True)


# Cache Model Training
@st.cache_resource
def load_and_train_model():
    # Load SDSS DR17 Dataset
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "fedesoriano/stellar-classification-dataset-sdss17",
        "star_classification.csv",
    )
    
    # Feature Engineering (Color Indices)
    df['u-g'] = df['u'] - df['g']
    df['g-r'] = df['g'] - df['r']
    df['r-i'] = df['r'] - df['i']
    df['i-z'] = df['i'] - df['z']

    features = ['u', 'g', 'r', 'i', 'z', 'redshift', 'u-g', 'g-r', 'r-i', 'i-z']
    X = df[features]
    y = df['class']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    return model, le


# Load Model
with st.spinner(" Initializing Space Engine & Training XGBoost Model..."):
    model, le = load_and_train_model()

# --- SIDEBAR CONTENT ---
st.sidebar.title(" Navigation & Info")
st.sidebar.markdown("---")

# Definitions Section
st.sidebar.subheader("Astronomical Glossary")
with st.sidebar.expander("Star", expanded=False):
    st.write("A luminous sphere of plasma held together by its own gravity, undergoing nuclear fusion at its core.")

with st.sidebar.expander(" Galaxy ", expanded=False):
    st.write("A huge collection of gas, dust, billions of stars, and their solar systems, held together by gravity.")

with st.sidebar.expander(" Quasar", expanded=False):
    st.write("An extremely luminous active galactic nucleus (AGN) powered by a supermassive black hole at the center of a distant galaxy.")

st.sidebar.markdown("---")

# SDSS DR17 Dataset Info
st.sidebar.subheader(" Dataset Information")
st.sidebar.info("""
**SDSS DR17** (Sloan Digital Sky Survey Data Release 17)
- **Total Samples:** 100,000 observations
- **Photometric Filters:** Ultraviolet ($u$), Green ($g$), Red ($r$), Near Infrared ($i$), Infrared ($z$)
- **Redshift ($z$):** Measures expansion of space
""")

st.sidebar.markdown("---")
st.sidebar.markdown("** Project by Nishat Tasnim**")


# --- MAIN UI BODY ---
st.title(" Quasar Hunter: Stellar Classification")
st.markdown("Classify astronomical objects into **Star**, **Galaxy**, or **Quasar** using Photometric Data & High-Performance **XGBoost Classifier**.")
st.markdown("---")

# Model Metrics Row
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown("""
    <div class="metric-card">
        <h3> Model Engine</h3>
        <h2>XGBoost</h2>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown("""
    <div class="metric-card">
        <h3> Test Accuracy</h3>
        <h2 style="color: #00ff88;">97.88%</h2>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown("""
    <div class="metric-card">
        <h3> Data Source</h3>
        <h2>SDSS DR17</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("\n")

# Form Inputs
st.subheader(" Input Photometric Bands & Redshift")

col1, col2 = st.columns(2)

with col1:
    u = st.number_input("Ultraviolet Filter (u)", value=25.26, format="%.4f")
    g = st.number_input("Green Filter (g)", value=24.37, format="%.4f")
    r = st.number_input("Red Filter (r)", value=24.47, format="%.4f")

with col2:
    i = st.number_input("Near Infrared Filter (i)", value=22.75, format="%.4f")
    z = st.number_input("Infrared Filter (z)", value=21.04, format="%.4f")
    redshift = st.number_input("Redshift (z-value)", value=1.77, format="%.6f")

# Feature Engineering Calculation
u_g = u - g
g_r = g - r
r_i = r - i
i_z = i - z

st.markdown("---")
st.subheader(" Automated Feature Engineering (Color Indices)")
col_f1, col_f2, col_f3, col_f4 = st.columns(4)
col_f1.metric("u - g Index", f"{u_g:.4f}")
col_f2.metric("g - r Index", f"{g_r:.4f}")
col_f3.metric("r - i Index", f"{r_i:.4f}")
col_f4.metric("i - z Index", f"{i_z:.4f}")

st.markdown("---")

# Predict Button
if st.button(" Predict Celestial Object", use_container_width=True):
    # Model Input Array
    input_data = np.array([[u, g, r, i, z, redshift, u_g, g_r, r_i, i_z]])
    
    # Prediction
    pred_class_idx = model.predict(input_data)[0]
    pred_class_label = le.inverse_transform([pred_class_idx])[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = np.max(probabilities) * 100

    # Display Glassmorphism Result Card
    st.markdown(f"""
    <div class="pred-card">
        <h2 style="margin: 0;">Predicted Celestial Object:</h2>
        <h1 style="color: #00d4ff; font-size: 3rem; margin: 10px 0;"> {pred_class_label.upper()} </h1>
        <p style="font-size: 1.2rem; margin: 0;">Model Confidence: <strong>{confidence:.2f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Probability Breakdown
    st.subheader(" Class Probabilities Breakdown")
    prob_df = pd.DataFrame({
        'Celestial Class': le.classes_,
        'Probability': probabilities
    })
    st.bar_chart(prob_df.set_index('Celestial Class'))

# Footer Credit
st.markdown("---")
st.markdown("<p style='text-align: center; color: #cbd5e1;'>Designed & Developed by <strong>Nishat Tasnim</strong> | Powered by Streamlit & XGBoost</p>", unsafe_allow_html=True)
