import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Page Config
st.set_page_config(
    page_title="Stellar Classification - Quasar Hunter",
    layout="wide"
)

# Custom CSS for UI Enhancement & Removing GitHub Header / Emojis
st.markdown("""
<style>
    /* Hide Streamlit Header & GitHub Icon */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Background Image Settings */
    .stApp {
        background-image: url('https://raw.githubusercontent.com/nishatml/Quasar/main/background.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Make Text Clean & White */
    body, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, span {
        color: #FFFFFF !important;
    }

    /* Remove Ugly White Containers and replace with Translucent Dark Containers */
    div[data-testid="stVerticalBlock"] > div {
        background: rgba(10, 15, 30, 0.75) !important;
        border-radius: 12px !important;
        padding: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 15, 30, 0.9) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# App Title & Description
st.title("Quasar Hunter: Stellar Classification")
st.write("Classify astronomical objects into **Star**, **Galaxy**, or **Quasar** using Photometric Data & XGBoost Classifier.")

# Model & Data Loader Function
@st.cache_resource
def load_and_train_model():
    path = kagglehub.dataset_download("fedesoriano/stellar-classification-dataset-sdss17")
    csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
    df = pd.read_csv(os.path.join(path, csv_file))

    # Feature Engineering
    df['u_g'] = df['u'] - df['g']
    df['g_r'] = df['g'] - df['r']
    df['r_i'] = df['r'] - df['i']
    df['i_z'] = df['i'] - df['z']

    features = ['u', 'g', 'r', 'i', 'z', 'redshift', 'u_g', 'g_r', 'r_i', 'i_z']
    X = df[features]
    y = df['class']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    return model, le, acc

with st.spinner("Initializing Space Engine & Training Model..."):
    model, le, accuracy = load_and_train_model()

# Metrics Row
col1, col2, col3 = st.columns(3)
col1.metric("Model Engine", "XGBoost")
col2.metric("Test Accuracy", f"{accuracy*100:.2f}%")
col3.metric("Data Source", "SDSS DR17")



# Input Section
st.subheader("Enter Photometric Bands & Redshift")

c1, c2, c3 = st.columns(3)
with c1:
    u = st.number_input("u band", value=25.26)
    g = st.number_input("g band", value=22.75)
    r = st.number_input("r band", value=21.03)
with c2:
    i = st.number_input("i band", value=20.21)
    z = st.number_input("z band", value=19.49)
with c3:
    redshift = st.number_input("Redshift (z)", value=2.14)

# Calculated Features Display
u_g = u - g
g_r = g - r
r_i = r - i
i_z = i - z

st.write("### Automated Features (Color Indices)")
st.write(f"**u-g:** `{u_g:.4f}` | **g-r:** `{g_r:.4f}` | **r-i:** `{r_i:.4f}` | **i-z:** `{i_z:.4f}`")

# Prediction
if st.button("Predict Object Class", use_container_width=True):
    input_data = np.array([[u, g, r, i, z, redshift, u_g, g_r, r_i, i_z]])
    pred = model.predict(input_data)[0]
    pred_class = le.inverse_transform([pred])[0]
    
    
    st.success(f"Predicted Object Class: {pred_class}")

# Sidebar Content
with st.sidebar:
    st.header("Navigation & Info")
    st.subheader("Astronomical Glossary")
    with st.expander("Star"):
        st.write("A luminous sphere of plasma held together by its own gravity.")
    with st.expander("Galaxy"):
        st.write("A huge system of stars, stellar remnants, interstellar gas, dust, and dark matter.")
    with st.expander("Quasar"):
        st.write("An extremely luminous active galactic nucleus (AGN) powered by a supermassive black hole.")

    st.markdown("---")
    st.markdown("**Designed & Developed by Nishat Tasnim**")
