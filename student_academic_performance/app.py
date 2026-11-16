import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Student Academic Performance Predictor",
    page_icon="🎓",
    layout="centered",
)

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model.keras"
SCALER_PATH = ARTIFACTS / "scaler.pkl"
FEATURES_PATH = ARTIFACTS / "features.pkl"
METADATA_PATH = ARTIFACTS / "metadata.json"

st.title("🎓 Student Academic Performance Predictor")
st.write(
    "Enter the student's academic, demographic and behavioural details "
    "to estimate their final-grade category."
)
st.caption(
    "This is an educational machine-learning demonstration, not a definitive "
    "assessment of a student's ability or future."
)

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    return model, scaler, features, metadata

required_files = [MODEL_PATH, SCALER_PATH, FEATURES_PATH]
missing_files = [p.name for p in required_files if not p.exists()]
if missing_files:
    st.error(
        "Trained model files are missing: " + ", ".join(missing_files) +
        ". Run `python train_model.py` first and make sure the artifacts folder "
        "is beside app.py."
    )
    st.stop()

try:
    model, scaler, features, metadata = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the trained model files: {exc}")
    st.stop()

with st.form("student_details"):
    st.subheader("Academic information")
    c1, c2 = st.columns(2)
    with c1:
        g1 = st.slider("First period grade (G1)", 0, 20, 10)
        g2 = st.slider("Second period grade (G2)", 0, 20, 10)
        studytime = st.selectbox(
            "Weekly study time",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Less than 2 hours",
                2: "2–5 hours",
                3: "5–10 hours",
                4: "More than 10 hours",
            }[x],
        )
        failures = st.number_input("Past class failures", 0, 4, 0)
        absences = st.number_input("Number of school absences", 0, 100, 4)
        age = st.number_input("Age", 15, 22, 17)
        freetime = st.slider("Free time after school (1–5)", 1, 5, 3)
    with c2:
        medu = st.slider("Mother's education (0–4)", 0, 4, 2)
        fedu = st.slider("Father's education (0–4)", 0, 4, 2)
        goout = st.slider("Going out with friends (1–5)", 1, 5, 3)
        dalc = st.slider("Workday alcohol consumption (1–5)", 1, 5, 1)
        health = st.slider("Current health status (1–5)", 1, 5, 3)
        higher = st.selectbox("Wants to pursue higher education?", ["yes", "no"])
        schoolsup = st.selectbox("Receives extra educational support?", ["yes", "no"])

    submitted = st.form_submit_button("Predict academic performance", type="primary")

if submitted:
    values = {
        "G1": g1,
        "G2": g2,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "age": age,
        "Medu": medu,
        "Fedu": fedu,
        "goout": goout,
        "Dalc": dalc,
        "health": health,
        "higher": 1 if higher == "yes" else 0,
        "schoolsup": 1 if schoolsup == "yes" else 0,
        "freetime": freetime,
    }
    input_df = pd.DataFrame([[values[col] for col in features]], columns=features)
    scaled_input = scaler.transform(input_df)
    probabilities = model.predict(scaled_input, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))
    labels = ["Low", "Medium", "High"]

    st.divider()
    st.subheader("Prediction")
    st.metric("Predicted academic performance", labels[predicted_class])
    st.write("Model confidence estimates")
    probability_df = pd.DataFrame({
        "Category": labels,
        "Estimated probability": probabilities,
    }).set_index("Category")
    st.bar_chart(probability_df)
    st.caption(
        "The predicted category is based on the model's learned patterns in "
        "the UCI Portuguese student dataset. It should not be used to make "
        "high-stakes educational decisions."
    )

with st.expander("About this project"):
    st.write(
        "The model uses a Deep Neural Network trained on the UCI Student "
        "Performance dataset. The target is derived from G3 (final grade): "
        "Low = 0–9, Medium = 10–14, High = 15–20."
    )
    if metadata:
        st.write(f"Training rows: {metadata.get('training_rows', 'N/A')}")
        st.write(f"Test rows: {metadata.get('test_rows', 'N/A')}")
        if "dnn_test_accuracy" in metadata:
            st.write(f"Recorded DNN test accuracy: {metadata['dnn_test_accuracy']:.3f}")
