import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Page config
st.set_page_config(page_title="Crime Prediction", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>🚔 Crime Type Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# Input section
st.subheader("📝 Enter Crime Details")

col1, col2 = st.columns(2)

with col1:
    city = st.text_input("📍 City")
    victim_age = st.number_input("👤 Victim Age", 1, 100)

with col2:
    victim_gender = st.selectbox("⚧ Gender", ["Male", "Female"])
    police = st.number_input("👮 Police Deployed", 0, 50)

weapon = st.text_input("🔪 Weapon Used")

st.markdown("---")

# Prepare input
input_data = pd.DataFrame({
    'City': [city],
    'Victim Age': [victim_age],
    'Victim Gender': [victim_gender],
    'Weapon Used': [weapon],
    'Police Deployed': [police],
    'Reported_DayOfWeek': [1],
    'Reported_Month': [1],
    'Reported_Year': [2024],
    'Occurred_DayOfWeek': [1],
    'Occurred_Month': [1],
    'Occurred_Year': [2024],
    'Time_Of_Occurrence_Minutes': [600],
    'Crime Code': [101],
    'Case Closed': ['No'],
    'Date Case Closed': ['01-01-2024']
})

# Predict button
if st.button("🔍 Predict Crime Type"):
    input_processed = preprocessor.transform(input_data)
    proba = model.predict_proba(input_processed)
    score = proba[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    st.write("🔢 Violence Probability:", round(score, 3))

    if score < 0.3:
        st.success("🟢 Low Risk Crime")
    elif score < 0.6:
        st.warning("🟡 Medium Risk Crime")
    else:
        st.error("🔴 High Risk Violent Crime")