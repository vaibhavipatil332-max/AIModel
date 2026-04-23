import streamlit as st
import joblib
import pandas as pd

# Load ONLY pipeline model
model = joblib.load("model.pkl")

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

input_data = pd.DataFrame([[
    city,                    # City
    101,                     # Crime Code
    victim_age,              # Victim Age
    victim_gender,           # Victim Gender
    weapon,                  # Weapon Used
    police,                  # Police Deployed
    'No',                    # Case Closed
    '2024-01-01',            # Date Case Closed (IMPORTANT format)
    1,                       # Reported_DayOfWeek
    1,                       # Reported_Month
    2024,                    # Reported_Year
    1,                       # Occurred_DayOfWeek
    1,                       # Occurred_Month
    2024,                    # Occurred_Year
    600                      # Time_Of_Occurrence_Minutes
]], columns=[
    'City', 'Crime Code', 'Victim Age', 'Victim Gender', 'Weapon Used',
    'Police Deployed', 'Case Closed', 'Date Case Closed',
    'Reported_DayOfWeek', 'Reported_Month', 'Reported_Year',
    'Occurred_DayOfWeek', 'Occurred_Month', 'Occurred_Year',
    'Time_Of_Occurrence_Minutes'
])

if st.button("🔍 Predict Crime Type"):

    if city == "" or weapon == "":
        st.warning("⚠️ Please fill all fields")

    else:
        proba = model.predict_proba(input_data)
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
