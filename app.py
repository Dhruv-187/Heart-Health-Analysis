import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from fpdf import FPDF
import datetime

# Load model and scaler
model = load_model('heart_model.h5')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Heart Health Monitor", layout="centered")
st.title("💓Heart Health Monitoring System")

# Patient Information
st.subheader("Enter Patient Details")
name = st.text_input("Name")
age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 200)
chol = st.number_input("Cholesterol (mg/dl)", 100, 400)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220)
exang = st.selectbox("Exercise Induced Angina", ["True", "False"])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, step=0.1)
slope = st.selectbox("Slope of ST", ["Upsloping", "Flat", "Downsloping"])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])

# Optional BMI Calculator
with st.expander("BMI Calculator"):
    height_cm = st.number_input("Height (in cm)", min_value=50.0, max_value=250.0, step=0.1)
    weight_kg = st.number_input("Weight (in kg)", min_value=10.0, max_value=300.0, step=0.1)
    if st.button("Calculate BMI"):
        height_m = height_cm / 100
        bmi = round(weight_kg / (height_m ** 2), 2)
        st.success(f"Your BMI is {bmi}")
        if bmi < 18.5:
            st.warning("You're underweight. A balanced diet is recommended.")
        elif 18.5 <= bmi < 24.9:
            st.info("You have a normal weight. Keep it up!")
        elif 25 <= bmi < 29.9:
            st.warning("You're overweight. Regular exercise is advised.")
        else:
            st.error("You are obese. Please consult a healthcare professional.")

# Encoding categorical inputs
sex = 1 if sex == "Male" else 0
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal": 2, "Asymptomatic": 3}
cp = cp_map[cp]
fbs = 1 if fbs == "True" else 0
restecg_map = {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2}
restecg = restecg_map[restecg]
exang = 1 if exang == "True" else 0
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope = slope_map[slope]
thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversable Defect": 3}
thal = thal_map[thal]

# Feature vector
features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                      exang, oldpeak, slope, ca, thal]])
scaled_input = scaler.transform(features)

# Prediction
if st.button("Predict"):
    prob = model.predict(scaled_input)[0][0]
    prediction = int(prob > 0.5)
    risk_percent = round(prob * 100, 2)

    # Risk classification and advice
    if prob < 0.3:
        risk_level = "Low Risk"
        advice = (
            "- Maintain a healthy diet.\n"
            "- Engage in regular physical activity.\n"
            "- Avoid smoking or alcohol abuse.\n"
            "- Keep stress levels low.\n"
        )
    elif 0.3 <= prob < 0.6:
        risk_level = "Moderate Risk"
        advice = (
            "- Regularly monitor blood pressure and cholesterol.\n"
            "- Reduce salt and saturated fat intake.\n"
            "- Have periodic medical checkups.\n"
            "- Consider a personalized fitness plan.\n"
        )
    else:
        risk_level = "High Risk"
        advice = (
            "- Seek immediate medical advice.\n"
            "- Follow a cardiac diet and avoid strenuous activity without clearance.\n"
            "- Take prescribed medications without fail.\n"
            "- Prioritize stress management and sleep hygiene.\n"
        )

    st.subheader("🔍 Prediction Result")
    st.write(f"**Risk Probability:** {risk_percent}%")
    st.write(f"**Risk Level:** {risk_level}")
    st.info(f"**Precaution/Advice:**\n{advice}")

    # Generate PDF report (excluding BMI)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Heart Disease Prediction Report", ln=True, align="C")

    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    if name:
        pdf.cell(0, 10, f"Name: {name}", ln=True)
    pdf.cell(0, 10, f"Age: {age}", ln=True)
    pdf.cell(0, 10, f"Sex: {'Male' if sex == 1 else 'Female'}", ln=True)
    pdf.cell(0, 10, f"Risk Probability: {risk_percent}%", ln=True)
    pdf.cell(0, 10, f"Risk Level: {risk_level}", ln=True)
    pdf.multi_cell(0, 10, f"Advice:\n{advice}")

    # Output the PDF safely
    pdf_file = "Heart_Report.pdf"
    pdf.output(pdf_file, 'F')

    with open(pdf_file, "rb") as f:
        st.download_button(label="📄Download PDF Report",
                           data=f,
                           file_name=pdf_file,
                           mime="application/pdf")
