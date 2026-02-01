🫀 AI-Powered Heart Health Monitoring System

Predict • Prevent • Protect
An interactive machine learning–based web application that predicts heart disease risk using clinical parameters, provides actionable health guidance, and generates downloadable medical reports. Built to demonstrate how AI can support early risk assessment in healthcare


🌟 Highlights

✨ Deep Learning–based risk prediction
📊 Real-time probability & risk classification
🧮 Optional BMI calculator
📄 Auto-generated PDF health report
🖥️ Clean & interactive Streamlit UI
⚙️ End-to-end ML pipeline implementation


🧠 Tech Stack

| Category        | Tools               |
| --------------- | ------------------- |
| Programming     | Python              |
| ML / DL         | TensorFlow, Keras   |
| Data Processing | NumPy, Scikit-learn |
| Web App         | Streamlit           |
| Reporting       | FPDF                |


📊 Clinical Features Used

The model predicts heart disease risk using 13 medical parameters:
Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Maximum Heart Rate, Exercise-Induced Angina, ST Depression, ST Slope, Number of Major Vessels, Thalassemia


🧭 System Workflow

1️⃣ User enters patient details
2️⃣ Data validation & preprocessing
3️⃣ Feature scaling using trained scaler
4️⃣ Deep learning model prediction
5️⃣ Risk classification (Low / Moderate / High)
6️⃣ Preventive guidance generation
7️⃣ PDF health report download


🚀 Getting Started

🔹 Clone the Repository
git clone https://github.com/your-username/heart-health-monitor.git
cd heart-health-monitor

🔹 Install Dependencies
pip install -r requirements.txt

🔹 Run the Application
streamlit run app.py


📁 Project Structure

heart-health-monitor/
│
├── app.py              # Streamlit web application
├── heart_model.h5      # Trained deep learning model
├── scaler.pkl          # Pre-fitted feature scaler
├── requirements.txt    # Project dependencies
└── README.md           # Documentation


📄 Output Preview

✔️ Heart disease risk probability
✔️ Risk category classification
✔️ Personalized precautionary advice
✔️ Downloadable PDF medical report


🎯 Purpose & Use Case

This project is built for academic, portfolio, and learning purposes, showcasing how machine learning models can be deployed in healthcare-related applications.

⚠️ Disclaimer
This application is not intended for real medical diagnosis and should not replace professional healthcare advice.


👨‍💻 Author

Dhruv Rangari
Computer Science Student | YCCE
Diploma Graduate – Government Polytechnic, Nagpur


🔮 Future Enhancements

🔁 Model retraining with larger datasets
☁️ Cloud deployment
📈 Advanced visual analytics
🔗 Integration with health APIs


⭐ Support

If you find this project useful, consider giving it a ⭐
Feedback, suggestions, and contributions are welcome!
