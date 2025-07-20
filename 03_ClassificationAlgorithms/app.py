import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# --- Page configuration ---
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Dark Theme Custom CSS ---
st.markdown("""
<style>
    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #ff6b6b, #ee5a24);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    /* Risk Boxes */
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2em;
        margin: 2rem 0;
        animation: pulse 2s infinite;
    }
    .risk-low {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2em;
        margin: 2rem 0;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    /* Expander Styling */
    .st-expander > div > div {
        background-color: transparent !important;
        color: white !important;
    }
    /* Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #3498db, #2980b9);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 18px;
        font-weight: bold;
        margin-top: 2rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- Load model, scaler, columns ---
def load_models():
    try:
        model = joblib.load("KNN_Heart.pkl")
        scaler = joblib.load("scaler.pkl")
        columns = joblib.load("columns.pkl")
        return model, scaler, columns
    except Exception as e:
        st.error(f"❌ Model or dependencies not found: {e}")
        st.stop()

model, scaler, expected_columns = load_models()

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>❤️ Heart Disease Risk Assessment</h1>
    <p>Advanced AI-powered cardiac risk evaluation system</p>
</div>
""", unsafe_allow_html=True)

# --- About Section ---
with st.expander("ℹ️ About This Assessment", expanded=False):
    st.markdown("""
    ⚠️ **Important Notice:**  
    This tool is for educational purposes only.  
    Always consult a certified healthcare provider for medical decisions.
    """)

# --- User Inputs (Clean No Background) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Basic Information")
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"], format_func=lambda x: "Yes" if x == "Y" else "No")

with col2:
    st.subheader("🩺 Clinical Measurements")
    resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting BS > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)

# --- Advanced Parameters (No Box) ---
st.subheader("📊 Advanced Parameters")
col3, col4, col5 = st.columns(3)

with col3:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
with col4:
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, 0.1)
with col5:
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# --- Summary Metrics ---
st.subheader("📈 Current Parameters Summary")
col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
with col_summary1:
    st.metric("Age", f"{age} years")
    st.metric("BP", f"{resting_bp} mm Hg")
with col_summary2:
    st.metric("Cholesterol", f"{cholesterol} mg/dL")
    st.metric("Max HR", f"{max_hr} bpm")
with col_summary3:
    st.metric("Sex", "Male" if sex == "M" else "Female")
    st.metric("Angina", "Yes" if exercise_angina == "Y" else "No")
with col_summary4:
    st.metric("Fasting BS", "Yes" if fasting_bs == 1 else "No")
    st.metric("Oldpeak", f"{oldpeak}")

# --- Prediction ---
if st.button("🔍 Analyze Heart Disease Risk"):
    try:
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            f'Sex_{sex}': 1,
            f'ChestPainType_{chest_pain}': 1,
            f'RestingECG_{resting_ecg}': 1,
            f'ExerciseAngina_{exercise_angina}': 1,
            f'ST_Slope_{st_slope}': 1
        }

        input_df = pd.DataFrame([raw_input])
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]

        st.markdown("---")
        st.subheader("🎯 Risk Assessment Results")
        risk_pct = prediction_proba[1] * 100 if prediction == 1 else prediction_proba[0] * 100

        if prediction == 1:
            st.markdown(f"""
            <div class="risk-high">
                <h2>⚠️ HIGH RISK</h2>
                <p><strong>Risk Score: {risk_pct:.1f}%</strong></p>
                <p>Model indicates a high probability of heart disease.</p>
            </div>
            """, unsafe_allow_html=True)
            st.warning("**Recommended Actions:**\n\n- 🏥 Visit a cardiologist\n- 🧪 Get advanced tests\n- 🏃‍♂️ Lifestyle changes\n- 🚭 Quit smoking")
        else:
            st.markdown(f"""
            <div class="risk-low">
                <h2>✅ LOW RISK</h2>
                <p><strong>Risk Score: {risk_pct:.1f}%</strong></p>
                <p>Model indicates a low probability of heart disease.</p>
            </div>
            """, unsafe_allow_html=True)
            st.info("Keep maintaining a healthy lifestyle!")

        # Risk Factors Section
        st.subheader("📊 Risk Factors Analysis")
        risk_factors = []
        if age > 65: risk_factors.append("Age > 65")
        if resting_bp > 140: risk_factors.append("High BP")
        if cholesterol > 240: risk_factors.append("High Cholesterol")
        if fasting_bs == 1: risk_factors.append("High Fasting Sugar")
        if exercise_angina == "Y": risk_factors.append("Exercise Angina")
        if max_hr < 100: risk_factors.append("Low Max HR")
        if risk_factors:
            st.warning("**Potential Risk Factors Detected:**")
            for rf in risk_factors:
                st.write(f"• {rf}")
        else:
            st.success("No major clinical red flags detected.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --- Footer ---
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; padding-top: 1rem'>
    <p><strong>Disclaimer:</strong> Educational tool only. Always consult a healthcare professional.</p>
    <p>🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>
""", unsafe_allow_html=True)
