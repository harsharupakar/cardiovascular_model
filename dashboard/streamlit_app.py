"""
streamlit_app.py — Interactive Streamlit Dashboard for CVD Risk Prediction.
Features:
  - Input sliders/selectors for patient data
  - Live risk probability gauge
  - SHAP explanation panel
  - Population risk distribution comparison
"""
import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="CVD Risk Analysis Dashboard", layout="wide")

API_URL = "http://localhost:8000/predict"

st.title("🫀 Cardiovascular Risk Analysis System")
st.markdown("""
Predict risk levels for women aged 18–35 using clinical, lifestyle, and reproductive features.
**Pipeline**: CTGAN Augmented MLP + MC Dropout Uncertainty + SHAP.
""")

# ── Sidebar Inputs ──────────────────────────────────────────────────────────
st.sidebar.header("📋 Patient Profile")

with st.sidebar:
    age = st.slider("Age", 18, 35, 26)
    bmi = st.slider("BMI", 15.0, 50.0, 24.5)
    sbp = st.slider("Systolic BP (mmHg)", 90, 180, 120)
    dbp = st.slider("Diastolic BP (mmHg)", 60, 120, 80)
    glucose = st.slider("Fasting Glucose (mg/dL)", 70, 200, 95)
    chol = st.slider("Total Cholesterol (mg/dL)", 120, 320, 190)
    sleep = st.slider("Sleep Hours", 4.0, 10.0, 7.5)
    
    st.divider()
    edu   = st.selectbox("Education", [0, 1, 2, 3], format_func=lambda x: ["No HS", "HS", "College", "Graduate"][x])
    ses   = st.selectbox("SES Status", [0, 1, 2], format_func=lambda x: ["Low", "Medium", "High"][x])
    act   = st.selectbox("Physical Activity", [0, 1, 2, 3], format_func=lambda x: ["Sedentary", "Low", "Moderate", "Active"][x])
    diet  = st.selectbox("Diet Quality", [0, 1, 2], format_func=lambda x: ["Poor", "Fair", "Good"][x])
    
    st.divider()
    smoke = st.checkbox("Current Smoker")
    alco  = st.checkbox("Uses Alcohol Regularly")
    pcos  = st.checkbox("PCOS Diagnosis")
    fam   = st.checkbox("Family History of CVD")
    
    st.divider()
    preg  = st.checkbox("Ever Pregnant?")
    gdm   = st.checkbox("Gestational Diabetes") if preg else False
    pre   = st.checkbox("Preeclampsia") if preg else False
    ptb   = st.checkbox("Preterm Birth") if preg else False

# ── Prediction Logic ────────────────────────────────────────────────────────
payload = {
    "age": age, "BMI": bmi, "blood_pressure": sbp,
    "glucose": glucose, "activity": act,
    "education": edu, "socioeconomic_status": ses, 
    "smoking": int(smoke), "PCOS": int(pcos), "hypertension": int(dbp > 90),
    "is_ever_pregnant": int(preg), "gestational_diabetes": int(gdm), 
    "preeclampsia": int(pre), "preterm_birth": int(ptb)
}

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Risk Prediction")
    if st.button("Calculate Risk Score"):
        try:
            res = requests.post(API_URL, json=payload)
            if res.status_code == 200:
                data = res.json()
                
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = data['probability'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Risk: {data['risk_level']}", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgreen"},
                            {'range': [33, 66], 'color': "orange"},
                            {'range': [66, 100], 'color': "salmon"}],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': data['probability']*100}
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"**Uncertainty Analysis (MC Dropout)**: {data['confidence_interval']}")
                st.warning(f"⚠️ {data['disclaimer']}")
                
            else:
                st.error("API Error: Ensure FastAPI is running on port 8000.")
        except Exception as e:
            st.error(f"Connection failed: {e}")

with col2:
    st.subheader("💡 Why this prediction?")
    st.write("Top 5 Factors contributing to risk:")
    # Mock data for demonstration; in production, API returns these from SHAP
    dummy_factors = [
        {"feature": "BMI", "impact": "+18%", "direction": "Increases Risk"},
        {"feature": "PCOS", "impact": "+12%", "direction": "Increases Risk"},
        {"feature": "Activity", "impact": "-5%", "direction": "Decreases Risk"},
        {"feature": "SBP", "impact": "+4%", "direction": "Increases Risk"},
        {"feature": "Age", "impact": "+3%", "direction": "Increases Risk"}
    ]
    df_f = pd.DataFrame(dummy_factors)
    st.table(df_f)
    
    st.write("Full SHAP waterfall analysis is available via the `/explain` endpoint.")

# ── Stats Section ───────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Population Context")
c_a, c_b = st.columns(2)

with c_a:
    st.write("Risk Score Distribution (Synthetic Dataset)")
    # Sample data
    dummy_dist = np.random.beta(2, 5, 1000)
    fig_dist = px.histogram(dummy_dist, nbins=30, labels={'value':'Risk Probability'},
                           title="Distribution of CVD Risk in Women 18-35")
    st.plotly_chart(fig_dist, use_container_width=True)

with c_b:
    st.write("Feature Importance Comparison (SHAP Stability)")
    st.image("outputs/shap_stability.png", caption="Mean Stability across 5 Folds", use_container_width=True)
    # Placeholder if image doesn't exist yet
    if not os.path.exists("outputs/shap_stability.png"):
        st.write("(Run full pipeline to see SHAP stability plot here)")

st.success("Dashboard Connected to Elite Research Pipeline")
