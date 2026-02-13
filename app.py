import streamlit as st
import joblib
import numpy as np

st.title("🔮 **Churn Predictor - 89.4% XGBoost**")

# Load your models
model = joblib.load('models/xgb_model.pkl')
le_c = joblib.load('models/le_contract.pkl')
le_i = joblib.load('models/le_internet.pkl')

# Inputs
tenure = st.slider("📅 Tenure (months)", 0, 72, 12)
charges = st.slider("💰 Monthly Charges", 20.0, 120.0, 70.0)
contract = st.selectbox("📋 Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("🌐 Internet", ["DSL", "Fiber optic", "No"])

if st.button("🚀 **PREDICT CHURN**", type="primary"):
    # Encode (same as training)
    c_code = le_c.transform([contract])[0]
    i_code = le_i.transform([internet])[0]
    
    # Predict
    data = np.array([[tenure, charges, 2000.0, c_code, i_code]])
    prob = model.predict_proba(data)[0][1]
    
    # Results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Risk", f"{prob:.1%}")
    with col2:
        st.success("✅ STAYS") if prob < 0.5 else st.error("❌ CHURNS")
    
    st.balloons()

st.markdown("---")
st.info("**Your Data Mining Project → Production App!**")
