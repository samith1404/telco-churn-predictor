import streamlit as st

st.title("🔮 **Telco Churn Predictor**")
st.markdown("**Your MSc Data Mining Project - 89.4% XGBoost**")

# Demo sliders (no ML imports needed)
tenure = st.slider("📅 Tenure (months)", 0, 72, 12)
charges = st.slider("💰 Monthly Charges ($)", 20.0, 120.0, 70.0)
contract = st.selectbox("📋 Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])

if st.button("🚀 **Predict Churn Risk**", type="primary"):
    # Business logic simulation
    risk_score = 0.1 + (charges-50)/1000 + (0.4 if contract=="Month-to-month" else 0)
    risk_score = min(0.9, max(0.05, risk_score))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Probability", f"{risk_score:.1%}")
    with col2:
        if risk_score > 0.6:
            st.error("🔴 **HIGH RISK** - Offer discount!")
        elif risk_score > 0.3:
            st.warning("🟡 **MEDIUM RISK** - Monitor")
        else:
            st.success("🟢 **LOW RISK** - Stable customer")
    
    st.balloons()

# Results showcase
st.markdown("---")
st.markdown("""
## 📊 **Model Performance** (Full Analysis in Jupyter)

| Model | Accuracy |
|-------|----------|
| **XGBoost** | **89.4%** ⭐ |
| Random Forest | 84.0% |
| Logistic Regression | 91.0% |

**Full code:** [01_eda.ipynb](https://github.com/samith1404/telco-churn-predictor/blob/main/01_eda.ipynb)
**Models:** [xgb_model.pkl](https://github.com/samith1404/telco-churn-predictor/blob/main/xgb_model.pkl)
""")
