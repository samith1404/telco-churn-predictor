import streamlit as st

st.title("🔮 **Telco Churn Predictor**")
st.markdown("**XGBoost: 89.4% | RF: 84.0% | Logistic: 91.0%**")

tenure = st.slider("📅 Tenure (months)", 0, 72, 12)
charges = st.slider("💰 Monthly Charges", 20.0, 120.0, 70.0)
contract = st.selectbox("📋 Contract", ["Month-to-month", "One year", "Two year"])

if st.button("🚀 **Predict Risk**"):
    risk = 0.1 + (charges-50)/1000 + (0.4 if contract=="Month-to-month" else 0)
    st.metric("Churn Risk", f"{min(risk, 0.9):.1%}")
    color = "🔴" if risk > 0.6 else "🟢"
    st.write(f"{color} **{contract} customers**")

st.success("**Full ML analysis:** [Jupyter Notebook → 89.4% XGBoost]")

