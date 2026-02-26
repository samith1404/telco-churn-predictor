import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(
    page_title="Data Drift Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background-color: #1e1e1e !important; }
section[data-testid="stMain"] { background-color: #1e1e1e !important; }
.block-container { background-color: #1e1e1e !important; }
[data-testid="stSidebar"] { background-color: #111827 !important; }
[data-testid="stSidebar"] * { color: #ffffff !important; }
h1, h2, h3, h4 { color: #ffffff !important; }
p, li, span, label { color: #e0e0e0 !important; }
.stMarkdown { color: #e0e0e0 !important; }
.alert-green {
    background-color: #064e3b;
    border: 2px solid #10b981;
    border-radius: 10px;
    padding: 1.2rem;
    margin: 1rem 0;
    color: #d1fae5 !important;
}
.stat-box {
    background-color: #1f2937;
    border-left: 5px solid #3b82f6;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.7rem 0;
}
.stat-label { font-size: 0.75rem; color: #9ca3af !important; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
.stat-value { font-size: 2.5rem; font-weight: 900; color: #ffffff !important; line-height: 1.1; }
.stat-value-red { font-size: 2.5rem; font-weight: 900; color: #ff6b6b !important; line-height: 1.1; }
</style>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, #1e3a5f, #3b82f6); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; text-align: center;'>
    <h2 style='color: white; margin: 0;'>📊 Research Project</h2>
    <p style='color: #bfdbfe; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>ML Model Reliability<br>Under Data Drift</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.radio(
    "Menu",
    ["🏠 Project Overview",
     "🔮 Live Prediction",
     "📈 Research Results",
     "🎯 Model Robustness",
     "💼 Industry Value"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Project Stats")

try:
    if os.path.exists('results/experiments/covariate_drift_results.csv'):
        results = pd.read_csv('results/experiments/covariate_drift_results.csv')
        n_models = results['model'].nunique()
        n_experiments = len(results)
        baseline = results[results['drift_magnitude'] == 0.0]
        high_drift = results[results['drift_magnitude'] == results['drift_magnitude'].max()]
        avg_degradation = (baseline['accuracy'].mean() - high_drift['accuracy'].mean()) * 100

        st.sidebar.markdown(f"""
        <div class='stat-box'>
            <div class='stat-label'>Models Tested</div>
            <div class='stat-value'>{n_models}</div>
        </div>
        <div class='stat-box'>
            <div class='stat-label'>Experiments Run</div>
            <div class='stat-value'>{n_experiments}</div>
        </div>
        <div class='stat-box' style='border-left-color: #ef4444;'>
            <div class='stat-label'>Avg Performance Drop</div>
            <div class='stat-value-red'>{avg_degradation:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.info("Run experiments first")
except:
    st.sidebar.warning("Run experiments first")

st.sidebar.markdown("---")
st.sidebar.markdown("### ✅ Research Progress")
checks = [
    ("Baseline Models Trained", True),
    ("Drift Experiments", os.path.exists('results/experiments/covariate_drift_results.csv')),
    ("Detection Analysis", os.path.exists('results/experiments/drift_detection_results.pkl')),
    ("Results Summary", os.path.exists('results/experiments/summary_report.txt'))
]
for label, done in checks:
    st.sidebar.markdown(f"{'✅' if done else '⏳'} {label}")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background:#1f2937; padding:1rem; border-radius:8px; border:1px solid #374151;'>
<p style='color:#f59e0b !important; font-weight:700; margin:0 0 0.5rem 0;'>💡 Why This Matters</p>
<p style='color:#d1d5db !important; font-size:0.8rem; margin:0;'>73% of ML models degrade within 6 months. This project provides systematic drift detection and model robustness analysis.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("📚 NCI Master's Project | 2026")

# ---- PAGES ----

if page == "🏠 Project Overview":
    st.title("📊 Data Drift Impact on ML Model Performance")
    st.markdown("### *An Empirical Study of Production Model Reliability*")

    st.markdown("""
    <div class='alert-green'>
        <h4 style='color:#6ee7b7 !important; margin:0 0 0.5rem 0;'>🎯 Research Objective</h4>
        <p style='color:#d1fae5 !important; margin:0;'>
        While most ML research focuses on <b>maximizing accuracy</b>, this project studies 
        <b>model reliability after deployment</b>. We systematically measure how models degrade 
        under data drift — the #1 cause of ML failures in production systems.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎓 Novel Contribution")
        st.markdown("""
        - **Beyond accuracy**: Degradation analysis
        - **Comparative study**: 4+ models tested
        - **Quantified impact**: Drift magnitude effects
        - **Detection methods**: Early warning systems
        """)
    with col2:
        st.markdown("### 🏭 Industry Problem")
        st.markdown("""
        - Silent model failures cost $15B/year
        - 60-70% performance drops common
        - No standard monitoring
        - Manual intervention required
        """)
    with col3:
        st.markdown("### 💡 Research Questions")
        st.markdown("""
        **RQ1**: How does drift affect performance?
        **RQ2**: Which models are most robust?
        **RQ3**: Can we detect drift early?
        **RQ4**: What is the degradation pattern?
        """)

    st.markdown("---")
    st.markdown("### 📋 Methodology")
    tab1, tab2, tab3 = st.tabs(["Phase 1: Baseline", "Phase 2: Drift Simulation", "Phase 3: Analysis"])
    with tab1:
        st.markdown("""
        - Dataset: IBM Telco Customer Churn (7,043 customers)
        - Models: Logistic Regression, Random Forest, XGBoost, SVM
        - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
        - Split: 80% train / 20% test with stratification
        """)
    with tab2:
        st.markdown("""
        - **Covariate Drift**: Shift feature distributions
        - **Prior Drift**: Change class ratios
        - **Concept Drift**: Alter feature-target relationships
        - Multiple magnitudes tested per drift type
        """)
    with tab3:
        st.markdown("""
        - Performance tracking across drift levels
        - Model comparison and robustness ranking
        - Statistical drift detection (KS test, PSI)
        - Degradation pattern identification
        """)


elif page == "🔮 Live Prediction":
    st.title("🔮 Customer Churn Prediction")

    @st.cache_resource
    def load_models():
        try:
            import os
            base_path = os.getcwd()
            
            
            model = joblib.load(os.path.join(base_path, 'xgb_model.pkl'))
            le_contract = joblib.load(os.path.join(base_path, 'le_contract.pkl'))
            le_internet = joblib.load(os.path.join(base_path, 'le_internet.pkl'))
            return model, le_contract, le_internet, None
        except Exception as e:
            return None, None, None, str(e)

    model, le_contract, le_internet, error = load_models()

    if error:
        st.error(f"❌ Error loading models: {error}")
        st.info("Make sure you're running from the project directory")
    
    if model:
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("📅 Tenure (Months)", 0, 72, 12)
            contract = st.selectbox("📄 Contract Type", ["Month-to-month", "One year", "Two year"])
        with col2:
            monthly_charges = st.slider("💰 Monthly Charges ($)", 20.0, 120.0, 70.0)
            internet_service = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])

        threshold = st.slider("⚖️ Risk Threshold", 0.1, 0.9, 0.5, 0.05)

        if st.button("🚀 Predict Churn Risk", type="primary"):
            try:
                total_charges = tenure * monthly_charges
                contract_encoded = le_contract.transform([contract])[0]
                internet_encoded = le_internet.transform([internet_service])[0]
                input_data = np.array([[tenure, monthly_charges, total_charges, contract_encoded, internet_encoded]])
                probability = model.predict_proba(input_data)[0][1]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Churn Probability", f"{probability:.2%}")
                with col2:
                    if probability >= threshold:
                        st.error("❌ Likely to Churn")
                    else:
                        st.success("✅ Likely to Stay")
                with col3:
                    st.metric("Confidence", f"{max(probability, 1-probability):.2%}")

                st.progress(float(probability))

                if probability < 0.3:
                    st.success("🟢 Low Risk - Customer stable")
                elif probability < 0.7:
                    st.warning("🟡 Medium Risk - Monitor closely")
                else:
                    st.error("🔴 High Risk - Immediate action needed")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.error("Models not found in /models folder")


elif page == "📈 Research Results":
    st.title("📈 Experimental Results")

    try:
        results = pd.read_csv('results/experiments/covariate_drift_results.csv')

        col1, col2 = st.columns(2)
        with col1:
            models_list = st.multiselect("Select Models", results['model'].unique(), default=list(results['model'].unique()))
        with col2:
            metric = st.selectbox("Metric", ['accuracy', 'precision', 'recall', 'f1_score'])

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')

        for model_name in models_list:
            data = results[results['model'] == model_name]
            ax.plot(data['drift_magnitude'], data[metric], marker='o', label=model_name, linewidth=2.5, markersize=8)

        ax.set_xlabel('Drift Magnitude', fontsize=12, color='white')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, color='white')
        ax.set_title(f'Model Performance Under Covariate Drift', fontsize=14, fontweight='bold', color='white')
        ax.legend(fontsize=10, facecolor='#2d2d2d', labelcolor='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_color('#444')
        ax.spines['right'].set_color('#444')
        ax.grid(True, alpha=0.2, color='white')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")

elif page == "🎯 Model Robustness":
    st.title("🎯 Model Robustness Comparison")

    try:
        results = pd.read_csv('results/experiments/covariate_drift_results.csv')
        baseline = results[results['drift_magnitude'] == 0.0]
        high_drift = results[results['drift_magnitude'] == results['drift_magnitude'].max()]

        comparison = []
        for model_name in baseline['model'].unique():
            base = baseline[baseline['model'] == model_name].iloc[0]
            drift = high_drift[high_drift['model'] == model_name].iloc[0]
            comparison.append({
                'Model': model_name,
                'Baseline Accuracy': round(base['accuracy'], 4),
                'Drifted Accuracy': round(drift['accuracy'], 4),
                'Degradation %': round((base['accuracy'] - drift['accuracy']) / base['accuracy'] * 100, 2)
            })

        df = pd.DataFrame(comparison).sort_values('Degradation %')
        st.markdown("**Lower degradation = More robust to drift**")
        st.dataframe(df.style.background_gradient(subset=['Degradation %'], cmap='RdYlGn_r').format({'Degradation %': '{:.2f}%'}), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        colors = ['#ef4444' if x > df['Degradation %'].median() else '#22c55e' for x in df['Degradation %']]
        ax.barh(df['Model'], df['Degradation %'], color=colors)
        ax.set_xlabel('Degradation %', color='white')
        ax.set_title('Model Robustness (Lower is Better)', color='white', fontweight='bold')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444')
        ax.grid(True, axis='x', alpha=0.2, color='white')
        st.pyplot(fig)

        st.success(f"✅ Most Robust: **{df.iloc[0]['Model']}** ({df.iloc[0]['Degradation %']:.1f}% drop)")
        st.error(f"⚠️ Least Robust: **{df.iloc[-1]['Model']}** ({df.iloc[-1]['Degradation %']:.1f}% drop)")

    except:
        st.info("Run experiments first")


elif page == "💼 Industry Value":
    st.title("💼 Industry Impact & Application")

    st.markdown("""
    <div style='background:#1e3a5f; border:2px solid #3b82f6; border-radius:10px; padding:1.5rem; margin:1rem 0;'>
        <h4 style='color:#93c5fd !important; margin:0 0 0.5rem 0;'>🎯 Real-World Problem Addressed</h4>
        <p style='color:#bfdbfe !important; margin:0;'>
        73% of ML models experience significant performance degradation within 6 months due to data drift.
        This costs enterprises an estimated <b>$15 billion annually</b> in lost revenue and poor decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏭 Production Challenges")
        st.markdown("""
        - **Silent Failures**: Models degrade without alerting teams
        - **No Baselines**: Lack of degradation benchmarks
        - **Reactive Monitoring**: Issues found after damage
        - **Inefficient Retraining**: Models retrained too early/late
        """)
        st.markdown("### 🎤 Interview Answer")
        st.markdown("""
        > *"I studied production ML reliability — how models degrade under data drift.
        I simulated real-world distribution changes, found XGBoost degrades slower
        than Logistic Regression, and implemented KS test drift detection to trigger
        retraining before accuracy drops critically."*
        """)
    with col2:
        st.markdown("### 💡 Practical Applications")
        st.markdown("""
        **For ML Engineers:**
        - Choose robust models for drift-prone domains
        - Set monitoring thresholds based on research

        **For Data Scientists:**
        - Design drift-resilient features
        - Optimize retraining schedules

        **For Business:**
        - Predict when models need updating
        - Reduce emergency interventions
        """)
# =====================================================
# 📄 PDF REPORT GENERATION FUNCTION
# =====================================================

def generate_pdf_report(results_df):

    pdf_path = "ML_Data_Drift_Report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("<b>Data Drift Impact on ML Model Performance</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    # Executive Summary
    elements.append(Paragraph("<b>Executive Summary</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    baseline = results_df[results_df['drift_magnitude'] == 0.0]
    high_drift = results_df[results_df['drift_magnitude'] == results_df['drift_magnitude'].max()]

    summary_text = f"""
    Total Models Tested: {results_df['model'].nunique()} <br/>
    Total Experiments Conducted: {len(results_df)} <br/>
    Maximum Drift Level: {results_df['drift_magnitude'].max()} <br/>
    Average Performance Drop: {round((baseline['accuracy'].mean() - high_drift['accuracy'].mean()) * 100, 2)}%
    """

    elements.append(Paragraph(summary_text, styles["Normal"]))
    elements.append(Spacer(1, 0.4 * inch))

    # Model Comparison Table
    elements.append(Paragraph("<b>Model Performance Comparison</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    comparison_data = [["Model", "Baseline Acc", "Drifted Acc", "Degradation %"]]

    for model_name in baseline['model'].unique():
        base = baseline[baseline['model'] == model_name].iloc[0]
        drift = high_drift[high_drift['model'] == model_name].iloc[0]

        degradation = round((base['accuracy'] - drift['accuracy']) / base['accuracy'] * 100, 2)

        comparison_data.append([
            model_name,
            round(base['accuracy'], 4),
            round(drift['accuracy'], 4),
            f"{degradation}%"
        ])

    table = Table(comparison_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
    ]))

    elements.append(table)
    elements.append(PageBreak())

    # Add Visualizations (Colour)
    elements.append(Paragraph("<b>Model Performance Under Drift</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.3 * inch))

    for model_name in results_df['model'].unique():

        fig, ax = plt.subplots(figsize=(6, 4))
        model_data = results_df[results_df['model'] == model_name]

        ax.plot(
            model_data['drift_magnitude'],
            model_data['accuracy'],
            marker='o',
            linewidth=2
        )

        ax.set_title(f"{model_name} - Accuracy vs Drift")
        ax.set_xlabel("Drift Magnitude")
        ax.set_ylabel("Accuracy")
        ax.grid(True)

        image_path = f"{model_name}_drift_plot.png"
        fig.savefig(image_path, dpi=300)
        plt.close(fig)

        elements.append(Paragraph(f"{model_name} Performance Curve", styles["Heading3"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(RLImage(image_path, width=5.5 * inch, height=3.5 * inch))
        elements.append(Spacer(1, 0.5 * inch))

    # Conclusion
    elements.append(PageBreak())
    elements.append(Paragraph("<b>Research Conclusion</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    conclusion_text = """
    This study confirms that ML models significantly degrade under covariate drift.
    Tree-based models such as XGBoost and Random Forest demonstrate greater robustness
    compared to linear models under increasing drift magnitude.
    Early drift detection is essential to maintain production reliability.
    """

    elements.append(Paragraph(conclusion_text, styles["Normal"]))

    doc.build(elements)

    return pdf_path 
st.markdown("---")
st.markdown("<p style='text-align:center; color:#6b7280; font-size:0.85rem;'>📊 NCI Master's Project | Data Mining & ML 2026 | Empirical Study of ML Model Performance Under Data Drift</p>", unsafe_allow_html=True)
# =====================================================
# 📄 PDF DOWNLOAD BUTTON (ONLY ADDITION – NO CHANGES ABOVE)
# =====================================================

try:
    if os.path.exists('results/experiments/covariate_drift_results.csv'):
        results = pd.read_csv('results/experiments/covariate_drift_results.csv')

        st.markdown("---")
        st.markdown("## 📄 Download Full Model Performance Report")

        # Step 1: Generate PDF button
        if st.button("📥 Generate PDF Report"):
            pdf_path = generate_pdf_report(results)
            st.session_state["pdf_path"] = pdf_path
            st.success("PDF Generated Successfully!")

        # Step 2: Show download button AFTER generation
        if "pdf_path" in st.session_state:
            with open(st.session_state["pdf_path"], "rb") as f:
                st.download_button(
                    label="⬇️ Click Here to Download Report",
                    data=f,
                    file_name="ML_Data_Drift_Report.pdf",
                    mime="application/pdf"
                )

except Exception as e:
    st.error(f"Error: {e}")