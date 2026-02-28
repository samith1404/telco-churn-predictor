import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime

st.set_page_config(
    page_title="Data Drift Research",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL CSS — Deep Space Aurora Theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

/* ── Base & Background ── */
.stApp {
    background: #020817 !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(56,189,248,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(139,92,246,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 50% 50%, rgba(16,185,129,0.04) 0%, transparent 70%) !important;
    background-attachment: fixed !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
section[data-testid="stMain"] { background: transparent !important; }
.block-container {
    background: transparent !important;
    padding-top: 2rem !important;
    max-width: 1200px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(2,8,23,0.95) !important;
    border-right: 1px solid rgba(56,189,248,0.15) !important;
    backdrop-filter: blur(20px) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 0.2rem !important; }
[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    padding: 0.6rem 1rem !important;
    margin: 0.15rem 0 !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(56,189,248,0.1) !important;
    border-color: rgba(56,189,248,0.3) !important;
}

/* ── Typography ── */
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; color: #f8fafc !important; letter-spacing: -1px !important; line-height: 1.1 !important; }
h2 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; color: #f1f5f9 !important; letter-spacing: -0.5px !important; }
h3, h4 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important; color: #e2e8f0 !important; }
p, li, span, label { font-family: 'Space Grotesk', sans-serif !important; color: #94a3b8 !important; }
.stMarkdown p { color: #94a3b8 !important; }
b, strong { color: #e2e8f0 !important; }

/* ── Hero Title Gradient ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1.5px;
    line-height: 1.1;
    margin: 0;
}
.hero-sub {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    color: #64748b;
    font-weight: 400;
    margin-top: 0.5rem;
    letter-spacing: 0.3px;
}
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 0.5rem;
    display: block;
}

/* ── Glass Cards ── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.75rem 0;
    backdrop-filter: blur(10px);
}
.glass-card-cyan {
    background: rgba(56,189,248,0.05);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.75rem 0;
}
.glass-card-violet {
    background: rgba(139,92,246,0.05);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.75rem 0;
}
.glass-card-green {
    background: rgba(52,211,153,0.05);
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.75rem 0;
}
.glass-card-amber {
    background: rgba(251,191,36,0.05);
    border: 1px solid rgba(251,191,36,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.75rem 0;
}
.glass-card-red {
    background: rgba(239,68,68,0.05);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.75rem 0;
}

/* ── Stat Boxes ── */
.stat-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin: 0.5rem 0;
    position: relative;
    overflow: hidden;
}
.stat-pill::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    border-radius: 12px 12px 0 0;
}
.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #475569 !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
    display: block;
    margin-bottom: 0.3rem;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #f1f5f9 !important;
    line-height: 1;
}
.stat-value-cyan {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.stat-value-red {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #f87171 !important;
    line-height: 1;
}

/* ── Sidebar Brand ── */
.sidebar-brand {
    background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(139,92,246,0.1));
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.brand-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    color: #f1f5f9 !important;
    margin: 0;
}
.brand-sub {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem;
    color: #64748b !important;
    margin: 0.4rem 0 0 0;
}

/* ── Code / Formula Box ── */
.formula-box {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #7dd3fc !important;
    margin: 0.5rem 0;
}

/* ── Recommendation Banner ── */
.rec-banner {
    background: linear-gradient(135deg, rgba(52,211,153,0.08) 0%, rgba(56,189,248,0.06) 100%);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}
.rec-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #34d399, #38bdf8, #818cf8);
}
.rec-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #34d399 !important;
    margin: 0 0 1rem 0;
}

/* ── Warning Banner ── */
.warn-banner {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 14px;
    padding: 1.4rem;
    margin: 0.75rem 0;
    border-left: 3px solid #fbbf24;
}

/* ── Gradient Divider ── */
.gradient-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.3), rgba(139,92,246,0.3), transparent);
    margin: 2rem 0;
    border: none;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(56,189,248,0.12) !important;
    color: #38bdf8 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(139,92,246,0.15)) !important;
    border: 1px solid rgba(56,189,248,0.4) !important;
    color: #38bdf8 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(56,189,248,0.25), rgba(139,92,246,0.25)) !important;
    border-color: rgba(56,189,248,0.6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(56,189,248,0.15) !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; }
[data-testid="stDataFrameResizable"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.02) !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.7rem !important; letter-spacing: 1px !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, #38bdf8, #818cf8) !important; border-radius: 99px !important; }
.stProgress > div { background: rgba(255,255,255,0.06) !important; border-radius: 99px !important; }

/* ── Footer ── */
.footer-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #1e293b !important;
    text-align: center;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 1rem 0;
}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}
span[data-baseweb="tag"] {
    background: rgba(56,189,248,0.15) !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    border-radius: 6px !important;
    color: #38bdf8 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #020817; }
::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.2); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(56,189,248,0.4); }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MATPLOTLIB THEME
# ============================================================
plt.rcParams.update({
    'figure.facecolor': '#020817',
    'axes.facecolor': '#040d1a',
    'axes.edgecolor': '#1e293b',
    'axes.labelcolor': '#64748b',
    'xtick.color': '#475569',
    'ytick.color': '#475569',
    'text.color': '#94a3b8',
    'grid.color': '#0f172a',
    'grid.linewidth': 1,
    'legend.facecolor': '#040d1a',
    'legend.edgecolor': '#1e293b',
    'legend.labelcolor': '#94a3b8',
    'font.family': 'monospace',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

MODEL_COLORS = {
    'Logistic Regression': '#38bdf8',
    'Random Forest': '#fb923c',
    'XGBoost': '#34d399',
    'SVM': '#a78bfa'
}

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("""
<div class='sidebar-brand'>
    <div style='font-size:1.8rem; margin-bottom:0.5rem;'>🌊</div>
    <p class='brand-title'>DATA DRIFT RESEARCH</p>
    <p class='brand-sub'>ML Model Reliability Study<br>NCI Master's Project · 2026</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<span class='section-label' style='padding-left:0.5rem; margin-bottom:0.5rem; display:block;'>Navigation</span>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Menu",
    ["🏠 Project Overview",
     "🔬 Methodology",
     "🔮 Live Prediction",
     "📈 Research Results",
     "🎯 Model Robustness",
     "💼 Industry Value"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<span class='section-label' style='padding-left:0.5rem; display:block;'>Project Stats</span>
""", unsafe_allow_html=True)

try:
    if os.path.exists('results/experiments/covariate_drift_results.csv'):
        _r = pd.read_csv('results/experiments/covariate_drift_results.csv')
        _n_models = _r['model'].nunique()
        _n_exp = len(_r)
        _base = _r[_r['drift_magnitude'] == 0.0]
        _high = _r[_r['drift_magnitude'] == _r['drift_magnitude'].max()]
        _avg_deg = (_base['accuracy'].mean() - _high['accuracy'].mean()) * 100
        st.sidebar.markdown(f"""
        <div class='stat-pill'>
            <span class='stat-label'>Models Tested</span>
            <div class='stat-value-cyan'>{_n_models}</div>
        </div>
        <div class='stat-pill'>
            <span class='stat-label'>Experiments Run</span>
            <div class='stat-value-cyan'>{_n_exp}</div>
        </div>
        <div class='stat-pill'>
            <span class='stat-label'>Avg Performance Drop</span>
            <div class='stat-value-red'>{_avg_deg:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.info("Run experiments to see stats")
except:
    st.sidebar.warning("Stats unavailable")

st.sidebar.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<span class='section-label' style='padding-left:0.5rem; display:block;'>Research Progress</span>
""", unsafe_allow_html=True)

checks = [
    ("Baseline Models Trained", True),
    ("Drift Experiments", os.path.exists('results/experiments/covariate_drift_results.csv')),
    ("Detection Analysis", os.path.exists('results/experiments/drift_detection_results.pkl')),
    ("Results Summary", os.path.exists('results/experiments/summary_report.txt'))
]
for label, done in checks:
    icon = "●" if done else "○"
    color = "#34d399" if done else "#334155"
    st.sidebar.markdown(
        f"<p style='color:{color} !important; font-family:JetBrains Mono,monospace; font-size:0.78rem; margin:0.3rem 0;'>{icon} {label}</p>",
        unsafe_allow_html=True
    )

st.sidebar.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='background:rgba(251,191,36,0.06); border:1px solid rgba(251,191,36,0.15); border-radius:10px; padding:1rem;'>
<p style='color:#fbbf24 !important; font-family:Space Grotesk,sans-serif; font-weight:600; font-size:0.8rem; margin:0 0 0.4rem 0;'>⚡ KEY FINDING</p>
<p style='color:#64748b !important; font-size:0.75rem; margin:0; line-height:1.5;'>Logistic Regression proved most resilient — accuracy improved under drift while others degraded. SVM appeared stable but was degenerate (F1 = 0).</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGE: PROJECT OVERVIEW
# ============================================================
if page == "🏠 Project Overview":
    st.markdown("""
    <span class='section-label'>NCI Master's Project · Data Mining & ML · 2026</span>
    <h1 class='hero-title'>Data Drift Impact on<br>ML Model Performance</h1>
    <p class='hero-sub'>An Empirical Study of Production Model Reliability</p>
    """, unsafe_allow_html=True)
    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card-cyan'>
        <span class='section-label'>Research Objective</span>
        <p style='color:#cbd5e1 !important; margin:0; font-size:1rem; line-height:1.7;'>
        While most ML research focuses on <strong>maximising accuracy at training time</strong>, this project
        studies <strong>model reliability after deployment</strong>. We systematically measure how models degrade
        under data drift — the #1 cause of ML failures in production systems — and identify which
        architecture withstands real-world distribution shifts.
        </p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='glass-card'>
            <span class='section-label'>Novel Contribution</span>
            <p style='color:#cbd5e1 !important; font-size:0.9rem; line-height:1.9; margin:0;'>
            → Beyond accuracy: Degradation analysis<br>
            → Comparative study: 4 architectures<br>
            → Quantified impact across 7 drift levels<br>
            → Early warning detection methods
            </p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='glass-card'>
            <span class='section-label'>Industry Problem</span>
            <p style='color:#cbd5e1 !important; font-size:0.9rem; line-height:1.9; margin:0;'>
            → Silent failures cost <strong>$15B/year</strong><br>
            → 60–70% performance drops common<br>
            → No standard monitoring protocol<br>
            → Manual intervention required
            </p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='glass-card'>
            <span class='section-label'>Research Questions</span>
            <p style='color:#cbd5e1 !important; font-size:0.9rem; line-height:1.9; margin:0;'>
            <strong>RQ1</strong> How does drift affect performance?<br>
            <strong>RQ2</strong> Which models are most robust?<br>
            <strong>RQ3</strong> Can we detect drift early?<br>
            <strong>RQ4</strong> What is the degradation pattern?
            </p>
        </div>""", unsafe_allow_html=True)
    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
    st.markdown("<span class='section-label'>Methodology Phases</span>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Phase 1 — Baseline", "Phase 2 — Drift Simulation", "Phase 3 — Analysis"])
    with tab1:
        st.markdown("""
        <div class='glass-card' style='margin-top:1rem;'>
        <p style='color:#94a3b8 !important; line-height:1.9; margin:0;'>
        → <strong>Dataset:</strong> IBM Telco Customer Churn (7,043 customers)<br>
        → <strong>Models:</strong> Logistic Regression, Random Forest, XGBoost, SVM<br>
        → <strong>Metrics:</strong> Accuracy, Precision, Recall, F1, ROC-AUC<br>
        → <strong>Split:</strong> 80% train / 20% test with stratification
        </p>
        </div>""", unsafe_allow_html=True)
    with tab2:
        st.markdown("""
        <div class='glass-card' style='margin-top:1rem;'>
        <p style='color:#94a3b8 !important; line-height:1.9; margin:0;'>
        → <strong>Covariate Drift:</strong> Gaussian noise injection + mean shift on numeric features<br>
        → <strong>Prior Drift:</strong> Change class ratios to simulate label imbalance shift<br>
        → <strong>Concept Drift:</strong> Alter feature-target relationships over time<br>
        → <strong>7 magnitudes</strong> tested: 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0
        </p>
        </div>""", unsafe_allow_html=True)
    with tab3:
        st.markdown("""
        <div class='glass-card' style='margin-top:1rem;'>
        <p style='color:#94a3b8 !important; line-height:1.9; margin:0;'>
        → Performance tracking across all drift levels per model<br>
        → Robustness ranking by degradation % (not raw accuracy)<br>
        → Statistical drift detection via KS test and PSI scoring<br>
        → Degradation curve analysis to determine retraining thresholds
        </p>
        </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE: METHODOLOGY
# ============================================================
elif page == "🔬 Methodology":
    st.markdown("""
    <span class='section-label'>Scientific Rigour</span>
    <h1 class='hero-title'>Research Methodology</h1>
    <p class='hero-sub'>Exactly how drift was simulated, which features were affected, and how the best model was determined</p>
    """, unsafe_allow_html=True)
    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card-cyan'>
        <span class='section-label'>Why This Matters</span>
        <p style='color:#cbd5e1 !important; margin:0; line-height:1.7;'>
        This section directly answers: <strong>where and how was the data drifted?</strong>
        Our results are reproducible and scientifically defensible because the drift process
        is fully parameterised, controlled, and documented here.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><span class='section-label'>Step 01</span>", unsafe_allow_html=True)
    st.markdown("<h2>Dataset & Features</h2>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, cls in zip(
        [c1, c2, c3, c4],
        ["Dataset", "Records", "Features", "Train/Test Split"],
        ["IBM Telco Churn", "7,043", "5", "80 / 20"],
        ["stat-value-cyan", "stat-value", "stat-value", "stat-value-cyan"]
    ):
        with col:
            st.markdown(f"""
            <div class='stat-pill'>
                <span class='stat-label'>{label}</span>
                <div class='{cls}' style='font-size:1.4rem;'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.dataframe(pd.DataFrame({
        "Feature": ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService"],
        "Type": ["Numeric", "Numeric", "Numeric", "Categorical", "Categorical"],
        "Drifted?": ["✅ Yes", "✅ Yes", "✅ Yes", "❌ No (encoded)", "❌ No (encoded)"],
        "Why It Drifts in Real Life": [
            "Customer base ages — avg tenure increases over time",
            "Price changes, promotions affect charge distributions",
            "Cumulative charges shift as pricing changes",
            "Contract preferences change seasonally",
            "Technology adoption changes service mix"
        ]
    }), use_container_width=True, hide_index=True)

    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
    st.markdown("<span class='section-label'>Step 02</span>", unsafe_allow_html=True)
    st.markdown("<h2>Drift Simulation Technique</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card-violet'>
        <span class='section-label'>Technique — Covariate Shift via Gaussian Noise + Mean Shift</span>
        <p style='color:#c4b5fd !important; margin:0; line-height:1.7;'>
        Controlled <strong>Gaussian noise injection</strong> + <strong>mean shift</strong> applied to all numeric features.
        Simulates real-world distribution changes without altering data labels — mimicking what happens when
        customer behaviour changes but the churn definition stays the same.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<span class='section-label'>The Formula</span>", unsafe_allow_html=True)
        st.markdown("""
        <div class='formula-box'>
        X_drifted = X<br>
        &nbsp;&nbsp;&nbsp;&nbsp;+ (magnitude × σ × ε)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;+ (magnitude × μ × 0.1)<br><br>
        <span style='color:#475569; font-size:0.75rem;'>
        X &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= original feature values<br>
        magnitude = drift level (0.0 → 1.0)<br>
        σ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= std deviation of feature<br>
        ε &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= Gaussian noise ~ N(0,1)<br>
        μ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= mean of feature<br>
        0.1 &nbsp;&nbsp;&nbsp;= 10% mean shift per unit
        </span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("<span class='section-label'>7 Drift Levels Tested</span>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Magnitude": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
            "Real-World Equivalent": [
                "Just deployed — baseline",
                "1–2 months post-deployment",
                "3–4 months — minor market shift",
                "5–6 months — noticeable change",
                "Post price-hike scenario",
                "Post competitor-entry",
                "Full market disruption"
            ]
        }), use_container_width=True, hide_index=True)

    st.markdown("<span class='section-label'>Visual Evidence — Before vs After Drift</span>", unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor('#020817')
    np.random.seed(42)
    orig = np.random.normal(loc=65, scale=20, size=1000)
    for idx, (mag, title, accent) in enumerate(zip(
        [0.0, 0.3, 1.0],
        ["Magnitude = 0.0\nNo Drift (Baseline)", "Magnitude = 0.3\nModerate Drift", "Magnitude = 1.0\nExtreme Drift"],
        ['#38bdf8', '#fbbf24', '#f87171']
    )):
        ax = axes[idx]
        ax.set_facecolor('#040d1a')
        noise = np.random.normal(0, 1, size=1000)
        drifted = orig + (mag * np.std(orig) * noise) + (mag * np.mean(orig) * 0.1)
        ax.hist(orig, bins=30, alpha=0.3, color='#475569', label='Original', density=True)
        ax.hist(drifted, bins=30, alpha=0.75, color=accent, label='Drifted', density=True)
        ax.set_title(title, color='#94a3b8', fontsize=9, pad=10)
        ax.set_xlabel('MonthlyCharges ($)', color='#475569', fontsize=8)
        ax.set_ylabel('Density', color='#475569', fontsize=8)
        ax.tick_params(colors='#334155', labelsize=7)
        ax.spines['bottom'].set_color('#1e293b')
        ax.spines['left'].set_color('#1e293b')
        ax.legend(fontsize=8, facecolor='#040d1a', labelcolor='#64748b', framealpha=0.8)
        ax.grid(True, alpha=0.15, color='#0f172a', linewidth=0.8)
    plt.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()

    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
    st.markdown("<span class='section-label'>Step 03</span>", unsafe_allow_html=True)
    st.markdown("<h2>Models Evaluated</h2>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost", "SVM"],
        "Architecture": ["Linear", "Ensemble (Bagging)", "Ensemble (Boosting)", "Kernel-based"],
        "Hypothesis": [
            "Sensitive to feature distribution shift",
            "Moderate robustness via feature averaging",
            "High robustness — gradient boosting self-corrects",
            "Unpredictable — depends on support vector positions"
        ],
        "Actual Result": ["Most robust ✅", "Least robust ❌", "Runner-up ⚡", "Degenerate 🚨"]
    }), use_container_width=True, hide_index=True)

    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
    st.markdown("<span class='section-label'>Step 04</span>", unsafe_allow_html=True)
    st.markdown("<h2>How We Determined the Best Model</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card-amber'>
        <span class='section-label'>Key Innovation — Degradation Score</span>
        <p style='color:#fef3c7 !important; margin:0; line-height:1.7;'>
        We don't just compare accuracy. We measure <strong>degradation % under drift</strong> —
        how stable a model stays as real-world data changes. A model with lower baseline accuracy
        but far lower degradation is <strong>more valuable in production</strong> than one that collapses under drift.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<span class='section-label'>Degradation Formula</span>", unsafe_allow_html=True)
        st.markdown("""
        <div class='formula-box'>
        Degradation % =<br>
        &nbsp;&nbsp;((Baseline Acc − Drifted Acc)<br>
        &nbsp;&nbsp;&nbsp;/ Baseline Acc) × 100<br><br>
        <span style='color:#475569; font-size:0.75rem;'>
        Negative = model improved under drift<br>
        0% = perfectly robust (SVM — but broken)<br>
        Positive = degraded (lower = better)<br>
        Drifted = accuracy at magnitude 1.0
        </span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("<span class='section-label'>Selection Criteria</span>", unsafe_allow_html=True)
        st.markdown("""
        <div class='glass-card'>
        <p style='color:#94a3b8 !important; font-size:0.9rem; line-height:2; margin:0;'>
        ✦ High baseline accuracy (competitive)<br>
        ✦ Low degradation % at max drift<br>
        ✦ Consistent F1 across all 7 levels<br>
        ✦ High ROC-AUC under stress<br>
        ✦ Detectable early via KS test / PSI
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-card-green'>
        <span class='section-label'>Summary — What Makes This Rigorous</span>
        <p style='color:#d1fae5 !important; margin:0; line-height:1.9; font-size:0.9rem;'>
        <strong>① Controlled drift</strong> — magnitude varies 0.0→1.0 in 7 steps, isolating the effect precisely<br>
        <strong>② Multiple drift types</strong> — covariate, prior, and concept drift independently tested<br>
        <strong>③ Multiple architectures</strong> — 4 structurally different models on equal footing<br>
        <strong>④ Multiple metrics</strong> — accuracy, precision, recall, F1, ROC-AUC all tracked<br>
        <strong>⑤ Statistical detection</strong> — KS test and PSI used to objectively measure drift onset
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE: LIVE PREDICTION
# ============================================================
elif page == "🔮 Live Prediction":
    st.markdown("""
    <span class='section-label'>XGBoost Model · Live Inference</span>
    <h1 class='hero-title'>Customer Churn<br>Prediction</h1>
    <p class='hero-sub'>Adjust the customer profile below and predict churn risk in real time</p>
    """, unsafe_allow_html=True)
    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)

    @st.cache_resource
    def load_models():
        try:
            base_path = os.getcwd()
            model = joblib.load(os.path.join(base_path, 'xgb_model.pkl'))
            le_contract = joblib.load(os.path.join(base_path, 'le_contract.pkl'))
            le_internet = joblib.load(os.path.join(base_path, 'le_internet.pkl'))
            return model, le_contract, le_internet, None
        except Exception as e:
            return None, None, None, str(e)

    model, le_contract, le_internet, error = load_models()
    if error:
        st.error(f"Error loading model: {error}")

    if model:
        st.markdown("<span class='section-label'>Customer Profile</span>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        with col2:
            monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 70.0)
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        threshold = st.slider("Risk Classification Threshold", 0.1, 0.9, 0.5, 0.05)

        if st.button("▶  Run Prediction"):
            try:
                total_charges = tenure * monthly_charges
                contract_encoded = le_contract.transform([contract])[0]
                internet_encoded = le_internet.transform([internet_service])[0]
                input_data = np.array([[tenure, monthly_charges, total_charges, contract_encoded, internet_encoded]])
                probability = model.predict_proba(input_data)[0][1]

                st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Churn Probability", f"{probability:.2%}")
                with col2:
                    if probability >= threshold:
                        st.error("⚠ Likely to Churn")
                    else:
                        st.success("✓ Likely to Stay")
                with col3:
                    st.metric("Model Confidence", f"{max(probability, 1-probability):.2%}")

                st.progress(float(probability))

                if probability < 0.3:
                    st.markdown("""<div class='glass-card-green'><p style='color:#34d399 !important; margin:0; font-weight:600;'>🟢 LOW RISK — Customer profile is stable. No intervention needed.</p></div>""", unsafe_allow_html=True)
                elif probability < 0.7:
                    st.markdown("""<div class='glass-card-amber'><p style='color:#fbbf24 !important; margin:0; font-weight:600;'>🟡 MEDIUM RISK — Monitor this account. Consider a retention offer.</p></div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class='glass-card-red'><p style='color:#f87171 !important; margin:0; font-weight:600;'>🔴 HIGH RISK — Immediate action recommended. Prioritise outreach.</p></div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.error("Model files not found. Ensure xgb_model.pkl is in the project root.")


# ============================================================
# PAGE: RESEARCH RESULTS
# ============================================================
elif page == "📈 Research Results":
    st.markdown("""
    <span class='section-label'>Experimental Findings</span>
    <h1 class='hero-title'>Research Results</h1>
    <p class='hero-sub'>Model accuracy across 7 drift magnitudes — from baseline (0.0) to extreme drift (1.0)</p>
    """, unsafe_allow_html=True)
    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)

    try:
        results = pd.read_csv('results/experiments/covariate_drift_results.csv')

        col1, col2 = st.columns(2)
        with col1:
            models_list = st.multiselect("Select Models", results['model'].unique(), default=list(results['model'].unique()))
        with col2:
            metric = st.selectbox("Performance Metric", ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'])

        st.markdown("""
        <div class='glass-card-cyan'>
            <span class='section-label'>How to Read This Chart</span>
            <p style='color:#7dd3fc !important; margin:0; font-size:0.9rem; line-height:1.6;'>
            X-axis = drift magnitude (0.0 baseline → 1.0 extreme distribution shift).
            <strong>Flat line = robust</strong>. <strong>Steep downward slope = sensitive to drift</strong>.
            SVM shown as dashed — degenerate classifier (F1 = 0 throughout).
            </p>
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(13, 5.5))
        fig.patch.set_facecolor('#020817')
        ax.set_facecolor('#040d1a')

        ax.axvspan(0.0, 0.2, alpha=0.06, color='#34d399')
        ax.axvspan(0.2, 0.6, alpha=0.06, color='#fbbf24')
        ax.axvspan(0.6, 1.0, alpha=0.06, color='#f87171')

        y_vals = results[metric].dropna()
        y_min = y_vals.min() if not y_vals.empty else 0

        ax.text(0.01, y_min, '▲ MILD', color='#34d399', fontsize=7, alpha=0.6, fontfamily='monospace', va='bottom')
        ax.text(0.21, y_min, '▲ MODERATE', color='#fbbf24', fontsize=7, alpha=0.6, fontfamily='monospace', va='bottom')
        ax.text(0.61, y_min, '▲ SEVERE', color='#f87171', fontsize=7, alpha=0.6, fontfamily='monospace', va='bottom')

        for model_name in models_list:
            data = results[results['model'] == model_name].sort_values('drift_magnitude')
            c = MODEL_COLORS.get(model_name, '#94a3b8')
            style = (0, (4, 3)) if model_name == 'SVM' else 'solid'
            ax.plot(data['drift_magnitude'], data[metric], marker='o',
                    label=model_name, linewidth=2.5, markersize=7,
                    color=c, linestyle=style,
                    markerfacecolor='#020817', markeredgewidth=2.5)

        ax.set_xlabel('Drift Magnitude  ·  0.0 = No Drift  →  1.0 = Extreme Drift', fontsize=10, color='#475569', labelpad=10)
        ax.set_ylabel(metric.replace('_', ' ').upper(), fontsize=10, color='#475569', labelpad=10)
        ax.set_title(f'MODEL PERFORMANCE UNDER COVARIATE DRIFT  ·  {metric.replace("_"," ").upper()}', fontsize=9, color='#475569', pad=12, fontfamily='monospace')
        ax.legend(fontsize=9, facecolor='#040d1a', labelcolor='#94a3b8', framealpha=0.9, edgecolor='#1e293b')
        ax.tick_params(colors='#334155', labelsize=8)
        ax.spines['bottom'].set_color('#1e293b')
        ax.spines['left'].set_color('#1e293b')
        ax.grid(True, alpha=0.12, color='#0f172a', linewidth=0.8)
        st.pyplot(fig)
        plt.close()

        st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
        st.markdown("<span class='section-label'>Summary — Baseline vs Max Drift</span>", unsafe_allow_html=True)

        baseline = results[results['drift_magnitude'] == 0.0]
        high_drift = results[results['drift_magnitude'] == results['drift_magnitude'].max()]
        summary_rows = []
        for m in results['model'].unique():
            b = baseline[baseline['model'] == m]
            d = high_drift[high_drift['model'] == m]
            if not b.empty and not d.empty:
                change = d.iloc[0][metric] - b.iloc[0][metric]
                summary_rows.append({
                    'Model': m,
                    f'Baseline {metric.upper()}': round(b.iloc[0][metric], 4),
                    f'Drifted {metric.upper()} (max)': round(d.iloc[0][metric], 4),
                    'Δ Change': f"{'▲' if change > 0 else '▼'} {abs(change):.4f}"
                })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error loading results: {e}")


# ============================================================
# PAGE: MODEL ROBUSTNESS
# ============================================================
elif page == "🎯 Model Robustness":
    st.markdown("""
    <span class='section-label'>Core Research Finding</span>
    <h1 class='hero-title'>Model Robustness<br>Under Drift</h1>
    <p class='hero-sub'>Lower degradation = more reliable in production. But accuracy alone can deceive.</p>
    """, unsafe_allow_html=True)
    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)

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
                'Degradation %': round((base['accuracy'] - drift['accuracy']) / base['accuracy'] * 100, 2),
                'Baseline F1': round(base['f1_score'], 4),
                'Drifted F1': round(drift['f1_score'], 4),
            })

        df = pd.DataFrame(comparison).sort_values('Degradation %')
        has_svm = 'SVM' in df['Model'].values

        st.markdown("<span class='section-label'>Robustness Ranking</span>", unsafe_allow_html=True)
        st.dataframe(
            df.style.background_gradient(subset=['Degradation %'], cmap='RdYlGn_r')
              .format({'Degradation %': '{:.2f}%', 'Baseline Accuracy': '{:.4f}',
                       'Drifted Accuracy': '{:.4f}', 'Baseline F1': '{:.4f}', 'Drifted F1': '{:.4f}'}),
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(11, 4.5))
        fig.patch.set_facecolor('#020817')
        ax.set_facecolor('#040d1a')
        bar_colors = [MODEL_COLORS.get(m, '#94a3b8') for m in df['Model']]
        bars = ax.barh(df['Model'], df['Degradation %'], color=bar_colors, alpha=0.85, height=0.5)
        for bar, val in zip(bars, df['Degradation %']):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:+.2f}%', va='center', color='#64748b', fontsize=9, fontfamily='monospace')
        ax.axvline(0, color='#1e293b', linewidth=1.5, linestyle='--')
        ax.set_xlabel('Degradation %  ·  Negative = Improved  ·  Positive = Degraded', fontsize=9, color='#475569', labelpad=10)
        ax.set_title('MODEL ROBUSTNESS UNDER MAXIMUM DRIFT', fontsize=10, color='#475569', pad=12, fontfamily='monospace')
        ax.tick_params(colors='#475569', labelsize=9)
        ax.spines['bottom'].set_color('#1e293b')
        ax.spines['left'].set_color('#1e293b')
        ax.grid(True, axis='x', alpha=0.1, color='#0f172a')
        st.pyplot(fig)
        plt.close()

        if has_svm:
            st.markdown("""
            <div class='warn-banner'>
                <span class='section-label'>⚠ Critical Finding — SVM Degenerate Behaviour</span>
                <p style='color:#fef3c7 !important; margin:0; font-size:0.9rem; line-height:1.7;'>
                SVM reports <strong>0.00% degradation</strong> — but this is a <strong>false positive</strong>.
                Precision, recall, and F1 are <strong>all 0.0 at every drift level</strong>, meaning it collapsed
                to predicting only the majority class (73.5% accuracy = always "no churn").
                This is a <strong>total model failure masked by accuracy</strong> — proving that
                <em>accuracy alone is insufficient to evaluate robustness under drift.</em>
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class='glass-card-green' style='margin:0.5rem 0;'><p style='color:#34d399 !important; margin:0; font-family:JetBrains Mono,monospace; font-size:0.88rem;'>✦ MOST ROBUST: Logistic Regression &nbsp;·&nbsp; −1.18% &nbsp;·&nbsp; accuracy actually improved under drift</p></div>
        <div class='glass-card-amber' style='margin:0.5rem 0;'><p style='color:#fbbf24 !important; margin:0; font-family:JetBrains Mono,monospace; font-size:0.88rem;'>⚡ RUNNER-UP: XGBoost &nbsp;·&nbsp; +1.09% &nbsp;·&nbsp; acceptable degradation, good recall preservation</p></div>
        <div class='glass-card-red' style='margin:0.5rem 0;'><p style='color:#f87171 !important; margin:0; font-family:JetBrains Mono,monospace; font-size:0.88rem;'>✗ LEAST ROBUST: Random Forest &nbsp;·&nbsp; +1.31% &nbsp;·&nbsp; F1 collapsed from 0.519 → 0.385 (−26%)</p></div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
        st.markdown("<span class='section-label'>Production Recommendation</span>", unsafe_allow_html=True)
        st.markdown("""
        <div class='rec-banner'>
            <p class='rec-title'>✦ Recommended for Production: Logistic Regression</p>
            <p style='color:#94a3b8 !important; margin:0 0 1.2rem 0; line-height:1.7; font-size:0.95rem;'>
            Based on empirical analysis across <strong>7 drift magnitudes</strong> and <strong>5 metrics</strong>,
            Logistic Regression is the optimal choice for deployment in drift-prone production environments.
            </p>
            <p style='color:#64748b !important; margin:0; font-size:0.88rem; font-family:JetBrains Mono,monospace; line-height:2;'>
            → Only model that improved under drift: 0.7793 → 0.7885 accuracy (+1.18%)<br>
            → Highest ROC-AUC at max drift: 0.8221 — best discrimination under stress<br>
            → Most consistent F1 across all 7 levels (0.541 → 0.516, only −4.6%)<br>
            → Interpretable — explainable when drift is detected<br>
            → Lowest risk of silent failure in long deployment cycles
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class='glass-card'>
                <span class='section-label'>Use Logistic Regression when</span>
                <p style='color:#94a3b8 !important; font-size:0.85rem; line-height:1.9; margin:0;'>
                ✦ Long deployment cycles (6+ months)<br>
                ✦ Limited retraining budget<br>
                ✦ Interpretability required<br>
                ✦ Gradual distribution shift expected
                </p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='glass-card'>
                <span class='section-label'>XGBoost as alternative</span>
                <p style='color:#94a3b8 !important; font-size:0.85rem; line-height:1.9; margin:0;'>
                ✦ 1.09% degradation — acceptable<br>
                ✦ Better recall under drift<br>
                ✦ Use when catching churners matters<br>
                ✦ Good if retraining is frequent
                </p>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class='glass-card'>
                <span class='section-label'>Avoid Random Forest when</span>
                <p style='color:#94a3b8 !important; font-size:0.85rem; line-height:1.9; margin:0;'>
                ✦ Distributions shift frequently<br>
                ✦ Retraining intervals are long<br>
                ✦ F1-score is the primary KPI<br>
                ✦ Operating in volatile markets
                </p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
        st.markdown("<span class='section-label'>Retraining Thresholds</span>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Drift Magnitude": ["0.0 – 0.2", "0.2 – 0.5", "0.5 – 0.7", "0.7 – 1.0"],
            "Action": ["✅ No action", "👁 Monitor F1 weekly", "⚠ Schedule retraining", "🚨 Retrain immediately"],
            "Evidence from Data": [
                "All models within 0.5% of baseline",
                "RF F1 drops 0.519 → 0.456 — first warning signal",
                "RF F1 at 0.460, XGB at 0.487 — meaningful degradation",
                "RF F1 collapsed to 0.385 (−26%). Only LR reliable."
            ]
        }), use_container_width=True, hide_index=True)

        st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
        st.markdown("<span class='section-label'>F1-Score — The Truth Accuracy Hides</span>", unsafe_allow_html=True)

        try:
            fig2, ax2 = plt.subplots(figsize=(13, 5))
            fig2.patch.set_facecolor('#020817')
            ax2.set_facecolor('#040d1a')
            ax2.axvspan(0.0, 0.2, alpha=0.06, color='#34d399')
            ax2.axvspan(0.2, 0.6, alpha=0.06, color='#fbbf24')
            ax2.axvspan(0.6, 1.0, alpha=0.06, color='#f87171')
            for model_name in results['model'].unique():
                data = results[results['model'] == model_name].sort_values('drift_magnitude')
                c = MODEL_COLORS.get(model_name, '#94a3b8')
                style = (0, (4, 3)) if model_name == 'SVM' else 'solid'
                ax2.plot(data['drift_magnitude'], data['f1_score'],
                        marker='o', label=model_name, linewidth=2.5, markersize=7,
                        color=c, linestyle=style, markerfacecolor='#020817', markeredgewidth=2.5)
            ax2.annotate('SVM F1 = 0.0\nacross all levels\n(degenerate classifier)',
                        xy=(0.5, 0.01), xytext=(0.35, 0.12),
                        arrowprops=dict(arrowstyle='->', color='#a78bfa', lw=1.5),
                        color='#a78bfa', fontsize=8, fontfamily='monospace')
            ax2.set_xlabel('Drift Magnitude  ·  0.0 = No Drift  →  1.0 = Extreme Drift', fontsize=10, color='#475569', labelpad=10)
            ax2.set_ylabel('F1-SCORE', fontsize=10, color='#475569', labelpad=10)
            ax2.set_title('F1-SCORE UNDER DRIFT  ·  SVM COLLAPSES TO ZERO  ·  RANDOM FOREST DEGRADES MOST', fontsize=9, color='#475569', pad=12, fontfamily='monospace')
            ax2.legend(fontsize=9, facecolor='#040d1a', labelcolor='#94a3b8', framealpha=0.9, edgecolor='#1e293b')
            ax2.tick_params(colors='#334155', labelsize=8)
            ax2.spines['bottom'].set_color('#1e293b')
            ax2.spines['left'].set_color('#1e293b')
            ax2.grid(True, alpha=0.12, color='#0f172a', linewidth=0.8)
            st.pyplot(fig2)
            plt.close()
        except:
            pass

    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================
# PAGE: INDUSTRY VALUE
# ============================================================
elif page == "💼 Industry Value":
    st.markdown("""
    <span class='section-label'>Real-World Impact</span>
    <h1 class='hero-title'>Industry Impact &<br>Application</h1>
    <p class='hero-sub'>How this research translates into business value and production ML decisions</p>
    """, unsafe_allow_html=True)
    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-card-cyan'>
        <span class='section-label'>Scale of the Problem</span>
        <p style='color:#7dd3fc !important; margin:0; font-size:1rem; line-height:1.7;'>
        <strong>73% of ML models</strong> experience significant performance degradation within 6 months due to data drift.
        This costs enterprises an estimated <strong>$15 billion annually</strong> in lost revenue and poor automated decisions.
        This research provides a systematic, empirical framework for understanding and preventing this.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='glass-card'>
            <span class='section-label'>Production Challenges</span>
            <p style='color:#94a3b8 !important; font-size:0.9rem; line-height:2; margin:0;'>
            <strong>Silent Failures</strong> — Models degrade without alerting anyone<br>
            <strong>No Baselines</strong> — No benchmarks for acceptable degradation<br>
            <strong>Reactive Monitoring</strong> — Issues detected after business damage<br>
            <strong>Inefficient Retraining</strong> — Models retrained too early or too late
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='glass-card-violet' style='margin-top:0.75rem;'>
            <span class='section-label'>🎤 Interview Answer</span>
            <p style='color:#c4b5fd !important; font-size:0.85rem; line-height:1.8; margin:0; font-style:italic;'>
            "I studied how 4 ML architectures respond to covariate drift in telco churn data.
            I simulated drift using Gaussian noise injection and mean shift on numeric features,
            testing 7 magnitudes from 0.0 to 1.0. My key finding: accuracy is misleading —
            SVM showed 0% degradation but was entirely degenerate (F1 = 0). Logistic Regression
            was genuinely most robust. Random Forest F1 collapsed 26% at maximum drift.
            This proves that model selection for production must prioritise degradation robustness,
            not just baseline accuracy."
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='glass-card'>
            <span class='section-label'>Practical Applications</span>
            <p style='color:#94a3b8 !important; font-size:0.9rem; line-height:2.1; margin:0;'>
            <strong>For ML Engineers</strong><br>
            → Choose Logistic Regression for drift-prone domains<br>
            → Set KS-test monitoring at drift magnitude 0.2<br><br>
            <strong>For Data Scientists</strong><br>
            → Use degradation % as primary deployment metric<br>
            → Design drift-resilient feature pipelines<br><br>
            <strong>For Business</strong><br>
            → Schedule model reviews at 3-month intervals<br>
            → Budget retraining when drift exceeds 0.5
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PDF REPORT GENERATION
# ============================================================
def generate_pdf_report(results_df):
    pdf_path = "ML_Data_Drift_Report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Data Drift Impact on ML Model Performance</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("<b>Methodology Summary</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(
        "Covariate drift was simulated using Gaussian noise injection combined with a mean shift "
        "applied to all numeric features (tenure, MonthlyCharges, TotalCharges). "
        "Drift magnitudes from 0.0 to 1.0 were tested across 7 levels. "
        "Primary evaluation metric: Degradation % — percentage drop in accuracy from baseline to maximum drift.",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 0.4 * inch))

    baseline = results_df[results_df['drift_magnitude'] == 0.0]
    high_drift = results_df[results_df['drift_magnitude'] == results_df['drift_magnitude'].max()]
    avg_drop = round((baseline['accuracy'].mean() - high_drift['accuracy'].mean()) * 100, 2)

    elements.append(Paragraph("<b>Executive Summary</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(
        f"Total Models Tested: {results_df['model'].nunique()}<br/>"
        f"Total Experiments: {len(results_df)}<br/>"
        f"Max Drift Level: {results_df['drift_magnitude'].max()}<br/>"
        f"Average Performance Drop at Max Drift: {avg_drop}%<br/>"
        f"Key Finding: SVM collapsed to majority-class prediction (F1=0). "
        f"Logistic Regression was genuinely most robust (-1.18% degradation). "
        f"Random Forest was least robust (+1.31%, F1 dropped 26%).",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 0.4 * inch))

    elements.append(Paragraph("<b>Model Robustness Comparison</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))
    comparison_data = [["Model", "Baseline Acc", "Drifted Acc", "Degradation %", "Recommendation"]]
    model_results = []
    for model_name in baseline['model'].unique():
        base = baseline[baseline['model'] == model_name].iloc[0]
        drift = high_drift[high_drift['model'] == model_name].iloc[0]
        degradation = round((base['accuracy'] - drift['accuracy']) / base['accuracy'] * 100, 2)
        model_results.append((model_name, round(base['accuracy'], 4), round(drift['accuracy'], 4), degradation))
    model_results.sort(key=lambda x: x[3])
    best = model_results[0][0]
    for idx, (mn, ba, da, deg) in enumerate(model_results):
        if mn == 'SVM':
            rec = "DEGENERATE (F1=0)"
        elif idx == 0:
            rec = "RECOMMENDED"
        elif idx == len(model_results) - 1:
            rec = "Avoid in production"
        else:
            rec = "Acceptable"
        comparison_data.append([mn, ba, da, f"{deg}%", rec])
    table = Table(comparison_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#1e293b')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#dcfce7')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(
        f"<b>Recommendation: {best}</b> — most robust for production in drift-prone environments. "
        f"Showed -1.18% degradation, highest ROC-AUC at max drift (0.8221), and consistent F1.",
        styles["Normal"]
    ))
    elements.append(PageBreak())

    elements.append(Paragraph("<b>Model Performance Curves</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.3 * inch))
    for model_name in results_df['model'].unique():
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        model_data = results_df[results_df['model'] == model_name].sort_values('drift_magnitude')
        ax.plot(model_data['drift_magnitude'], model_data['accuracy'], marker='o', linewidth=2, color='#0f172a')
        ax.set_title(f"{model_name} — Accuracy vs Drift Magnitude", color='#0f172a')
        ax.set_xlabel("Drift Magnitude (0.0 = No Drift → 1.0 = Extreme)")
        ax.set_ylabel("Accuracy")
        ax.tick_params(colors='#0f172a')
        ax.grid(True, alpha=0.3)
        image_path = f"{model_name}_drift_plot.png"
        fig.savefig(image_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        elements.append(Paragraph(f"{model_name} — Performance Curve", styles["Heading3"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(RLImage(image_path, width=5.5 * inch, height=3.5 * inch))
        elements.append(Spacer(1, 0.5 * inch))

    elements.append(PageBreak())
    elements.append(Paragraph("<b>Research Conclusion</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(
        "This study empirically demonstrates that ML model robustness under covariate drift varies "
        "significantly across architectures, and that accuracy alone is insufficient to evaluate "
        "production reliability. The critical SVM finding — 0% accuracy degradation masking total "
        "F1 collapse — underscores the need for multi-metric evaluation. Logistic Regression is "
        "recommended for deployment in drift-prone environments. Organisations should implement "
        "F1-score monitoring with the drift magnitude thresholds identified in this study to "
        "trigger automated retraining before critical performance failure occurs.",
        styles["Normal"]
    ))
    doc.build(elements)
    return pdf_path


# ============================================================
# FOOTER + PDF
# ============================================================
st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<p class='footer-text'>NCI Master's Project &nbsp;·&nbsp; Data Mining & ML 2026 &nbsp;·&nbsp; Empirical Study of ML Model Performance Under Data Drift</p>
""", unsafe_allow_html=True)

try:
    if os.path.exists('results/experiments/covariate_drift_results.csv'):
        results = pd.read_csv('results/experiments/covariate_drift_results.csv')
        st.markdown("<span class='section-label'>Export</span>", unsafe_allow_html=True)
        st.markdown("<h3>Download Full Research Report</h3>", unsafe_allow_html=True)
        if st.button("⬇  Generate & Download PDF Report"):
            pdf_path = generate_pdf_report(results)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download  ML_Data_Drift_Report.pdf",
                    data=f,
                    file_name="ML_Data_Drift_Report.pdf",
                    mime="application/pdf"
                )
except:
    pass

