# Add to existing statsmodels/arch imports
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.outliers_influence import reset_ramsey
from arch.unitroot import PhillipsPerron
"""
NEXUS KERNEL — Professional Time-Series Econometrics Platform
Research by Ahmed Hisham
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import io
import base64
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="NEXUS KERNEL",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Lazy imports (keep startup fast) ─────────────────────────────────────────
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import jarque_bera, shapiro

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, ccf
from statsmodels.tsa.ardl import ARDL, ardl_select_order
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_white,
    het_breuschpagan,
)
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import pmdarima as pm

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — Midnight Navy / Steel Cyan / EViews-style light blue canvas
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&family=Orbitron:wght@700;900&display=swap');

/* ── Root palette ── */
:root {
    --navy:   #0F172A;
    --cyan:   #38BDF8;
    --cyan2:  #0EA5E9;
    --gold:   #F59E0B;
    --canvas: #E0E7FF;
    --card:   #F8FAFF;
    --border: #C7D2FE;
    --muted:  #64748B;
    --pass:   #10B981;
    --fail:   #EF4444;
    --warn:   #F59E0B;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--canvas) !important;
    color: var(--navy) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: 3px solid var(--cyan) !important;
}
section[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 2px solid var(--border) !important;
    border-left: 5px solid var(--cyan) !important;
    border-radius: 6px !important;
    padding: 14px 18px !important;
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.35rem !important;
    color: var(--navy) !important;
}

/* ── Brutalist result cards ── */
.brutalist-card {
    background: var(--card);
    border: 2px solid var(--navy);
    border-radius: 0;
    box-shadow: 5px 5px 0 var(--navy);
    padding: 20px 24px;
    margin-bottom: 20px;
}
.brutalist-card-cyan {
    background: var(--navy);
    border: 2px solid var(--cyan);
    border-radius: 0;
    box-shadow: 5px 5px 0 var(--cyan);
    padding: 20px 24px;
    margin-bottom: 20px;
    color: #F1F5F9 !important;
}
.brutalist-card-cyan * { color: #F1F5F9 !important; }

.badge-pass { background:#D1FAE5; color:#065F46; border:1.5px solid #6EE7B7;
    padding:3px 10px; border-radius:2px; font-size:0.75rem; font-weight:700;
    font-family:'Space Mono',monospace; text-transform:uppercase; }
.badge-fail { background:#FEE2E2; color:#991B1B; border:1.5px solid #FCA5A5;
    padding:3px 10px; border-radius:2px; font-size:0.75rem; font-weight:700;
    font-family:'Space Mono',monospace; text-transform:uppercase; }
.badge-warn { background:#FEF3C7; color:#92400E; border:1.5px solid #FCD34D;
    padding:3px 10px; border-radius:2px; font-size:0.75rem; font-weight:700;
    font-family:'Space Mono',monospace; text-transform:uppercase; }

/* ── Section headers ── */
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--navy);
    letter-spacing: 0.12em;
    border-bottom: 3px solid var(--cyan);
    padding-bottom: 6px;
    margin-bottom: 18px;
    text-transform: uppercase;
}
.section-title-inv {
    font-family: 'Orbitron', monospace;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--cyan);
    letter-spacing: 0.12em;
    border-bottom: 2px solid var(--cyan);
    padding-bottom: 4px;
    margin-bottom: 14px;
    text-transform: uppercase;
}

/* ── LaTeX equation block ── */
.eq-block {
    background: #1E293B;
    border-left: 4px solid var(--cyan);
    padding: 14px 20px;
    margin: 14px 0 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: var(--cyan);
    border-radius: 0 4px 4px 0;
    overflow-x: auto;
}

/* ── AI interpretation block ── */
.ai-block {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 100%);
    border: 1.5px solid var(--cyan);
    border-radius: 4px;
    padding: 18px 22px;
    margin-top: 14px;
    color: #E2E8F0 !important;
    font-size: 0.9rem;
    line-height: 1.7;
}
.ai-block h4 { color: var(--cyan) !important; font-family:'Orbitron',monospace;
    font-size:0.8rem; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:8px; }
.ai-block p { color: #CBD5E1 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--navy) !important;
    border-radius: 4px 4px 0 0;
    padding: 4px 6px 0;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #94A3B8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-bottom: 2px solid transparent !important;
    padding: 8px 14px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
    background: rgba(56,189,248,0.08) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--canvas) !important;
    border: 2px solid var(--navy);
    border-top: none;
    padding: 20px !important;
}

/* ── Tables ── */
.coef-table { width: 100%; border-collapse: collapse; font-family: 'Space Mono', monospace;
    font-size: 0.8rem; }
.coef-table th { background: var(--navy); color: var(--cyan); padding: 8px 12px;
    text-align: left; text-transform: uppercase; letter-spacing: 0.08em; }
.coef-table td { padding: 7px 12px; border-bottom: 1px solid var(--border); }
.coef-table tr:hover td { background: #EEF2FF; }
.sig { color: var(--pass); font-weight: 700; }
.insig { color: var(--fail); }

/* ── Dataframe ── */
.stDataFrame { border: 2px solid var(--border) !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--navy) !important;
    color: var(--cyan) !important;
    border: 2px solid var(--cyan) !important;
    border-radius: 2px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 700;
    padding: 8px 20px !important;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: var(--cyan) !important;
    color: var(--navy) !important;
    box-shadow: 4px 4px 0 var(--navy);
}

/* ── Download button ── */
.stDownloadButton > button {
    background: var(--gold) !important;
    color: var(--navy) !important;
    border: 2px solid var(--navy) !important;
    border-radius: 2px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: var(--navy) !important;
    color: var(--cyan) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Info / warning boxes ── */
.stAlert { border-radius: 2px !important; border-left-width: 5px !important; }

/* ── Number inputs ── */
.stNumberInput input, .stTextInput input {
    font-family: 'Space Mono', monospace !important;
    background: #F8FAFF !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 2px !important;
    color: var(--navy) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #F8FAFF !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 2px !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background: var(--cyan) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--canvas); }
::-webkit-scrollbar-thumb { background: var(--cyan); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "authenticated": False,
        "user_name": "",
        "user_email": "",
        "user_occ": "",
        "raw_df": None,
        "clean_df": None,
        "freq": None,
        "ols_results": None,
        "ardl_results": None,
        "var_results": None,
        "garch_results": None,
        "arima_results": None,
        "report_sections": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def pval_badge(p, alpha=0.05):
    if p < 0.01:   return '<span class="badge-pass">★★★ p<0.01</span>'
    if p < 0.05:   return '<span class="badge-pass">★★ p<0.05</span>'
    if p < 0.10:   return '<span class="badge-warn">★ p<0.10</span>'
    return '<span class="badge-fail">Insig.</span>'

def stat_verdict(p, alpha=0.05, reject_label="Reject H₀", fail_label="Fail to Reject H₀"):
    if p < alpha:
        return f'<span class="badge-pass">{reject_label}</span>'
    return f'<span class="badge-fail">{fail_label}</span>'

def fmt(x, decimals=4):
    try:
        return f"{float(x):.{decimals}f}"
    except:
        return str(x)

def navy_fig(fig, height=420):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F8FAFF",
        font=dict(family="DM Sans", color="#0F172A"),
        title_font=dict(family="Orbitron", size=13, color="#0F172A"),
        xaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", zerolinecolor="#CBD5E1"),
        yaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", zerolinecolor="#CBD5E1"),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#CBD5E1", borderwidth=1),
    )
    return fig

CYAN = "#38BDF8"
NAVY = "#0F172A"
GOLD = "#F59E0B"
TEAL = "#14B8A6"
RED  = "#EF4444"
GRN  = "#10B981"


# ══════════════════════════════════════════════════════════════════════════════
# ① LOGIN GATEWAY
# ══════════════════════════════════════════════════════════════════════════════
def login_page():
    st.markdown("""
    <div style="max-width:520px;margin:60px auto 0;">
        <div style="text-align:center;margin-bottom:32px;">
            <p style="font-family:'Orbitron',monospace;font-size:2.6rem;font-weight:900;
               color:#0F172A;letter-spacing:0.15em;margin-bottom:0;line-height:1;">
               ⬡ NEXUS KERNEL
            </p>
            <p style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#38BDF8;
               letter-spacing:0.25em;text-transform:uppercase;margin-top:6px;">
               Professional Time-Series Econometrics Platform
            </p>
            <p style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#64748B;
               margin-top:4px;">Research by Ahmed Hisham</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        with st.form("login_form"):
            st.markdown('<p class="section-title">Access Portal</p>', unsafe_allow_html=True)
            fname = st.text_input("First Name")
            lname = st.text_input("Last Name")
            email = st.text_input("Email Address")
            occ   = st.selectbox("Occupation", [
                "Select…", "Researcher / Academic", "Economist",
                "Financial Analyst", "Policy Analyst", "Student",
                "Data Scientist", "Other"
            ])
            submitted = st.form_submit_button("▶  ENTER NEXUS KERNEL", use_container_width=True)
            if submitted:
                if fname and lname and "@" in email and occ != "Select…":
                    st.session_state.authenticated = True
                    st.session_state.user_name  = f"{fname} {lname}"
                    st.session_state.user_email = email
                    st.session_state.user_occ   = occ
                    st.rerun()
                else:
                    st.error("Please complete all fields with valid information.")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:18px 0 10px;text-align:center;border-bottom:1px solid #1E3A5F;">
            <p style="font-family:'Orbitron',monospace;font-size:1.25rem;font-weight:900;
               color:#38BDF8;letter-spacing:0.2em;margin:0;line-height:1.2;">⬡ NEXUS</p>
            <p style="font-family:'Orbitron',monospace;font-size:1.0rem;font-weight:700;
               color:#38BDF8;letter-spacing:0.3em;margin:0;">KERNEL</p>
            <p style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#475569;
               letter-spacing:0.15em;margin-top:4px;">RESEARCH BY AHMED HISHAM</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="padding:12px 8px;border-bottom:1px solid #1E3A5F;margin-bottom:10px;">
            <p style="font-size:0.72rem;color:#64748B;margin:0;letter-spacing:0.08em;
               font-family:'Space Mono',monospace;text-transform:uppercase;">Session</p>
            <p style="font-size:0.88rem;color:#CBD5E1;margin:2px 0 0;font-weight:600;">
               {st.session_state.user_name}</p>
            <p style="font-size:0.72rem;color:#475569;margin:0;">{st.session_state.user_occ}</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.clean_df is not None:
            df = st.session_state.clean_df
            st.markdown("""
            <p style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#38BDF8;
               letter-spacing:0.15em;text-transform:uppercase;margin-bottom:6px;">
               ◈ Active Dataset</p>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#0A1628;border:1px solid #1E3A5F;border-radius:3px;
                 padding:10px 12px;margin-bottom:12px;">
                <p style="margin:0;font-size:0.78rem;color:#94A3B8;font-family:'Space Mono',monospace;">
                   Obs: <span style="color:#38BDF8;">{len(df)}</span> &nbsp;|&nbsp;
                   Vars: <span style="color:#38BDF8;">{len(df.columns)}</span> &nbsp;|&nbsp;
                   Freq: <span style="color:#F59E0B;">{st.session_state.freq or 'Unknown'}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Navigation hint
        st.markdown("""
        <p style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#334155;
           letter-spacing:0.12em;text-transform:uppercase;margin-top:8px;">
           ◈ Module Navigator</p>
        <p style="font-size:0.72rem;color:#475569;line-height:1.6;margin-bottom:10px;">
           Use the tabs above to switch between modules. Data persists across tabs.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("⏻  Sign Out", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        st.markdown("""
        <div style="margin-top:auto;padding-top:20px;text-align:center;">
            <p style="font-size:0.6rem;color:#1E3A5F;font-family:'Space Mono',monospace;
               letter-spacing:0.1em;">NEXUS KERNEL v2.0 · EViews-grade · 2025</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ② DATA INGESTION & CLEANING TAB
# ══════════════════════════════════════════════════════════════════════════════
def tab_data():
    st.markdown('<p class="section-title">⬡ Data Ingestion & Cleaning Engine</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload CSV or XLSX file",
        type=["csv", "xlsx"],
        help="Upload your time-series data. First column should be a date/period column."
    )

    if uploaded:
        try:
            if uploaded.name.endswith(".xlsx"):
                raw = pd.read_excel(uploaded)
            else:
                raw = pd.read_csv(uploaded)
            st.session_state.raw_df = raw.copy()
            st.success(f"✓  Loaded **{uploaded.name}** — {raw.shape[0]} rows × {raw.shape[1]} columns")
        except Exception as e:
            st.error(f"Read error: {e}")
            return

    if st.session_state.raw_df is None:
        st.markdown("""
        <div class="brutalist-card" style="text-align:center;padding:40px;">
            <p style="font-family:'Orbitron',monospace;color:#94A3B8;font-size:0.9rem;
               letter-spacing:0.15em;">NO DATA LOADED</p>
            <p style="color:#64748B;font-size:0.85rem;">Upload a CSV or XLSX file to begin analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    raw = st.session_state.raw_df.copy()
    st.markdown('<p class="section-title">Raw Preview</p>', unsafe_allow_html=True)
    st.dataframe(raw.head(10), use_container_width=True)

    st.markdown('<p class="section-title">⬡ Cleaning Configuration</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        date_col = st.selectbox("Date / Period Column", raw.columns.tolist())
    with c2:
        freq_choice = st.selectbox("Data Frequency", ["Auto-Detect", "Annual", "Quarterly", "Monthly", "Daily"])
    with c3:
        missing_method = st.selectbox("Missing Data Method", ["Linear Interpolation", "Forward Fill", "Backward Fill", "Drop Rows"])

    strip_symbols = st.checkbox("Strip symbols ($, %, commas) and convert to numeric", value=True)
    seasonal_adj  = st.checkbox("Apply Seasonal Adjustment (X-11 decomposition proxy)", value=False)

    if st.button("▶  RUN CLEANING ENGINE", use_container_width=True):
        with st.spinner("Processing..."):
            df = raw.copy()

            # 1. Date index
            try:
                df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
                df = df.set_index(date_col).sort_index()
            except Exception:
                try:
                    df = df.set_index(date_col).sort_index()
                except:
                    st.warning("Could not parse date column. Using row index.")

            # 2. Symbol stripping
            if strip_symbols:
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = (df[col].astype(str)
                                   .str.replace(r'[$%,\s]', '', regex=True)
                                   .str.replace(r'[^\d\.\-]', '', regex=True))
                        df[col] = pd.to_numeric(df[col], errors='coerce')

            # 3. Frequency detection
            if freq_choice == "Auto-Detect":
                try:
                    inferred = pd.infer_freq(df.index)
                    if inferred in ['A', 'AS', 'YE', 'YS', 'BA', 'BAS']:
                        freq = "Annual"
                    elif inferred in ['Q', 'QS', 'BQ', 'BQS', 'QE']:
                        freq = "Quarterly"
                    elif inferred in ['M', 'MS', 'BM', 'BMS', 'ME']:
                        freq = "Monthly"
                    elif inferred in ['D', 'B']:
                        freq = "Daily"
                    else:
                        freq = inferred or "Unknown"
                except:
                    freq = "Unknown"
            else:
                freq = freq_choice
            st.session_state.freq = freq

            # 4. Missing data
            if missing_method == "Linear Interpolation":
                df = df.interpolate(method='linear')
            elif missing_method == "Forward Fill":
                df = df.ffill()
            elif missing_method == "Backward Fill":
                df = df.bfill()
            elif missing_method == "Drop Rows":
                df = df.dropna()

            # 5. Seasonal adjustment (STL-based proxy)
            if seasonal_adj and freq in ["Quarterly", "Monthly"]:
                period = 4 if freq == "Quarterly" else 12
                for col in df.select_dtypes(include=np.number).columns:
                    try:
                        from statsmodels.tsa.seasonal import STL
                        stl = STL(df[col].dropna(), period=period)
                        res = stl.fit()
                        df[col + "_SA"] = df[col] - res.seasonal
                    except:
                        pass

            st.session_state.clean_df = df

        # ── Stats summary ──
        st.markdown('<p class="section-title">Summary Statistics</p>', unsafe_allow_html=True)
        desc = df.describe().T
        st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)

        # ── Missing heatmap ──
        miss = df.isnull().sum()
        if miss.sum() > 0:
            st.warning(f"⚠ Remaining missing values after treatment: {miss.sum()}")
        else:
            st.success("✓ No missing values remaining.")

        # ── Frequency badge ──
        st.markdown(f"""
        <div class="brutalist-card">
            <p style="margin:0;font-family:'Space Mono',monospace;font-size:0.8rem;">
               Detected Frequency: <span style="color:#38BDF8;font-weight:700;">{freq}</span> &nbsp;|&nbsp;
               Observations: <span style="color:#F59E0B;font-weight:700;">{len(df)}</span> &nbsp;|&nbsp;
               Numeric Variables: <span style="color:#10B981;font-weight:700;">
               {len(df.select_dtypes(include=np.number).columns)}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Time-series plot ──
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            fig = go.Figure()
            for i, col in enumerate(num_cols[:6]):
                colors_list = [CYAN, GOLD, TEAL, RED, GRN, "#A78BFA"]
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], name=col,
                    line=dict(color=colors_list[i % 6], width=2),
                    mode='lines'
                ))
            fig = navy_fig(fig, height=380)
            fig.update_layout(title="Time-Series Overview (first 6 variables)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<p class="section-title">Cleaned Dataset</p>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ③ STATIONARITY VAULT TAB
# ══════════════════════════════════════════════════════════════════════════════
def run_adf(series, maxlag, regression, autolag):
    kw = dict(regression=regression)
    if autolag == "Fixed":
        kw["maxlag"] = maxlag
        kw["autolag"] = None
    else:
        kw["autolag"] = autolag
    res = adfuller(series.dropna(), **kw)
    # res = (adf_stat, pvalue, usedlag, nobs, critical_values, icbest)
    return res[0], res[1], res[2], res[4]  # stat, pval, lags, crits

# Replace the old run_pp function with this:
def run_pp(series, regression):
    """True Phillips-Perron test using arch.unitroot with Newey-West standard errors."""
    try:
        # arch.unitroot uses 'c' for constant, 'ct' for trend, 'n' for none
        pp = PhillipsPerron(series.dropna(), trend=regression)
        # critical values are stored in a dict-like object
        crits = {
            '1%': pp.critical_values.get('1%', 0),
            '5%': pp.critical_values.get('5%', 0),
            '10%': pp.critical_values.get('10%', 0)
        }
        return pp.stat, pp.pvalue, pp.lags, crits
    except Exception as e:
        return None, None, None, {}

def tab_stationarity():
    st.markdown('<p class="section-title">⬡ Stationarity Vault — Unit Root Testing</p>', unsafe_allow_html=True)

    if st.session_state.clean_df is None:
        st.info("Please load and clean your data in the **Data** tab first.")
        return

    df = st.session_state.clean_df
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.warning("No numeric columns found.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sel_var = st.selectbox("Variable", num_cols)
    with c2:
        diff_order = st.selectbox("Transformation", ["Level", "1st Difference", "2nd Difference"])
    with c3:
        regression = st.selectbox("Deterministic Terms", ["c (Intercept)", "ct (Trend+Intercept)", "n (None)", "ctt (Quad Trend)"])
        reg_map = {"c (Intercept)": "c", "ct (Trend+Intercept)": "ct", "n (None)": "n", "ctt (Quad Trend)": "ctt"}
        reg_code = reg_map[regression]
    with c4:
        lag_method = st.selectbox("Lag Selection", ["AIC", "BIC", "t-stat", "Fixed"])

    fixed_lags = 1
    if lag_method == "Fixed":
        fixed_lags = st.slider("Fixed Lag Length", 1, 12, 2)

    if st.button("▶  RUN UNIT ROOT TESTS", use_container_width=True):
        series = df[sel_var].copy().dropna()

        if diff_order == "1st Difference":
            series = series.diff().dropna()
            label = f"Δ{sel_var}"
        elif diff_order == "2nd Difference":
            series = series.diff().diff().dropna()
            label = f"Δ²{sel_var}"
        else:
            label = sel_var

        # ── Time-series plot ──
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=label,
                                  line=dict(color=CYAN, width=2)))
        fig.add_hline(y=series.mean(), line=dict(color=GOLD, dash='dash', width=1),
                      annotation_text="Mean", annotation_position="bottom right")
        fig = navy_fig(fig, 300)
        fig.update_layout(title=f"Series Plot: {label}")
        st.plotly_chart(fig, use_container_width=True)

        # ── ADF ──
        try:
            adf_stat, adf_p, adf_lags, adf_crits = run_adf(series, fixed_lags, reg_code, lag_method if lag_method != "Fixed" else "Fixed")
            has_adf = True
        except Exception as e:
            has_adf = False
            st.error(f"ADF failed: {e}")

        # ── PP (using ADF with BIC as proxy) ──
        try:
            pp_stat, pp_p, pp_lags, pp_crits = run_pp(series, reg_code)
            has_pp = True
        except:
            has_pp = False

        # ── KPSS ──
        try:
            kpss_reg = "c" if reg_code in ["c", "n"] else "ct"
            kpss_stat, kpss_p, kpss_lags, kpss_crits = kpss(series.dropna(), regression=kpss_reg, nlags="auto")
            has_kpss = True
        except Exception as e:
            has_kpss = False

        # ── Results cards ──
        col_a, col_b = st.columns(2)

        with col_a:
            if has_adf:
                verdict_adf = "STATIONARY" if adf_p < 0.05 else "NON-STATIONARY"
                badge_cls   = "badge-pass" if adf_p < 0.05 else "badge-fail"
                st.markdown(f"""
                <div class="brutalist-card">
                    <p class="section-title" style="font-size:0.85rem;">
                       Augmented Dickey-Fuller Test</p>
                    <table class="coef-table">
                        <tr><th>Statistic</th><th>Value</th></tr>
                        <tr><td>ADF Statistic</td><td style="font-family:'Space Mono',monospace;">
                            {fmt(adf_stat)}</td></tr>
                        <tr><td>p-value</td><td style="font-family:'Space Mono',monospace;">
                            {fmt(adf_p)}</td></tr>
                        <tr><td>Lags Used</td><td style="font-family:'Space Mono',monospace;">
                            {adf_lags}</td></tr>
                        <tr><td>Crit. 1%</td><td style="font-family:'Space Mono',monospace;">
                            {fmt(adf_crits.get('1%',0))}</td></tr>
                        <tr><td>Crit. 5%</td><td style="font-family:'Space Mono',monospace;">
                            {fmt(adf_crits.get('5%',0))}</td></tr>
                        <tr><td>Crit. 10%</td><td style="font-family:'Space Mono',monospace;">
                            {fmt(adf_crits.get('10%',0))}</td></tr>
                    </table>
                    <p style="margin-top:10px;font-family:'Space Mono',monospace;font-size:0.8rem;">
                       H₀: Unit root present &nbsp;|&nbsp;
                       <span class="{badge_cls}">{verdict_adf}</span>
                    </p>
                    <p style="font-size:0.75rem;color:#64748B;margin-top:6px;">
                       ADF stat {'<' if adf_stat < adf_crits.get('5%',0) else '≥'} 5% critical value
                       ({fmt(adf_crits.get('5%',0))})
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with col_b:
            if has_kpss:
                verdict_kpss = "STATIONARY" if kpss_p > 0.05 else "NON-STATIONARY"
                badge_cls_k  = "badge-pass" if kpss_p > 0.05 else "badge-fail"
                st.markdown(f"""
                <div class="brutalist-card">
                    <p class="section-title" style="font-size:0.85rem;">
                       KPSS Test (H₀: Stationary)</p>
                    <table class="coef-table">
                        <tr><th>Statistic</th><th>Value</th></tr>
                        <tr><td>KPSS Statistic</td><td style="font-family:'Space Mono',monospace;">
                            {fmt(kpss_stat)}</td></tr>
                        <tr><td>p-value (approx.)</td><td style="font-family:'Space Mono',monospace;">
                            {fmt(kpss_p)}</td></tr>
                        <tr><td>Lags Used</td><td style="font-family:'Space Mono',monospace;">
                            {kpss_lags}</td></tr>
                        <tr><td>Crit. 1%</td><td style="font-family:'Space Mono',monospace;">
                            {fmt(kpss_crits.get('1%',0))}</td></tr>
                        <tr><td>Crit. 5%</td><td style="font-family:'Space Mono',monospace;">
                            {fmt(kpss_crits.get('5%',0))}</td></tr>
                        <tr><td>Crit. 10%</td><td style="font-family:'Space Mono',monospace;">
                            {fmt(kpss_crits.get('10%',0))}</td></tr>
                    </table>
                    <p style="margin-top:10px;font-family:'Space Mono',monospace;font-size:0.8rem;">
                       H₀: Series is Stationary &nbsp;|&nbsp;
                       <span class="{badge_cls_k}">{verdict_kpss}</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)

        # ── ACF / PACF ──
        st.markdown('<p class="section-title">ACF & PACF Correlograms</p>', unsafe_allow_html=True)
        max_lags = min(20, len(series) // 4)
        acf_vals  = acf(series.dropna(),  nlags=max_lags, fft=True)
        pacf_vals = pacf(series.dropna(), nlags=max_lags)
        lags_range = list(range(len(acf_vals)))
        conf = 1.96 / np.sqrt(len(series))

        fig2 = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"])
        for name, vals, col_idx in [("ACF", acf_vals, 1), ("PACF", pacf_vals, 2)]:
            fig2.add_trace(go.Bar(x=lags_range, y=vals, name=name,
                                   marker_color=CYAN, opacity=0.8), row=1, col=col_idx)
            fig2.add_hline(y=conf,  line=dict(color=RED, dash='dash', width=1), row=1, col=col_idx)
            fig2.add_hline(y=-conf, line=dict(color=RED, dash='dash', width=1), row=1, col=col_idx)
            fig2.add_hline(y=0,     line=dict(color=NAVY, width=0.5), row=1, col=col_idx)
        fig2 = navy_fig(fig2, 340)
        fig2.update_layout(showlegend=False, title=f"Correlograms — {label}")
        st.plotly_chart(fig2, use_container_width=True)

        # ── AI interpretation ──
        if has_adf:
            interp = _interpret_stationarity(adf_stat, adf_p, adf_crits, label, diff_order,
                                              kpss_p if has_kpss else None)
            st.markdown(f"""
            <div class="ai-block">
                <h4>⬡ AI Interpretation</h4>
                {interp}
            </div>
            """, unsafe_allow_html=True)


def _interpret_stationarity(stat, pval, crits, var, diff, kpss_p):
    verdict = "stationary" if pval < 0.05 else "non-stationary (contains a unit root)"
    rec = ""
    if pval >= 0.05 and diff == "Level":
        rec = "<p>📌 <strong>Recommendation:</strong> First-difference the series before estimation. Re-run the ADF on Δ" + var + ".</p>"
    elif pval < 0.05 and diff == "1st Difference":
        rec = f"<p>📌 <strong>Recommendation:</strong> {var} is integrated of order <strong>I(1)</strong>. Suitable for cointegration analysis or ARDL Bounds Test.</p>"
    elif pval < 0.05 and diff == "Level":
        rec = f"<p>📌 <strong>Recommendation:</strong> {var} is <strong>I(0)</strong> — stationary in levels. May be used directly in OLS or ARDL.</p>"

    kpss_note = ""
    if kpss_p is not None:
        if kpss_p > 0.05 and pval < 0.05:
            kpss_note = "<p>✅ <strong>KPSS confirms stationarity</strong> — both tests agree. High confidence.</p>"
        elif kpss_p < 0.05 and pval >= 0.05:
            kpss_note = "<p>⚠️ <strong>Mixed signals:</strong> ADF fails to reject unit root, KPSS rejects stationarity. Consider structural breaks.</p>"

    return f"""
    <p>The <strong>Augmented Dickey-Fuller test</strong> on <em>{var}</em> ({diff}) produces a
    test statistic of <strong>{stat:.4f}</strong> with a p-value of <strong>{pval:.4f}</strong>.
    At the conventional 5% significance level, the series appears <strong>{verdict}</strong>.</p>
    {kpss_note}
    {rec}
    <p style="font-size:0.8rem;color:#94A3B8;margin-top:8px;">
    Critical values: 1% = {crits.get('1%',0):.4f} | 5% = {crits.get('5%',0):.4f} | 10% = {crits.get('10%',0):.4f}</p>
    """


# ══════════════════════════════════════════════════════════════════════════════
# ④ ESTIMATION KERNEL TAB
# ══════════════════════════════════════════════════════════════════════════════
def tab_estimation():
    st.markdown('<p class="section-title">⬡ Estimation Kernel</p>', unsafe_allow_html=True)

    if st.session_state.clean_df is None:
        st.info("Please load and clean your data in the **Data** tab first.")
        return

    df   = st.session_state.clean_df
    cols = df.select_dtypes(include=np.number).columns.tolist()

    model_type = st.selectbox("Select Estimation Model", [
        "OLS — Ordinary Least Squares",
        "ARDL — Autoregressive Distributed Lag",
        "VAR — Vector Autoregression",
        "VECM — Vector Error Correction Model",
        "GARCH(1,1) — Volatility Model",
        "ARIMA / SARIMA — Forecasting",
    ])

    st.markdown("---")

    if model_type.startswith("OLS"):
        _est_ols(df, cols)
    elif model_type.startswith("ARDL"):
        _est_ardl(df, cols)
    elif model_type.startswith("VAR"):
        _est_var(df, cols)
    elif model_type.startswith("VECM"):
        _est_vecm(df, cols)
    elif model_type.startswith("GARCH"):
        _est_garch(df, cols)
    elif model_type.startswith("ARIMA"):
        _est_arima(df, cols)


# ── OLS ───────────────────────────────────────────────────────────────────────
def _est_ols(df, cols):
    st.markdown('<p class="section-title">OLS Configuration</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        dep_var  = st.selectbox("Dependent Variable (Y)", cols)
    with c2:
        indep_vars = st.multiselect("Independent Variables (X)", [c for c in cols if c != dep_var])

    add_const = st.checkbox("Include Constant (Intercept)", value=True)
    log_transform = st.multiselect("Apply log() to selected variables", [dep_var] + indep_vars)

    if st.button("▶  ESTIMATE OLS", use_container_width=True):
        if not indep_vars:
            st.error("Select at least one independent variable.")
            return

        data = df[[dep_var] + indep_vars].dropna().copy()
        for col in log_transform:
            if col in data.columns:
                # Safe log transform: filter out non-positive values to prevent -inf/NaN crashes
                valid_mask = data[col] > 0
                if not valid_mask.all():
                    st.warning(f"⚠️ Variable '{col}' contains zero or negative values. These observations were dropped for log transformation.")
                data = data[valid_mask]
                data[col] = np.log(data[col])
        data = data.dropna()

        Y = data[dep_var]
        X = data[indep_vars]
        if add_const:
            X = sm.add_constant(X)

        try:
            model  = OLS(Y, X).fit(cov_type='HC1')
        except Exception as e:
            st.error(f"Estimation failed: {e}")
            return

        st.session_state.ols_results = model

        # ── LaTeX equation ──
        vars_str = " + ".join([f"\\beta_{{{i+1}}} \\cdot {v}" for i, v in enumerate(indep_vars)])
        st.markdown(f"""
        <div class="eq-block">
            <p style="margin:0;font-size:0.82rem;">𝑴𝒐𝒅𝒆𝒍 𝑺𝒑𝒆𝒄𝒊𝒇𝒊𝒄𝒂𝒕𝒊𝒐𝒏:</p>
            <p style="margin:4px 0 0;font-size:1rem;color:#F59E0B;">
               {dep_var} = {'β₀ + ' if add_const else ''}{' + '.join([f'β{i+1}·{v}' for i,v in enumerate(indep_vars)])} + ε
            </p>
            <p style="margin:4px 0 0;font-size:0.72rem;color:#64748B;">
               Standard errors: HC1 (Heteroskedasticity-Consistent White robust SEs)</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Goodness-of-fit ──
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("R²",       f"{model.rsquared:.4f}")
        c2.metric("Adj. R²",  f"{model.rsquared_adj:.4f}")
        c3.metric("F-stat",   f"{model.fvalue:.3f}")
        c4.metric("Prob(F)",  f"{model.f_pvalue:.4f}")
        c5.metric("AIC",      f"{model.aic:.2f}")

        c6, c7, c8, c9 = st.columns(4)
        c6.metric("Log-Lik",  f"{model.llf:.3f}")
        c7.metric("BIC",      f"{model.bic:.2f}")
        c8.metric("SSR",      f"{model.ssr:.4f}")
        c9.metric("Obs (N)",  f"{int(model.nobs)}")

        # ── Coefficient table ──
        st.markdown('<p class="section-title">Coefficient Table</p>', unsafe_allow_html=True)
        rows = ""
        for var_name, coef, se, tstat, pval in zip(
            model.params.index, model.params, model.bse, model.tvalues, model.pvalues
        ):
            ci_lo, ci_hi = model.conf_int().loc[var_name]
            sig_cls = "sig" if pval < 0.05 else "insig"
            rows += f"""
            <tr>
                <td style="font-family:'Space Mono',monospace;">{var_name}</td>
                <td class="{sig_cls}">{fmt(coef)}</td>
                <td>{fmt(se)}</td>
                <td>{fmt(tstat)}</td>
                <td>{fmt(pval)}</td>
                <td>{fmt(ci_lo)}</td>
                <td>{fmt(ci_hi)}</td>
                <td>{pval_badge(pval)}</td>
            </tr>"""
        st.markdown(f"""
        <div class="brutalist-card">
            <table class="coef-table">
                <tr>
                    <th>Variable</th><th>Coefficient</th><th>Std. Error</th>
                    <th>t-Statistic</th><th>p-Value</th>
                    <th>CI Lower (95%)</th><th>CI Upper (95%)</th><th>Significance</th>
                </tr>
                {rows}
            </table>
        </div>
        """, unsafe_allow_html=True)

        # ── Residual plots ──
        _residual_plots(model.resid.values, model.fittedvalues.values, dep_var)

        # ── Diagnostics ──
        _run_diagnostics(model, Y, X)

        # ── AI interpretation ──
        ai_text = _interpret_ols(model, dep_var, indep_vars, log_transform)
        st.markdown(f"""
        <div class="ai-block">
            <h4>⬡ AI Interpretation & Analysis</h4>
            {ai_text}
        </div>
        """, unsafe_allow_html=True)

        # Store for report
        st.session_state.report_sections["OLS"] = {
            "model": "OLS",
            "dep": dep_var,
            "indep": indep_vars,
            "r2": model.rsquared,
            "adj_r2": model.rsquared_adj,
            "fstat": model.fvalue,
            "fp": model.f_pvalue,
            "nobs": model.nobs,
            "params": dict(zip(model.params.index, model.params)),
            "pvals": dict(zip(model.pvalues.index, model.pvalues)),
            "ai": ai_text,
        }


def _interpret_ols(model, dep, indep, log_vars):
    sig_count = (model.pvalues < 0.05).sum()
    insig = [v for v in indep if model.pvalues.get(v, 1) >= 0.05]
    sig   = [v for v in indep if model.pvalues.get(v, 1) < 0.05]

    coef_text = ""
    for v in indep:
        c = model.params.get(v, 0)
        p = model.pvalues.get(v, 1)
        if v in log_vars and dep in log_vars:
            interp = f"a 1% increase in {v} is associated with a <strong>{c:.4f}%</strong> change in {dep} (elasticity)"
        elif v in log_vars:
            interp = f"a 1% increase in {v} changes {dep} by approximately <strong>{c/100:.6f}</strong> units (semi-elasticity)"
        elif dep in log_vars:
            interp = f"a 1-unit increase in {v} changes {dep} by approximately <strong>{c*100:.4f}%</strong>"
        else:
            interp = f"a 1-unit increase in {v} is associated with a <strong>{c:.4f}</strong>-unit change in {dep}"

        sig_label = "statistically significant at 5%" if p < 0.05 else "not statistically significant"
        coef_text += f"<li><strong>{v}</strong> (β={c:.4f}, p={p:.4f}): {interp}. This coefficient is <em>{sig_label}</em>.</li>"

    quality = "strong" if model.rsquared_adj > 0.7 else "moderate" if model.rsquared_adj > 0.4 else "weak"
    f_verdict = "statistically significant as a whole" if model.f_pvalue < 0.05 else "not jointly significant"

    return f"""
    <p>The OLS model regresses <strong>{dep}</strong> on {len(indep)} regressor(s). The model explains
    <strong>{model.rsquared*100:.2f}%</strong> of variance in the dependent variable
    (Adj. R² = {model.rsquared_adj:.4f}), indicating <em>{quality}</em> explanatory power.
    The F-statistic ({model.fvalue:.3f}, p={model.f_pvalue:.4f}) shows the model is {f_verdict}.</p>
    <p><strong>Coefficient Interpretation:</strong></p>
    <ul>{coef_text}</ul>
    {'<p>⚠️ Variables ' + ', '.join(insig) + ' are not significant at 5%. Consider dropping or testing joint restrictions.</p>' if insig else ''}
    <p style="font-size:0.8rem;color:#94A3B8;">
       Standard errors are HC1 (White-corrected), guarding against heteroskedasticity.
       Always verify Breusch-Pagan and White tests in the Diagnostic Suite tab.</p>
    """


# ── ARDL ──────────────────────────────────────────────────────────────────────
def _est_ardl(df, cols):
    st.markdown('<p class="section-title">ARDL Configuration</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        dep_var   = st.selectbox("Dependent Variable (Y)", cols)
    with c2:
        indep_vars = st.multiselect("Independent Variables", [c for c in cols if c != dep_var])

    c3, c4, c5 = st.columns(3)
    with c3:
        max_lag_y = st.slider("Max Lags for Y", 1, 8, 4)
    with c4:
        max_lag_x = st.slider("Max Lags for X", 0, 8, 4)
    with c5:
        ic_crit   = st.selectbox("Lag Selection Criterion", ["aic", "bic", "hqic"])

    bounds_test = st.checkbox("Run PSS Bounds Test for Cointegration", value=True)

    if st.button("▶  ESTIMATE ARDL", use_container_width=True):
        if not indep_vars:
            st.error("Select at least one independent variable.")
            return
        try:
            data = df[[dep_var] + indep_vars].dropna()
            order_sel = ardl_select_order(
                data[dep_var], max_lag_y,
                data[indep_vars], max_lag_x,
                ic=ic_crit, trend='c'
            )
            best_order = order_sel.ardl_order

            model = ARDL(
                data[dep_var], best_order[0],
                data[indep_vars], best_order[1:],
                trend='c'
            ).fit()

            st.session_state.ardl_results = model

            # ── LaTeX ──
            p = best_order[0]
            x_lags = dict(zip(indep_vars, best_order[1:] if len(best_order) > 1 else [max_lag_x]*len(indep_vars)))
            lag_str = " + ".join([f"α{i}·{dep_var}(t-{i})" for i in range(1, p+1)])
            x_str   = " + ".join([f"Σβ·{v}(t-j)" for v in indep_vars])
            st.markdown(f"""
            <div class="eq-block">
                <p style="margin:0;font-size:0.82rem;">ARDL({p},{','.join(str(q) for q in best_order[1:])}) Model:</p>
                <p style="margin:4px 0 0;color:#F59E0B;">
                   {dep_var}(t) = c + {lag_str} + {x_str} + ε(t)
                </p>
                <p style="margin:4px 0 0;font-size:0.72rem;color:#64748B;">
                   Optimal lags selected by {ic_crit.upper()}. Trend = constant only.</p>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("AIC", f"{model.aic:.3f}")
            c2.metric("BIC", f"{model.bic:.3f}")
            c3.metric("Log-Lik", f"{model.llf:.3f}")
            c4.metric("Observations", f"{int(model.nobs)}")

            # ── Coefficients ──
            st.markdown('<p class="section-title">ARDL Coefficient Table</p>', unsafe_allow_html=True)
            rows = ""
            for vname, coef, se, tval, pval in zip(
                model.params.index, model.params, model.bse, model.tvalues, model.pvalues
            ):
                sig_cls = "sig" if pval < 0.05 else "insig"
                rows += f"""<tr>
                    <td style="font-family:'Space Mono',monospace;">{vname}</td>
                    <td class="{sig_cls}">{fmt(coef)}</td>
                    <td>{fmt(se)}</td>
                    <td>{fmt(tval)}</td>
                    <td>{fmt(pval)}</td>
                    <td>{pval_badge(pval)}</td></tr>"""
            st.markdown(f"""
            <div class="brutalist-card">
                <table class="coef-table">
                    <tr><th>Variable</th><th>Coefficient</th><th>Std Error</th>
                        <th>t-Stat</th><th>p-Value</th><th>Sig.</th></tr>
                    {rows}
                </table>
            </div>""", unsafe_allow_html=True)

            # ── Long-Run Coefficients ──
            try:
                lr = model.params.copy()
                # Simple long-run approximation
                const_coef = lr.get('const', 0)
                lag_sum = sum([lr.get(f'{dep_var}.L{i}', 0) for i in range(1, p+1)])
                denom = 1 - lag_sum if abs(1 - lag_sum) > 1e-6 else 1e-6

                st.markdown('<p class="section-title">Long-Run Coefficients (Derived)</p>', unsafe_allow_html=True)
                lr_rows = ""
                lr_const = const_coef / denom
                lr_rows += f"<tr><td>Constant</td><td style='font-family:Space Mono,monospace;'>{fmt(lr_const)}</td><td>—</td></tr>"
                for v in indep_vars:
                    x_sum = sum([lr.get(f'{v}.L{j}', lr.get(v, 0)) for j in range(0, x_lags.get(v, 0)+1)])
                    lr_coef = x_sum / denom
                    lr_rows += f"<tr><td>{v}</td><td style='font-family:Space Mono,monospace;'>{fmt(lr_coef)}</td><td>Long-run elasticity/effect</td></tr>"

                st.markdown(f"""
                <div class="brutalist-card">
                    <table class="coef-table">
                        <tr><th>Variable</th><th>Long-Run Coefficient</th><th>Interpretation</th></tr>
                        {lr_rows}
                    </table>
                </div>""", unsafe_allow_html=True)
            except:
                pass

            # ── Bounds Test ──
            if bounds_test:
                try:
                    bt = model.bounds_test(n_obs=len(data), case=3, alpha=0.05)
                    st.markdown('<p class="section-title">PSS Bounds Test (Cointegration)</p>', unsafe_allow_html=True)
                    f_stat_bt = bt.stat
                    lower_b   = bt.crit_vals.get('lower', [None, None, None])
                    upper_b   = bt.crit_vals.get('upper', [None, None, None])
                    conclusion = "Evidence of cointegration (F > I(1) bound)" if bt.conclusion == "cointegration" else "Inconclusive / No cointegration"
                    badge_bt  = "badge-pass" if bt.conclusion == "cointegration" else "badge-warn"
                    st.markdown(f"""
                    <div class="brutalist-card">
                        <p style="font-family:'Space Mono',monospace;">
                           F-statistic: <strong>{fmt(f_stat_bt)}</strong> &nbsp;|&nbsp;
                           <span class="{badge_bt}">{conclusion}</span>
                        </p>
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.info(f"Bounds test could not run automatically: {e}")

            # ── Residuals ──
            _residual_plots(model.resid.values, model.fittedvalues.values, dep_var)

            # ── AI Interpretation ──
            ai_text = _interpret_ardl(model, dep_var, indep_vars, p, denom, ic_crit)
            st.markdown(f"""
            <div class="ai-block">
                <h4>⬡ AI Interpretation — ARDL & Error Correction</h4>
                {ai_text}
            </div>""", unsafe_allow_html=True)

            st.session_state.report_sections["ARDL"] = {
                "model": "ARDL", "dep": dep_var, "indep": indep_vars,
                "order": best_order, "ic": ic_crit, "ai": ai_text
            }

        except Exception as e:
            st.error(f"ARDL estimation failed: {e}")


def _interpret_ardl(model, dep, indep, p, denom, ic):
    ect_approx = 1 - denom  # speed of adjustment proxy
    ect_pct = abs(ect_approx) * 100
    direction = "toward" if denom > 0 else "away from"

    sig_vars = [v for v in indep if any(model.pvalues.get(f'{v}.L{j}', 1) < 0.05 for j in range(0, p+1))]

    return f"""
    <p>The <strong>ARDL</strong> model was estimated using optimal lag selection via <strong>{ic.upper()}</strong>.
    The model includes <strong>{p}</strong> lag(s) of the dependent variable and distributed lags
    of {len(indep)} regressor(s): {', '.join(indep)}.</p>

    <p><strong>Error Correction / Speed of Adjustment:</strong><br>
    The derived adjustment coefficient suggests that approximately <strong>{ect_pct:.1f}%</strong>
    of any short-run deviation from the long-run equilibrium is corrected each period,
    as the system moves <em>{direction}</em> equilibrium. A speed above 50% per period indicates
    rapid mean reversion.</p>

    <p><strong>Significant short-run drivers:</strong> {', '.join(sig_vars) if sig_vars else 'None at 5% level.'}</p>

    <p><strong>Interpretation note:</strong> ARDL coefficients on lagged X terms represent short-run
    dynamics. Divide by (1 − sum of lagged Y coefficients) to recover long-run multipliers.
    The Bounds Test output (above) determines whether a stable long-run relationship exists.</p>

    <p style="font-size:0.8rem;color:#94A3B8;">
    Reference: Pesaran, Shin & Smith (2001) — "Bounds testing approaches to the analysis of level relationships."</p>
    """


# ── VAR ───────────────────────────────────────────────────────────────────────
def _est_var(df, cols):
    st.markdown('<p class="section-title">VAR Configuration</p>', unsafe_allow_html=True)
    endog_vars = st.multiselect("Endogenous Variables (select 2–6)", cols)
    c1, c2 = st.columns(2)
    with c1:
        max_lags = st.slider("Maximum Lags to Test", 1, 12, 4)
    with c2:
        ic = st.selectbox("Information Criterion", ["aic", "bic", "hqic", "fpe"])

    irf_periods = st.slider("IRF Periods", 5, 30, 15)

    if st.button("▶  ESTIMATE VAR", use_container_width=True):
        if len(endog_vars) < 2:
            st.error("Select at least 2 endogenous variables.")
            return
        try:
            data = df[endog_vars].dropna()
            var_model = VAR(data)
            ic_res    = var_model.select_order(max_lags)
            best_lag  = getattr(ic_res, ic)

            fitted = var_model.fit(best_lag)
            st.session_state.var_results = fitted

            # ── LaTeX ──
            k = len(endog_vars)
            st.markdown(f"""
            <div class="eq-block">
                <p style="margin:0;">VAR({best_lag}) System — {k} Endogenous Variables:</p>
                <p style="color:#F59E0B;margin:4px 0 0;">
                   Y(t) = c + A₁·Y(t-1) + A₂·Y(t-2) + ... + A{best_lag}·Y(t-{best_lag}) + ε(t)
                </p>
                <p style="font-size:0.72rem;color:#64748B;margin-top:4px;">
                   where Y(t) = [{', '.join(endog_vars)}]ᵀ | Selected lag: {best_lag} via {ic.upper()}</p>
            </div>""", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Optimal Lag",   str(best_lag))
            c2.metric("AIC",           f"{fitted.aic:.4f}")
            c3.metric("BIC",           f"{fitted.bic:.4f}")

            # ── Summary per equation ──
            st.markdown('<p class="section-title">VAR Equation Summaries</p>', unsafe_allow_html=True)
            for eq_var in endog_vars:
                with st.expander(f"▸ Equation: {eq_var}"):
                    eq_coefs   = fitted.coefs_exog if hasattr(fitted,'coefs_exog') else []
                    params_df  = fitted.params
                    rows = ""
                    for vname in params_df.index:
                        coef = params_df.loc[vname, eq_var]
                        rows += f"<tr><td style='font-family:Space Mono,monospace;'>{vname}</td><td>{fmt(coef)}</td></tr>"
                    st.markdown(f"""
                    <table class="coef-table">
                        <tr><th>Regressor</th><th>Coefficient ({eq_var})</th></tr>{rows}
                    </table>""", unsafe_allow_html=True)

            # ── IRF ──
            st.markdown('<p class="section-title">Impulse Response Functions</p>', unsafe_allow_html=True)
            try:
                irf = fitted.irf(irf_periods)
                impulse_var = st.selectbox("Impulse from:", endog_vars, key="irf_imp")
                response_var = st.selectbox("Response to:", endog_vars, key="irf_resp")
                imp_idx  = endog_vars.index(impulse_var)
                resp_idx = endog_vars.index(response_var)

                irf_vals = irf.irfs[:, resp_idx, imp_idx]
                periods  = list(range(len(irf_vals)))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=periods, y=irf_vals, name="IRF",
                                          line=dict(color=CYAN, width=2.5)))
                fig.add_hline(y=0, line=dict(color=NAVY, dash='dash', width=1))
                fig.add_trace(go.Scatter(
                    x=periods + periods[::-1],
                    y=[v*1.2 for v in irf_vals] + [v*0.8 for v in irf_vals[::-1]],
                    fill='toself', fillcolor='rgba(56,189,248,0.1)',
                    line=dict(color='rgba(0,0,0,0)'), name="Approx. CI"
                ))
                fig = navy_fig(fig, 380)
                fig.update_layout(
                    title=f"IRF: Response of {response_var} to {impulse_var} shock",
                    xaxis_title="Periods", yaxis_title="Response"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"IRF computation failed: {e}")

            ai_text = _interpret_var(fitted, endog_vars, best_lag)
            st.markdown(f"""
            <div class="ai-block">
                <h4>⬡ AI Interpretation — VAR System</h4>
                {ai_text}
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"VAR estimation failed: {e}")


def _interpret_var(fitted, endog, lag):
    return f"""
    <p>The <strong>VAR({lag})</strong> system jointly models {len(endog)} endogenous variables:
    <strong>{', '.join(endog)}</strong>. Each variable is regressed on its own past values
    and the past values of all other variables in the system, capturing dynamic
    interdependencies that single-equation OLS would miss.</p>

    <p><strong>Granger Causality:</strong> To determine which variables "Granger-cause" others,
    examine whether lagged values of variable X significantly predict variable Y (over and above Y's
    own lags). This can be formally tested with the Wald test on equation-specific lag blocks.</p>

    <p><strong>Impulse Response Functions (IRFs):</strong> The IRF plots above trace the response of
    one variable to a one-standard-deviation shock in another, holding all else constant.
    A response that decays to zero over time is consistent with system stability.</p>

    <p><strong>Forecast Error Variance Decomposition (FEVD)</strong> (not shown) would further
    reveal what fraction of each variable's forecast uncertainty is attributable to shocks in the other variables.</p>

    <p style="font-size:0.8rem;color:#94A3B8;">
    Model diagnostics (residual autocorrelation, normality) are critical for valid VAR inference.
    Use the Diagnostic Suite tab for residual-level checks.</p>
    """


# ── VECM ──────────────────────────────────────────────────────────────────────
def _est_vecm(df, cols):
    st.markdown('<p class="section-title">VECM Configuration & Johansen Test</p>', unsafe_allow_html=True)
    endog_vars = st.multiselect("Endogenous Variables (I(1) series required)", cols)
    c1, c2, c3 = st.columns(3)
    with c1:
        k_ar_diff = st.slider("Lags in Differences (k)", 1, 8, 2)
    with c2:
        det_order  = st.selectbox("Deterministics", ["ci (Const in CE)", "li (Trend in CE)", "nc (No Const)"], index=0)
        det_map = {"ci (Const in CE)": -1, "li (Trend in CE)": 0, "nc (No Const)": -1} 
    
    if st.button("▶  RUN JOHANSEN TEST & ESTIMATE VECM", use_container_width=True):
        if len(endog_vars) < 2:
            st.error("Select at least 2 variables.")
            return
        try:
            data = df[endog_vars].dropna()
            
            # ── 1. Johansen Cointegration Test ──
            with st.spinner("Running Johansen Cointegration Test..."):
                j_res = coint_johansen(data, det_order=det_map[det_order], k_ar_diff=k_ar_diff)
                
                trace_stat = j_res.lr1
                max_eig = j_res.lr2
                cv_trace = j_res.cvt[:, 1]  # 5% critical values for trace
                
                # Calculate suggested rank based on Trace statistic at 5% level
                coint_rank = sum(trace_stat > cv_trace)
                
            st.markdown('<p class="section-title">Johansen Cointegration Test (Trace)</p>', unsafe_allow_html=True)
            
            trace_rows = ""
            for i in range(len(trace_stat)):
                sig_cls = "sig" if trace_stat[i] > cv_trace[i] else "insig"
                trace_rows += f"""
                <tr>
                    <td style="font-family:'Space Mono',monospace;">r ≤ {i}</td>
                    <td class="{sig_cls}">{fmt(trace_stat[i])}</td>
                    <td>{fmt(j_res.cvt[i, 0])}</td>
                    <td>{fmt(j_res.cvt[i, 1])}</td>
                    <td>{fmt(j_res.cvt[i, 2])}</td>
                </tr>"""
                
            rank_badge = "badge-pass" if coint_rank > 0 else "badge-fail"
            st.markdown(f"""
            <div class="brutalist-card">
                <table class="coef-table">
                    <tr><th>Null Hypothesis</th><th>Trace Statistic</th><th>Crit 10%</th><th>Crit 5%</th><th>Crit 1%</th></tr>
                    {trace_rows}
                </table>
                <p style="margin-top:10px;font-family:'Space Mono',monospace;font-size:0.85rem;">
                   Suggested Cointegration Rank (r) at 5% level: <span class="{rank_badge}">{coint_rank}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

            if coint_rank == 0:
                st.warning("⚠️ No cointegration detected. A VAR model in first differences is econometrically recommended over a VECM. Proceeding with r=1 for demonstration purposes only.")
                coint_rank = 1

            # ── 2. VECM Estimation ──
            st.markdown('<p class="section-title">VECM Estimation</p>', unsafe_allow_html=True)
            det_vecm_map = {"ci (Const in CE)": "ci", "li (Trend in CE)": "li", "nc (No Const)": "nc"}
            vecm_model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic=det_vecm_map[det_order])
            fitted = vecm_model.fit()

            st.markdown(f"""
            <div class="eq-block">
                <p style="margin:0;">VECM({k_ar_diff}) — Rank r={coint_rank}:</p>
                <p style="color:#F59E0B;margin:4px 0 0;">
                   ΔY(t) = αβᵀY(t-1) + Γ₁ΔY(t-1) + ... + Γ{k_ar_diff}ΔY(t-{k_ar_diff}) + ε(t)
                </p>
            </div>""", unsafe_allow_html=True)

            st.markdown('<p class="section-title">Cointegrating Vector(s) β (Normalized)</p>', unsafe_allow_html=True)
            beta_df = pd.DataFrame(fitted.beta, index=endog_vars[:len(fitted.beta)],
                                    columns=[f"CI Vector {i+1}" for i in range(coint_rank)])
            st.dataframe(beta_df.style.format("{:.6f}"), use_container_width=True)

            st.markdown('<p class="section-title">Adjustment Coefficients α (Loadings)</p>', unsafe_allow_html=True)
            alpha_df = pd.DataFrame(fitted.alpha, index=endog_vars[:len(fitted.alpha)],
                                     columns=[f"CI Vector {i+1}" for i in range(coint_rank)])
            st.dataframe(alpha_df.style.format("{:.6f}"), use_container_width=True)

            ai_text = _interpret_vecm(fitted, endog_vars, coint_rank)
            st.markdown(f"""
            <div class="ai-block">
                <h4>⬡ AI Interpretation — VECM & Cointegration</h4>
                {ai_text}
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"VECM estimation failed: {e}")


def _interpret_vecm(fitted, endog, r):
    alpha_vals = fitted.alpha.flatten()[:r]
    adj_str = ", ".join([f"{v}: α={a:.4f}" for v, a in zip(endog[:r], alpha_vals)])
    return f"""
    <p>The <strong>VECM</strong> imposes cointegration rank <strong>r={r}</strong>, meaning the
    model identifies {r} stable long-run equilibrium relationship(s) among the I(1) variables:
    <strong>{', '.join(endog)}</strong>.</p>

    <p><strong>Cointegrating Vector (β):</strong> The β matrix above defines the long-run
    linear combination(s) that are stationary (I(0)). Normalize by the first variable to interpret
    coefficients as the long-run relationship of each variable relative to the first.</p>

    <p><strong>Adjustment Coefficients (α — Loading Matrix):</strong><br>
    {adj_str}<br>
    These measure how quickly each variable adjusts toward the long-run equilibrium when the
    cointegrating relationship is out of balance. A negative α indicates the variable adjusts
    <em>toward</em> equilibrium (error-correcting behavior).</p>

    <p style="font-size:0.8rem;color:#94A3B8;">
    The VECM is the appropriate specification when variables are I(1) and cointegrated.
    If cointegration rank is uncertain, run the Johansen trace/max-eigenvalue test first.</p>
    """


# ── GARCH ─────────────────────────────────────────────────────────────────────
def _est_garch(df, cols):
    st.markdown('<p class="section-title">GARCH(1,1) Configuration</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        vol_var   = st.selectbox("Return / Volatility Series", cols)
    with c2:
        mean_mdl  = st.selectbox("Mean Model", ["Constant", "AR(1)", "Zero"])
    with c3:
        dist      = st.selectbox("Error Distribution", ["normal", "t", "skewt", "ged"])

    returns_pct = st.checkbox("Multiply series by 100 (for daily returns in %)", value=True)

    if st.button("▶  ESTIMATE GARCH(1,1)", use_container_width=True):
        try:
            series = df[vol_var].dropna()
            if returns_pct:
                series = series * 100

            mean_map = {"Constant": "Constant", "AR(1)": "AR", "Zero": "Zero"}
            am = arch_model(series, mean=mean_map[mean_mdl], lags=1 if mean_mdl=="AR(1)" else 0,
                            vol='Garch', p=1, q=1, dist=dist)
            res = am.fit(disp='off')
            st.session_state.garch_results = res

            st.markdown(f"""
            <div class="eq-block">
                <p style="margin:0;">GARCH(1,1) Specification:</p>
                <p style="color:#F59E0B;margin:4px 0 0;">
                   rₜ = μ + εₜ &nbsp;|&nbsp; εₜ = σₜ·zₜ, &nbsp; zₜ ~ {dist.upper()}
                </p>
                <p style="color:#38BDF8;margin:4px 0 0;">
                   σ²ₜ = ω + α₁·ε²ₜ₋₁ + β₁·σ²ₜ₋₁
                </p>
                <p style="font-size:0.72rem;color:#64748B;margin-top:4px;">
                   Persistence: α₁ + β₁ &lt; 1 required for stationarity</p>
            </div>""", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            p = res.params
            alpha1 = p.get('alpha[1]', p.get('alpha1', 0))
            beta1  = p.get('beta[1]',  p.get('beta1',  0))
            omega  = p.get('omega', 0)
            persist = alpha1 + beta1

            c1.metric("ω (baseline vol.)",  f"{omega:.6f}")
            c2.metric("α₁ (ARCH effect)",   f"{alpha1:.4f}")
            c3.metric("β₁ (GARCH effect)",  f"{beta1:.4f}")
            c4.metric("Persistence α+β",    f"{persist:.4f}",
                      delta="Stationary" if persist < 1 else "Non-stationary",
                      delta_color="normal" if persist < 1 else "inverse")

            # ── Summary table ──
            st.markdown('<p class="section-title">Parameter Estimates</p>', unsafe_allow_html=True)
            rows = ""
            for nm, val, se, tv, pv in zip(
                res.params.index, res.params, res.std_err, res.tvalues, res.pvalues
            ):
                sig_cls = "sig" if pv < 0.05 else "insig"
                rows += f"""<tr>
                    <td style="font-family:'Space Mono',monospace;">{nm}</td>
                    <td class="{sig_cls}">{fmt(val,6)}</td>
                    <td>{fmt(se,6)}</td>
                    <td>{fmt(tv)}</td>
                    <td>{fmt(pv)}</td>
                    <td>{pval_badge(pv)}</td></tr>"""
            st.markdown(f"""
            <div class="brutalist-card">
                <table class="coef-table">
                    <tr><th>Parameter</th><th>Estimate</th><th>Std Error</th>
                        <th>t-Stat</th><th>p-Value</th><th>Sig.</th></tr>
                    {rows}
                </table>
            </div>""", unsafe_allow_html=True)

            # ── Conditional Volatility plot ──
            cond_vol = res.conditional_volatility
            fig = make_subplots(rows=2, cols=1, subplot_titles=[
                f"Returns: {vol_var}", "Conditional Volatility (σₜ)"])
            fig.add_trace(go.Scatter(x=series.index, y=series.values,
                                      name="Returns", line=dict(color=NAVY, width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=series.index, y=cond_vol,
                                      name="Cond. Volatility", line=dict(color=CYAN, width=2),
                                      fill='tozeroy', fillcolor='rgba(56,189,248,0.15)'), row=2, col=1)
            fig = navy_fig(fig, 500)
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            ai_text = _interpret_garch(omega, alpha1, beta1, persist, dist)
            st.markdown(f"""
            <div class="ai-block">
                <h4>⬡ AI Interpretation — Volatility Dynamics</h4>
                {ai_text}
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"GARCH failed: {e}")


def _interpret_garch(omega, alpha1, beta1, persist, dist):
    half_life = -np.log(2) / np.log(persist) if 0 < persist < 1 else float('inf')
    vol_type  = "highly persistent" if persist > 0.95 else "moderately persistent" if persist > 0.85 else "mean-reverting"
    return f"""
    <p>The <strong>GARCH(1,1)</strong> model characterizes volatility clustering in the series.
    Key findings:</p>

    <p>• <strong>ARCH effect (α₁ = {alpha1:.4f}):</strong> This measures how much yesterday's
    unexpected shock (ε²ₜ₋₁) feeds into today's variance. A significant α₁ confirms the presence
    of ARCH effects — volatility reacts to recent shocks.</p>

    <p>• <strong>GARCH effect (β₁ = {beta1:.4f}):</strong> This captures the inertia or "memory"
    in volatility. High β₁ means volatility is slow to decay once it spikes.</p>

    <p>• <strong>Persistence (α₁ + β₁ = {persist:.4f}):</strong> The series exhibits
    <em>{vol_type}</em> volatility dynamics.
    {'The half-life of a volatility shock is approximately ' + f'{half_life:.1f} periods.' if persist < 1 else 'The process is integrated in variance (IGARCH).'}
    </p>

    <p>• <strong>Error Distribution ({dist.upper()}):</strong>
    {"Fat tails are accounted for via the Student-t distribution, which is appropriate for financial return series." if dist == "t" else "Normal errors assumed — consider switching to Student-t if JB normality test fails."}
    </p>

    <p style="font-size:0.8rem;color:#94A3B8;">
    Run Ljung-Box on squared residuals to confirm no remaining ARCH effects post-estimation.</p>
    """


# ── ARIMA ─────────────────────────────────────────────────────────────────────
def _est_arima(df, cols):
    st.markdown('<p class="section-title">ARIMA / SARIMA Configuration</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        target_var = st.selectbox("Target Series", cols)
    with c2:
        mode = st.selectbox("Mode", ["Auto ARIMA (pmdarima)", "Manual Specification"])

    if mode == "Manual Specification":
        c3, c4, c5 = st.columns(3)
        with c3:
            p_val = st.slider("p (AR)",  0, 6, 1)
        with c4:
            d_val = st.slider("d (I)",   0, 2, 1)
        with c5:
            q_val = st.slider("q (MA)",  0, 6, 1)
        seasonal = st.checkbox("Include Seasonal Component (SARIMA)")
        if seasonal:
            c6, c7, c8, c9 = st.columns(4)
            with c6: P_s = st.slider("P (seasonal AR)",  0, 3, 1)
            with c7: D_s = st.slider("D (seasonal I)",   0, 2, 1)
            with c8: Q_s = st.slider("Q (seasonal MA)",  0, 3, 1)
            with c9: m   = st.slider("m (season length)", 2, 52, 12)
        else:
            P_s = D_s = Q_s = m = 0

    fc_periods = st.slider("Forecast Periods", 4, 60, 12)

    if st.button("▶  ESTIMATE ARIMA", use_container_width=True):
        try:
            series = df[target_var].dropna()
            split  = int(len(series) * 0.85)
            train  = series.iloc[:split]
            test   = series.iloc[split:]

            if mode == "Auto ARIMA (pmdarima)":
                with st.spinner("Running auto_arima search (may take a moment)..."):
                    auto_model = pm.auto_arima(
                        train, seasonal=(st.session_state.freq in ["Monthly","Quarterly"]),
                        m=12 if st.session_state.freq == "Monthly" else 4,
                        stepwise=True, suppress_warnings=True,
                        error_action='ignore', information_criterion='aic'
                    )
                p_val, d_val, q_val = auto_model.order
                if hasattr(auto_model, 'seasonal_order'):
                    P_s, D_s, Q_s, m = auto_model.seasonal_order
                else:
                    P_s = D_s = Q_s = 0; m = 0
            else:
                auto_model = None

            # Fit statsmodels ARIMA
            order_s = (P_s, D_s, Q_s, m) if m > 0 else None
            if order_s:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                sm_model = SARIMAX(train, order=(p_val,d_val,q_val),
                                    seasonal_order=(P_s,D_s,Q_s,m)).fit(disp=False)
            else:
                sm_model = ARIMA(train, order=(p_val,d_val,q_val)).fit()

            st.session_state.arima_results = sm_model

            model_name = f"SARIMA({p_val},{d_val},{q_val})×({P_s},{D_s},{Q_s})[{m}]" if m > 0 else f"ARIMA({p_val},{d_val},{q_val})"
            st.markdown(f"""
            <div class="eq-block">
                <p style="margin:0;">Selected Model: <strong>{model_name}</strong></p>
                <p style="color:#F59E0B;margin:4px 0 0;">
                   φ(B)Φ(Bˢ)∇ᵈ∇ˢᴰyₜ = θ(B)Θ(Bˢ)εₜ
                </p>
                <p style="font-size:0.72rem;color:#64748B;margin-top:4px;">
                   B = backshift operator | ∇ = differencing | ε ~ WN(0,σ²)</p>
            </div>""", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("AIC",   f"{sm_model.aic:.3f}")
            c2.metric("BIC",   f"{sm_model.bic:.3f}")
            c3.metric("Train N", str(len(train)))
            c4.metric("Test N",  str(len(test)))

            # ── Forecast ──
            fc = sm_model.forecast(steps=fc_periods + len(test))
            test_fc = fc[:len(test)]
            future_fc = sm_model.forecast(steps=fc_periods)

            if len(test) > 0:
                rmse = np.sqrt(np.mean((test.values - test_fc.values[:len(test)])**2))
                mae  = np.mean(np.abs(test.values - test_fc.values[:len(test)]))
                c5, c6 = st.columns(2)
                c5.metric("Out-of-Sample RMSE", f"{rmse:.4f}")
                c6.metric("Out-of-Sample MAE",  f"{mae:.4f}")

            # ── Forecast chart ──
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train.values, name="Training",
                                      line=dict(color=NAVY, width=2)))
            if len(test) > 0:
                fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual (Test)",
                                          line=dict(color=GRN, width=2, dash='dot')))

            last_idx = series.index[-1]
            try:
                freq_alias = pd.infer_freq(series.index) or 'ME'
                future_idx = pd.date_range(start=last_idx, periods=fc_periods+1, freq=freq_alias)[1:]
            except:
                future_idx = range(len(series), len(series)+fc_periods)

            fig.add_trace(go.Scatter(x=future_idx, y=future_fc.values, name="Forecast",
                                      line=dict(color=CYAN, width=2.5, dash='dash')))
            fig = navy_fig(fig, 420)
            fig.update_layout(title=f"{model_name} Forecast — {target_var}",
                               xaxis_title="Period", yaxis_title=target_var)
            st.plotly_chart(fig, use_container_width=True)

            # ── Residual diagnostics ──
            _residual_plots(sm_model.resid.values, sm_model.fittedvalues.values, target_var)

            ai_text = _interpret_arima(model_name, p_val, d_val, q_val, sm_model, fc_periods,
                                        rmse if len(test)>0 else None)
            st.markdown(f"""
            <div class="ai-block">
                <h4>⬡ AI Interpretation — ARIMA Forecasting</h4>
                {ai_text}
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"ARIMA failed: {e}")


def _interpret_arima(model_name, p, d, q, fitted, fc_periods, rmse):
    d_interp = {0: "already stationary in levels", 1: "required first-differencing (I(1))", 2: "required second-differencing (I(2))"}
    return f"""
    <p>The <strong>{model_name}</strong> model was selected as the optimal specification.
    The series {d_interp.get(d, f'required {d} differences to achieve stationarity')}.</p>

    <p>• <strong>AR({p}) component:</strong> The current value depends on the past {p} observation(s),
    capturing the autocorrelation (persistence) structure in the series.</p>

    <p>• <strong>MA({q}) component:</strong> The current value incorporates the past {q} forecast
    error(s), smoothing out the noise and correcting for moving average dynamics.</p>

    {'<p>• <strong>Out-of-Sample Accuracy:</strong> RMSE = ' + f'{rmse:.4f}' + '. Compare this against a naïve random-walk benchmark to assess genuine predictive value.</p>' if rmse else ''}

    <p>• <strong>Forecast Horizon:</strong> The model projects {fc_periods} periods ahead.
    Uncertainty intervals widen at longer horizons — interpret point forecasts with caution
    beyond 4–6 periods for most macroeconomic series.</p>

    <p style="font-size:0.8rem;color:#94A3B8;">
    Check Ljung-Box Q-test on residuals to confirm white-noise errors. If autocorrelation persists,
    increase p or q. If residuals are non-normal, consider a GED or t distribution.</p>
    """


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ DIAGNOSTIC SUITE TAB
# ══════════════════════════════════════════════════════════════════════════════
def _residual_plots(resid, fitted, label):
    st.markdown('<p class="section-title">Residual Analysis</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fitted, y=resid, mode='markers',
                                  marker=dict(color=CYAN, size=5, opacity=0.7), name="Residuals"))
        fig.add_hline(y=0, line=dict(color=NAVY, dash='dash', width=1.5))
        fig = navy_fig(fig, 300)
        fig.update_layout(title="Residuals vs. Fitted", xaxis_title="Fitted", yaxis_title="Residual")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Q-Q plot
        osm, osr = stats.probplot(resid, dist="norm")[:2]
        qq_x = osm[0]; qq_y = osm[1]
        slope, intercept, r, p, se = stats.linregress(qq_x, qq_y)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=qq_x, y=qq_y, mode='markers',
                                   marker=dict(color=CYAN, size=5, opacity=0.8), name="Sample"))
        fig2.add_trace(go.Scatter(x=[min(qq_x), max(qq_x)],
                                   y=[slope*min(qq_x)+intercept, slope*max(qq_x)+intercept],
                                   line=dict(color=GOLD, width=2, dash='dash'), name="Normal line"))
        fig2 = navy_fig(fig2, 300)
        fig2.update_layout(title="Normal Q-Q Plot", xaxis_title="Theoretical Quantiles",
                            yaxis_title="Sample Quantiles")
        st.plotly_chart(fig2, use_container_width=True)

    # Residual histogram
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=resid, nbinsx=30, name="Residuals",
                                 marker_color=CYAN, opacity=0.75))
    x_range = np.linspace(min(resid), max(resid), 200)
    norm_pdf = stats.norm.pdf(x_range, np.mean(resid), np.std(resid)) * len(resid) * (max(resid)-min(resid))/30
    fig3.add_trace(go.Scatter(x=x_range, y=norm_pdf, name="Normal fit",
                               line=dict(color=GOLD, width=2)))
    fig3 = navy_fig(fig3, 280)
    fig3.update_layout(title="Residual Distribution with Normal Fit", barmode='overlay')
    st.plotly_chart(fig3, use_container_width=True)


def tab_diagnostics():
    st.markdown('<p class="section-title">⬡ Diagnostic Suite — Model Police</p>', unsafe_allow_html=True)

    if st.session_state.clean_df is None:
        st.info("Load and clean data first. Then estimate a model in the Estimation Kernel tab.")
        return

    model_choice = st.selectbox("Select Model to Diagnose", [
        "OLS Results", "ARDL Results", "ARIMA Results", "GARCH Results"
    ])

    model_map = {
        "OLS Results":   "ols_results",
        "ARDL Results":  "ardl_results",
        "ARIMA Results": "arima_results",
        "GARCH Results": "garch_results",
    }
    model_key = model_map[model_choice]
    fitted_model = st.session_state.get(model_key)

    if fitted_model is None:
        st.warning(f"No {model_choice} found. Please estimate the model first in the Estimation Kernel tab.")
        return

    try:
        resid  = fitted_model.resid.values if hasattr(fitted_model.resid, 'values') else np.array(fitted_model.resid)
        fvals  = fitted_model.fittedvalues.values if hasattr(fitted_model.fittedvalues, 'values') else np.zeros(len(resid))
    except:
        st.error("Could not extract residuals from this model.")
        return

    resid = resid[~np.isnan(resid)]
    n     = len(resid)

    st.markdown('<p class="section-title">① Autocorrelation Tests</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        dw = durbin_watson(resid)
        dw_verdict = "No autocorrelation" if 1.5 < dw < 2.5 else ("Positive autocorr." if dw < 1.5 else "Negative autocorr.")
        badge_dw   = "badge-pass" if 1.5 < dw < 2.5 else "badge-fail"
        st.markdown(f"""
        <div class="brutalist-card">
            <p class="section-title" style="font-size:0.82rem;">Durbin-Watson Test</p>
            <p style="font-family:'Space Mono',monospace;font-size:1.6rem;color:#0F172A;margin:4px 0;">
               {dw:.4f}</p>
            <p style="font-size:0.78rem;color:#64748B;">
               Rule: DW ≈ 2.0 → no autocorrelation | DW → 0 → positive | DW → 4 → negative</p>
            <span class="{badge_dw}">{dw_verdict}</span>
        </div>""", unsafe_allow_html=True)

    with c2:
        try:
            if model_choice == "OLS Results" and hasattr(fitted_model, 'model'):
                bg_lm, bg_p, bg_f, bg_fp = acorr_breusch_godfrey(fitted_model, nlags=4)
                bg_verdict = "No autocorrelation (H₀ not rejected)" if bg_p > 0.05 else "Autocorrelation detected (H₀ rejected)"
                badge_bg   = "badge-pass" if bg_p > 0.05 else "badge-fail"
                st.markdown(f"""
                <div class="brutalist-card">
                    <p class="section-title" style="font-size:0.82rem;">Breusch-Godfrey LM Test</p>
                    <table class="coef-table">
                        <tr><th>Statistic</th><th>Value</th></tr>
                        <tr><td>LM Statistic</td><td>{fmt(bg_lm)}</td></tr>
                        <tr><td>p-value</td><td>{fmt(bg_p)}</td></tr>
                        <tr><td>F-statistic</td><td>{fmt(bg_f)}</td></tr>
                        <tr><td>F p-value</td><td>{fmt(bg_fp)}</td></tr>
                    </table>
                    <span class="{badge_bg}">{bg_verdict}</span>
                </div>""", unsafe_allow_html=True)
        except:
            st.info("Breusch-Godfrey requires an OLS model with regressors.")

    st.markdown('<p class="section-title">② Heteroskedasticity Tests</p>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        try:
            if model_choice == "OLS Results":
                X_exog = fitted_model.model.exog
                bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(resid, X_exog)
                bp_verdict = "Homoskedastic (H₀ not rejected)" if bp_p > 0.05 else "Heteroskedasticity detected"
                badge_bp   = "badge-pass" if bp_p > 0.05 else "badge-fail"
                st.markdown(f"""
                <div class="brutalist-card">
                    <p class="section-title" style="font-size:0.82rem;">Breusch-Pagan Test</p>
                    <table class="coef-table">
                        <tr><th>Statistic</th><th>Value</th></tr>
                        <tr><td>LM Statistic</td><td>{fmt(bp_lm)}</td></tr>
                        <tr><td>p-value</td><td>{fmt(bp_p)}</td></tr>
                    </table>
                    <span class="{badge_bp}">{bp_verdict}</span>
                </div>""", unsafe_allow_html=True)
        except:
            st.info("Breusch-Pagan available for OLS models only.")

    with c4:
        try:
            if model_choice == "OLS Results":
                X_exog = fitted_model.model.exog
                wh_lm, wh_p, wh_f, wh_fp = het_white(resid, X_exog)
                wh_verdict = "Homoskedastic (H₀ not rejected)" if wh_p > 0.05 else "Heteroskedasticity detected"
                badge_wh   = "badge-pass" if wh_p > 0.05 else "badge-fail"
                st.markdown(f"""
                <div class="brutalist-card">
                    <p class="section-title" style="font-size:0.82rem;">White's Test</p>
                    <table class="coef-table">
                        <tr><th>Statistic</th><th>Value</th></tr>
                        <tr><td>LM Statistic</td><td>{fmt(wh_lm)}</td></tr>
                        <tr><td>p-value</td><td>{fmt(wh_p)}</td></tr>
                    </table>
                    <span class="{badge_wh}">{wh_verdict}</span>
                </div>""", unsafe_allow_html=True)
        except:
            st.info("White's test available for OLS models only.")
st.markdown('<p class="section-title">⑤ Specification Error — Ramsey RESET</p>', unsafe_allow_html=True)
    try:
        if model_choice == "OLS Results" and hasattr(fitted_model, 'model'):
            reset_res = reset_ramsey(fitted_model, degree=3)
            reset_f = reset_res.stat
            reset_p = reset_res.pvalue
            
            reset_verdict = "No specification error (H₀ not rejected)" if reset_p > 0.05 else "Specification error detected (non-linear terms needed)"
            badge_reset   = "badge-pass" if reset_p > 0.05 else "badge-fail"
            
            st.markdown(f"""
            <div class="brutalist-card">
                <p class="section-title" style="font-size:0.82rem;">Ramsey RESET Test</p>
                <table class="coef-table">
                    <tr><th>Statistic</th><th>Value</th></tr>
                    <tr><td>F-Statistic</td><td>{fmt(reset_f)}</td></tr>
                    <tr><td>p-value</td><td>{fmt(reset_p)}</td></tr>
                </table>
                <p style="margin-top:8px;">H₀: Model has no omitted non-linear variables</p>
                <span class="{badge_reset}">{reset_verdict}</span>
            </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.info("Ramsey RESET test is available for standard OLS models only.")

    st.markdown('<p class="section-title">③ Normality Test</p>', unsafe_allow_html=True)
    jb_stat, jb_p = jarque_bera(resid)
    skew  = stats.skew(resid)
    kurt  = stats.kurtosis(resid)
    jb_verdict = "Normally distributed (H₀ not rejected)" if jb_p > 0.05 else "Non-normal residuals"
    badge_jb   = "badge-pass" if jb_p > 0.05 else "badge-warn"
    c5, c6 = st.columns(2)
    with c5:
        st.markdown(f"""
        <div class="brutalist-card">
            <p class="section-title" style="font-size:0.82rem;">Jarque-Bera Normality Test</p>
            <table class="coef-table">
                <tr><th>Statistic</th><th>Value</th></tr>
                <tr><td>JB Statistic</td><td>{fmt(jb_stat)}</td></tr>
                <tr><td>p-value</td><td>{fmt(jb_p)}</td></tr>
                <tr><td>Skewness</td><td>{fmt(skew)}</td></tr>
                <tr><td>Excess Kurtosis</td><td>{fmt(kurt)}</td></tr>
            </table>
            <p style="margin-top:8px;">H₀: residuals are normally distributed</p>
            <span class="{badge_jb}">{jb_verdict}</span>
        </div>""", unsafe_allow_html=True)

with c6:
        # Histogram with normal overlay
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=resid, nbinsx=25, name="Residuals",
                                    marker_color=CYAN, opacity=0.7, histnorm='probability density'))
        x_rng = np.linspace(min(resid), max(resid), 200)
        fig.add_trace(go.Scatter(x=x_rng, y=stats.norm.pdf(x_rng, np.mean(resid), np.std(resid)),
                                  name="Normal PDF", line=dict(color=GOLD, width=2.5)))
        fig = navy_fig(fig, 300)
        fig.update_layout(title="Residual Histogram vs. Normal", barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)

st.markdown('<p class="section-title">④ Structural Stability — CUSUM</p>', unsafe_allow_html=True)
try:
        cusum = np.cumsum(resid)
        sigma = np.std(resid)
        n_obs = len(resid)
        conf_band = 0.948 * sigma * np.sqrt(n_obs) * np.array([
            2 * i / n_obs - 1 for i in range(1, n_obs + 1)
        ])
        t_axis = list(range(1, n_obs + 1))

        fig4 = make_subplots(rows=1, cols=2, subplot_titles=["CUSUM", "CUSUM of Squares"])
        fig4.add_trace(go.Scatter(x=t_axis, y=cusum, name="CUSUM",
                                   line=dict(color=CYAN, width=2)), row=1, col=1)
        fig4.add_trace(go.Scatter(x=t_axis, y=conf_band, name="5% Upper",
                                   line=dict(color=RED, dash='dash', width=1)), row=1, col=1)
        fig4.add_trace(go.Scatter(x=t_axis, y=-conf_band, name="5% Lower",
                                   line=dict(color=RED, dash='dash', width=1)), row=1, col=1)

        cusum_sq = np.cumsum(resid**2) / np.sum(resid**2)
        expected = np.linspace(0, 1, n_obs)
        fig4.add_trace(go.Scatter(x=t_axis, y=cusum_sq, name="CUSUM²",
                                   line=dict(color=TEAL, width=2)), row=1, col=2)
        fig4.add_trace(go.Scatter(x=t_axis, y=expected, name="Expected",
                                   line=dict(color=GOLD, dash='dash', width=1)), row=1, col=2)

        fig4 = navy_fig(fig4, 380)
        fig4.update_layout(showlegend=True)
        st.plotly_chart(fig4, use_container_width=True)

        # Check for breaches
        breaches = sum([1 for c, b in zip(cusum, conf_band) if abs(c) > abs(b)])
        stability = "✅ Structurally stable" if breaches < 5 else "⚠️ Potential structural break detected"
        badge_s   = "badge-pass" if breaches < 5 else "badge-warn"
        st.markdown(f'<span class="{badge_s}">{stability}</span>', unsafe_allow_html=True)
except Exception as e:
        st.warning(f"CUSUM computation error: {e}")

    # ── AI Diagnostic Summary ──
try:
        diag_summary = _interpret_diagnostics(
            dw,
            jb_p, skew, kurt,
            bp_p if 'bp_p' in dir() else None,
            wh_p if 'wh_p' in dir() else None,
            bg_p if 'bg_p' in dir() else None,
            breaches if 'breaches' in dir() else 0
        )
        st.markdown(f"""
        <div class="ai-block">
            <h4>⬡ AI Diagnostic Summary — Model Report Card</h4>
            {diag_summary}
        </div>""", unsafe_allow_html=True)
except:
        pass


def _interpret_diagnostics(dw, jb_p, skew, kurt, bp_p, wh_p, bg_p, cusum_breaches):
    issues = []
    passes = []

    if 1.5 < dw < 2.5:
        passes.append("DW test shows no first-order autocorrelation")
    else:
        issues.append(f"DW = {dw:.3f} indicates {'positive' if dw<2 else 'negative'} serial correlation — consider Newey-West SEs or AR errors")

    if jb_p > 0.05:
        passes.append("Residuals are normally distributed (JB test passed)")
    else:
        issues.append(f"Non-normal residuals (JB p={jb_p:.4f}, skew={skew:.3f}, kurtosis={kurt:.3f}) — OLS still BLUE by CLM, but inference may be unreliable in small samples")

    if bp_p is not None:
        if bp_p > 0.05:
            passes.append("Breusch-Pagan: homoskedastic errors confirmed")
        else:
            issues.append(f"Heteroskedasticity detected (BP p={bp_p:.4f}) — use HC-robust standard errors")

    if cusum_breaches < 5:
        passes.append("CUSUM test: no evidence of structural breaks")
    else:
        issues.append("CUSUM test flags potential structural instability — consider dummy variables or Chow test")

    pass_html  = "".join([f"<li>✅ {p}</li>" for p in passes])
    issue_html = "".join([f"<li>⚠️ {i}</li>" for i in issues])

    grade = "A" if len(issues) == 0 else "B" if len(issues) == 1 else "C" if len(issues) == 2 else "D"
    grade_col = {"A": "#10B981", "B": "#F59E0B", "C": "#EF4444", "D": "#7F1D1D"}.get(grade, "#94A3B8")

    return f"""
    <p style="font-family:'Orbitron',monospace;font-size:1.2rem;color:{grade_col};">
       MODEL GRADE: {grade}</p>

    <p><strong>Diagnostic Passes:</strong></p>
    <ul>{pass_html if pass_html else '<li>None</li>'}</ul>

    <p><strong>Issues Identified:</strong></p>
    <ul>{issue_html if issue_html else '<li>No issues — model passes all diagnostic tests ✅</li>'}</ul>

    <p style="font-size:0.82rem;color:#94A3B8;margin-top:8px;">
    Even if all tests pass, economic theory and variable selection remain paramount.
    Diagnostics confirm the statistical properties of the model, not its economic validity.</p>
    """


# ══════════════════════════════════════════════════════════════════════════════
# ⑥ REPORTING & EXPORT TAB
# ══════════════════════════════════════════════════════════════════════════════
def tab_report():
    st.markdown('<p class="section-title">⬡ Reporting & PDF Export</p>', unsafe_allow_html=True)

    user   = st.session_state.user_name
    email  = st.session_state.user_email
    occ    = st.session_state.user_occ
    freq   = st.session_state.freq or "N/A"
    sects  = st.session_state.report_sections

    if st.session_state.clean_df is None:
        st.info("No data or model results to export yet. Load data and run at least one model.")
        return

    df     = st.session_state.clean_df
    cols   = df.select_dtypes(include=np.number).columns.tolist()

    st.markdown('<p class="section-title">Report Configuration</p>', unsafe_allow_html=True)
    report_title = st.text_input("Report Title", value="Macroeconomic Time-Series Analysis Report")
    include_desc = st.checkbox("Include Descriptive Statistics", value=True)
    include_models = st.multiselect("Include Model Results", list(sects.keys()) if sects else ["None"], default=list(sects.keys()) if sects else [])

    c1, c2 = st.columns(2)
    with c1:
        author_note = st.text_area("Author Notes / Abstract", height=100,
                                    value="This report was generated using NEXUS KERNEL — a professional econometrics platform.")
    with c2:
        institution = st.text_input("Institution / Organization", value="")

    if st.button("▶  GENERATE PDF REPORT", use_container_width=True):
        with st.spinner("Building PDF..."):
            pdf_bytes = _build_pdf(
                title=report_title,
                user=user, email=email, occ=occ,
                institution=institution,
                author_note=author_note,
                df=df, freq=freq,
                sects=sects,
                include_desc=include_desc,
                include_models=include_models
            )
        st.success("✓ PDF report generated successfully!")
        st.download_button(
            label="⬇  DOWNLOAD PDF REPORT",
            data=pdf_bytes,
            file_name=f"NEXUS_KERNEL_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )


def _build_pdf(title, user, email, occ, institution, author_note, df, freq, sects,
               include_desc, include_models):
    buf = io.BytesIO()

    NAVY_RL  = colors.HexColor("#0F172A")
    CYAN_RL  = colors.HexColor("#38BDF8")
    GOLD_RL  = colors.HexColor("#F59E0B")
    LIGHT    = colors.HexColor("#E0E7FF")
    WHITE    = colors.white
    GRAY     = colors.HexColor("#64748B")
    GREEN_RL = colors.HexColor("#10B981")
    RED_RL   = colors.HexColor("#EF4444")

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('NKTitle', parent=styles['Title'],
                                  fontSize=22, fontName='Helvetica-Bold',
                                  textColor=NAVY_RL, spaceAfter=4, alignment=TA_CENTER)
    sub_style   = ParagraphStyle('NKSub', parent=styles['Normal'],
                                  fontSize=10, fontName='Helvetica',
                                  textColor=CYAN_RL, spaceAfter=2, alignment=TA_CENTER)
    h1_style    = ParagraphStyle('NKH1', parent=styles['Heading1'],
                                  fontSize=13, fontName='Helvetica-Bold',
                                  textColor=NAVY_RL, spaceBefore=14, spaceAfter=6,
                                  borderPad=4)
    h2_style    = ParagraphStyle('NKH2', parent=styles['Heading2'],
                                  fontSize=11, fontName='Helvetica-Bold',
                                  textColor=CYAN_RL, spaceBefore=10, spaceAfter=4)
    body_style  = ParagraphStyle('NKBody', parent=styles['Normal'],
                                  fontSize=9, fontName='Helvetica',
                                  textColor=NAVY_RL, spaceAfter=6,
                                  leading=14, alignment=TA_JUSTIFY)
    mono_style  = ParagraphStyle('NKMono', parent=styles['Normal'],
                                  fontSize=8, fontName='Courier',
                                  textColor=NAVY_RL, spaceAfter=4, leading=12)
    footer_s    = ParagraphStyle('NKFoot', parent=styles['Normal'],
                                  fontSize=7, fontName='Helvetica',
                                  textColor=GRAY, alignment=TA_CENTER)

    story = []

    # ── Cover ──
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("⬡ NEXUS KERNEL", ParagraphStyle('Brand',
        parent=styles['Normal'], fontSize=28, fontName='Helvetica-Bold',
        textColor=CYAN_RL, alignment=TA_CENTER, spaceAfter=4)))
    story.append(Paragraph("Professional Time-Series Econometrics Platform",
                             ParagraphStyle('BrandSub', parent=styles['Normal'],
                                            fontSize=11, textColor=GRAY, alignment=TA_CENTER,
                                            spaceAfter=4)))
    story.append(HRFlowable(width="100%", thickness=3, color=CYAN_RL, spaceAfter=12))

    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.3*cm))

    meta_data = [
        ["Prepared By:", user],
        ["Email:", email],
        ["Occupation:", occ],
        ["Institution:", institution or "—"],
        ["Data Frequency:", freq],
        ["Generated:", datetime.now().strftime("%B %d, %Y — %H:%M")],
    ]
    meta_table = Table(meta_data, colWidths=[4*cm, 12*cm])
    meta_table.setStyle(TableStyle([
        ('FONTNAME',  (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE',  (0,0), (-1,-1), 9),
        ('FONTNAME',  (0,0), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,0), (0,-1), NAVY_RL),
        ('TEXTCOLOR', (1,0), (1,-1), GRAY),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [WHITE, LIGHT]),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=GRAY, spaceAfter=10))

    story.append(Paragraph("Author Abstract", h1_style))
    story.append(Paragraph(author_note, body_style))
    story.append(PageBreak())

    # ── Descriptive Stats ──
    if include_desc:
        story.append(Paragraph("1. Dataset Overview & Descriptive Statistics", h1_style))
        desc = df.describe().T.round(4)
        story.append(Paragraph(f"Dataset contains <b>{len(df)}</b> observations and "
                                f"<b>{len(df.columns)}</b> variables at <b>{freq}</b> frequency.",
                                body_style))

        table_data = [["Variable"] + list(desc.columns)]
        for var in desc.index:
            row = [var] + [str(round(v, 4)) for v in desc.loc[var]]
            table_data.append(row)

        col_widths = [4*cm] + [(15*cm / len(desc.columns))] * len(desc.columns)
        t = Table(table_data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0), NAVY_RL),
            ('TEXTCOLOR',     (0,0), (-1,0), CYAN_RL),
            ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0), (-1,-1), 7.5),
            ('FONTNAME',      (0,1), (-1,-1), 'Helvetica'),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHT]),
            ('GRID',          (0,0), (-1,-1), 0.3, GRAY),
            ('TOPPADDING',    (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('LEFTPADDING',   (0,0), (-1,-1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4*cm))

    # ── Model sections ──
    for i, sec_key in enumerate(include_models, 2):
        if sec_key not in sects:
            continue
        sec = sects[sec_key]
        story.append(PageBreak())
        story.append(Paragraph(f"{i}. {sec_key} Estimation Results", h1_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=CYAN_RL, spaceAfter=8))

        if "dep" in sec:
            story.append(Paragraph(f"<b>Dependent Variable:</b> {sec['dep']}", body_style))
        if "indep" in sec:
            story.append(Paragraph(f"<b>Regressors:</b> {', '.join(sec['indep'])}", body_style))
        if "r2" in sec:
            story.append(Paragraph(
                f"<b>R-squared:</b> {sec['r2']:.4f} &nbsp;&nbsp; "
                f"<b>Adj. R²:</b> {sec['adj_r2']:.4f} &nbsp;&nbsp; "
                f"<b>F-stat:</b> {sec['fstat']:.3f} (p={sec['fp']:.4f}) &nbsp;&nbsp; "
                f"<b>N:</b> {int(sec['nobs'])}",
                body_style))

        if "params" in sec and "pvals" in sec:
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph("Coefficient Table:", h2_style))
            coef_data = [["Variable", "Coefficient", "p-Value", "Significance"]]
            for vname, coef in sec["params"].items():
                p = sec["pvals"].get(vname, 1)
                sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
                coef_data.append([vname, f"{coef:.6f}", f"{p:.4f}", sig])
            ct = Table(coef_data, colWidths=[6*cm, 4*cm, 3.5*cm, 2.5*cm])
            ct.setStyle(TableStyle([
                ('BACKGROUND',    (0,0), (-1,0), NAVY_RL),
                ('TEXTCOLOR',     (0,0), (-1,0), CYAN_RL),
                ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTNAME',      (0,1), (-1,-1), 'Courier'),
                ('FONTSIZE',      (0,0), (-1,-1), 8),
                ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHT]),
                ('GRID',          (0,0), (-1,-1), 0.3, GRAY),
                ('TOPPADDING',    (0,0), (-1,-1), 3),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ]))
            story.append(ct)

        if "ai" in sec and sec["ai"]:
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph("AI Interpretation:", h2_style))
            # Strip HTML tags for PDF
            import re
            clean_ai = re.sub(r'<[^>]+>', '', sec["ai"])
            clean_ai = clean_ai.replace("&nbsp;", " ").strip()
            for para in clean_ai.split("\n"):
                if para.strip():
                    story.append(Paragraph(para.strip(), body_style))

    # ── Footer page ──
    story.append(PageBreak())
    story.append(Paragraph("Disclaimer & Methodology Notes", h1_style))
    story.append(Paragraph(
        "This report was generated by NEXUS KERNEL, a professional-grade time-series econometrics "
        "platform built by Ahmed Hisham. All statistical tests and models conform to established "
        "econometric standards. Results should be interpreted in the context of economic theory and "
        "data quality. NEXUS KERNEL is not responsible for decisions made on the basis of this output.",
        body_style))
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=GRAY, spaceAfter=8))
    story.append(Paragraph(
        f"NEXUS KERNEL © {datetime.now().year} · Research by Ahmed Hisham · "
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        footer_s))

    doc.build(story)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# ⑦ SUMMARY STATISTICS TAB
# ══════════════════════════════════════════════════════════════════════════════
def tab_summary():
    st.markdown('<p class="section-title">⬡ Summary Statistics & Visualization Studio</p>', unsafe_allow_html=True)

    if st.session_state.clean_df is None:
        st.info("Load data first.")
        return

    df   = st.session_state.clean_df
    cols = df.select_dtypes(include=np.number).columns.tolist()

    tab_a, tab_b, tab_c = st.tabs(["📊 Descriptive Stats", "📈 Time-Series Plots", "🔗 Correlation Matrix"])

    with tab_a:
        desc = df[cols].describe().T.round(4)
        desc['skewness'] = df[cols].skew().round(4)
        desc['kurtosis'] = df[cols].kurt().round(4)
        desc['missing']  = df[cols].isnull().sum()
        st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)

        # Distribution plots
        sel = st.selectbox("Inspect Distribution of:", cols)
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Distribution", "Box Plot"])
        fig.add_trace(go.Histogram(x=df[sel].dropna(), name=sel,
                                    marker_color=CYAN, opacity=0.75), row=1, col=1)
        fig.add_trace(go.Box(y=df[sel].dropna(), name=sel,
                              marker_color=CYAN, boxpoints='outliers'), row=1, col=2)
        fig = navy_fig(fig, 350)
        st.plotly_chart(fig, use_container_width=True)

    with tab_b:
        sel_vars = st.multiselect("Select variables to plot", cols, default=cols[:3])
        if sel_vars:
            normalize = st.checkbox("Normalize to index (=100 at start)")
            fig = go.Figure()
            color_cycle = [CYAN, GOLD, TEAL, RED, GRN, "#A78BFA", "#F472B6"]
            for i, v in enumerate(sel_vars):
                y = df[v].dropna()
                if normalize:
                    y = y / y.iloc[0] * 100
                fig.add_trace(go.Scatter(x=y.index, y=y.values, name=v,
                                          line=dict(color=color_cycle[i % 7], width=2)))
            fig = navy_fig(fig, 450)
            fig.update_layout(title="Time-Series Plot",
                               xaxis_title="Period", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)

    with tab_c:
        if len(cols) < 2:
            st.info("Need at least 2 numeric variables for correlation analysis.")
        else:
            corr_method = st.selectbox("Method", ["pearson", "spearman", "kendall"])
            corr = df[cols].corr(method=corr_method).round(4)

            fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale=[[0, NAVY], [0.5, "#F1F5F9"], [1, CYAN]],
                text=corr.values.round(2), texttemplate="%{text}",
                textfont=dict(size=9), zmin=-1, zmax=1,
                hoverongaps=False
            ))
            fig = navy_fig(fig, 500)
            fig.update_layout(title=f"{corr_method.title()} Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

            # Scatter matrix
            if st.checkbox("Show Scatter Matrix (first 5 variables)"):
                sub_cols = cols[:5]
                fig2 = go.Figure(go.Splom(
                    dimensions=[dict(label=c, values=df[c]) for c in sub_cols],
                    marker=dict(color=CYAN, size=4, opacity=0.6),
                    showupperhalf=False
                ))
                fig2 = navy_fig(fig2, 600)
                fig2.update_layout(title="Scatter Plot Matrix")
                st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if not st.session_state.authenticated:
        login_page()
        return

    render_sidebar()

    st.markdown("""
    <div style="background:linear-gradient(90deg,#0F172A 0%,#1E3A5F 100%);
         padding:16px 24px;margin-bottom:20px;border-bottom:3px solid #38BDF8;">
        <p style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:900;
           color:#38BDF8;letter-spacing:0.18em;margin:0;display:inline;">⬡ NEXUS KERNEL</p>
        <p style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#475569;
           letter-spacing:0.2em;margin:0;display:inline;margin-left:20px;">
           PROFESSIONAL TIME-SERIES ECONOMETRICS PLATFORM</p>
        <span style="float:right;font-family:'Space Mono',monospace;font-size:0.7rem;
              color:#64748B;">Research by Ahmed Hisham</span>
    </div>
    """, unsafe_allow_html=True)

    tab_labels = [
        "📂 Data",
        "📉 Stationarity",
        "⚙ Estimation",
        "🔬 Diagnostics",
        "📊 Statistics",
        "📄 Report",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        try:
            tab_data()
        except Exception as e:
            st.error(f"Data module error: {e}")

    with tabs[1]:
        try:
            tab_stationarity()
        except Exception as e:
            st.error(f"Stationarity module error: {e}")

    with tabs[2]:
        try:
            tab_estimation()
        except Exception as e:
            st.error(f"Estimation module error: {e}")

    with tabs[3]:
        try:
            tab_diagnostics()
        except Exception as e:
            st.error(f"Diagnostics module error: {e}")

    with tabs[4]:
        try:
            tab_summary()
        except Exception as e:
            st.error(f"Statistics module error: {e}")

    with tabs[5]:
        try:
            tab_report()
        except Exception as e:
            st.error(f"Report module error: {e}")


if __name__ == "__main__":
    main()
