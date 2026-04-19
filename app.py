"""
NEXUS KERNEL v3.0 — Professional Time-Series Econometrics Platform
Research by Ahmed Hisham
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="NEXUS KERNEL v3.0",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Lazy Imports for later modules ───────────────────────────────────────────
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import statsmodels.api as sm

# ══════════════════════════════════════════════════════════════════════════════
# 1. GLOBAL CSS — 2026 Midnight Navy & Steel Cyan Desktop Architecture
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&family=Orbitron:wght@700;900&display=swap');

:root {
    --navy:   #0F172A;
    --navy-light: #1E293B;
    --cyan:   #38BDF8;
    --gold:   #F59E0B;
    --canvas: #F1F5F9; /* Desktop gray canvas */
    --card:   #FFFFFF;
    --border: #CBD5E1;
    --muted:  #64748B;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--canvas) !important;
    color: var(--navy) !important;
}

/* ── Top App Bar (Mimicking Desktop Software) ── */
.top-menu-bar {
    background: var(--navy);
    border-bottom: 3px solid var(--cyan);
    padding: 12px 24px;
    margin: -3rem -3rem 2rem -3rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.brand-text {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 900;
    color: var(--cyan);
    letter-spacing: 0.15em;
    margin: 0;
}
.author-text {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Sidebar Styling ── */
section[data-testid="stSidebar"] {
    background: var(--navy-light) !important;
    border-right: 1px solid var(--navy) !important;
}
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }

/* ── Brutalist Panel Cards ── */
.panel-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 4px solid var(--navy);
    box-shadow: 2px 2px 10px rgba(15,23,42,0.05);
    padding: 24px;
    margin-bottom: 24px;
}

.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.0rem;
    font-weight: 700;
    color: var(--navy);
    letter-spacing: 0.1em;
    border-bottom: 2px solid var(--cyan);
    padding-bottom: 6px;
    margin-bottom: 16px;
    text-transform: uppercase;
}

/* ── Chatbot UI overrides ── */
.stChatMessage {
    background-color: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "active_menu": "Data Management",
        "raw_df": None,
        "clean_df": None,
        "freq": None,
        "chat_history": [{"role": "assistant", "content": "I am your AI Econometrician. How can I help you analyze your series today?"}],
        # Model storage
        "ols_results": None,
        "ardl_results": None,
        "var_results": None,
        "vecm_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════════════════════
# 3. AI ECONOMETRICIAN CHATBOT (Sidebar)
# ══════════════════════════════════════════════════════════════════════════════
def render_ai_copilot():
    st.sidebar.markdown("""
    <div style="margin-top:20px;border-bottom:2px solid #38BDF8;padding-bottom:5px;margin-bottom:15px;">
        <p style="font-family:'Orbitron',monospace;font-size:0.9rem;font-weight:700;color:#38BDF8;margin:0;">
        ⬡ AI ECONOMETRICIAN</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat history container
    chat_container = st.sidebar.container(height=400)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Chat Input
    if prompt := st.sidebar.chat_input("Ask about models, tests, or results..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Simple echo/placeholder logic for Part 1. (Will be wired to an LLM/logic tree later)
        response = f"I see you are asking about '{prompt}'. Ensure your data is stationary before proceeding to standard VAR models. If it is I(1), consider the VECM module."
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 4. DESKTOP ROUTER & NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
def render_top_bar():
    st.markdown("""
    <div class="top-menu-bar">
        <div>
            <p class="brand-text">⬡ NEXUS KERNEL v3.0</p>
        </div>
        <div style="text-align:right;">
            <p class="author-text">Research by Ahmed Hisham</p>
            <p class="author-text" style="color:#38BDF8;">Enterprise Edition</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_navigation():
    st.sidebar.markdown('<p style="font-family:\'Space Mono\',monospace;color:#94A3B8;font-size:0.75rem;margin-bottom:5px;">WORKSPACE MENU</p>', unsafe_allow_html=True)
    
    menus = [
        "Data Management", 
        "Stationarity Vault", 
        "Estimation Kernel", 
        "Diagnostic Police", 
        "Forecasting Engine", 
        "Reporting"
    ]
    
    choice = st.sidebar.radio("Navigate", menus, label_visibility="collapsed")
    st.session_state.active_menu = choice
    
    if st.session_state.clean_df is not None:
        st.sidebar.success(f"✅ Data Loaded: {len(st.session_state.clean_df)} obs")


# ══════════════════════════════════════════════════════════════════════════════
# 5. DATA INGESTION MODULE
# ══════════════════════════════════════════════════════════════════════════════
def module_data():
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Data Ingestion & Engineering</p>', unsafe_allow_html=True)
    
    uploaded = st.file_uploader("Import Workspace Data (CSV/XLSX)", type=["csv", "xlsx"])
    
    if uploaded:
        try:
            raw = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
            st.session_state.raw_df = raw.copy()
            st.success(f"File loaded: {uploaded.name}")
        except Exception as e:
            st.error(f"Read error: {e}")
            
    if st.session_state.raw_df is not None:
        raw = st.session_state.raw_df
        c1, c2, c3 = st.columns(3)
        date_col = c1.selectbox("Time Index Column", raw.columns)
        freq_choice = c2.selectbox("Frequency", ["Auto", "Annual", "Quarterly", "Monthly", "Daily"])
        miss_method = c3.selectbox("Missing Data Handling", ["Interpolate", "Drop", "Forward Fill"])
        
        if st.button("▶ Initialize & Clean Workspace", use_container_width=True):
            with st.spinner("Engineering data..."):
                df = raw.copy()
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
                
                # Convert to numeric, strip symbols
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
                
                # Missing
                if miss_method == "Interpolate": df = df.interpolate()
                elif miss_method == "Drop": df = df.dropna()
                elif miss_method == "Forward Fill": df = df.ffill()
                
                st.session_state.clean_df = df
                st.session_state.freq = freq_choice
                st.rerun()

    if st.session_state.clean_df is not None:
        st.markdown('<p class="section-title" style="margin-top:20px;">Active Workspace Data</p>', unsafe_allow_html=True)
        st.dataframe(st.session_state.clean_df.head(), use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
def main():
    render_top_bar()
    render_navigation()
    render_ai_copilot()
    
    # Router
    if st.session_state.active_menu == "Data Management":
        module_data()
    else:
        st.markdown(f"""
        <div class="panel-card" style="text-align:center; padding: 60px;">
            <p style="font-family:'Orbitron', monospace; color:#38BDF8; font-size:1.5rem;">{st.session_state.active_menu}</p>
            <p style="color:#64748B;">This module is awaiting integration. Please load your data first.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
