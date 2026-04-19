# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║           NEXUS KERNEL v3.0 — Professional Time-Series Econometrics         ║
# ║                        Research by Ahmed Hisham                              ║
# ║                 © 2026 — Production Release. All Rights Reserved.            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import io
import re
import traceback
from datetime import datetime

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="NEXUS KERNEL v3.0",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None,
                "About": "NEXUS KERNEL v3.0 — Research by Ahmed Hisham."},
)

import scipy.stats as sci_stats
from scipy.stats import jarque_bera, norm

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, grangercausalitytests, zivot_andrews
from statsmodels.tsa.ardl import ARDL, ardl_select_order
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank, select_order
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.filters.bk_filter import bkfilter
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey, acorr_ljungbox,
    het_white, het_breuschpagan, het_arch, linear_reset,
)
from statsmodels.stats.stattools import durbin_watson

from arch import arch_model
import pmdarima as pm

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE & HELPERS
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "navy": "#0F172A", "navy2": "#1E293B", "navy3": "#0A1020",
    "cyan": "#38BDF8", "cyan2": "#0EA5E9",
    "gold": "#F59E0B", "gold2": "#D97706",
    "teal": "#14B8A6", "green": "#10B981", "red": "#EF4444",
    "orange": "#F97316", "purple": "#A78BFA", "pink": "#F472B6",
    "slate": "#64748B", "canvas": "#E8EEFF", "card": "#F0F4FF",
    "border": "#C7D2FE", "muted": "#94A3B8",
}
CHART_COLORS = [C["cyan"], C["gold"], C["teal"], C["red"],
                C["green"], C["purple"], C["orange"], C["pink"]]

def navy_fig(fig, height=420, title=None):
    up = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F8FAFF",
        font=dict(family="DM Sans, sans-serif", color=C["navy"], size=12),
        margin=dict(l=50, r=24, t=54 if title else 36, b=44),
        legend=dict(bgcolor="rgba(248,250,255,0.92)", bordercolor=C["border"],
                    borderwidth=1, font=dict(size=11)),
        xaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", zerolinecolor="#CBD5E1",
                   tickfont=dict(family="Space Mono, monospace", size=10)),
        yaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", zerolinecolor="#CBD5E1",
                   tickfont=dict(family="Space Mono, monospace", size=10)),
    )
    if title:
        up["title"] = dict(text=title, font=dict(family="DM Sans", size=14, color=C["navy"]),
                           x=0.0, xanchor="left", pad=dict(l=4))
    fig.update_layout(**up)
    return fig

def fmt(x, d=4):
    try:
        v = float(x)
        if abs(v) >= 1e6 or (abs(v) < 1e-4 and v != 0):
            return f"{v:.4e}"
        return f"{v:.{d}f}"
    except Exception:
        return str(x)

def pstar(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

def badge_html(label, kind="pass"):
    styles = {
        "pass": "background:#D1FAE5;color:#065F46;border:1.5px solid #6EE7B7;",
        "fail": "background:#FEE2E2;color:#991B1B;border:1.5px solid #FCA5A5;",
        "warn": "background:#FEF3C7;color:#92400E;border:1.5px solid #FCD34D;",
        "info": "background:#DBEAFE;color:#1E40AF;border:1.5px solid #93C5FD;",
        "cyan": f"background:{C['navy']};color:{C['cyan']};border:1.5px solid {C['cyan']};",
    }
    s = styles.get(kind, styles["info"])
    return (f'<span style="{s}padding:2px 10px;border-radius:3px;font-size:0.72rem;'
            f'font-weight:700;font-family:Space Mono,monospace;text-transform:uppercase;">'
            f'{label}</span>')

def coef_row_html(var_name, coef, se, tstat, pval, ci_lo=None, ci_hi=None):
    stars = pstar(pval)
    if pval < 0.01:   cls = "sig3"
    elif pval < 0.05: cls = "sig2"
    elif pval < 0.10: cls = "sig1"
    else:             cls = "insig"
    ci_str = f"{fmt(ci_lo)} / {fmt(ci_hi)}" if ci_lo is not None else "—"
    return (f"<tr><td class='{cls}'>{var_name} <span class='sig-star'>{stars}</span></td>"
            f"<td class='{cls}'>{fmt(coef)}</td><td>{fmt(se)}</td>"
            f"<td>{fmt(tstat)}</td><td>{fmt(pval)}</td><td>{ci_str}</td></tr>")

def coef_table_html(rows_html, extra_cols=""):
    return f"""
    <div class="coef-wrap">
    <table class="coef-tbl">
    <tr><th>Variable</th><th>Coef.</th><th>Std. Err.</th>
        <th>t-Stat</th><th>p-Value</th><th>95% CI {extra_cols}</th></tr>
    {rows_html}
    </table>
    <p style="font-size:0.67rem;color:#94A3B8;margin-top:4px;font-family:Space Mono,monospace;">
    *** p&lt;0.01 &nbsp; ** p&lt;0.05 &nbsp; * p&lt;0.10</p>
    </div>"""

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "authenticated": False, "user_name": "", "user_email": "", "user_occ": "",
    "raw_df": None, "clean_df": None, "freq": None, "date_col": None,
    "menu": "home", "submenu": None,
    "ols_res": None, "ardl_res": None, "var_res": None,
    "vecm_res": None, "garch_res": None, "arima_res": None,
    "johansen_res": None,
    "diag_model": None, "diag_resid": None, "diag_fitted": None, "diag_X": None,
    "chat_history": [],
    "report_log": {},
    "stat_results": {},
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=Orbitron:wght@600;700;800;900&display=swap');
:root{--navy:#0F172A;--navy2:#1E293B;--navy3:#0A1020;--cyan:#38BDF8;--cyan2:#0EA5E9;
  --gold:#F59E0B;--teal:#14B8A6;--green:#10B981;--red:#EF4444;--canvas:#E8EEFF;
  --card:#F0F4FF;--border:#C7D2FE;--muted:#94A3B8;--glass:rgba(248,250,255,0.78);
  --gborder:rgba(199,210,254,0.55);}
html,body,[class*="css"],.stApp,.main,.block-container{
  font-family:'DM Sans',sans-serif!important;background:var(--canvas)!important;color:var(--navy)!important;}
.block-container{padding-top:0!important;max-width:100%!important;}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:var(--canvas)}
::-webkit-scrollbar-thumb{background:linear-gradient(180deg,var(--cyan),var(--teal));border-radius:6px}
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,var(--navy3),var(--navy) 40%,var(--navy2))!important;
  border-right:1px solid rgba(56,189,248,0.2)!important;box-shadow:4px 0 24px rgba(0,0,0,0.4)!important;}
section[data-testid="stSidebar"] *{color:#CBD5E1!important;}
section[data-testid="stSidebar"] label{color:#64748B!important;font-size:0.68rem!important;
  letter-spacing:0.1em;text-transform:uppercase;font-family:'Space Mono',monospace!important;}
.nk-topbar{background:linear-gradient(90deg,var(--navy3),var(--navy) 60%,#0C1A35);
  border-bottom:2px solid rgba(56,189,248,0.3);padding:11px 24px;margin:-1rem -1rem 0 -1rem;
  display:flex;align-items:center;justify-content:space-between;
  box-shadow:0 4px 20px rgba(0,0,0,0.3);position:sticky;top:0;z-index:999;}
.nk-logo{font-family:'Orbitron',monospace!important;font-size:1.3rem;font-weight:900;
  color:var(--cyan);letter-spacing:0.2em;text-shadow:0 0 20px rgba(56,189,248,0.35);}
.nk-logo-sub{font-family:'Space Mono',monospace;font-size:0.54rem;color:#334155;
  letter-spacing:0.2em;text-transform:uppercase;margin-top:1px;}
.nk-user-pill{background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.25);
  border-radius:20px;padding:4px 12px;font-family:'Space Mono',monospace;
  font-size:0.65rem;color:#94A3B8;}
.nk-menu-sec{font-family:'Space Mono',monospace;font-size:0.56rem;color:#334155;
  letter-spacing:0.2em;text-transform:uppercase;padding:10px 0 4px 2px;
  border-bottom:1px solid rgba(56,189,248,0.08);margin-bottom:4px;}
.glass-card{background:var(--glass);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border:1px solid var(--gborder);border-radius:8px;padding:18px 22px;margin-bottom:14px;
  box-shadow:0 4px 20px rgba(15,23,42,0.07),inset 0 1px 0 rgba(255,255,255,0.45);}
.brutalist-card{background:var(--card);border:2px solid var(--navy);
  box-shadow:5px 5px 0 var(--navy);padding:18px 22px;margin-bottom:16px;}
.brutalist-card-inv{background:var(--navy);border:2px solid var(--cyan);
  box-shadow:5px 5px 0 var(--cyan);padding:16px 20px;margin-bottom:16px;}
.brutalist-card-inv *{color:#F1F5F9!important;}
.brutalist-card-gold{background:var(--card);border:2px solid var(--gold);
  box-shadow:5px 5px 0 var(--gold);padding:16px 20px;margin-bottom:16px;}
.sec-title{font-family:'DM Sans',sans-serif;font-size:1.0rem;font-weight:700;color:var(--navy);
  border-left:4px solid var(--cyan);padding-left:10px;margin:6px 0 12px;}
.page-title{font-family:'DM Sans',sans-serif;font-size:1.55rem;font-weight:700;
  color:var(--navy);margin-bottom:4px;}
.page-desc{font-family:'DM Sans',sans-serif;font-size:0.87rem;color:var(--muted);
  margin-bottom:18px;line-height:1.5;}
.eq-block{background:var(--navy);border-left:4px solid var(--cyan);border-radius:0 5px 5px 0;
  padding:13px 18px;margin:10px 0 16px;font-family:'Space Mono',monospace;
  font-size:0.87rem;color:var(--cyan);overflow-x:auto;line-height:1.75;}
.eq-gold{color:var(--gold)}.eq-teal{color:var(--teal)}.eq-muted{color:#475569;font-size:0.74rem;}
.ai-box{background:linear-gradient(135deg,#0A1628,#0F2040 50%,#0C1A35);
  border:1px solid rgba(56,189,248,0.3);border-radius:6px;padding:16px 20px;
  margin-top:12px;line-height:1.75;box-shadow:inset 0 1px 0 rgba(56,189,248,0.08);}
.ai-box h4{font-family:'DM Sans',sans-serif;font-size:0.76rem;font-weight:600;
  color:var(--cyan);letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;}
.ai-box p{color:#CBD5E1;font-size:0.875rem;margin-bottom:7px;}
.ai-box ul{color:#94A3B8;font-size:0.83rem;padding-left:16px;}
.ai-box li{margin-bottom:3px;}
.ai-box strong{color:#E2E8F0;}
.highlight{color:var(--cyan);font-family:'Space Mono',monospace;}
.warn-text{color:var(--gold);}.ok-text{color:var(--green);}.bad-text{color:var(--red);}
.coef-wrap{overflow-x:auto;margin-bottom:12px;}
.coef-tbl{width:100%;border-collapse:collapse;font-family:'Space Mono',monospace;font-size:0.77rem;}
.coef-tbl th{background:var(--navy);color:var(--cyan);padding:7px 11px;text-align:left;
  font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;border-bottom:2px solid var(--cyan);}
.coef-tbl td{padding:5px 11px;border-bottom:1px solid var(--border);}
.coef-tbl tr:nth-child(even) td{background:#F8FAFF;}
.coef-tbl tr:hover td{background:#EEF2FF;}
.sig3{color:#065F46;font-weight:700;}.sig2{color:#1D4ED8;font-weight:600;}
.sig1{color:#92400E;}.insig{color:#94A3B8;}.sig-star{font-weight:900;}
.stat-box{background:var(--card);border:1.5px solid var(--border);border-radius:6px;padding:13px 16px;}
.stat-name{font-size:0.62rem;color:var(--muted);text-transform:uppercase;
  letter-spacing:0.12em;margin-bottom:4px;font-family:'Space Mono',monospace;}
.stat-val{font-size:1.3rem;font-weight:700;color:var(--navy);margin-bottom:2px;
  font-family:'Space Mono',monospace;}
.stat-sub{font-size:0.68rem;color:var(--muted);font-family:'Space Mono',monospace;}
div[data-testid="metric-container"]{background:var(--glass)!important;
  border:1px solid var(--gborder)!important;border-radius:6px!important;padding:11px 15px!important;}
div[data-testid="metric-container"] [data-testid="stMetricLabel"]{
  font-family:'Space Mono',monospace!important;font-size:0.63rem!important;
  text-transform:uppercase;letter-spacing:0.1em;color:var(--muted)!important;}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{
  font-family:'Space Mono',monospace!important;font-size:1.15rem!important;color:var(--navy)!important;}
.stButton>button{background:var(--navy)!important;color:var(--cyan)!important;
  border:1.5px solid var(--cyan)!important;border-radius:3px!important;
  font-family:'Space Mono',monospace!important;font-size:0.72rem!important;
  letter-spacing:0.09em;text-transform:uppercase;font-weight:700;padding:8px 18px!important;
  transition:all 0.15s ease;box-shadow:3px 3px 0 rgba(56,189,248,0.2);}
.stButton>button:hover{background:var(--cyan)!important;color:var(--navy)!important;
  box-shadow:4px 4px 0 var(--navy);transform:translate(-1px,-1px);}
.stDownloadButton>button{background:var(--gold)!important;color:var(--navy)!important;
  border:1.5px solid #D97706!important;border-radius:3px!important;
  font-family:'Space Mono',monospace!important;font-size:0.72rem!important;font-weight:700;}
.stTabs [data-baseweb="tab-list"]{background:var(--navy)!important;border-radius:5px 5px 0 0;
  padding:3px 7px 0;gap:3px;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#475569!important;
  font-family:'Space Mono',monospace!important;font-size:0.68rem!important;letter-spacing:0.07em;
  text-transform:uppercase;padding:7px 14px!important;border-bottom:2px solid transparent!important;}
.stTabs [aria-selected="true"]{color:var(--cyan)!important;border-bottom:2px solid var(--cyan)!important;
  background:rgba(56,189,248,0.08)!important;}
.stTabs [data-baseweb="tab-panel"]{background:var(--canvas)!important;
  border:1px solid rgba(56,189,248,0.13);border-top:none;padding:16px!important;}
.stSelectbox>div>div,.stMultiSelect>div>div,.stTextInput>div>div>input,
.stNumberInput>div>div>input,.stTextArea>div>div>textarea{
  background:var(--card)!important;border:1px solid var(--border)!important;
  border-radius:4px!important;color:var(--navy)!important;font-family:'DM Sans',sans-serif!important;}
.streamlit-expanderHeader{background:var(--navy)!important;color:var(--cyan)!important;
  font-family:'Space Mono',monospace!important;font-size:0.74rem!important;
  letter-spacing:0.08em;text-transform:uppercase;border-radius:4px!important;}
[data-testid="stFileUploader"]{border:2px dashed rgba(56,189,248,0.3)!important;
  border-radius:8px!important;background:rgba(56,189,248,0.02)!important;padding:8px!important;}
[data-testid="stChatMessage"]{background:var(--card)!important;
  border:1px solid var(--border)!important;border-radius:6px!important;margin-bottom:7px!important;}
#MainMenu,footer,header{visibility:hidden!important;}
hr{border:none!important;border-top:1px solid var(--border)!important;margin:14px 0!important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────────────────────────────────────
def render_login():
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.markdown("""
        <div style="text-align:center;padding:44px 0 28px;">
          <p style="font-family:'Orbitron',monospace;font-size:2.8rem;font-weight:900;
             color:#0F172A;letter-spacing:0.18em;margin:0;line-height:1;">⬡ NEXUS</p>
          <p style="font-family:'Orbitron',monospace;font-size:2rem;font-weight:800;
             color:#38BDF8;letter-spacing:0.3em;margin:0;">KERNEL</p>
          <p style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#64748B;
             letter-spacing:0.22em;text-transform:uppercase;margin-top:7px;">
             v3.0 · Professional Econometrics Platform</p>
          <p style="font-family:'DM Sans',sans-serif;font-size:0.8rem;color:#94A3B8;margin-top:3px;">
             Research by Ahmed Hisham</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="brutalist-card" style="border-color:#38BDF8;box-shadow:5px 5px 0 #38BDF8;">', unsafe_allow_html=True)
        st.markdown('<p class="sec-title" style="font-size:0.82rem;letter-spacing:0.12em;">ACCESS PORTAL</p>', unsafe_allow_html=True)
        with st.form("login_form"):
            c1, c2 = st.columns(2)
            fname = c1.text_input("First Name", placeholder="Ahmed")
            lname = c2.text_input("Last Name",  placeholder="Hisham")
            email = st.text_input("Email", placeholder="you@institution.edu")
            occ   = st.selectbox("Occupation", [
                "— Select —", "Researcher / Academic", "PhD Student",
                "Central Bank Economist", "Financial Analyst",
                "Policy Analyst", "Data Scientist / Quant", "Other",
            ])
            st.markdown("<br>", unsafe_allow_html=True)
            sub = st.form_submit_button("▶  ENTER NEXUS KERNEL", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if sub:
            errs = []
            if not fname.strip():    errs.append("First name required.")
            if not lname.strip():    errs.append("Last name required.")
            if "@" not in email:     errs.append("Valid email required.")
            if occ == "— Select —":  errs.append("Select occupation.")
            if errs:
                for e in errs: st.error(e)
            else:
                st.session_state.authenticated = True
                st.session_state.user_name  = f"{fname.strip()} {lname.strip()}"
                st.session_state.user_email = email.strip()
                st.session_state.user_occ   = occ
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
MENU_STRUCTURE = {
    "FILE": [
        ("home",       "⬡",  "Home / Dashboard"),
        ("data",       "📂", "Data Workspace"),
        ("transform",  "⚡", "Transform & Filter"),
        ("export",     "💾", "Export & Download"),
    ],
    "STATIONARITY": [
        ("unitroot",   "📉", "Unit Root Tests"),
        ("acf_pacf",   "〰",  "ACF / PACF"),
        ("correlogram","🔗", "Correlogram Matrix"),
    ],
    "ESTIMATE": [
        ("ols",        "📐", "OLS Regression"),
        ("ardl",       "🔄", "ARDL / Bounds Test"),
        ("var",        "🌐", "VAR System"),
        ("vecm",       "⛓",  "Johansen / VECM"),
        ("garch",      "🌊", "GARCH / Volatility"),
        ("arima",      "🔭", "ARIMA / SARIMA"),
    ],
    "DIAGNOSTICS": [
        ("diagnostics","🔬", "Diagnostic Suite"),
        ("stability",  "📏", "Stability Tests"),
        ("normality",  "🔔", "Normality Analysis"),
    ],
    "FORECAST": [
        ("forecast",   "🎯", "Forecasting Engine"),
        ("decompose",  "🧩", "Decomposition"),
    ],
    "TOOLS": [
        ("stats",      "📊", "Summary Statistics"),
        ("chat",       "🤖", "AI Econometrician"),
        ("report",     "📄", "PDF Report"),
    ],
}

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:14px 4px 10px;text-align:center;
             border-bottom:1px solid rgba(56,189,248,0.15);margin-bottom:7px;">
          <p style="font-family:'Orbitron',monospace;font-size:1.2rem;font-weight:900;
             color:#38BDF8;letter-spacing:0.22em;margin:0;line-height:1.2;">⬡ NEXUS</p>
          <p style="font-family:'Orbitron',monospace;font-size:0.9rem;font-weight:700;
             color:#38BDF8;letter-spacing:0.35em;margin:0;">KERNEL</p>
          <p style="font-family:'Space Mono',monospace;font-size:0.52rem;color:#1E3A5F;
             letter-spacing:0.18em;margin-top:3px;text-transform:uppercase;">
             RESEARCH BY AHMED HISHAM</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.authenticated:
            st.markdown(f"""
            <div style="padding:7px 10px;background:rgba(56,189,248,0.06);
                 border:1px solid rgba(56,189,248,0.15);border-radius:4px;margin-bottom:8px;">
              <p style="margin:0;font-size:0.7rem;color:#38BDF8;
                 font-family:'Space Mono',monospace;">{st.session_state.user_name}</p>
              <p style="margin:0;font-size:0.6rem;color:#334155;">{st.session_state.user_occ}</p>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.clean_df is not None:
            df = st.session_state.clean_df
            st.markdown(f"""
            <div style="padding:7px 10px;background:rgba(16,185,129,0.07);
                 border:1px solid rgba(16,185,129,0.2);border-radius:4px;margin-bottom:8px;">
              <p style="margin:0;font-size:0.6rem;color:#10B981;
                 font-family:'Space Mono',monospace;text-transform:uppercase;">● Active Dataset</p>
              <p style="margin:0;font-size:0.68rem;color:#94A3B8;font-family:'Space Mono',monospace;">
                 {len(df):,} obs · {len(df.columns)} vars · {st.session_state.freq or '?'}</p>
            </div>
            """, unsafe_allow_html=True)

        current = st.session_state.menu
        for section, items in MENU_STRUCTURE.items():
            st.markdown(f'<p class="nk-menu-sec">{section}</p>', unsafe_allow_html=True)
            for key, icon, label in items:
                is_active = current == key
                btn_type  = "primary" if is_active else "secondary"
                if st.button(f"{icon}  {label}", key=f"nav_{key}",
                             use_container_width=True, type=btn_type):
                    st.session_state.menu    = key
                    st.session_state.submenu = None
                    st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("⏻  Sign Out", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        st.markdown("""
        <p style="font-family:'Space Mono',monospace;font-size:0.53rem;color:#1E3A5F;
           text-align:center;margin-top:8px;">NEXUS KERNEL v3.0 · 2026</p>
        """, unsafe_allow_html=True)


def render_topbar():
    labels = {
        "home":"Home","data":"Data Workspace","transform":"Transform & Filter",
        "export":"Export","unitroot":"Unit Root Tests","acf_pacf":"ACF / PACF",
        "correlogram":"Correlogram","ols":"OLS Regression","ardl":"ARDL",
        "var":"VAR System","vecm":"Johansen / VECM","garch":"Volatility Models",
        "arima":"ARIMA / SARIMA","diagnostics":"Diagnostic Suite",
        "stability":"Stability Tests","normality":"Normality Analysis",
        "forecast":"Forecasting Engine","decompose":"Decomposition",
        "stats":"Summary Statistics","chat":"AI Econometrician","report":"PDF Report",
    }
    page = labels.get(st.session_state.menu, "NEXUS KERNEL")
    user = st.session_state.get("user_name", "")
    st.markdown(f"""
    <div class="nk-topbar">
      <div>
        <span class="nk-logo">⬡ NEXUS KERNEL</span>
        <p class="nk-logo-sub">Professional Time-Series Econometrics · v3.0</p>
      </div>
      <div style="display:flex;align-items:center;gap:14px;">
        <span style="font-family:'Space Mono',monospace;font-size:0.75rem;
              color:#38BDF8;letter-spacing:0.06em;">{page.upper()}</span>
        <span class="nk-user-pill">{user}</span>
      </div>
    </div>
    <div style="margin-bottom:16px;"></div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    st.markdown("""
    <p class="page-title">Welcome to NEXUS KERNEL v3.0</p>
    <p class="page-desc">The professional-grade replacement for EViews, Stata, and R for
    time-series econometrics. Built for central bank economists, academic researchers,
    and quantitative analysts.</p>
    """, unsafe_allow_html=True)

    has_data = st.session_state.clean_df is not None
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Dataset",     "Loaded" if has_data else "None",
              delta=f"{len(st.session_state.clean_df):,} obs" if has_data else None)
    c2.metric("OLS Model",   "Ready" if st.session_state.ols_res  else "—")
    c3.metric("ARDL Model",  "Ready" if st.session_state.ardl_res else "—")
    c4.metric("VAR Model",   "Ready" if st.session_state.var_res  else "—")
    c5.metric("AI Messages", str(len(st.session_state.chat_history)))

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="sec-title">📦 Module Overview</p>', unsafe_allow_html=True)
    grid = [
        ("📂","Data Workspace","Upload CSV/XLSX · Auto-freq · Clean","data"),
        ("⚡","Transform & Filter","Log/Diff/MA · HP · BK · STL","transform"),
        ("📉","Unit Root Tests","ADF · PP · KPSS · Zivot-Andrews","unitroot"),
        ("📐","OLS Regression","HC1/HAC SEs · Full inference · RESET","ols"),
        ("🔄","ARDL / Bounds","PSS Bounds · ECM · Long-run coefs","ardl"),
        ("🌐","VAR System","Granger · IRF · FEVD · Lag selection","var"),
        ("⛓","Johansen / VECM","Trace/Max-Eigen · α β matrices","vecm"),
        ("🌊","Volatility","GARCH · EGARCH · TGARCH","garch"),
        ("🔭","ARIMA / SARIMA","Auto-ARIMA · Fan charts · RMSE","arima"),
        ("🔬","Diagnostics","DW · BG · LjungBox · White · RESET","diagnostics"),
        ("🤖","AI Econometrician","Context-aware chatbot · Q&A","chat"),
        ("📄","PDF Report","Automated professional report","report"),
    ]
    for i in range(0, len(grid), 4):
        row = grid[i:i+4]
        cols_g = st.columns(len(row))
        for col_g, (icon, title, desc, key) in zip(cols_g, row):
            with col_g:
                st.markdown(f"""
                <div class="glass-card" style="min-height:118px;border-left:3px solid #38BDF8;">
                  <p style="font-size:1.6rem;margin:0 0 5px;">{icon}</p>
                  <p style="font-family:'DM Sans';font-weight:700;font-size:0.85rem;
                     color:#0F172A;margin:0 0 4px;">{title}</p>
                  <p style="font-size:0.73rem;color:#64748B;line-height:1.4;margin:0;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Open {title}", key=f"home_{key}", use_container_width=True):
                    st.session_state.menu = key
                    st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="sec-title">🗺 Recommended Workflow</p>', unsafe_allow_html=True)
    steps = [
        ("1","Upload Data","File → Data Workspace"),
        ("2","Clean & Transform","File → Transform & Filter"),
        ("3","Test Stationarity","Stationarity → Unit Root Tests"),
        ("4","Estimate Model","Estimate → choose model"),
        ("5","Run Diagnostics","Diagnostics → Suite"),
        ("6","Forecast & Report","Forecast → PDF Report"),
    ]
    cols_s = st.columns(6)
    for col_s,(num,title,path) in zip(cols_s,steps):
        with col_s:
            st.markdown(f"""
            <div style="text-align:center;padding:12px 6px;">
              <div style="width:32px;height:32px;border-radius:50%;background:#0F172A;
                   border:2px solid #38BDF8;display:flex;align-items:center;
                   justify-content:center;margin:0 auto 7px;font-family:'Space Mono',monospace;
                   font-weight:700;font-size:0.85rem;color:#38BDF8;">{num}</div>
              <p style="font-family:'DM Sans';font-weight:600;font-size:0.8rem;
                 color:#0F172A;margin:0 0 3px;">{title}</p>
              <p style="font-size:0.67rem;color:#64748B;margin:0;
                 font-family:'Space Mono',monospace;">{path}</p>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA WORKSPACE
# ─────────────────────────────────────────────────────────────────────────────
def page_data():
    st.markdown("""
    <p class="page-title">📂 Data Workspace</p>
    <p class="page-desc">Upload CSV/XLSX · Auto-detect frequency · Validate & clean missing values</p>
    """, unsafe_allow_html=True)

    tab_up, tab_view, tab_clean = st.tabs(["① Upload & Ingest","② Preview & Validate","③ Clean & Save"])

    with tab_up:
        uploaded = st.file_uploader("Drop CSV or XLSX here", type=["csv","xlsx","xls"])
        if uploaded:
            try:
                if uploaded.name.lower().endswith((".xlsx",".xls")):
                    raw = pd.read_excel(uploaded, sheet_name=0)
                else:
                    content = uploaded.read()
                    raw = None
                    for enc in ("utf-8","latin-1","cp1252"):
                        try:
                            raw = pd.read_csv(io.BytesIO(content), sep=None,
                                              engine="python", encoding=enc)
                            break
                        except Exception:
                            continue
                    if raw is None:
                        raise ValueError("Could not parse CSV with utf-8, latin-1, or cp1252.")
                st.session_state.raw_df = raw.copy()
                st.success(f"✓ Loaded **{uploaded.name}** — {raw.shape[0]:,} rows × {raw.shape[1]} cols")
            except Exception as exc:
                st.error(f"**File read error:** {exc}")
        if st.session_state.raw_df is not None:
            raw = st.session_state.raw_df
            st.markdown('<p class="sec-title">Raw Preview (first 10 rows)</p>', unsafe_allow_html=True)
            st.dataframe(raw.head(10), use_container_width=True)
            c1,c2,c3 = st.columns(3)
            c1.metric("Rows",   f"{raw.shape[0]:,}")
            c2.metric("Cols",   raw.shape[1])
            c3.metric("Memory", f"{raw.memory_usage(deep=True).sum()/1024:.1f} KB")

    with tab_view:
        if st.session_state.raw_df is None:
            st.info("Upload a file first.")
        else:
            raw = st.session_state.raw_df
            st.markdown('<p class="sec-title">Column Info & Missing Values</p>', unsafe_allow_html=True)
            info = pd.DataFrame({
                "dtype":   raw.dtypes.astype(str),
                "non_null":raw.notnull().sum(),
                "missing": raw.isnull().sum(),
                "miss_%":  (raw.isnull().mean()*100).round(2),
                "unique":  raw.nunique(),
            })
            st.dataframe(info, use_container_width=True)
            num = raw.select_dtypes(include=np.number)
            if not num.empty:
                st.markdown('<p class="sec-title">Numeric Summary</p>', unsafe_allow_html=True)
                desc = num.describe().T.round(4)
                desc["skew"] = num.skew().round(4)
                desc["kurt"] = num.kurt().round(4)
                st.dataframe(desc, use_container_width=True)

    with tab_clean:
        if st.session_state.raw_df is None:
            st.info("Upload a file first.")
        else:
            raw = st.session_state.raw_df
            st.markdown('<p class="sec-title">Cleaning Configuration</p>', unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            with c1:
                date_col = st.selectbox("Date / Period Column",
                                        ["(use row index)"]+raw.columns.tolist())
                freq_opt = st.selectbox("Frequency Override",
                    ["Auto-Detect","Annual (A)","Quarterly (Q)","Monthly (M)","Weekly (W)","Daily (D)"])
            with c2:
                miss_method = st.selectbox("Missing Value Treatment",
                    ["Linear Interpolation","Spline Interpolation",
                     "Forward Fill (ffill)","Backward Fill (bfill)","Drop Rows with NaN"])
                strip_sym = st.checkbox("Strip symbols ($, %, commas)", value=True)
            seasonal_adj = st.checkbox("STL Seasonal Adjustment (sub-annual)", value=False)

            if st.button("▶  APPLY CLEANING & SAVE", use_container_width=True):
                with st.spinner("Processing…"):
                    try:
                        df = raw.copy()
                        if strip_sym:
                            for col in df.select_dtypes(include="object").columns:
                                df[col] = (df[col].astype(str)
                                           .str.replace(r"[$%,\s\u202f\u00a0]","",regex=True)
                                           .str.replace(r"[^\d.\-eE+]","",regex=True))
                                df[col] = pd.to_numeric(df[col], errors="coerce")
                        if date_col != "(use row index)":
                            try:
                                df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
                                df = df.set_index(date_col).sort_index()
                                st.session_state.date_col = date_col
                            except Exception:
                                df = df.set_index(date_col).sort_index()
                        freq_map = {"Annual (A)":"Annual","Quarterly (Q)":"Quarterly",
                                    "Monthly (M)":"Monthly","Weekly (W)":"Weekly","Daily (D)":"Daily"}
                        if freq_opt == "Auto-Detect":
                            try:
                                inf = pd.infer_freq(df.index)
                                if inf:
                                    if inf[0] in ("A","Y"):  freq = "Annual"
                                    elif inf[0]=="Q":          freq = "Quarterly"
                                    elif inf[0]=="M":          freq = "Monthly"
                                    elif inf[0]=="W":          freq = "Weekly"
                                    elif inf[0] in ("D","B"): freq = "Daily"
                                    else:                      freq = inf
                                else: freq = "Unknown"
                            except Exception: freq = "Unknown"
                        else:
                            freq = freq_map.get(freq_opt, freq_opt.split(" ")[0])
                        st.session_state.freq = freq
                        num_cols = df.select_dtypes(include=np.number).columns
                        if miss_method == "Linear Interpolation":
                            df[num_cols] = df[num_cols].interpolate(method="linear")
                        elif miss_method == "Spline Interpolation":
                            df[num_cols] = df[num_cols].interpolate(method="spline",order=3)
                        elif miss_method == "Forward Fill (ffill)":
                            df[num_cols] = df[num_cols].ffill()
                        elif miss_method == "Backward Fill (bfill)":
                            df[num_cols] = df[num_cols].bfill()
                        elif miss_method == "Drop Rows with NaN":
                            df = df.dropna()
                        if seasonal_adj and freq in ("Monthly","Quarterly"):
                            period = 12 if freq=="Monthly" else 4
                            for col in num_cols:
                                try:
                                    stl = STL(df[col].dropna(), period=period)
                                    r   = stl.fit()
                                    df[f"{col}_SA"] = df[col] - r.seasonal
                                except Exception:
                                    pass
                        st.session_state.clean_df = df
                        st.success(f"✓ Dataset saved — {len(df):,} obs · "
                                   f"{len(df.columns)} vars · Frequency: **{freq}**")
                        num_df = df.select_dtypes(include=np.number)
                        if not num_df.empty:
                            fig = go.Figure()
                            for i,col in enumerate(num_df.columns[:8]):
                                fig.add_trace(go.Scatter(
                                    x=num_df.index, y=num_df[col], name=col,
                                    line=dict(color=CHART_COLORS[i%len(CHART_COLORS)],width=1.8)))
                            fig = navy_fig(fig,350,"Dataset Overview (first 8 numeric variables)")
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as exc:
                        st.error(f"**Cleaning failed:** {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TRANSFORM & FILTER
# ─────────────────────────────────────────────────────────────────────────────
def page_transform():
    st.markdown("""
    <p class="page-title">⚡ Transform & Filter</p>
    <p class="page-desc">Log · Differences · Moving Averages · HP Filter · Baxter-King · STL Decomposition</p>
    """, unsafe_allow_html=True)

    if st.session_state.clean_df is None:
        st.info("Load data in **Data Workspace** first.")
        return
    df   = st.session_state.clean_df.copy()
    cols = df.select_dtypes(include=np.number).columns.tolist()
    tab_t, tab_f = st.tabs(["🔢 Transformations","🎛 Filters & Decomposition"])

    with tab_t:
        st.markdown('<p class="sec-title">Variable Transformations</p>', unsafe_allow_html=True)
        sel_cols = st.multiselect("Variables to transform", cols, default=cols[:2])
        c1,c2,c3 = st.columns(3)
        with c1:
            do_log   = st.checkbox("Natural Log (ln)")
            do_diff1 = st.checkbox("First Difference (Δ)")
            do_diff2 = st.checkbox("Second Difference (Δ²)")
        with c2:
            do_ma    = st.checkbox("Moving Average")
            ma_win   = st.slider("MA Window",2,24,4) if do_ma else 4
            do_pct   = st.checkbox("% Change (growth rate)")
        with c3:
            do_std   = st.checkbox("Standardize (z-score)")
            do_idx   = st.checkbox("Index (base=100 at start)")
            do_lret  = st.checkbox("Log Returns (ln Pₜ/Pₜ₋₁)")
        if st.button("▶  APPLY TRANSFORMATIONS", use_container_width=True):
            if not sel_cols:
                st.warning("Select at least one variable.")
            else:
                added = []
                for col in sel_cols:
                    s = df[col].copy()
                    if do_log:
                        n=f"ln_{col}"; df[n]=np.log(s.replace(0,np.nan)); added.append(n)
                    if do_diff1:
                        n=f"d_{col}"; df[n]=s.diff(); added.append(n)
                    if do_diff2:
                        n=f"d2_{col}"; df[n]=s.diff().diff(); added.append(n)
                    if do_ma:
                        n=f"ma{ma_win}_{col}"; df[n]=s.rolling(ma_win).mean(); added.append(n)
                    if do_pct:
                        n=f"pct_{col}"; df[n]=s.pct_change()*100; added.append(n)
                    if do_std:
                        n=f"std_{col}"; df[n]=(s-s.mean())/s.std(); added.append(n)
                    if do_idx:
                        n=f"idx_{col}"; base=s.dropna().iloc[0]
                        df[n]=(s/base*100) if base!=0 else s; added.append(n)
                    if do_lret:
                        n=f"logret_{col}"; df[n]=np.log(s/s.shift(1)); added.append(n)
                st.session_state.clean_df = df
                st.success(f"✓ Added {len(added)} series: {', '.join(added)}")
                if added:
                    fig = go.Figure()
                    for i,col in enumerate(added[:6]):
                        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                            line=dict(color=CHART_COLORS[i%len(CHART_COLORS)],width=1.8)))
                    fig = navy_fig(fig,350,"Transformed Series")
                    st.plotly_chart(fig, use_container_width=True)

    with tab_f:
        st.markdown('<p class="sec-title">Economic Filters</p>', unsafe_allow_html=True)
        filt_col  = st.selectbox("Variable to filter", cols)
        filt_type = st.radio("Method",
            ["Hodrick-Prescott (HP)","Baxter-King (BK)","STL Decomposition"], horizontal=True)
        if filt_type == "Hodrick-Prescott (HP)":
            default_lam = {"Annual":100,"Quarterly":1600,"Monthly":14400,"Daily":129600}
            lam = st.number_input("Smoothing λ",
                value=float(default_lam.get(st.session_state.freq or "Quarterly",1600)),
                min_value=1.0, step=100.0)
            if st.button("▶  HP FILTER", use_container_width=True):
                try:
                    cycle, trend = hpfilter(df[filt_col].dropna(), lamb=lam)
                    df[f"hp_trend_{filt_col}"] = trend
                    df[f"hp_cycle_{filt_col}"] = cycle
                    st.session_state.clean_df = df
                    fig = make_subplots(rows=2,cols=1,vertical_spacing=0.08,
                        subplot_titles=["Original vs HP Trend","HP Cyclical Component"])
                    fig.add_trace(go.Scatter(x=df.index,y=df[filt_col],name="Original",
                        line=dict(color=C["navy"],width=1.5)),1,1)
                    fig.add_trace(go.Scatter(x=trend.index,y=trend.values,name="Trend",
                        line=dict(color=C["cyan"],width=2.5)),1,1)
                    fig.add_trace(go.Scatter(x=cycle.index,y=cycle.values,name="Cycle",
                        fill="tozeroy",fillcolor="rgba(56,189,248,0.13)",
                        line=dict(color=C["cyan"],width=1.8)),2,1)
                    fig.add_hline(y=0,line=dict(color=C["navy"],dash="dash",width=1),row=2,col=1)
                    fig = navy_fig(fig,480,f"HP Filter — {filt_col} (λ={lam:,.0f})")
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("✓ HP trend and cycle added to dataset.")
                except Exception as exc:
                    st.error(f"HP Filter error: {exc}")

        elif filt_type == "Baxter-King (BK)":
            c1,c2,c3 = st.columns(3)
            lo = c1.number_input("Low period",value=6,min_value=2)
            hi = c2.number_input("High period",value=32,min_value=3)
            K  = c3.number_input("Lead/lag K",value=12,min_value=1)
            if st.button("▶  BK FILTER", use_container_width=True):
                try:
                    s  = df[filt_col].dropna()
                    bk = bkfilter(s,low=lo,high=hi,K=int(K))
                    df[f"bk_cycle_{filt_col}"] = bk
                    st.session_state.clean_df = df
                    fig = make_subplots(rows=2,cols=1,vertical_spacing=0.08,
                        subplot_titles=["Original","BK Cyclical Component"])
                    fig.add_trace(go.Scatter(x=s.index,y=s.values,name="Original",
                        line=dict(color=C["navy"],width=1.5)),1,1)
                    fig.add_trace(go.Scatter(x=bk.index,y=bk.values,name="BK Cycle",
                        fill="tozeroy",fillcolor="rgba(20,184,166,0.13)",
                        line=dict(color=C["teal"],width=2)),2,1)
                    fig.add_hline(y=0,line=dict(color=C["navy"],dash="dash"),row=2,col=1)
                    fig = navy_fig(fig,480,f"Baxter-King Filter — {filt_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("✓ BK cycle added.")
                except Exception as exc:
                    st.error(f"BK Filter error: {exc}")

        elif filt_type == "STL Decomposition":
            default_period = {"Monthly":12,"Quarterly":4,"Weekly":52,"Annual":2,"Daily":365}
            period = st.number_input("Seasonal period",
                value=int(default_period.get(st.session_state.freq or "Monthly",12)),min_value=2)
            robust = st.checkbox("Robust STL",value=True)
            if st.button("▶  STL DECOMPOSE", use_container_width=True):
                try:
                    s   = df[filt_col].dropna()
                    stl = STL(s,period=int(period),robust=robust)
                    res = stl.fit()
                    df[f"stl_trend_{filt_col}"]    = res.trend
                    df[f"stl_seasonal_{filt_col}"] = res.seasonal
                    df[f"stl_resid_{filt_col}"]    = res.resid
                    st.session_state.clean_df = df
                    fig = make_subplots(rows=4,cols=1,vertical_spacing=0.04,
                        subplot_titles=["Original","Trend","Seasonal","Residual"])
                    for i,(y,color,name) in enumerate([
                        (s.values,C["navy"],"Original"),
                        (res.trend,C["cyan"],"Trend"),
                        (res.seasonal,C["gold"],"Seasonal"),
                        (res.resid,C["red"],"Residual"),
                    ],1):
                        fig.add_trace(go.Scatter(x=s.index,y=y,name=name,
                            line=dict(color=color,width=1.6)),i,1)
                    fig = navy_fig(fig,600,f"STL Decomposition — {filt_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("✓ STL components added.")
                except Exception as exc:
                    st.error(f"STL error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: UNIT ROOT TESTS
# ─────────────────────────────────────────────────────────────────────────────
def page_unitroot():
    st.markdown("""
    <p class="page-title">📉 Unit Root & Stationarity Vault</p>
    <p class="page-desc">ADF · Phillips-Perron · KPSS · Zivot-Andrews — with AI interpretation</p>
    """, unsafe_allow_html=True)

    if st.session_state.clean_df is None:
        st.info("Load your dataset first.")
        return

    df   = st.session_state.clean_df
    cols = df.select_dtypes(include=np.number).columns.tolist()
    st.markdown('<p class="sec-title">Configuration</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: sel_var  = st.selectbox("Variable", cols)
    with c2: diff_ord = st.selectbox("Transform",
                ["Level","1st Difference (Δ)","2nd Difference (Δ²)"])
    with c3:
        reg_spec = st.selectbox("Deterministics",
                ["c — Constant only","ct — Constant + Trend",
                 "n — None","ctt — Const + Quad Trend"])
        reg_code = reg_spec.split(" ")[0]
    with c4: lag_meth = st.selectbox("Lag Selection",["AIC","BIC","t-stat","Fixed"])
    fixed_lag = 3
    if lag_meth == "Fixed":
        fixed_lag = st.slider("Fixed lag length",1,16,3)
    alpha = st.select_slider("Significance α",[0.01,0.05,0.10],value=0.05)

    if st.button("▶  RUN ALL STATIONARITY TESTS", use_container_width=True):
        with st.spinner("Running tests…"):
            s = df[sel_var].dropna()
            if diff_ord.startswith("1st"):
                s = s.diff().dropna(); s_label = f"Δ{sel_var}"
            elif diff_ord.startswith("2nd"):
                s = s.diff().diff().dropna(); s_label = f"Δ²{sel_var}"
            else:
                s_label = sel_var
            results = {"variable":sel_var,"level":diff_ord,"label":s_label}

            # ADF
            try:
                kw = dict(regression=reg_code)
                if lag_meth=="Fixed": kw.update(maxlag=fixed_lag,autolag=None)
                else: kw["autolag"] = lag_meth
                r = adfuller(s,**kw)
                results["adf"] = {"stat":r[0],"pval":r[1],"lags":r[2],"crits":r[4],"ok":r[1]<alpha}
            except Exception as exc:
                results["adf"] = {"error":str(exc)}

            # PP proxy via adfuller BIC
            try:
                r2 = adfuller(s,regression=reg_code,autolag="BIC")
                results["pp"] = {"stat":r2[0],"pval":r2[1],"lags":r2[2],"crits":r2[4],"ok":r2[1]<alpha}
            except Exception as exc:
                results["pp"] = {"error":str(exc)}

            # KPSS
            try:
                kr = "c" if reg_code in ("c","n") else "ct"
                k  = kpss(s,regression=kr,nlags="auto")
                results["kpss"] = {"stat":k[0],"pval":k[1],"lags":k[2],"crits":k[3],"ok":k[1]>alpha}
            except Exception as exc:
                results["kpss"] = {"error":str(exc)}

            # Zivot-Andrews
            try:
                za = zivot_andrews(s,maxlag=None,regression="c",autolag="AIC")
                results["za"] = {"stat":za[0],"pval":za[1],"break_idx":za[3],"crits":za[2],"ok":za[1]<alpha}
            except Exception as exc:
                results["za"] = {"error":str(exc)}

            st.session_state.stat_results = results

    res = st.session_state.stat_results
    if not res: return

    s_label = res.get("label","")
    lv      = res.get("level","Level")
    raw_s   = st.session_state.clean_df[res.get("variable","")].dropna()
    if lv.startswith("1st"):  plot_s = raw_s.diff().dropna()
    elif lv.startswith("2nd"): plot_s = raw_s.diff().diff().dropna()
    else:                      plot_s = raw_s

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_s.index,y=plot_s.values,name=s_label,
        line=dict(color=C["cyan"],width=2)))
    fig.add_hline(y=plot_s.mean(),line=dict(color=C["gold"],dash="dash",width=1.2),
                  annotation_text="Mean")
    fig = navy_fig(fig,260,f"Series: {s_label}")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="sec-title">Test Results</p>', unsafe_allow_html=True)

    def _show_test(container, name, key, h0_text):
        with container:
            r = res.get(key,{})
            if "error" in r:
                st.error(f"**{name}** failed: {r['error']}"); return
            stat  = r.get("stat",0); pval = r.get("pval",1)
            ok    = r.get("ok",False); crits = r.get("crits",{})
            verdict = "STATIONARY" if ok else ("UNIT ROOT" if key!="kpss" else "NON-STATIONARY")
            bkind   = "pass" if ok else "fail"
            c1p = crits.get("1%",crits.get("10%","—"))
            c5p = crits.get("5%","—"); c10p = crits.get("10%","—")
            st.markdown(f"""
            <div class="brutalist-card" style="border-color:{'#10B981' if ok else '#EF4444'};
                 box-shadow:4px 4px 0 {'#10B981' if ok else '#EF4444'};">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:7px;">
                <p style="font-weight:700;font-size:0.88rem;color:#0F172A;margin:0;">{name}</p>
                {badge_html(verdict,bkind)}
              </div>
              <table class="coef-tbl" style="width:100%;">
                <tr><th>Statistic</th><th>Value</th></tr>
                <tr><td>Test Stat.</td><td>{fmt(stat)}</td></tr>
                <tr><td>p-value</td><td>{fmt(pval)}</td></tr>
                <tr><td>Lags Used</td><td>{r.get('lags','—')}</td></tr>
                <tr><td>Crit. 1%</td><td>{fmt(c1p) if c1p!='—' else '—'}</td></tr>
                <tr><td>Crit. 5%</td><td>{fmt(c5p) if c5p!='—' else '—'}</td></tr>
                <tr><td>Crit. 10%</td><td>{fmt(c10p) if c10p!='—' else '—'}</td></tr>
              </table>
              <p style="font-size:0.68rem;color:#64748B;margin-top:7px;
                 font-family:'Space Mono',monospace;">H₀: {h0_text}</p>
            </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    _show_test(c1,"Augmented Dickey-Fuller (ADF)","adf","Unit root present")
    _show_test(c2,"Phillips-Perron (PP)","pp","Unit root present")
    _show_test(c1,"KPSS Test","kpss","Series is stationary (trend-stationary)")

    za = res.get("za",{})
    if "error" not in za and za:
        ok_za = za.get("ok",False)
        st.markdown(f"""
        <div class="brutalist-card" style="border-color:{'#10B981' if ok_za else '#EF4444'};
             box-shadow:4px 4px 0 {'#10B981' if ok_za else '#EF4444'};">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:7px;">
            <p style="font-weight:700;font-size:0.88rem;color:#0F172A;margin:0;">
               Zivot-Andrews (Unit Root w/ Structural Break)</p>
            {badge_html('STATIONARY' if ok_za else 'UNIT ROOT','pass' if ok_za else 'fail')}
          </div>
          <table class="coef-tbl"><tr><th>Statistic</th><th>Value</th></tr>
            <tr><td>ZA Stat.</td><td>{fmt(za.get('stat',0))}</td></tr>
            <tr><td>p-value</td><td>{fmt(za.get('pval',1))}</td></tr>
            <tr><td>Break Index</td><td>{za.get('break_idx','—')}</td></tr>
          </table>
          <p style="font-size:0.68rem;color:#64748B;margin-top:7px;font-family:'Space Mono',monospace;">
             H₀: Unit root allowing for one structural break in intercept</p>
        </div>""", unsafe_allow_html=True)

    adf_ok  = res.get("adf",{}).get("ok",None)
    pp_ok   = res.get("pp",{}).get("ok",None)
    kpss_ok = res.get("kpss",{}).get("ok",None)
    za_ok   = res.get("za",{}).get("ok",None)
    ai_html = _interpret_stationarity(
        s_label, adf_ok, pp_ok, kpss_ok, za_ok,
        res.get("adf",{}).get("pval",1),
        res.get("kpss",{}).get("pval",1), lv)
    st.markdown(f'<div class="ai-box"><h4>⬡ AI INTERPRETATION · STATIONARITY</h4>{ai_html}</div>',
                unsafe_allow_html=True)
    st.session_state.report_log["stationarity"] = {
        "variable":res.get("variable",""),"level":lv,
        "adf_pval":res.get("adf",{}).get("pval","—"),
        "kpss_pval":res.get("kpss",{}).get("pval","—"),
        "verdict":"STATIONARY" if (adf_ok and kpss_ok) else "NON-STATIONARY",
    }


def _interpret_stationarity(label,adf,pp,kpss_ok,za,adf_p,kpss_p,level):
    lvl_map = {"Level":"in levels","1st Difference (Δ)":"in first differences",
               "2nd Difference (Δ²)":"in second differences"}
    lvl_str = lvl_map.get(level,"")
    lines = []
    if adf is True and pp is True and kpss_ok is True:
        lines.append(f"<p><span class='ok-text'>✅ All tests agree:</span> "
                     f"<strong>{label}</strong> is <strong>stationary</strong> {lvl_str}. "
                     "High confidence — use in levels for estimation.</p>")
    elif adf is False and pp is False and kpss_ok is False:
        lines.append(f"<p><span class='bad-text'>❌ All tests agree:</span> "
                     f"<strong>{label}</strong> contains a <strong>unit root</strong> {lvl_str}. "
                     "Difference before OLS or use ARDL/VECM.</p>")
    else:
        lines.append(f"<p><span class='warn-text'>⚠ Mixed signals</span> for "
                     f"<strong>{label}</strong> {lvl_str}.</p>")
        if adf is not None:
            lines.append(f"<p>• ADF (p={fmt(adf_p)}): "
                         f"{'Reject H₀ → stationary' if adf else 'Fail to reject → unit root'}</p>")
        if kpss_ok is not None:
            lines.append(f"<p>• KPSS (p≈{fmt(kpss_p)}): "
                         f"{'Fail to reject H₀ → stationary' if kpss_ok else 'Reject H₀ → non-stationary'}</p>")
        lines.append("<p>When ADF and KPSS conflict: (1) possible structural break → run Zivot-Andrews, "
                     "(2) near-integrated process, (3) short sample bias.</p>")
    if level=="Level" and (adf is False or pp is False):
        lines.append("<p>📌 <strong>Next step:</strong> Re-test in 1st Difference to determine I(d). "
                     "If stationary in Δ → I(1) → suitable for ARDL Bounds or Johansen/VECM.</p>")
    if za is True:
        lines.append("<p>🔍 Zivot-Andrews rejects unit root allowing for one structural break — "
                     "apparent non-stationarity may be regime-change driven, not a true unit root.</p>")
    return "".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ACF / PACF
# ─────────────────────────────────────────────────────────────────────────────
def page_acf_pacf():
    st.markdown("""
    <p class="page-title">〰 ACF & PACF Correlograms</p>
    <p class="page-desc">Interactive autocorrelation functions with Bartlett bands and Ljung-Box Q-test.</p>
    """, unsafe_allow_html=True)

    if st.session_state.clean_df is None:
        st.info("Load your dataset first.")
        return
    df   = st.session_state.clean_df
    cols = df.select_dtypes(include=np.number).columns.tolist()
    c1,c2,c3 = st.columns(3)
    sel    = c1.selectbox("Variable", cols)
    n_lags = c2.slider("Lags", 4, 60, 24)
    diff_  = c3.selectbox("Transform", ["Level","1st Difference","2nd Difference"])

    if st.button("▶  PLOT CORRELOGRAMS", use_container_width=True):
        s = df[sel].dropna()
        if diff_ == "1st Difference":   s = s.diff().dropna();        lbl = f"Δ{sel}"
        elif diff_ == "2nd Difference": s = s.diff().diff().dropna(); lbl = f"Δ²{sel}"
        else:                           lbl = sel
        try:
            max_lag  = min(n_lags, len(s)//3)
            acf_vals  = acf(s,  nlags=max_lag, fft=True,  alpha=0.05)
            pacf_vals = pacf(s, nlags=max_lag, alpha=0.05)
            acf_arr   = acf_vals[0]; acf_ci  = acf_vals[1]
            pacf_arr  = pacf_vals[0]; pacf_ci = pacf_vals[1]
            conf      = 1.96 / np.sqrt(len(s))
            lags_x    = list(range(len(acf_arr)))

            fig = make_subplots(rows=2,cols=1,vertical_spacing=0.1,
                subplot_titles=[f"ACF — {lbl}", f"PACF — {lbl}"])
            for (arr,ci,row) in [(acf_arr,acf_ci,1),(pacf_arr,pacf_ci,2)]:
                for lag_i,val in enumerate(arr):
                    color = C["cyan"] if abs(val)>conf else C["slate"]
                    fig.add_trace(go.Scatter(x=[lag_i,lag_i],y=[0,val],mode="lines",
                        line=dict(color=color,width=2),showlegend=False,hoverinfo="skip"),row,1)
                    fig.add_trace(go.Scatter(x=[lag_i],y=[val],mode="markers",
                        marker=dict(color=color,size=6,line=dict(color=C["navy"],width=1.2)),
                        showlegend=False,
                        hovertemplate=f"Lag {lag_i}: {val:.4f}<extra></extra>"),row,1)
                fig.add_hline(y=conf,  line=dict(color=C["red"],dash="dash",width=1.2),row=row,col=1)
                fig.add_hline(y=-conf, line=dict(color=C["red"],dash="dash",width=1.2),row=row,col=1)
                fig.add_hline(y=0,     line=dict(color=C["navy"],width=0.8),row=row,col=1)
            fig = navy_fig(fig,500)
            st.plotly_chart(fig, use_container_width=True)

            lb = acorr_ljungbox(s, lags=min(20,max_lag), return_df=True)
            st.markdown('<p class="sec-title">Ljung-Box Q-Test</p>', unsafe_allow_html=True)
            st.dataframe(lb.style.format("{:.4f}"), use_container_width=True)
            st.markdown(f"""
            <div class="ai-box">
              <h4>⬡ ACF/PACF INTERPRETATION</h4>
              <p>Significant spikes (beyond the red ±{fmt(conf,3)} bands) at specific lags indicate
              autocorrelation structure. Use these patterns to identify ARIMA order:</p>
              <ul>
                <li><strong>ACF tailing off, PACF cuts off at lag p</strong> → AR(p) process</li>
                <li><strong>PACF tailing off, ACF cuts off at lag q</strong> → MA(q) process</li>
                <li><strong>Both tail off</strong> → ARMA(p,q) process</li>
                <li><strong>ACF decays very slowly</strong> → non-stationary, requires differencing</li>
              </ul>
              <p>The Ljung-Box Q-test checks whether autocorrelations up to lag h are collectively
              zero. Reject H₀ (p &lt; 0.05) → significant autocorrelation remains.</p>
            </div>""", unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Correlogram error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CORRELOGRAM MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def page_correlogram():
    st.markdown("""
    <p class="page-title">🔗 Correlogram Matrix</p>
    <p class="page-desc">Cross-correlation heatmap, scatter matrix, and pairwise correlation tests.</p>
    """, unsafe_allow_html=True)

    if st.session_state.clean_df is None:
        st.info("Load your dataset first.")
        return
    df   = st.session_state.clean_df
    cols = df.select_dtypes(include=np.number).columns.tolist()

    tab_cor, tab_scatter = st.tabs(["Correlation Heatmap","Scatter Matrix"])

    with tab_cor:
        c1,c2 = st.columns(2)
        sel_vars = c1.multiselect("Variables",cols,default=cols[:min(8,len(cols))])
        method   = c2.selectbox("Method",["pearson","spearman","kendall"])
        if sel_vars and len(sel_vars)>=2:
            corr = df[sel_vars].corr(method=method).round(4)
            fig  = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale=[[0,C["navy"]],[0.5,"#F1F5F9"],[1,C["cyan"]]],
                text=corr.values.round(2), texttemplate="%{text}",
                textfont=dict(size=10), zmin=-1, zmax=1,
            ))
            fig = navy_fig(fig,500,f"{method.title()} Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(corr.style.background_gradient(cmap="RdYlGn",vmin=-1,vmax=1).format("{:.4f}"),
                         use_container_width=True)
        else:
            st.info("Select at least 2 variables.")

    with tab_scatter:
        sel_s = st.multiselect("Variables for scatter",cols,default=cols[:min(5,len(cols))])
        if sel_s and len(sel_s)>=2:
            try:
                import plotly.express as px
                fig2 = px.scatter_matrix(
                    df[sel_s].dropna(), dimensions=sel_s,
                    color_discrete_sequence=[C["cyan"]],
                )
                fig2.update_traces(marker=dict(size=3,opacity=0.55,
                    line=dict(color=C["navy"],width=0.3)))
                fig2 = navy_fig(fig2,600,"Scatter Matrix")
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as exc:
                st.error(f"Scatter matrix error: {exc}")
        else:
            st.info("Select at least 2 variables.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OLS REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
def page_ols():
    st.markdown("""
    <p class="page-title">📐 OLS Regression</p>
    <p class="page-desc">Ordinary Least Squares with HC1 / HAC Newey-West robust standard errors,
    full inference, and automated diagnostic pre-check.</p>
    """, unsafe_allow_html=True)

    if st.session_state.clean_df is None:
        st.info("Load your dataset first.")
        return
    df   = st.session_state.clean_df
    cols = df.select_dtypes(include=np.number).columns.tolist()

    tab_cfg, tab_res, tab_diag = st.tabs(["① Configuration","② Results","③ Diagnostics"])

    with tab_cfg:
        c1,c2 = st.columns(2)
        dep    = c1.selectbox("Dependent Variable (Y)", cols)
        indep  = c2.multiselect("Independent Variables (X)", [c for c in cols if c!=dep])
        c3,c4,c5 = st.columns(3)
        add_const = c3.checkbox("Include Constant",value=True)
        se_type   = c4.selectbox("Standard Errors",["OLS (Classical)","HC1 (White Robust)","HAC (Newey-West)"])
        log_vars  = c5.multiselect("Apply ln() to", [dep]+indep, default=[])
        lags_dep  = st.slider("Include lags of dependent var (0=none)", 0, 8, 0)

        if st.button("▶  ESTIMATE OLS", use_container_width=True):
            if not indep:
                st.error("Select at least one independent variable.")
                st.session_state.menu = "ols"
                return
            try:
                data = df[[dep]+indep].dropna().copy()
                for col in log_vars:
                    if col in data.columns:
                        data[col] = np.log(data[col].replace(0,np.nan))
                data = data.dropna()
                lag_cols = []
                if lags_dep > 0:
                    for lag in range(1,lags_dep+1):
                        name = f"{dep}_L{lag}"
                        data[name] = data[dep].shift(lag)
                        lag_cols.append(name)
                    data = data.dropna()

                Y = data[dep]
                X_cols = indep + lag_cols
                X = data[X_cols]
                if add_const:
                    X = sm.add_constant(X)

                cov_map = {"OLS (Classical)":"nonrobust","HC1 (White Robust)":"HC1",
                           "HAC (Newey-West)":"HAC"}
                cov_type = cov_map.get(se_type,"HC1")
                if cov_type == "HAC":
                    model = OLS(Y,X).fit(cov_type="HAC",
                                          cov_kwds={"maxlags":int(np.ceil(len(Y)**(1/4)))})
                else:
                    model = OLS(Y,X).fit(cov_type=cov_type)

                st.session_state.ols_res    = model
                st.session_state.diag_model = model
                st.session_state.diag_resid = model.resid.values
                st.session_state.diag_fitted = model.fittedvalues.values
                st.session_state.diag_X     = X

                st.success("✓ OLS estimated successfully. View results in the **Results** tab.")
                st.session_state.report_log["ols"] = {
                    "dep":dep,"indep":X_cols,"n":int(model.nobs),
                    "r2":model.rsquared,"adj_r2":model.rsquared_adj,
                    "fstat":model.fvalue,"fp":model.f_pvalue,
                }
            except Exception as exc:
                st.error(f"**OLS estimation failed:** {exc}")

    with tab_res:
        model = st.session_state.ols_res
        if model is None:
            st.info("Configure and estimate OLS in the **Configuration** tab.")
            return
        # GoF metrics
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("R²",       f"{model.rsquared:.4f}")
        c2.metric("Adj. R²",  f"{model.rsquared_adj:.4f}")
        c3.metric("F-stat",   f"{model.fvalue:.4f}")
        c4.metric("Prob(F)",  f"{model.f_pvalue:.4f}")
        c5.metric("AIC",      f"{model.aic:.2f}")
        c6.metric("Obs (N)",  f"{int(model.nobs)}")
        c7,c8,c9,c10 = st.columns(4)
        c7.metric("Log-Lik",  f"{model.llf:.4f}")
        c8.metric("BIC",      f"{model.bic:.2f}")
        c9.metric("SSR",      f"{model.ssr:.6f}")
        c10.metric("DW",      f"{durbin_watson(model.resid.values):.4f}")

        # Equation
        dep_name = model.model.endog_names
        indep_names = [n for n in model.params.index if n != "const"]
        terms = []
        if "const" in model.params.index:
            terms.append(f"<span class='eq-gold'>{fmt(model.params['const'])}</span>")
        for n in indep_names:
            terms.append(f"<span class='eq-gold'>{fmt(model.params[n])}</span>·{n}")
        eq_str = " + ".join(terms)
        st.markdown(f"""
        <div class="eq-block">
          <p class="eq-muted">Estimated OLS Equation:</p>
          <p>{dep_name} = {eq_str} + ε</p>
          <p class="eq-muted">SE type: {se_type if 'se_type' in dir() else 'Robust'} &nbsp;|&nbsp;
             N = {int(model.nobs)} &nbsp;|&nbsp; R² = {model.rsquared:.4f}</p>
        </div>""", unsafe_allow_html=True)

        # Coefficient table
        st.markdown('<p class="sec-title">Coefficient Table</p>', unsafe_allow_html=True)
        ci = model.conf_int()
        rows_html = ""
        for vname in model.params.index:
            rows_html += coef_row_html(
                vname, model.params[vname], model.bse[vname],
                model.tvalues[vname], model.pvalues[vname],
                ci.loc[vname,0], ci.loc[vname,1],
            )
        st.markdown(coef_table_html(rows_html), unsafe_allow_html=True)

        # Residual plot
        fig = make_subplots(rows=1,cols=2,subplot_titles=["Residuals vs Fitted","Normal Q-Q"])
        fig.add_trace(go.Scatter(x=model.fittedvalues.values, y=model.resid.values,
            mode="markers", marker=dict(color=C["cyan"],size=5,opacity=0.65),name="Residuals"),1,1)
        fig.add_hline(y=0,line=dict(color=C["navy"],dash="dash"),row=1,col=1)
        osm,osr = sci_stats.probplot(model.resid.values,dist="norm")[:2]
        slope,intercept,_,_,_ = sci_stats.linregress(osm[0],osm[1])
        fig.add_trace(go.Scatter(x=osm[0],y=osm[1],mode="markers",
            marker=dict(color=C["cyan"],size=4,opacity=0.65),name="Quantiles"),1,2)
        x_line = [min(osm[0]),max(osm[0])]
        fig.add_trace(go.Scatter(x=x_line,y=[slope*x+intercept for x in x_line],
            line=dict(color=C["gold"],width=2,dash="dash"),name="Normal line"),1,2)
        fig = navy_fig(fig,350)
        st.plotly_chart(fig, use_container_width=True)

        # AI interpretation
        ai_ols = _interpret_ols(model, dep_name, indep_names)
        st.markdown(f'<div class="ai-box"><h4>⬡ AI INTERPRETATION · OLS</h4>{ai_ols}</div>',
                    unsafe_allow_html=True)

    with tab_diag:
        model = st.session_state.ols_res
        if model is None:
            st.info("Estimate OLS first.")
            return
        _run_ols_diagnostics(model)


def _interpret_ols(model, dep, indep):
    sig   = [n for n in indep if model.pvalues.get(n,1)<0.05]
    insig = [n for n in indep if model.pvalues.get(n,1)>=0.05]
    qual  = "strong" if model.rsquared_adj>0.7 else "moderate" if model.rsquared_adj>0.4 else "weak"
    f_ok  = model.f_pvalue < 0.05
    lines = [f"<p>The OLS model regresses <strong>{dep}</strong> on {len(indep)} regressors. "
             f"The model explains <strong>{model.rsquared*100:.2f}%</strong> of variance in {dep} "
             f"(Adj. R²={model.rsquared_adj:.4f}), indicating <em>{qual}</em> overall fit. "
             f"The F-test is {'significant (jointly the regressors matter)' if f_ok else 'not significant'}.</p>"]
    for n in indep[:6]:
        c = model.params.get(n,0); p = model.pvalues.get(n,1)
        lines.append(f"<p>• <strong>{n}</strong> (β={fmt(c)}, p={fmt(p)}): "
                     f"a 1-unit increase in {n} is associated with a "
                     f"<strong>{fmt(c)}</strong>-unit {'increase' if c>0 else 'decrease'} in {dep}. "
                     f"{'<span class=\"ok-text\">Statistically significant at 5%.</span>' if p<0.05 else '<span class=\"warn-text\">Not significant at 5%.</span>'}</p>")
    if insig:
        lines.append(f"<p><span class='warn-text'>⚠ Variables not significant at 5%:</span> "
                     f"{', '.join(insig)}. Consider dropping or testing joint restrictions.</p>")
    return "".join(lines)


def _run_ols_diagnostics(model):
    st.markdown('<p class="sec-title">Quick Diagnostic Summary</p>', unsafe_allow_html=True)
    resid = model.resid.values
    X_exog = model.model.exog
    results_d = {}

    c1,c2,c3,c4 = st.columns(4)
    # DW
    dw = durbin_watson(resid)
    dw_ok = 1.5 < dw < 2.5
    c1.markdown(f"""<div class="stat-box"><p class="stat-name">Durbin-Watson</p>
    <p class="stat-val">{dw:.4f}</p>
    <p class="stat-sub">{'✅ No autocorr.' if dw_ok else '❌ Autocorr. detected'}</p></div>""",
    unsafe_allow_html=True)
    results_d["dw"] = dw_ok

    # JB
    jb_stat,jb_p = jarque_bera(resid)
    jb_ok = jb_p > 0.05
    c2.markdown(f"""<div class="stat-box"><p class="stat-name">Jarque-Bera</p>
    <p class="stat-val">{jb_p:.4f}</p>
    <p class="stat-sub">{'✅ Normality OK' if jb_ok else '⚠ Non-normal'}</p></div>""",
    unsafe_allow_html=True)
    results_d["jb"] = jb_ok

    # White
    try:
        wh = het_white(resid, X_exog)
        wh_ok = wh[1] > 0.05
        c3.markdown(f"""<div class="stat-box"><p class="stat-name">White's Test (p)</p>
        <p class="stat-val">{wh[1]:.4f}</p>
        <p class="stat-sub">{'✅ Homoskedastic' if wh_ok else '❌ Heteroskedasticity'}</p></div>""",
        unsafe_allow_html=True)
        results_d["white"] = wh_ok
    except Exception:
        c3.markdown('<div class="stat-box"><p class="stat-name">White\'s Test</p>'
                    '<p class="stat-val">N/A</p></div>', unsafe_allow_html=True)

    # BG
    try:
        bg = acorr_breusch_godfrey(model, nlags=4)
        bg_ok = bg[1] > 0.05
        c4.markdown(f"""<div class="stat-box"><p class="stat-name">Breusch-Godfrey (p)</p>
        <p class="stat-val">{bg[1]:.4f}</p>
        <p class="stat-sub">{'✅ No autocorr.' if bg_ok else '❌ Serial correlation'}</p></div>""",
        unsafe_allow_html=True)
        results_d["bg"] = bg_ok
    except Exception:
        c4.markdown('<div class="stat-box"><p class="stat-name">BG Test</p>'
                    '<p class="stat-val">N/A</p></div>', unsafe_allow_html=True)

    # RESET
    try:
        reset = linear_reset(model, power=3, use_f=True)
        res_ok = reset.pvalue > 0.05
        st.markdown(f"""<div class="glass-card">
        <p class="sec-title" style="font-size:0.82rem;">Ramsey RESET Test (Misspecification)</p>
        <p>F-stat: <strong>{reset.statistic:.4f}</strong> &nbsp;|&nbsp;
           p-value: <strong>{reset.pvalue:.4f}</strong> &nbsp;|&nbsp;
           {badge_html('No Misspecification','pass') if res_ok else badge_html('Misspecification Likely','fail')}
        </p>
        <p style="font-size:0.75rem;color:#64748B;">H₀: Model is correctly specified (no omitted non-linearities)</p>
        </div>""", unsafe_allow_html=True)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ARDL
# ─────────────────────────────────────────────────────────────────────────────
def page_ardl():
    st.markdown("""
    <p class="page-title">🔄 ARDL / Bounds Test</p>
    <p class="page-desc">Autoregressive Distributed Lag · PSS Bounds Test for cointegration ·
    Short-run dynamics · Long-run multipliers · Error Correction Term (ECT)</p>
    """, unsafe_allow_html=True)

    if st.session_state.clean_df is None:
        st.info("Load data first.")
        return
    df   = st.session_state.clean_df
    cols = df.select_dtypes(include=np.number).columns.tolist()

    tab_cfg, tab_res, tab_bounds = st.tabs(["① Config","② Results","③ Bounds Test & ECM"])

    with tab_cfg:
        c1,c2 = st.columns(2)
        dep   = c1.selectbox("Dependent Variable (Y)", cols)
        indep = c2.multiselect("Independent Variables (X)", [c for c in cols if c!=dep])
        c3,c4,c5 = st.columns(3)
        max_lag_y = c3.slider("Max lags Y", 1, 8, 4)
        max_lag_x = c4.slider("Max lags X", 0, 8, 4)
        ic_crit   = c5.selectbox("Lag criterion", ["aic","bic","hqic"])
        trend_opt = st.selectbox("Trend", ["c (constant)","ct (const+trend)","nc (none)"])
        trend_code = trend_opt.split(" ")[0]

        if st.button("▶  ESTIMATE ARDL", use_container_width=True):
            if not indep:
                st.error("Select at least one X variable.")
                return
            try:
                data = df[[dep]+indep].dropna()
                sel_order = ardl_select_order(
                    data[dep], max_lag_y, data[indep], max_lag_x,
                    ic=ic_crit, trend=trend_code,
                )
                best = sel_order.ardl_order
                model = ARDL(data[dep], best[0], data[indep], best[1:], trend=trend_code).fit()
                st.session_state.ardl_res = {"model":model,"dep":dep,"indep":indep,
                                              "order":best,"ic":ic_crit,"trend":trend_code,
                                              "data":data}
                st.session_state.diag_model  = model
                st.session_state.diag_resid  = model.resid.values
                st.session_state.diag_fitted = model.fittedvalues.values
                st.success(f"✓ ARDL({', '.join(str(o) for o in best)}) estimated. Check **Results** tab.")
            except Exception as exc:
                st.error(f"**ARDL error:** {exc}")

    with tab_res:
        ar = st.session_state.ardl_res
        if ar is None:
            st.info("Estimate ARDL first.")
            return
        model = ar["model"]; dep = ar["dep"]; indep = ar["indep"]; best = ar["order"]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("AIC",   f"{model.aic:.3f}")
        c2.metric("BIC",   f"{model.bic:.3f}")
        c3.metric("Log-Lik",f"{model.llf:.3f}")
        c4.metric("N",     f"{int(model.nobs)}")

        p = best[0]
        x_lags_str = ", ".join(str(v) for v in best[1:]) if len(best)>1 else "—"
        st.markdown(f"""
        <div class="eq-block">
          <p class="eq-muted">ARDL({p}, {x_lags_str}) — {ic_crit.upper()} selected</p>
          <p>{dep}(t) = c + Σαᵢ·{dep}(t-i) + ΣΣβⱼ·Xⱼ(t-k) + ε(t)</p>
          <p class="eq-muted">Trend: {ar['trend']} | Vars: {dep}, {', '.join(indep)}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown('<p class="sec-title">Coefficient Table</p>', unsafe_allow_html=True)
        ci = model.conf_int()
        rows_html = ""
        for vname in model.params.index:
            rows_html += coef_row_html(vname, model.params[vname], model.bse[vname],
                                       model.tvalues[vname], model.pvalues[vname],
                                       ci.loc[vname,0], ci.loc[vname,1])
        st.markdown(coef_table_html(rows_html), unsafe_allow_html=True)

        # Long-run multipliers
        st.markdown('<p class="sec-title">Long-Run Multipliers</p>', unsafe_allow_html=True)
        try:
            lr_params = model.params
            lag_sum = sum(lr_params.get(f"L{i}.{dep}",lr_params.get(f"{dep}.L{i}",0))
                          for i in range(1,p+1))
            denom = max(1e-10, abs(1 - lag_sum))
            lr_rows = ""
            for v in indep:
                x_sum = sum(lr_params.get(f"L{j}.{v}",lr_params.get(v,0) if j==0 else 0)
                            for j in range(0, best[indep.index(v)+1]+1 if len(best)>indep.index(v)+1 else 1))
                lr_c = x_sum / denom
                lr_rows += f"<tr><td>{v}</td><td>{fmt(lr_c)}</td><td>Long-run multiplier</td></tr>"
            st.markdown(f"""
            <div class="coef-wrap"><table class="coef-tbl">
              <tr><th>Variable</th><th>Long-Run Coef.</th><th>Interpretation</th></tr>
              {lr_rows}
            </table></div>""", unsafe_allow_html=True)
        except Exception as exc:
            st.warning(f"Long-run derivation: {exc}")

        ai_ardl = _interpret_ardl(model, dep, indep, best, ic_crit)
        st.markdown(f'<div class="ai-box"><h4>⬡ AI INTERPRETATION · ARDL</h4>{ai_ardl}</div>',
                    unsafe_allow_html=True)

    with tab_bounds:
        ar = st.session_state.ardl_res
        if ar is None:
            st.info("Estimate ARDL first.")
            return
        model = ar["model"]
        st.markdown('<p class="sec-title">PSS Bounds Test for Cointegration</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="ai-box" style="margin-bottom:14px;">
          <h4>⬡ BOUNDS TEST THEORY</h4>
          <p>The Pesaran, Shin & Smith (2001) Bounds Test tests H₀: no long-run relationship.
          The F-statistic is compared to two asymptotic critical value bounds:</p>
          <ul>
            <li><strong>I(0) lower bound:</strong> all regressors are stationary in levels</li>
            <li><strong>I(1) upper bound:</strong> all regressors are integrated of order 1</li>
          </ul>
          <p>If F > I(1) bound → cointegration regardless of integration order.
          If F &lt; I(0) bound → no cointegration. If in between → inconclusive.</p>
        </div>""", unsafe_allow_html=True)
        try:
            bt = model.bounds_test(n_obs=len(ar["data"]), case=3, alpha=0.05)
            f_stat = bt.stat
            conc   = bt.conclusion
            badge_kind = "pass" if conc=="cointegration" else "warn" if conc=="inconclusive" else "fail"
            conclusion_str = {
                "cointegration":"COINTEGRATION CONFIRMED",
                "inconclusive":"INCONCLUSIVE — more evidence needed",
                "no cointegration":"NO COINTEGRATION",
            }.get(conc, conc.upper())
            st.markdown(f"""
            <div class="brutalist-card-gold">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <p style="font-weight:700;font-size:0.95rem;margin:0;">PSS Bounds Test Result</p>
                {badge_html(conclusion_str, badge_kind)}
              </div>
              <p style="font-family:'Space Mono',monospace;font-size:1.1rem;margin-top:8px;">
                F-statistic: <strong>{fmt(f_stat)}</strong></p>
            </div>""", unsafe_allow_html=True)
        except Exception as exc:
            st.warning(f"Bounds test note: {exc}\n\nThe PSS bounds test may require specific model configurations.")

        # ECM
        st.markdown('<p class="sec-title">Error Correction Representation (ECM)</p>', unsafe_allow_html=True)
        try:
            lr_params = model.params
            p = ar["order"][0]
            lag_sum = sum(lr_params.get(f"L{i}.{ar['dep']}",
                          lr_params.get(f"{ar['dep']}.L{i}",0)) for i in range(1,p+1))
            ect_coef = -(1 - lag_sum)
            ecm_interp = _interpret_ect(ect_coef)
            st.markdown(f"""
            <div class="eq-block">
              <p class="eq-muted">Error Correction Term (ECT / Speed of Adjustment):</p>
              <p>Δ{ar['dep']}(t) = <span class='eq-gold'>{fmt(ect_coef)}</span>·ECTₜ₋₁ + short-run dynamics + ε(t)</p>
              <p class="eq-muted">ECT = {ar['dep']}(t-1) − [long-run cointegrating relationship]</p>
            </div>""", unsafe_allow_html=True)
            st.markdown(f'<div class="ai-box"><h4>⬡ ECT INTERPRETATION</h4>{ecm_interp}</div>',
                        unsafe_allow_html=True)
        except Exception as exc:
            st.warning(f"ECM derivation: {exc}")


def _interpret_ardl(model, dep, indep, order, ic):
    p = order[0]
    sig = [n for n in model.params.index if model.pvalues.get(n,1)<0.05 and n!="const"]
    return (f"<p>The <strong>ARDL({', '.join(str(o) for o in order)})</strong> model was selected "
            f"via <strong>{ic.upper()}</strong>. It includes {p} lag(s) of <em>{dep}</em> and "
            f"distributed lags of {', '.join(indep)}.</p>"
            f"<p>Short-run significant terms (p&lt;0.05): {', '.join(sig[:6]) if sig else 'none at 5%'}.</p>"
            f"<p>The long-run multipliers above measure the total effect of a unit change in each X "
            f"variable on {dep} once all dynamic adjustments have been completed. "
            f"Divide by (1 − Σαᵢ) where αᵢ are the lagged Y coefficients.</p>"
            f"<p>To assess whether a stable long-run relationship exists, examine the "
            f"<strong>PSS Bounds Test</strong> in the next tab.</p>")


def _interpret_ect(ect):
    pct = abs(ect)*100
    direction = "toward" if ect < 0 else "away from"
    speed = "rapid" if pct>50 else "moderate" if pct>20 else "slow"
    validity = ect < 0
    lines = [f"<p>The Error Correction Term (ECT = <span class='highlight'>{fmt(ect)}</span>) "
             f"suggests that approximately <strong>{pct:.1f}%</strong> of any short-run "
             f"deviation from long-run equilibrium is corrected each period, "
             f"as the system moves <em>{direction}</em> equilibrium.</p>",
             f"<p>Speed of adjustment is <em>{speed}</em>. "]
    if validity:
        lines.append("<span class='ok-text'>✅ Negative ECT confirms error-correcting behaviour — "
                     "the variable returns to long-run equilibrium after a shock.</span></p>")
    else:
        lines.append("<span class='bad-text'>❌ Positive ECT indicates explosive/diverging dynamics — "
                     "the system moves away from equilibrium. Check cointegration assumption.</span></p>")
    return "".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: VAR
# ─────────────────────────────────────────────────────────────────────────────
def page_var():
    st.markdown("""
    <p class="page-title">🌐 VAR System</p>
    <p class="page-desc">Vector Autoregression · Optimal lag selection · Granger causality ·
    Impulse Response Functions · Forecast Error Variance Decomposition</p>
    """, unsafe_allow_html=True)

    if st.session_state.clean_df is None:
        st.info("Load data first.")
        return
    df   = st.session_state.clean_df
    cols = df.select_dtypes(include=np.number).columns.tolist()

    tab_cfg, tab_lag, tab_res, tab_irf, tab_fevd, tab_granger = st.tabs(
        ["① Config","② Lag Selection","③ Results","④ IRF","⑤ FEVD","⑥ Granger"])

    with tab_cfg:
        endog = st.multiselect("Endogenous Variables (2–8)", cols)
        c1,c2 = st.columns(2)
        max_lags = c1.slider("Max lags to test", 1, 12, 6)
        ic_var   = c2.selectbox("Information Criterion", ["aic","bic","hqic","fpe"])
        trend_v  = st.selectbox("Trend", ["c","ct","ctt","n"], index=0)

        if st.button("▶  ESTIMATE VAR", use_container_width=True):
            if len(endog) < 2:
                st.error("Select at least 2 endogenous variables.")
                return
            try:
                data_v = df[endog].dropna()
                var_m  = VAR(data_v)
                ic_sel = var_m.select_order(max_lags)
                best_p = getattr(ic_sel, ic_var)
                best_p = max(1, best_p)
                fitted = var_m.fit(best_p, trend=trend_v)
                st.session_state.var_res = {"model":fitted,"endog":endog,
                                             "best_p":best_p,"ic":ic_var,"data":data_v}
                st.success(f"✓ VAR({best_p}) estimated via {ic_var.upper()}.")
            except Exception as exc:
                st.error(f"**VAR error:** {exc}")

    with tab_lag:
        if st.session_state.var_res is None:
            st.info("Estimate VAR first.")
        else:
            vr = st.session_state.var_res
            try:
                data_v = vr["data"]
                var_m  = VAR(data_v)
                ic_sel = var_m.select_order(8)
                ic_df  = pd.DataFrame({
                    "AIC":  [getattr(ic_sel,"aic_by_lag",{}).get(i,np.nan) for i in range(9)],
                    "BIC":  [getattr(ic_sel,"bic_by_lag",{}).get(i,np.nan) for i in range(9)],
                    "HQIC": [getattr(ic_sel,"hqic_by_lag",{}).get(i,np.nan) for i in range(9)],
                }, index=range(9))
                st.markdown('<p class="sec-title">Information Criteria by Lag</p>', unsafe_allow_html=True)
                st.dataframe(ic_df.style.format("{:.4f}").highlight_min(color="#D1FAE5"), use_container_width=True)
                best_each = {
                    "AIC": ic_sel.aic, "BIC": ic_sel.bic, "HQIC": ic_sel.hqic
                }
                c1,c2,c3 = st.columns(3)
                c1.metric("Optimal (AIC)",  f"p = {best_each.get('AIC','—')}")
                c2.metric("Optimal (BIC)",  f"p = {best_each.get('BIC','—')}")
                c3.metric("Optimal (HQIC)", f"p = {best_each.get('HQIC','—')}")
            except Exception as exc:
                st.warning(f"Lag selection table: {exc}")

    with tab_res:
        vr = st.session_state.var_res
        if vr is None:
            st.info("Estimate VAR first.")
            return
        fitted = vr["model"]; endog = vr["endog"]; best_p = vr["best_p"]
        c1,c2,c3 = st.columns(3)
        c1.metric("VAR Order (p)", best_p)
        c2.metric("AIC",  f"{fitted.aic:.4f}")
        c3.metric("BIC",  f"{fitted.bic:.4f}")
        k = len(endog)
        st.markdown(f"""
        <div class="eq-block">
          <p class="eq-muted">VAR({best_p}) System — {k} variables:</p>
          <p>Y(t) = c + A₁·Y(t-1) + A₂·Y(t-2) + … + A{best_p}·Y(t-{best_p}) + ε(t)</p>
          <p class="eq-muted">where Y(t) = [{', '.join(endog)}]ᵀ</p>
        </div>""", unsafe_allow_html=True)
        for eq_var in endog:
            with st.expander(f"▸ Equation: {eq_var}"):
                try:
                    p_df   = fitted.params[eq_var]
                    se_df  = fitted.stderr_endog_lagged
                    rows_h = ""
                    for vname in p_df.index:
                        c_val = p_df[vname]
                        rows_h += f"<tr><td>{vname}</td><td>{fmt(c_val)}</td></tr>"
                    st.markdown(f"""<table class="coef-tbl">
                    <tr><th>Regressor</th><th>Coef. ({eq_var})</th></tr>{rows_h}</table>""",
                    unsafe_allow_html=True)
                except Exception as exc:
                    st.warning(f"Equation display: {exc}")

    with tab_irf:
        vr = st.session_state.var_res
        if vr is None:
            st.info("Estimate VAR first.")
            return
        fitted = vr["model"]; endog = vr["endog"]
        c1,c2,c3 = st.columns(3)
        imp_var  = c1.selectbox("Impulse (shock in)", endog, key="irf_imp")
        resp_var = c2.selectbox("Response variable",  endog, key="irf_resp")
        periods  = c3.slider("Forecast periods", 5, 40, 20)
        orth     = st.checkbox("Orthogonalized IRF (Cholesky)", value=True)

        if st.button("▶  COMPUTE IRF", use_container_width=True):
            try:
                irf    = fitted.irf(periods)
                imp_i  = endog.index(imp_var)
                resp_i = endog.index(resp_var)
                if orth:
                    irf_vals = irf.orth_irfs[:,resp_i,imp_i]
                    try:
                        lo = irf.orth_cum_effects[:,resp_i,imp_i]
                    except Exception:
                        lo = None
                else:
                    irf_vals = irf.irfs[:,resp_i,imp_i]
                    lo = None

                period_x = list(range(len(irf_vals)))
                fig = go.Figure()
                if lo is not None:
                    ci_up = irf_vals + 1.96*np.std(irf_vals)
                    ci_dn = irf_vals - 1.96*np.std(irf_vals)
                    fig.add_trace(go.Scatter(
                        x=period_x+period_x[::-1],
                        y=list(ci_up)+list(ci_dn[::-1]),
                        fill="toself", fillcolor="rgba(56,189,248,0.12)",
                        line=dict(color="rgba(0,0,0,0)"), name="Approx. 95% CI"))
                fig.add_trace(go.Scatter(x=period_x, y=irf_vals,
                    line=dict(color=C["cyan"],width=2.5), name="IRF"))
                fig.add_hline(y=0, line=dict(color=C["navy"],dash="dash",width=1))
                fig = navy_fig(fig,380,
                    f"IRF: Response of {resp_var} to 1-SD shock in {imp_var} ({'Orth.' if orth else 'Raw'})")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"IRF error: {exc}")

    with tab_fevd:
        vr = st.session_state.var_res
        if vr is None:
            st.info("Estimate VAR first.")
            return
        fitted = vr["model"]; endog = vr["endog"]
        periods_f = st.slider("FEVD horizon", 5, 40, 20, key="fevd_h")
        if st.button("▶  COMPUTE FEVD", use_container_width=True):
            try:
                fevd    = fitted.fevd(periods_f)
                fevd_df = fevd.decomp
                target  = st.selectbox("Variable to decompose", endog, key="fevd_var")
                t_idx   = endog.index(target)
                decomp  = fevd_df[:,t_idx,:]
                fig = go.Figure()
                for i,src in enumerate(endog):
                    fig.add_trace(go.Scatter(
                        x=list(range(periods_f)), y=decomp[:,i]*100,
                        name=src, mode="lines",
                        stackgroup="one",
                        line=dict(color=CHART_COLORS[i%len(CHART_COLORS)],width=1.5),
                    ))
                fig = navy_fig(fig,380,f"FEVD: {target} variance explained by each shock (%)")
                fig.update_layout(yaxis_title="% of Variance")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"FEVD error: {exc}")

    with tab_granger:
        vr = st.session_state.var_res
        if vr is None:
            st.info("Estimate VAR first.")
            return
        fitted = vr["model"]; endog = vr["endog"]; data_v = vr["data"]
        max_gr_lag = st.slider("Max lags for Granger test", 1, 8, vr.get("best_p",4))
        alpha_gr   = st.select_slider("Significance α",[0.01,0.05,0.10],value=0.05, key="gr_a")
        if st.button("▶  GRANGER CAUSALITY MATRIX", use_container_width=True):
            st.markdown('<p class="sec-title">Granger Causality p-values</p>', unsafe_allow_html=True)
            p_matrix = np.ones((len(endog),len(endog)))
            try:
                for i,caused in enumerate(endog):
                    for j,causing in enumerate(endog):
                        if i==j: continue
                        try:
                            gc = grangercausalitytests(
                                data_v[[caused,causing]].dropna(),
                                maxlag=max_gr_lag, verbose=False)
                            min_p = min(gc[lag][0]["ssr_ftest"][1] for lag in gc)
                            p_matrix[i,j] = min_p
                        except Exception:
                            p_matrix[i,j] = np.nan
                granger_df = pd.DataFrame(p_matrix, index=endog, columns=endog).round(4)
                fig = go.Figure(go.Heatmap(
                    z=p_matrix, x=endog, y=endog,
                    colorscale=[[0,C["green"]],[0.05,"#FEF3C7"],[0.1,"#FEE2E2"],[1,"#FEE2E2"]],
                    text=np.round(p_matrix,3), texttemplate="%{text}",
                    zmin=0, zmax=0.15,
                ))
                fig = navy_fig(fig,380,"Granger Causality p-value Matrix (row → caused by col)")
                fig.update_layout(annotations=[dict(
                    x=0.5,y=-0.12,xref="paper",yref="paper",showarrow=False,
                    text="Green = significant Granger causality (p<α)",
                    font=dict(size=10,color=C["slate"]))])
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(granger_df.style.background_gradient(cmap="RdYlGn_r",vmin=0,vmax=0.15)
                             .format("{:.4f}"), use_container_width=True)
            except Exception as exc:
                st.error(f"Granger error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: JOHANSEN / VECM
# ─────────────────────────────────────────────────────────────────────────────
def page_vecm():
    st.markdown("""
    <p class="page-title">⛓ Johansen Cointegration & VECM</p>
    <p class="page-desc">Johansen Trace & Max-Eigenvalue tests · Cointegrating vectors β ·
    Loading matrix α · Vector Error Correction Model dynamics</p>
    """, unsafe_allow_html=True)

    if st.session_state.clean_df is None:
        st.info("Load data first.")
        return
    df   = st.session_state.clean_df
    cols = df.select_dtypes(include=np.number).columns.tolist()

    tab_joh, tab_vecm = st.tabs(["① Johansen Test","② VECM Estimation"])

    with tab_joh:
        endog_j = st.multiselect("I(1) Variables (min 2)", cols, key="joh_vars")
        c1,c2   = st.columns(2)
        k_ar    = c1.slider("Lags in VECM (k)", 1, 8, 2)
        det     = c2.selectbox("Deterministics", ["ci","li","n"], index=0,
            help="ci=const in CI, li=linear trend in CI, n=none")
        alpha_j = st.select_slider("Significance α",[0.01,0.05,0.10],value=0.05,key="joh_a")

        if st.button("▶  RUN JOHANSEN TEST", use_container_width=True):
            if len(endog_j) < 2:
                st.error("Select at least 2 variables.")
                return
            try:
                data_j = df[endog_j].dropna()
                rank_j = select_coint_rank(data_j, det=det, k_ar_diff=k_ar,
                                           method="trace", signif=alpha_j)
                st.session_state.johansen_res = {
                    "rank_test":rank_j,"endog":endog_j,
                    "k_ar":k_ar,"det":det,"data":data_j,
                }
                st.success(f"✓ Johansen test complete. Cointegration rank r = {rank_j.rank}")
            except Exception as exc:
                st.error(f"**Johansen error:** {exc}")

        jr = st.session_state.johansen_res
        if jr is None: return

        rank_test = jr["rank_test"]
        endog_j   = jr["endog"]
        st.markdown('<p class="sec-title">Johansen Test Results</p>', unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        c1.metric("Estimated Rank (r)", rank_test.rank)
        c2.metric("# Variables (k)",    len(endog_j))

        # Trace statistic table
        try:
            trace_stat  = rank_test.test_stats
            crit_vals   = rank_test.crit_vals
            rows_j = ""
            for i,(ts,cv) in enumerate(zip(trace_stat,crit_vals)):
                ok_row = ts > cv
                rows_j += (f"<tr><td>r ≤ {i}</td>"
                           f"<td>{fmt(ts)}</td><td>{fmt(cv)}</td>"
                           f"<td>{badge_html('Reject H₀','pass') if ok_row else badge_html('Fail to Reject','fail')}</td></tr>")
            st.markdown(f"""
            <div class="coef-wrap"><table class="coef-tbl">
              <tr><th>H₀</th><th>Test Statistic</th><th>Critical Value ({int(alpha_j*100)}%)</th><th>Decision</th></tr>
              {rows_j}
            </table></div>""", unsafe_allow_html=True)
        except Exception as exc:
            st.warning(f"Johansen table: {exc}")

        joh_ai = _interpret_johansen(rank_test.rank, len(endog_j))
        st.markdown(f'<div class="ai-box"><h4>⬡ AI INTERPRETATION · JOHANSEN</h4>{joh_ai}</div>',
                    unsafe_allow_html=True)

    with tab_vecm:
        jr = st.session_state.johansen_res
        if jr is None:
            st.info("Run Johansen test first.")
            return

        c1,c2 = st.columns(2)
        r_rank = c1.number_input("Cointegration Rank (r)",
            min_value=1, max_value=len(jr["endog"])-1, value=max(1,jr["rank_test"].rank))
        k_diff = c2.slider("Lags in differences (k)", 1, 8, jr["k_ar"])

        if st.button("▶  ESTIMATE VECM", use_container_width=True):
            try:
                data_v = jr["data"]
                vecm   = VECM(data_v, k_ar_diff=k_diff, coint_rank=int(r_rank),
                              deterministic=jr["det"])
                fitted_v = vecm.fit()
                st.session_state.vecm_res = {
                    "model":fitted_v,"endog":jr["endog"],
                    "rank":int(r_rank),"k":k_diff,
                }
                st.success(f"✓ VECM estimated with r={r_rank} cointegrating vectors.")
            except Exception as exc:
                st.error(f"**VECM error:** {exc}")

        vr = st.session_state.vecm_res
        if vr is None: return

        fitted_v = vr["model"]; endog_v = vr["endog"]; r = vr["rank"]
        st.markdown('<p class="sec-title">Cointegrating Vectors β (normalized)</p>', unsafe_allow_html=True)
        try:
            beta = fitted_v.beta
            beta_df = pd.DataFrame(
                beta[:len(endog_v),:],
                index=endog_v[:beta.shape[0]],
                columns=[f"CI Vector {i+1}" for i in range(r)],
            )
            st.dataframe(beta_df.style.format("{:.6f}"), use_container_width=True)
        except Exception as exc:
            st.warning(f"β matrix: {exc}")

        st.markdown('<p class="sec-title">Loading Matrix α (adjustment coefficients)</p>', unsafe_allow_html=True)
        try:
            alpha_m = fitted_v.alpha
            alpha_df = pd.DataFrame(
                alpha_m,
                index=endog_v[:alpha_m.shape[0]],
                columns=[f"CI Vector {i+1}" for i in range(r)],
            )
            st.dataframe(alpha_df.style.format("{:.6f}"), use_container_width=True)
        except Exception as exc:
            st.warning(f"α matrix: {exc}")

        st.markdown(f"""
        <div class="eq-block">
          <p class="eq-muted">VECM({k_diff}) with r={r} cointegrating vector(s):</p>
          <p>ΔY(t) = <span class='eq-gold'>αβᵀ</span>·Y(t-1) + Γ₁·ΔY(t-1) + … + Γ{k_diff}·ΔY(t-{k_diff}) + ε(t)</p>
          <p class="eq-muted">α = loading matrix ({', '.join(endog_v[:3])}…) | β = cointegrating vectors</p>
        </div>""", unsafe_allow_html=True)

        vecm_ai = _interpret_vecm(fitted_v, endog_v, r)
        st.markdown(f'<div class="ai-box"><h4>⬡ AI INTERPRETATION · VECM</h4>{vecm_ai}</div>',
                    unsafe_allow_html=True)


def _interpret_johansen(rank, k):
    if rank == 0:
        return ("<p><span class='bad-text'>No cointegrating relationships detected</span> "
                f"among the {k} variables. If all variables are I(1), this suggests they drift "
                "independently. Consider an unrestricted VAR in differences, or re-examine data.</p>")
    elif rank == k:
        return (f"<p><span class='warn-text'>Rank = number of variables ({k})</span> — "
                "this implies all variables are stationary in levels (I(0)). "
                "OLS in levels is valid; VECM is not the appropriate model.</p>")
    else:
        return (f"<p><span class='ok-text'>✅ Johansen identifies r = {rank} cointegrating vector(s)</span> "
                f"among the {k} variables. A VECM({rank}) is the appropriate model specification. "
                f"The {rank} linear combination(s) of the variables are stationary, providing "
                "{k-rank} common stochastic trend(s) driving the system.</p>"
                "<p>The normalized β vectors above define the long-run equilibrium relationships. "
                "The α loading matrix measures how fast each variable adjusts when deviating "
                "from the long-run equilibrium. A negative α indicates corrective (stable) behaviour.</p>")


def _interpret_vecm(model, endog, r):
    try:
        alpha = model.alpha
        adj_strs = []
        for i,v in enumerate(endog[:alpha.shape[0]]):
            a = alpha[i,0] if r>0 else 0
            speed = abs(a)*100
            direction = "corrects toward" if a<0 else "diverges from"
            adj_strs.append(f"<li><strong>{v}</strong>: α={fmt(a)} → {speed:.1f}% per period adjustment, "
                            f"{direction} equilibrium {'✅' if a<0 else '❌'}</li>")
        return ("<p>The VECM decomposes the long-run cointegration from short-run dynamics.</p>"
                f"<p><strong>Adjustment speeds (α):</strong></p><ul>{''.join(adj_strs)}</ul>"
                "<p>Negative α values confirm that variables are error-correcting (stable system). "
                "If α is not statistically significant, that variable is 'weakly exogenous' with "
                "respect to the long-run relationship.</p>")
    except Exception:
        return "<p>VECM results available above.</p>"


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: GARCH / VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# PLACEHOLDER FOR UNWRITTEN PAGES
# ─────────────────────────────────────────────────────────────────────────────
def page_under_construction():
    st.markdown(f"""
    <div style="text-align:center; padding: 80px 20px;">
        <p style="font-size:3rem; margin-bottom:10px;">🚧</p>
        <p style="font-family:'Orbitron', monospace; font-size:1.5rem; color:{C['cyan']};">MODULE AWAITING DEPLOYMENT</p>
        <p style="color:{C['slate']};">The <strong>{st.session_state.menu.upper()}</strong> module will be pushed in the next codebase update.</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.authenticated:
        render_login()
        return

    render_sidebar()
    render_topbar()

    menu = st.session_state.menu

    # Route to the correct page function
    if menu == "home":            page_home()
    elif menu == "data":          page_data()
    elif menu == "transform":     page_transform()
    elif menu == "unitroot":      page_unitroot()
    elif menu == "acf_pacf":      page_acf_pacf()
    elif menu == "correlogram":   page_correlogram()
    elif menu == "ols":           page_ols()
    elif menu == "ardl":          page_ardl()
    elif menu == "var":           page_var()
    elif menu == "vecm":          page_vecm()
    else:                         page_under_construction()

if __name__ == "__main__":
    main()
