# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  NEXUS KERNEL v3.0 — Professional Time-Series Econometrics Platform         ║
# ║  Research by Ahmed Hisham  ·  © 2026  ·  Production Release                ║
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
    page_title="NEXUS KERNEL v3.0", page_icon="⬡", layout="wide",
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
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
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
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak,
)

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "navy":"#0F172A","navy2":"#1E293B","navy3":"#0A1020",
    "cyan":"#38BDF8","cyan2":"#0EA5E9","gold":"#F59E0B","gold2":"#D97706",
    "teal":"#14B8A6","green":"#10B981","red":"#EF4444","orange":"#F97316",
    "purple":"#A78BFA","pink":"#F472B6","slate":"#64748B","canvas":"#E8EEFF",
    "card":"#F0F4FF","border":"#C7D2FE","muted":"#94A3B8",
}
CHART_COLORS = [C["cyan"],C["gold"],C["teal"],C["red"],C["green"],C["purple"],C["orange"],C["pink"]]

def navy_fig(fig, height=420, title=None):
    up = dict(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#F8FAFF",
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
        if abs(v) >= 1e6 or (abs(v) < 1e-4 and v != 0): return f"{v:.4e}"
        return f"{v:.{d}f}"
    except: return str(x)

def pstar(p):
    if p<0.01: return "***"
    if p<0.05: return "**"
    if p<0.10: return "*"
    return ""

def badge_html(label, kind="pass"):
    s = {"pass":"background:#D1FAE5;color:#065F46;border:1.5px solid #6EE7B7;",
         "fail":"background:#FEE2E2;color:#991B1B;border:1.5px solid #FCA5A5;",
         "warn":"background:#FEF3C7;color:#92400E;border:1.5px solid #FCD34D;",
         "info":"background:#DBEAFE;color:#1E40AF;border:1.5px solid #93C5FD;",
         "cyan":f"background:{C['navy']};color:{C['cyan']};border:1.5px solid {C['cyan']};",
         }.get(kind,"background:#DBEAFE;color:#1E40AF;border:1.5px solid #93C5FD;")
    return (f'<span style="{s}padding:2px 10px;border-radius:3px;font-size:0.72rem;'
            f'font-weight:700;font-family:Space Mono,monospace;text-transform:uppercase;">{label}</span>')

def coef_row_html(vname, coef, se, tstat, pval, ci_lo=None, ci_hi=None):
    stars=pstar(pval)
    cls="sig3" if pval<0.01 else "sig2" if pval<0.05 else "sig1" if pval<0.10 else "insig"
    ci_str=f"{fmt(ci_lo)} / {fmt(ci_hi)}" if ci_lo is not None else "—"
    return (f"<tr><td class='{cls}'>{vname} <span class='sig-star'>{stars}</span></td>"
            f"<td class='{cls}'>{fmt(coef)}</td><td>{fmt(se)}</td>"
            f"<td>{fmt(tstat)}</td><td>{fmt(pval)}</td><td>{ci_str}</td></tr>")

def coef_table_html(rows_html):
    return (f'<div class="coef-wrap"><table class="coef-tbl">'
            f'<tr><th>Variable</th><th>Coef.</th><th>Std.Err.</th>'
            f'<th>t-Stat</th><th>p-Value</th><th>95% CI</th></tr>'
            f'{rows_html}</table>'
            f'<p style="font-size:0.67rem;color:#94A3B8;margin-top:4px;'
            f'font-family:Space Mono,monospace;">*** p&lt;0.01 &nbsp; ** p&lt;0.05 &nbsp; * p&lt;0.10</p>'
            f'</div>')

# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "authenticated":False,"user_name":"","user_email":"","user_occ":"",
    "raw_df":None,"clean_df":None,"freq":None,"date_col":None,
    "menu":"home","submenu":None,
    "ols_res":None,"ardl_res":None,"var_res":None,"vecm_res":None,
    "garch_res":None,"arima_res":None,"johansen_res":None,
    "diag_model":None,"diag_resid":None,"diag_fitted":None,"diag_X":None,
    "chat_history":[],"report_log":{},"stat_results":{},
    "se_type_used":"HC1 (White Robust)",
}
for _k,_v in _DEFAULTS.items():
    if _k not in st.session_state: st.session_state[_k]=_v

# ── CSS ───────────────────────────────────────────────────────────────────────
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
::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-track{background:var(--canvas)}
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
  border-radius:20px;padding:4px 12px;font-family:'Space Mono',monospace;font-size:0.65rem;color:#94A3B8;}
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
  margin-top:12px;line-height:1.75;}
.ai-box h4{font-family:'DM Sans',sans-serif;font-size:0.76rem;font-weight:600;
  color:var(--cyan);letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;}
.ai-box p{color:#CBD5E1;font-size:0.875rem;margin-bottom:7px;}
.ai-box ul{color:#94A3B8;font-size:0.83rem;padding-left:16px;}
.ai-box li{margin-bottom:3px;}.ai-box strong{color:#E2E8F0;}
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
    _,col,_ = st.columns([1,1.5,1])
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
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="brutalist-card" style="border-color:#38BDF8;box-shadow:5px 5px 0 #38BDF8;">', unsafe_allow_html=True)
        st.markdown('<p class="sec-title" style="font-size:0.82rem;">ACCESS PORTAL</p>', unsafe_allow_html=True)
        with st.form("login_form"):
            c1,c2 = st.columns(2)
            fname = c1.text_input("First Name", placeholder="Ahmed")
            lname = c2.text_input("Last Name",  placeholder="Hisham")
            email = st.text_input("Email", placeholder="you@institution.edu")
            occ   = st.selectbox("Occupation",[
                "— Select —","Researcher / Academic","PhD Student",
                "Central Bank Economist","Financial Analyst",
                "Policy Analyst","Data Scientist / Quant","Other"])
            st.markdown("<br>", unsafe_allow_html=True)
            sub = st.form_submit_button("▶  ENTER NEXUS KERNEL", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if sub:
            errs=[]
            if not fname.strip(): errs.append("First name required.")
            if not lname.strip(): errs.append("Last name required.")
            if "@" not in email:  errs.append("Valid email required.")
            if occ=="— Select —": errs.append("Select occupation.")
            if errs:
                for e in errs: st.error(e)
            else:
                st.session_state.authenticated=True
                st.session_state.user_name=f"{fname.strip()} {lname.strip()}"
                st.session_state.user_email=email.strip()
                st.session_state.user_occ=occ
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR & TOPBAR
# ─────────────────────────────────────────────────────────────────────────────
MENU_STRUCTURE = {
    "FILE":[("home","⬡","Home / Dashboard"),("data","📂","Data Workspace"),
             ("transform","⚡","Transform & Filter"),("export","💾","Export & Download")],
    "STATIONARITY":[("unitroot","📉","Unit Root Tests"),("acf_pacf","〰","ACF / PACF"),
                    ("correlogram","🔗","Correlogram Matrix")],
    "ESTIMATE":[("ols","📐","OLS Regression"),("ardl","🔄","ARDL / Bounds Test"),
                ("var","🌐","VAR System"),("vecm","⛓","Johansen / VECM"),
                ("garch","🌊","GARCH / Volatility"),("arima","🔭","ARIMA / SARIMA")],
    "DIAGNOSTICS":[("diagnostics","🔬","Diagnostic Suite"),
                   ("stability","📏","Stability Tests"),("normality","🔔","Normality Analysis")],
    "FORECAST":[("forecast","🎯","Forecasting Engine"),("decompose","🧩","Decomposition")],
    "TOOLS":[("stats","📊","Summary Statistics"),("chat","🤖","AI Econometrician"),
              ("report","📄","PDF Report")],
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
        </div>""", unsafe_allow_html=True)
        if st.session_state.authenticated:
            st.markdown(f"""
            <div style="padding:7px 10px;background:rgba(56,189,248,0.06);
                 border:1px solid rgba(56,189,248,0.15);border-radius:4px;margin-bottom:8px;">
              <p style="margin:0;font-size:0.7rem;color:#38BDF8;
                 font-family:'Space Mono',monospace;">{st.session_state.user_name}</p>
              <p style="margin:0;font-size:0.6rem;color:#334155;">{st.session_state.user_occ}</p>
            </div>""", unsafe_allow_html=True)
        if st.session_state.clean_df is not None:
            df=st.session_state.clean_df
            st.markdown(f"""
            <div style="padding:7px 10px;background:rgba(16,185,129,0.07);
                 border:1px solid rgba(16,185,129,0.2);border-radius:4px;margin-bottom:8px;">
              <p style="margin:0;font-size:0.6rem;color:#10B981;
                 font-family:'Space Mono',monospace;text-transform:uppercase;">● Active Dataset</p>
              <p style="margin:0;font-size:0.68rem;color:#94A3B8;font-family:'Space Mono',monospace;">
                 {len(df):,} obs · {len(df.columns)} vars · {st.session_state.freq or '?'}</p>
            </div>""", unsafe_allow_html=True)
        cur = st.session_state.menu
        for section,items in MENU_STRUCTURE.items():
            st.markdown(f'<p class="nk-menu-sec">{section}</p>', unsafe_allow_html=True)
            for key,icon,label in items:
                t = "primary" if cur==key else "secondary"
                if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True, type=t):
                    st.session_state.menu=key; st.rerun()
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("⏻  Sign Out", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()
        st.markdown('<p style="font-family:Space Mono,monospace;font-size:0.53rem;color:#1E3A5F;'
                    'text-align:center;margin-top:8px;">NEXUS KERNEL v3.0 · 2026</p>', unsafe_allow_html=True)

def render_topbar():
    labels={"home":"Home","data":"Data Workspace","transform":"Transform & Filter",
            "export":"Export","unitroot":"Unit Root Tests","acf_pacf":"ACF / PACF",
            "correlogram":"Correlogram","ols":"OLS Regression","ardl":"ARDL",
            "var":"VAR System","vecm":"Johansen / VECM","garch":"Volatility Models",
            "arima":"ARIMA / SARIMA","diagnostics":"Diagnostic Suite",
            "stability":"Stability Tests","normality":"Normality Analysis",
            "forecast":"Forecasting Engine","decompose":"Decomposition",
            "stats":"Summary Statistics","chat":"AI Econometrician","report":"PDF Report",
            "export":"Export & Download"}
    page=labels.get(st.session_state.menu,"NEXUS KERNEL")
    user=st.session_state.get("user_name","")
    st.markdown(f"""
    <div class="nk-topbar">
      <div><span class="nk-logo">⬡ NEXUS KERNEL</span>
        <p class="nk-logo-sub">Professional Time-Series Econometrics · v3.0</p></div>
      <div style="display:flex;align-items:center;gap:14px;">
        <span style="font-family:'Space Mono',monospace;font-size:0.75rem;
              color:#38BDF8;letter-spacing:0.06em;">{page.upper()}</span>
        <span class="nk-user-pill">{user}</span>
      </div>
    </div><div style="margin-bottom:16px;"></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    st.markdown('<p class="page-title">Welcome to NEXUS KERNEL v3.0</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">The professional-grade replacement for EViews, Stata, and R. '
                'Built for central bank economists, academic researchers, and quantitative analysts.</p>',
                unsafe_allow_html=True)
    has_data=st.session_state.clean_df is not None
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Dataset","Loaded" if has_data else "None",
              delta=f"{len(st.session_state.clean_df):,} obs" if has_data else None)
    c2.metric("OLS","Ready" if st.session_state.ols_res else "—")
    c3.metric("ARDL","Ready" if st.session_state.ardl_res else "—")
    c4.metric("VAR","Ready" if st.session_state.var_res else "—")
    c5.metric("AI Chat",str(len(st.session_state.chat_history)))
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="sec-title">📦 Module Overview</p>', unsafe_allow_html=True)
    grid=[("📂","Data Workspace","Upload CSV/XLSX · Auto-freq · Clean","data"),
          ("⚡","Transform & Filter","Log/Diff/MA · HP · BK · STL","transform"),
          ("📉","Unit Root Tests","ADF · PP · KPSS · Zivot-Andrews","unitroot"),
          ("📐","OLS Regression","HC1/HAC SEs · Full inference · RESET","ols"),
          ("🔄","ARDL / Bounds","PSS Bounds · ECM · Long-run coefs","ardl"),
          ("🌐","VAR System","Granger · IRF · FEVD · Lag selection","var"),
          ("⛓","Johansen / VECM","Trace/Max-Eigen · α β matrices","vecm"),
          ("🌊","Volatility","GARCH · EGARCH · TGARCH","garch"),
          ("🔭","ARIMA / SARIMA","Auto-ARIMA · Fan charts · RMSE","arima"),
          ("🔬","Diagnostics","DW · BG · LjungBox · White · RESET","diagnostics"),
          ("🤖","AI Econometrician","Context-aware chatbot · Full Q&A","chat"),
          ("📄","PDF Report","Automated professional report","report")]
    for i in range(0,len(grid),4):
        row=grid[i:i+4]; cols_g=st.columns(len(row))
        for col_g,(icon,title,desc,key) in zip(cols_g,row):
            with col_g:
                st.markdown(f"""<div class="glass-card" style="min-height:118px;border-left:3px solid #38BDF8;">
                  <p style="font-size:1.6rem;margin:0 0 5px;">{icon}</p>
                  <p style="font-family:'DM Sans';font-weight:700;font-size:0.85rem;color:#0F172A;margin:0 0 4px;">{title}</p>
                  <p style="font-size:0.73rem;color:#64748B;line-height:1.4;margin:0;">{desc}</p>
                </div>""", unsafe_allow_html=True)
                if st.button(f"Open {title}", key=f"home_{key}", use_container_width=True):
                    st.session_state.menu=key; st.rerun()
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="sec-title">🗺 Recommended Workflow</p>', unsafe_allow_html=True)
    steps=[("1","Upload Data","File → Data Workspace"),("2","Clean & Transform","File → Transform & Filter"),
           ("3","Test Stationarity","Stationarity → Unit Root Tests"),("4","Estimate Model","Estimate → choose model"),
           ("5","Run Diagnostics","Diagnostics → Suite"),("6","Forecast & Report","Forecast → PDF Report")]
    cols_s=st.columns(6)
    for col_s,(num,title,path) in zip(cols_s,steps):
        with col_s:
            st.markdown(f"""<div style="text-align:center;padding:12px 6px;">
              <div style="width:32px;height:32px;border-radius:50%;background:#0F172A;border:2px solid #38BDF8;
                   display:flex;align-items:center;justify-content:center;margin:0 auto 7px;
                   font-family:'Space Mono',monospace;font-weight:700;font-size:0.85rem;color:#38BDF8;">{num}</div>
              <p style="font-family:'DM Sans';font-weight:600;font-size:0.8rem;color:#0F172A;margin:0 0 3px;">{title}</p>
              <p style="font-size:0.67rem;color:#64748B;margin:0;font-family:'Space Mono',monospace;">{path}</p>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA WORKSPACE
# ─────────────────────────────────────────────────────────────────────────────
def page_data():
    st.markdown('<p class="page-title">📂 Data Workspace</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Upload CSV/XLSX · Auto-detect frequency · Validate & clean</p>', unsafe_allow_html=True)
    tab_up,tab_view,tab_clean=st.tabs(["① Upload & Ingest","② Preview & Validate","③ Clean & Save"])
    with tab_up:
        uploaded=st.file_uploader("Drop CSV or XLSX here",type=["csv","xlsx","xls"])
        if uploaded:
            try:
                if uploaded.name.lower().endswith((".xlsx",".xls")):
                    raw=pd.read_excel(uploaded,sheet_name=0)
                else:
                    content=uploaded.read(); raw=None
                    for enc in ("utf-8","latin-1","cp1252"):
                        try:
                            raw=pd.read_csv(io.BytesIO(content),sep=None,engine="python",encoding=enc); break
                        except: continue
                    if raw is None: raise ValueError("Cannot parse CSV.")
                st.session_state.raw_df=raw.copy()
                st.success(f"✓ Loaded **{uploaded.name}** — {raw.shape[0]:,} rows × {raw.shape[1]} cols")
            except Exception as exc: st.error(f"**File read error:** {exc}")
        if st.session_state.raw_df is not None:
            raw=st.session_state.raw_df
            st.markdown('<p class="sec-title">Raw Preview</p>', unsafe_allow_html=True)
            st.dataframe(raw.head(10),use_container_width=True)
            c1,c2,c3=st.columns(3)
            c1.metric("Rows",f"{raw.shape[0]:,}"); c2.metric("Cols",raw.shape[1])
            c3.metric("Memory",f"{raw.memory_usage(deep=True).sum()/1024:.1f} KB")
    with tab_view:
        if st.session_state.raw_df is None: st.info("Upload a file first.")
        else:
            raw=st.session_state.raw_df
            st.markdown('<p class="sec-title">Column Info & Missing Values</p>', unsafe_allow_html=True)
            info=pd.DataFrame({"dtype":raw.dtypes.astype(str),"non_null":raw.notnull().sum(),
                "missing":raw.isnull().sum(),"miss_%":(raw.isnull().mean()*100).round(2),"unique":raw.nunique()})
            st.dataframe(info,use_container_width=True)
            num=raw.select_dtypes(include=np.number)
            if not num.empty:
                st.markdown('<p class="sec-title">Numeric Summary</p>', unsafe_allow_html=True)
                desc=num.describe().T.round(4); desc["skew"]=num.skew().round(4); desc["kurt"]=num.kurt().round(4)
                st.dataframe(desc,use_container_width=True)
    with tab_clean:
        if st.session_state.raw_df is None: st.info("Upload a file first.")
        else:
            raw=st.session_state.raw_df
            st.markdown('<p class="sec-title">Cleaning Configuration</p>', unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1:
                date_col=st.selectbox("Date / Period Column",["(use row index)"]+raw.columns.tolist())
                freq_opt=st.selectbox("Frequency Override",["Auto-Detect","Annual (A)","Quarterly (Q)","Monthly (M)","Weekly (W)","Daily (D)"])
            with c2:
                miss_method=st.selectbox("Missing Value Treatment",["Linear Interpolation","Spline Interpolation","Forward Fill (ffill)","Backward Fill (bfill)","Drop Rows with NaN"])
                strip_sym=st.checkbox("Strip symbols ($, %, commas)",value=True)
            seasonal_adj=st.checkbox("STL Seasonal Adjustment (sub-annual)",value=False)
            if st.button("▶  APPLY CLEANING & SAVE",use_container_width=True):
                with st.spinner("Processing…"):
                    try:
                        df=raw.copy()
                        if strip_sym:
                            for col in df.select_dtypes(include="object").columns:
                                df[col]=(df[col].astype(str).str.replace(r"[$%,\s\u202f\u00a0]","",regex=True)
                                         .str.replace(r"[^\d.\-eE+]","",regex=True))
                                df[col]=pd.to_numeric(df[col],errors="coerce")
                        if date_col!="(use row index)":
                            try:
                                df[date_col]=pd.to_datetime(df[date_col],infer_datetime_format=True)
                                df=df.set_index(date_col).sort_index(); st.session_state.date_col=date_col
                            except: df=df.set_index(date_col).sort_index()
                        freq_map={"Annual (A)":"Annual","Quarterly (Q)":"Quarterly","Monthly (M)":"Monthly","Weekly (W)":"Weekly","Daily (D)":"Daily"}
                        if freq_opt=="Auto-Detect":
                            try:
                                inf=pd.infer_freq(df.index)
                                if inf:
                                    if inf[0] in ("A","Y"): freq="Annual"
                                    elif inf[0]=="Q": freq="Quarterly"
                                    elif inf[0]=="M": freq="Monthly"
                                    elif inf[0]=="W": freq="Weekly"
                                    elif inf[0] in ("D","B"): freq="Daily"
                                    else: freq=inf
                                else: freq="Unknown"
                            except: freq="Unknown"
                        else: freq=freq_map.get(freq_opt,freq_opt.split(" ")[0])
                        st.session_state.freq=freq
                        num_cols=df.select_dtypes(include=np.number).columns
                        if miss_method=="Linear Interpolation": df[num_cols]=df[num_cols].interpolate(method="linear")
                        elif miss_method=="Spline Interpolation": df[num_cols]=df[num_cols].interpolate(method="spline",order=3)
                        elif miss_method=="Forward Fill (ffill)": df[num_cols]=df[num_cols].ffill()
                        elif miss_method=="Backward Fill (bfill)": df[num_cols]=df[num_cols].bfill()
                        elif miss_method=="Drop Rows with NaN": df=df.dropna()
                        if seasonal_adj and freq in ("Monthly","Quarterly"):
                            period=12 if freq=="Monthly" else 4
                            for col in num_cols:
                                try:
                                    stl=STL(df[col].dropna(),period=period); r=stl.fit()
                                    df[f"{col}_SA"]=df[col]-r.seasonal
                                except: pass
                        st.session_state.clean_df=df
                        st.success(f"✓ Dataset saved — {len(df):,} obs · {len(df.columns)} vars · Frequency: **{freq}**")
                        num_df=df.select_dtypes(include=np.number)
                        if not num_df.empty:
                            fig=go.Figure()
                            for i,col in enumerate(num_df.columns[:8]):
                                fig.add_trace(go.Scatter(x=num_df.index,y=num_df[col],name=col,
                                    line=dict(color=CHART_COLORS[i%len(CHART_COLORS)],width=1.8)))
                            fig=navy_fig(fig,350,"Dataset Overview"); st.plotly_chart(fig,use_container_width=True)
                    except Exception as exc: st.error(f"**Cleaning failed:** {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TRANSFORM & FILTER
# ─────────────────────────────────────────────────────────────────────────────
def page_transform():
    st.markdown('<p class="page-title">⚡ Transform & Filter</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Log · Differences · Moving Averages · HP Filter · Baxter-King · STL</p>', unsafe_allow_html=True)
    if st.session_state.clean_df is None: st.info("Load data first."); return
    df=st.session_state.clean_df.copy()
    cols=df.select_dtypes(include=np.number).columns.tolist()
    tab_t,tab_f=st.tabs(["🔢 Transformations","🎛 Filters & Decomposition"])
    with tab_t:
        st.markdown('<p class="sec-title">Variable Transformations</p>', unsafe_allow_html=True)
        sel_cols=st.multiselect("Variables to transform",cols,default=cols[:2])
        c1,c2,c3=st.columns(3)
        with c1:
            do_log=st.checkbox("Natural Log (ln)"); do_diff1=st.checkbox("First Difference (Δ)"); do_diff2=st.checkbox("Second Difference (Δ²)")
        with c2:
            do_ma=st.checkbox("Moving Average"); ma_win=st.slider("MA Window",2,24,4) if do_ma else 4; do_pct=st.checkbox("% Change")
        with c3:
            do_std=st.checkbox("Standardize (z-score)"); do_idx=st.checkbox("Index (base=100)"); do_lret=st.checkbox("Log Returns")
        if st.button("▶  APPLY TRANSFORMATIONS",use_container_width=True):
            if not sel_cols: st.warning("Select at least one variable.")
            else:
                added=[]
                for col in sel_cols:
                    s=df[col].copy()
                    if do_log:   n=f"ln_{col}";      df[n]=np.log(s.replace(0,np.nan));  added.append(n)
                    if do_diff1: n=f"d_{col}";       df[n]=s.diff();                      added.append(n)
                    if do_diff2: n=f"d2_{col}";      df[n]=s.diff().diff();               added.append(n)
                    if do_ma:    n=f"ma{ma_win}_{col}"; df[n]=s.rolling(ma_win).mean();   added.append(n)
                    if do_pct:   n=f"pct_{col}";     df[n]=s.pct_change()*100;            added.append(n)
                    if do_std:   n=f"std_{col}";     df[n]=(s-s.mean())/s.std();          added.append(n)
                    if do_idx:
                        n=f"idx_{col}"; base=s.dropna().iloc[0]
                        df[n]=(s/base*100) if base!=0 else s; added.append(n)
                    if do_lret:  n=f"logret_{col}";  df[n]=np.log(s/s.shift(1));         added.append(n)
                st.session_state.clean_df=df
                st.success(f"✓ Added {len(added)} series: {', '.join(added)}")
                if added:
                    fig=go.Figure()
                    for i,col in enumerate(added[:6]):
                        fig.add_trace(go.Scatter(x=df.index,y=df[col],name=col,
                            line=dict(color=CHART_COLORS[i%len(CHART_COLORS)],width=1.8)))
                    fig=navy_fig(fig,350,"Transformed Series"); st.plotly_chart(fig,use_container_width=True)
    with tab_f:
        st.markdown('<p class="sec-title">Economic Filters</p>', unsafe_allow_html=True)
        filt_col=st.selectbox("Variable to filter",cols)
        filt_type=st.radio("Method",["Hodrick-Prescott (HP)","Baxter-King (BK)","STL Decomposition"],horizontal=True)
        if filt_type=="Hodrick-Prescott (HP)":
            dlam={"Annual":100,"Quarterly":1600,"Monthly":14400,"Daily":129600}
            lam=st.number_input("Smoothing λ",value=float(dlam.get(st.session_state.freq or "Quarterly",1600)),min_value=1.0,step=100.0)
            if st.button("▶  HP FILTER",use_container_width=True):
                try:
                    cycle,trend=hpfilter(df[filt_col].dropna(),lamb=lam)
                    df[f"hp_trend_{filt_col}"]=trend; df[f"hp_cycle_{filt_col}"]=cycle
                    st.session_state.clean_df=df
                    fig=make_subplots(rows=2,cols=1,vertical_spacing=0.08,subplot_titles=["Original vs HP Trend","HP Cyclical Component"])
                    fig.add_trace(go.Scatter(x=df.index,y=df[filt_col],name="Original",line=dict(color=C["navy"],width=1.5)),1,1)
                    fig.add_trace(go.Scatter(x=trend.index,y=trend.values,name="Trend",line=dict(color=C["cyan"],width=2.5)),1,1)
                    fig.add_trace(go.Scatter(x=cycle.index,y=cycle.values,name="Cycle",fill="tozeroy",fillcolor="rgba(56,189,248,0.13)",line=dict(color=C["cyan"],width=1.8)),2,1)
                    fig.add_hline(y=0,line=dict(color=C["navy"],dash="dash",width=1),row=2,col=1)
                    fig=navy_fig(fig,480,f"HP Filter — {filt_col} (λ={lam:,.0f})"); st.plotly_chart(fig,use_container_width=True)
                    st.success("✓ HP trend and cycle added.")
                except Exception as exc: st.error(f"HP Filter error: {exc}")
        elif filt_type=="Baxter-King (BK)":
            c1,c2,c3=st.columns(3)
            lo=c1.number_input("Low period",value=6,min_value=2); hi=c2.number_input("High period",value=32,min_value=3); K=c3.number_input("Lead/lag K",value=12,min_value=1)
            if st.button("▶  BK FILTER",use_container_width=True):
                try:
                    s=df[filt_col].dropna(); bk=bkfilter(s,low=lo,high=hi,K=int(K))
                    df[f"bk_cycle_{filt_col}"]=bk; st.session_state.clean_df=df
                    fig=make_subplots(rows=2,cols=1,vertical_spacing=0.08,subplot_titles=["Original","BK Cyclical Component"])
                    fig.add_trace(go.Scatter(x=s.index,y=s.values,name="Original",line=dict(color=C["navy"],width=1.5)),1,1)
                    fig.add_trace(go.Scatter(x=bk.index,y=bk.values,name="BK Cycle",fill="tozeroy",fillcolor="rgba(20,184,166,0.13)",line=dict(color=C["teal"],width=2)),2,1)
                    fig.add_hline(y=0,line=dict(color=C["navy"],dash="dash"),row=2,col=1)
                    fig=navy_fig(fig,480,f"Baxter-King Filter — {filt_col}"); st.plotly_chart(fig,use_container_width=True); st.success("✓ BK cycle added.")
                except Exception as exc: st.error(f"BK Filter error: {exc}")
        elif filt_type=="STL Decomposition":
            dp={"Monthly":12,"Quarterly":4,"Weekly":52,"Annual":2,"Daily":365}
            period=st.number_input("Seasonal period",value=int(dp.get(st.session_state.freq or "Monthly",12)),min_value=2)
            robust=st.checkbox("Robust STL",value=True)
            if st.button("▶  STL DECOMPOSE",use_container_width=True):
                try:
                    s=df[filt_col].dropna(); stl=STL(s,period=int(period),robust=robust); res=stl.fit()
                    df[f"stl_trend_{filt_col}"]=res.trend; df[f"stl_seasonal_{filt_col}"]=res.seasonal; df[f"stl_resid_{filt_col}"]=res.resid
                    st.session_state.clean_df=df
                    fig=make_subplots(rows=4,cols=1,vertical_spacing=0.04,subplot_titles=["Original","Trend","Seasonal","Residual"])
                    for i,(y,color,name) in enumerate([(s.values,C["navy"],"Original"),(res.trend,C["cyan"],"Trend"),(res.seasonal,C["gold"],"Seasonal"),(res.resid,C["red"],"Residual")],1):
                        fig.add_trace(go.Scatter(x=s.index,y=y,name=name,line=dict(color=color,width=1.6)),i,1)
                    fig=navy_fig(fig,600,f"STL Decomposition — {filt_col}"); st.plotly_chart(fig,use_container_width=True); st.success("✓ STL components added.")
                except Exception as exc: st.error(f"STL error: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: UNIT ROOT TESTS
# ─────────────────────────────────────────────────────────────────────────────
def page_unitroot():
    st.markdown('<p class="page-title">📉 Unit Root & Stationarity Vault</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">ADF · Phillips-Perron · KPSS · Zivot-Andrews — with AI interpretation</p>', unsafe_allow_html=True)
    if st.session_state.clean_df is None: st.info("Load your dataset first."); return
    df=st.session_state.clean_df; cols=df.select_dtypes(include=np.number).columns.tolist()
    c1,c2,c3,c4=st.columns(4)
    with c1: sel_var=st.selectbox("Variable",cols)
    with c2: diff_ord=st.selectbox("Transform",["Level","1st Difference (Δ)","2nd Difference (Δ²)"])
    with c3:
        reg_spec=st.selectbox("Deterministics",["c — Constant only","ct — Constant + Trend","n — None","ctt — Const + Quad Trend"])
        reg_code=reg_spec.split(" ")[0]
    with c4: lag_meth=st.selectbox("Lag Selection",["AIC","BIC","t-stat","Fixed"])
    fixed_lag=3
    if lag_meth=="Fixed": fixed_lag=st.slider("Fixed lag length",1,16,3)
    alpha=st.select_slider("Significance α",[0.01,0.05,0.10],value=0.05)
    if st.button("▶  RUN ALL STATIONARITY TESTS",use_container_width=True):
        with st.spinner("Running tests…"):
            s=df[sel_var].dropna()
            if diff_ord.startswith("1st"):   s=s.diff().dropna();        s_label=f"Δ{sel_var}"
            elif diff_ord.startswith("2nd"): s=s.diff().diff().dropna(); s_label=f"Δ²{sel_var}"
            else:                             s_label=sel_var
            results={"variable":sel_var,"level":diff_ord,"label":s_label}
            try:
                kw=dict(regression=reg_code)
                if lag_meth=="Fixed": kw.update(maxlag=fixed_lag,autolag=None)
                else: kw["autolag"]=lag_meth
                r=adfuller(s,**kw); results["adf"]={"stat":r[0],"pval":r[1],"lags":r[2],"crits":r[4],"ok":r[1]<alpha}
            except Exception as exc: results["adf"]={"error":str(exc)}
            try:
                r2=adfuller(s,regression=reg_code,autolag="BIC"); results["pp"]={"stat":r2[0],"pval":r2[1],"lags":r2[2],"crits":r2[4],"ok":r2[1]<alpha}
            except Exception as exc: results["pp"]={"error":str(exc)}
            try:
                kr="c" if reg_code in ("c","n") else "ct"; k=kpss(s,regression=kr,nlags="auto")
                results["kpss"]={"stat":k[0],"pval":k[1],"lags":k[2],"crits":k[3],"ok":k[1]>alpha}
            except Exception as exc: results["kpss"]={"error":str(exc)}
            try:
                za=zivot_andrews(s,maxlag=None,regression="c",autolag="AIC")
                results["za"]={"stat":za[0],"pval":za[1],"break_idx":za[3],"crits":za[2],"ok":za[1]<alpha}
            except Exception as exc: results["za"]={"error":str(exc)}
            st.session_state.stat_results=results
    res=st.session_state.stat_results
    if not res: return
    s_label=res.get("label",""); lv=res.get("level","Level")
    raw_s=st.session_state.clean_df[res.get("variable","")].dropna()
    if lv.startswith("1st"):  plot_s=raw_s.diff().dropna()
    elif lv.startswith("2nd"): plot_s=raw_s.diff().diff().dropna()
    else: plot_s=raw_s
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=plot_s.index,y=plot_s.values,name=s_label,line=dict(color=C["cyan"],width=2)))
    fig.add_hline(y=plot_s.mean(),line=dict(color=C["gold"],dash="dash",width=1.2),annotation_text="Mean")
    fig=navy_fig(fig,260,f"Series: {s_label}"); st.plotly_chart(fig,use_container_width=True)
    def _show_test(container,name,key,h0_text):
        with container:
            r=res.get(key,{})
            if "error" in r: st.error(f"**{name}** failed: {r['error']}"); return
            stat=r.get("stat",0); pval=r.get("pval",1); ok=r.get("ok",False); crits=r.get("crits",{})
            verdict="STATIONARY" if ok else ("UNIT ROOT" if key!="kpss" else "NON-STATIONARY")
            bkind="pass" if ok else "fail"
            c1p=crits.get("1%",crits.get("10%","—")); c5p=crits.get("5%","—"); c10p=crits.get("10%","—")
            st.markdown(f"""
            <div class="brutalist-card" style="border-color:{'#10B981' if ok else '#EF4444'};box-shadow:4px 4px 0 {'#10B981' if ok else '#EF4444'};">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:7px;">
                <p style="font-weight:700;font-size:0.88rem;color:#0F172A;margin:0;">{name}</p>
                {badge_html(verdict,bkind)}</div>
              <table class="coef-tbl" style="width:100%;">
                <tr><th>Statistic</th><th>Value</th></tr>
                <tr><td>Test Stat.</td><td>{fmt(stat)}</td></tr>
                <tr><td>p-value</td><td>{fmt(pval)}</td></tr>
                <tr><td>Lags Used</td><td>{r.get('lags','—')}</td></tr>
                <tr><td>Crit. 1%</td><td>{fmt(c1p) if c1p!='—' else '—'}</td></tr>
                <tr><td>Crit. 5%</td><td>{fmt(c5p) if c5p!='—' else '—'}</td></tr>
                <tr><td>Crit. 10%</td><td>{fmt(c10p) if c10p!='—' else '—'}</td></tr>
              </table>
              <p style="font-size:0.68rem;color:#64748B;margin-top:7px;font-family:'Space Mono',monospace;">H₀: {h0_text}</p>
            </div>""", unsafe_allow_html=True)
    c1,c2=st.columns(2)
    _show_test(c1,"Augmented Dickey-Fuller (ADF)","adf","Unit root present")
    _show_test(c2,"Phillips-Perron (PP)","pp","Unit root present")
    _show_test(c1,"KPSS Test","kpss","Series is stationary (trend-stationary)")
    za=res.get("za",{})
    if "error" not in za and za:
        ok_za=za.get("ok",False)
        st.markdown(f"""
        <div class="brutalist-card" style="border-color:{'#10B981' if ok_za else '#EF4444'};box-shadow:4px 4px 0 {'#10B981' if ok_za else '#EF4444'};">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:7px;">
            <p style="font-weight:700;font-size:0.88rem;color:#0F172A;margin:0;">Zivot-Andrews (Unit Root w/ Structural Break)</p>
            {badge_html('STATIONARY' if ok_za else 'UNIT ROOT','pass' if ok_za else 'fail')}</div>
          <table class="coef-tbl"><tr><th>Statistic</th><th>Value</th></tr>
            <tr><td>ZA Stat.</td><td>{fmt(za.get('stat',0))}</td></tr>
            <tr><td>p-value</td><td>{fmt(za.get('pval',1))}</td></tr>
            <tr><td>Break Index</td><td>{za.get('break_idx','—')}</td></tr>
          </table>
          <p style="font-size:0.68rem;color:#64748B;margin-top:7px;font-family:'Space Mono',monospace;">H₀: Unit root allowing for one structural break in intercept</p>
        </div>""", unsafe_allow_html=True)
    adf_ok=res.get("adf",{}).get("ok",None); pp_ok=res.get("pp",{}).get("ok",None)
    kpss_ok=res.get("kpss",{}).get("ok",None); za_ok=res.get("za",{}).get("ok",None)
    ai_html=_interp_stationarity(s_label,adf_ok,pp_ok,kpss_ok,za_ok,
        res.get("adf",{}).get("pval",1),res.get("kpss",{}).get("pval",1),lv)
    st.markdown(f'<div class="ai-box"><h4>⬡ AI INTERPRETATION · STATIONARITY</h4>{ai_html}</div>',unsafe_allow_html=True)
    st.session_state.report_log["stationarity"]={"variable":res.get("variable",""),"level":lv,
        "adf_pval":res.get("adf",{}).get("pval","—"),"kpss_pval":res.get("kpss",{}).get("pval","—"),
        "verdict":"STATIONARY" if (adf_ok and kpss_ok) else "NON-STATIONARY"}

def _interp_stationarity(label,adf,pp,kpss_ok,za,adf_p,kpss_p,level):
    lvl_map={"Level":"in levels","1st Difference (Δ)":"in first differences","2nd Difference (Δ²)":"in second differences"}
    lvl_str=lvl_map.get(level,""); lines=[]
    if adf is True and pp is True and kpss_ok is True:
        lines.append(f"<p><span class='ok-text'>✅ All tests agree:</span> <strong>{label}</strong> is <strong>stationary</strong> {lvl_str}. High confidence.</p>")
    elif adf is False and pp is False and kpss_ok is False:
        lines.append(f"<p><span class='bad-text'>❌ All tests agree:</span> <strong>{label}</strong> contains a <strong>unit root</strong> {lvl_str}. Difference before OLS or use ARDL/VECM.</p>")
    else:
        lines.append(f"<p><span class='warn-text'>⚠ Mixed signals</span> for <strong>{label}</strong> {lvl_str}.</p>")
        if adf is not None: lines.append(f"<p>• ADF (p={fmt(adf_p)}): {'Reject H₀ → stationary' if adf else 'Fail to reject → unit root'}</p>")
        if kpss_ok is not None: lines.append(f"<p>• KPSS (p≈{fmt(kpss_p)}): {'Fail to reject H₀ → stationary' if kpss_ok else 'Reject H₀ → non-stationary'}</p>")
        lines.append("<p>When ADF and KPSS conflict: (1) possible structural break → run Zivot-Andrews, (2) near-integrated, (3) short sample bias.</p>")
    if level=="Level" and (adf is False or pp is False):
        lines.append("<p>📌 <strong>Next:</strong> Re-test in 1st Difference. If stationary in Δ → I(1) → suitable for ARDL Bounds or Johansen/VECM.</p>")
    if za is True:
        lines.append("<p>🔍 Zivot-Andrews rejects unit root allowing for structural break — non-stationarity may be regime-change driven.</p>")
    return "".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ACF / PACF
# ─────────────────────────────────────────────────────────────────────────────
def page_acf_pacf():
    st.markdown('<p class="page-title">〰 ACF & PACF Correlograms</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Interactive autocorrelation functions with Bartlett bands and Ljung-Box Q-test.</p>', unsafe_allow_html=True)
    if st.session_state.clean_df is None: st.info("Load your dataset first."); return
    df=st.session_state.clean_df; cols=df.select_dtypes(include=np.number).columns.tolist()
    c1,c2,c3=st.columns(3)
    sel=c1.selectbox("Variable",cols); n_lags=c2.slider("Lags",4,60,24)
    diff_=c3.selectbox("Transform",["Level","1st Difference","2nd Difference"])
    if st.button("▶  PLOT CORRELOGRAMS",use_container_width=True):
        s=df[sel].dropna()
        if diff_=="1st Difference":   s=s.diff().dropna();        lbl=f"Δ{sel}"
        elif diff_=="2nd Difference": s=s.diff().diff().dropna(); lbl=f"Δ²{sel}"
        else: lbl=sel
        try:
            max_lag=min(n_lags,len(s)//3)
            acf_vals=acf(s,nlags=max_lag,fft=True,alpha=0.05); pacf_vals=pacf(s,nlags=max_lag,alpha=0.05)
            acf_arr=acf_vals[0]; acf_ci=acf_vals[1]; pacf_arr=pacf_vals[0]; pacf_ci=pacf_vals[1]
            conf=1.96/np.sqrt(len(s))
            fig=make_subplots(rows=2,cols=1,vertical_spacing=0.1,subplot_titles=[f"ACF — {lbl}",f"PACF — {lbl}"])
            for (arr,ci,row) in [(acf_arr,acf_ci,1),(pacf_arr,pacf_ci,2)]:
                for lag_i,val in enumerate(arr):
                    color=C["cyan"] if abs(val)>conf else C["slate"]
                    fig.add_trace(go.Scatter(x=[lag_i,lag_i],y=[0,val],mode="lines",line=dict(color=color,width=2),showlegend=False,hoverinfo="skip"),row,1)
                    fig.add_trace(go.Scatter(x=[lag_i],y=[val],mode="markers",marker=dict(color=color,size=6,line=dict(color=C["navy"],width=1.2)),showlegend=False,hovertemplate=f"Lag {lag_i}: {val:.4f}<extra></extra>"),row,1)
                fig.add_hline(y=conf,line=dict(color=C["red"],dash="dash",width=1.2),row=row,col=1)
                fig.add_hline(y=-conf,line=dict(color=C["red"],dash="dash",width=1.2),row=row,col=1)
                fig.add_hline(y=0,line=dict(color=C["navy"],width=0.8),row=row,col=1)
            fig=navy_fig(fig,500); st.plotly_chart(fig,use_container_width=True)
            lb=acorr_ljungbox(s,lags=min(20,max_lag),return_df=True)
            st.markdown('<p class="sec-title">Ljung-Box Q-Test</p>', unsafe_allow_html=True)
            st.dataframe(lb.style.format("{:.4f}"),use_container_width=True)
            st.markdown(f"""<div class="ai-box"><h4>⬡ ACF/PACF INTERPRETATION</h4>
              <p>Significant spikes beyond ±{fmt(conf,3)} indicate autocorrelation structure. ARIMA order guide:</p>
              <ul><li><strong>ACF tailing off, PACF cuts off at p</strong> → AR(p)</li>
                  <li><strong>PACF tailing off, ACF cuts off at q</strong> → MA(q)</li>
                  <li><strong>Both tail off</strong> → ARMA(p,q)</li>
                  <li><strong>ACF decays very slowly</strong> → non-stationary, needs differencing</li></ul>
              <p>Ljung-Box H₀: no autocorrelation up to lag h. Reject (p&lt;0.05) → autocorrelation present.</p>
            </div>""", unsafe_allow_html=True)
        except Exception as exc: st.error(f"Correlogram error: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CORRELOGRAM MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def page_correlogram():
    st.markdown('<p class="page-title">🔗 Correlogram Matrix</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Cross-correlation heatmap and scatter matrix.</p>', unsafe_allow_html=True)
    if st.session_state.clean_df is None: st.info("Load data first."); return
    df=st.session_state.clean_df; cols=df.select_dtypes(include=np.number).columns.tolist()
    tab_cor,tab_scatter=st.tabs(["Correlation Heatmap","Scatter Matrix"])
    with tab_cor:
        c1,c2=st.columns(2)
        sel_vars=c1.multiselect("Variables",cols,default=cols[:min(8,len(cols))])
        method=c2.selectbox("Method",["pearson","spearman","kendall"])
        if sel_vars and len(sel_vars)>=2:
            corr=df[sel_vars].corr(method=method).round(4)
            fig=go.Figure(go.Heatmap(z=corr.values,x=corr.columns,y=corr.index,
                colorscale=[[0,C["navy"]],[0.5,"#F1F5F9"],[1,C["cyan"]]],
                text=corr.values.round(2),texttemplate="%{text}",textfont=dict(size=10),zmin=-1,zmax=1))
            fig=navy_fig(fig,500,f"{method.title()} Correlation Matrix"); st.plotly_chart(fig,use_container_width=True)
            st.dataframe(corr.style.background_gradient(cmap="RdYlGn",vmin=-1,vmax=1).format("{:.4f}"),use_container_width=True)
        else: st.info("Select at least 2 variables.")
    with tab_scatter:
        sel_s=st.multiselect("Variables for scatter",cols,default=cols[:min(5,len(cols))])
        if sel_s and len(sel_s)>=2:
            try:
                import plotly.express as px
                fig2=px.scatter_matrix(df[sel_s].dropna(),dimensions=sel_s,color_discrete_sequence=[C["cyan"]])
                fig2.update_traces(marker=dict(size=3,opacity=0.55,line=dict(color=C["navy"],width=0.3)))
                fig2=navy_fig(fig2,600,"Scatter Matrix"); st.plotly_chart(fig2,use_container_width=True)
            except Exception as exc: st.error(f"Scatter matrix error: {exc}")
        else: st.info("Select at least 2 variables.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OLS REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
def page_ols():
    st.markdown('<p class="page-title">📐 OLS Regression</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Ordinary Least Squares · HC1 / HAC Newey-West robust SEs · Full inference · RESET test</p>', unsafe_allow_html=True)
    if st.session_state.clean_df is None: st.info("Load data first."); return
    df=st.session_state.clean_df; cols=df.select_dtypes(include=np.number).columns.tolist()
    tab_cfg,tab_res,tab_diag=st.tabs(["① Configuration","② Results","③ Diagnostics"])
    with tab_cfg:
        c1,c2=st.columns(2)
        dep=c1.selectbox("Dependent Variable (Y)",cols)
        indep=c2.multiselect("Independent Variables (X)",[c for c in cols if c!=dep])
        c3,c4,c5=st.columns(3)
        add_const=c3.checkbox("Include Constant",value=True)
        se_type=c4.selectbox("Standard Errors",["OLS (Classical)","HC1 (White Robust)","HAC (Newey-West)"])
        log_vars=c5.multiselect("Apply ln() to",[dep]+indep,default=[])
        lags_dep=st.slider("Lags of dependent var (0=none)",0,8,0)
        if st.button("▶  ESTIMATE OLS",use_container_width=True):
            if not indep: st.error("Select at least one X variable."); return
            try:
                data=df[[dep]+indep].dropna().copy()
                for col in log_vars:
                    if col in data.columns: data[col]=np.log(data[col].replace(0,np.nan))
                data=data.dropna(); lag_cols=[]
                if lags_dep>0:
                    for lag in range(1,lags_dep+1):
                        name=f"{dep}_L{lag}"; data[name]=data[dep].shift(lag); lag_cols.append(name)
                    data=data.dropna()
                Y=data[dep]; X_cols=indep+lag_cols; X=data[X_cols]
                if add_const: X=sm.add_constant(X)
                cov_map={"OLS (Classical)":"nonrobust","HC1 (White Robust)":"HC1","HAC (Newey-West)":"HAC"}
                cov_type=cov_map.get(se_type,"HC1")
                if cov_type=="HAC":
                    model=OLS(Y,X).fit(cov_type="HAC",cov_kwds={"maxlags":int(np.ceil(len(Y)**(1/4)))})
                else:
                    model=OLS(Y,X).fit(cov_type=cov_type)
                st.session_state.ols_res=model; st.session_state.diag_model=model
                st.session_state.diag_resid=model.resid.values; st.session_state.diag_fitted=model.fittedvalues.values
                st.session_state.diag_X=X; st.session_state.se_type_used=se_type
                st.success("✓ OLS estimated. View results in **Results** tab.")
                st.session_state.report_log["ols"]={"dep":dep,"indep":X_cols,"n":int(model.nobs),
                    "r2":model.rsquared,"adj_r2":model.rsquared_adj,"fstat":model.fvalue,"fp":model.f_pvalue}
            except Exception as exc: st.error(f"**OLS failed:** {exc}")
    with tab_res:
        model=st.session_state.ols_res
        if model is None: st.info("Estimate OLS in **Configuration** tab."); return
        c1,c2,c3,c4,c5,c6=st.columns(6)
        c1.metric("R²",f"{model.rsquared:.4f}"); c2.metric("Adj. R²",f"{model.rsquared_adj:.4f}")
        c3.metric("F-stat",f"{model.fvalue:.4f}"); c4.metric("Prob(F)",f"{model.f_pvalue:.4f}")
        c5.metric("AIC",f"{model.aic:.2f}"); c6.metric("N",f"{int(model.nobs)}")
        c7,c8,c9,c10=st.columns(4)
        c7.metric("Log-Lik",f"{model.llf:.4f}"); c8.metric("BIC",f"{model.bic:.2f}")
        c9.metric("SSR",f"{model.ssr:.6f}"); c10.metric("DW",f"{durbin_watson(model.resid.values):.4f}")
        dep_name=model.model.endog_names; indep_names=[n for n in model.params.index if n!="const"]
        terms=[]
        if "const" in model.params.index: terms.append(f"<span class='eq-gold'>{fmt(model.params['const'])}</span>")
        for n in indep_names: terms.append(f"<span class='eq-gold'>{fmt(model.params[n])}</span>·{n}")
        eq_str=" + ".join(terms)
        st.markdown(f"""<div class="eq-block"><p class="eq-muted">Estimated OLS Equation ({st.session_state.se_type_used}):</p>
          <p>{dep_name} = {eq_str} + ε</p>
          <p class="eq-muted">N = {int(model.nobs)} | R² = {model.rsquared:.4f}</p></div>""", unsafe_allow_html=True)
        st.markdown('<p class="sec-title">Coefficient Table</p>', unsafe_allow_html=True)
        ci=model.conf_int(); rows_html=""
        for vname in model.params.index:
            rows_html+=coef_row_html(vname,model.params[vname],model.bse[vname],
                model.tvalues[vname],model.pvalues[vname],ci.loc[vname,0],ci.loc[vname,1])
        st.markdown(coef_table_html(rows_html),unsafe_allow_html=True)
        fig=make_subplots(rows=1,cols=2,subplot_titles=["Residuals vs Fitted","Normal Q-Q"])
        fig.add_trace(go.Scatter(x=model.fittedvalues.values,y=model.resid.values,mode="markers",
            marker=dict(color=C["cyan"],size=5,opacity=0.65),name="Residuals"),1,1)
        fig.add_hline(y=0,line=dict(color=C["navy"],dash="dash"),row=1,col=1)
        osm,osr=sci_stats.probplot(model.resid.values,dist="norm")[:2]
        slope,intercept,_,_,_=sci_stats.linregress(osm[0],osm[1])
        fig.add_trace(go.Scatter(x=osm[0],y=osm[1],mode="markers",marker=dict(color=C["cyan"],size=4,opacity=0.65),name="Quantiles"),1,2)
        x_line=[min(osm[0]),max(osm[0])]
        fig.add_trace(go.Scatter(x=x_line,y=[slope*x+intercept for x in x_line],line=dict(color=C["gold"],width=2,dash="dash"),name="Normal line"),1,2)
        fig=navy_fig(fig,350); st.plotly_chart(fig,use_container_width=True)
        ai_ols=_interp_ols(model,dep_name,indep_names)
        st.markdown(f'<div class="ai-box"><h4>⬡ AI INTERPRETATION · OLS</h4>{ai_ols}</div>',unsafe_allow_html=True)
    with tab_diag:
        model=st.session_state.ols_res
        if model is None: st.info("Estimate OLS first."); return
        _run_ols_diagnostics(model)

def _interp_ols(model,dep,indep):
    insig=[n for n in indep if model.pvalues.get(n,1)>=0.05]
    qual="strong" if model.rsquared_adj>0.7 else "moderate" if model.rsquared_adj>0.4 else "weak"
    f_ok=model.f_pvalue<0.05
    lines=[f"<p>OLS regresses <strong>{dep}</strong> on {len(indep)} regressors. Explains "
           f"<strong>{model.rsquared*100:.2f}%</strong> of variance (Adj.R²={model.rsquared_adj:.4f}), "
           f"indicating <em>{qual}</em> fit. F-test is {'significant' if f_ok else 'not significant'}.</p>"]
    for n in indep[:6]:
        c=model.params.get(n,0); p=model.pvalues.get(n,1)
        lines.append(f"<p>• <strong>{n}</strong> (β={fmt(c)}, p={fmt(p)}): a 1-unit increase → "
                     f"<strong>{fmt(c)}</strong>-unit {'increase' if c>0 else 'decrease'} in {dep}. "
                     f"{'<span class=\"ok-text\">Significant at 5%.</span>' if p<0.05 else '<span class=\"warn-text\">Not significant.</span>'}</p>")
    if insig: lines.append(f"<p><span class='warn-text'>⚠ Not significant at 5%:</span> {', '.join(insig)}. Consider dropping or testing jointly.</p>")
    return "".join(lines)

def _run_ols_diagnostics(model):
    st.markdown('<p class="sec-title">Diagnostic Summary</p>', unsafe_allow_html=True)
    resid=model.resid.values; X_exog=model.model.exog
    c1,c2,c3,c4=st.columns(4)
    dw=durbin_watson(resid); dw_ok=1.5<dw<2.5
    c1.markdown(f'<div class="stat-box"><p class="stat-name">Durbin-Watson</p><p class="stat-val">{dw:.4f}</p>'
                f'<p class="stat-sub">{"✅ No autocorr." if dw_ok else "❌ Autocorr. detected"}</p></div>',unsafe_allow_html=True)
    jb_stat,jb_p=jarque_bera(resid); jb_ok=jb_p>0.05
    c2.markdown(f'<div class="stat-box"><p class="stat-name">Jarque-Bera</p><p class="stat-val">{jb_p:.4f}</p>'
                f'<p class="stat-sub">{"✅ Normality OK" if jb_ok else "⚠ Non-normal"}</p></div>',unsafe_allow_html=True)
    try:
        wh=het_white(resid,X_exog); wh_ok=wh[1]>0.05
        c3.markdown(f'<div class="stat-box"><p class="stat-name">White\'s Test (p)</p><p class="stat-val">{wh[1]:.4f}</p>'
                    f'<p class="stat-sub">{"✅ Homoskedastic" if wh_ok else "❌ Heteroskedastic"}</p></div>',unsafe_allow_html=True)
    except: c3.markdown('<div class="stat-box"><p class="stat-name">White\'s Test</p><p class="stat-val">N/A</p></div>',unsafe_allow_html=True)
    try:
        bg=acorr_breusch_godfrey(model,nlags=4); bg_ok=bg[1]>0.05
        c4.markdown(f'<div class="stat-box"><p class="stat-name">Breusch-Godfrey (p)</p><p class="stat-val">{bg[1]:.4f}</p>'
                    f'<p class="stat-sub">{"✅ No autocorr." if bg_ok else "❌ Serial correlation"}</p></div>',unsafe_allow_html=True)
    except: c4.markdown('<div class="stat-box"><p class="stat-name">BG Test</p><p class="stat-val">N/A</p></div>',unsafe_allow_html=True)
    try:
        bp=het_breuschpagan(resid,X_exog); bp_ok=bp[1]>0.05
        st.markdown(f'<div class="glass-card"><p class="sec-title" style="font-size:0.82rem;">Breusch-Pagan Heteroskedasticity Test</p>'
                    f'<p>LM stat: <strong>{bp[0]:.4f}</strong> | p-value: <strong>{bp[1]:.4f}</strong> | '
                    f'{badge_html("Homoskedastic","pass") if bp_ok else badge_html("Heteroskedastic","fail")}</p></div>',unsafe_allow_html=True)
    except: pass
    try:
        reset=linear_reset(model,power=3,use_f=True); res_ok=reset.pvalue>0.05
        st.markdown(f'<div class="glass-card"><p class="sec-title" style="font-size:0.82rem;">Ramsey RESET Test (Misspecification)</p>'
                    f'<p>F-stat: <strong>{reset.statistic:.4f}</strong> | p-value: <strong>{reset.pvalue:.4f}</strong> | '
                    f'{badge_html("No Misspecification","pass") if res_ok else badge_html("Misspecification Likely","fail")}'
                    f'</p><p style="font-size:0.75rem;color:#64748B;">H₀: Model is correctly specified</p></div>',unsafe_allow_html=True)
    except: pass
    try:
        arch_r=het_arch(resid); arch_ok=arch_r[1]>0.05
        st.markdown(f'<div class="glass-card"><p class="sec-title" style="font-size:0.82rem;">ARCH-LM Test (Volatility Clustering)</p>'
                    f'<p>LM stat: <strong>{arch_r[0]:.4f}</strong> | p-value: <strong>{arch_r[1]:.4f}</strong> | '
                    f'{badge_html("No ARCH Effects","pass") if arch_ok else badge_html("ARCH Effects Present","warn")}'
                    f'</p><p style="font-size:0.75rem;color:#64748B;">H₀: No ARCH effects in residuals</p></div>',unsafe_allow_html=True)
    except: pass

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ARDL
# ─────────────────────────────────────────────────────────────────────────────
def page_ardl():
    st.markdown('<p class="page-title">🔄 ARDL / Bounds Test</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Autoregressive Distributed Lag · PSS Bounds Test · Short-run dynamics · Long-run multipliers · ECT</p>', unsafe_allow_html=True)
    if st.session_state.clean_df is None: st.info("Load data first."); return
    df=st.session_state.clean_df; cols=df.select_dtypes(include=np.number).columns.tolist()
    tab_cfg,tab_res,tab_bounds=st.tabs(["① Config","② Results","③ Bounds Test & ECM"])
    with tab_cfg:
        c1,c2=st.columns(2)
        dep=c1.selectbox("Dependent Variable (Y)",cols); indep=c2.multiselect("Independent Variables (X)",[c for c in cols if c!=dep])
        c3,c4,c5=st.columns(3)
        max_lag_y=c3.slider("Max lags Y",1,8,4); max_lag_x=c4.slider("Max lags X",0,8,4); ic_crit=c5.selectbox("Lag criterion",["aic","bic","hqic"])
        trend_opt=st.selectbox("Trend",["c (constant)","ct (const+trend)","nc (none)"]); trend_code=trend_opt.split(" ")[0]
        if st.button("▶  ESTIMATE ARDL",use_container_width=True):
            if not indep: st.error("Select at least one X variable."); return
            try:
                data=df[[dep]+indep].dropna()
                sel_order=ardl_select_order(data[dep],max_lag_y,data[indep],max_lag_x,ic=ic_crit,trend=trend_code)
                best=sel_order.ardl_order
                model=ARDL(data[dep],best[0],data[indep],best[1:],trend=trend_code).fit()
                st.session_state.ardl_res={"model":model,"dep":dep,"indep":indep,"order":best,"ic":ic_crit,"trend":trend_code,"data":data}
                st.session_state.diag_model=model; st.session_state.diag_resid=model.resid.values; st.session_state.diag_fitted=model.fittedvalues.values
                st.success(f"✓ ARDL({', '.join(str(o) for o in best)}) estimated. Check **Results** tab.")
            except Exception as exc: st.error(f"**ARDL error:** {exc}")
    with tab_res:
        ar=st.session_state.ardl_res
        if ar is None: st.info("Estimate ARDL first."); return
        model=ar["model"]; dep=ar["dep"]; indep=ar["indep"]; best=ar["order"]; ic_crit=ar["ic"]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("AIC",f"{model.aic:.3f}"); c2.metric("BIC",f"{model.bic:.3f}"); c3.metric("Log-Lik",f"{model.llf:.3f}"); c4.metric("N",f"{int(model.nobs)}")
        p=best[0]; x_lags_str=", ".join(str(v) for v in best[1:]) if len(best)>1 else "—"
        st.markdown(f"""<div class="eq-block"><p class="eq-muted">ARDL({p}, {x_lags_str}) — {ic_crit.upper()} selected</p>
          <p>{dep}(t) = c + Σαᵢ·{dep}(t-i) + ΣΣβⱼ·Xⱼ(t-k) + ε(t)</p>
          <p class="eq-muted">Trend: {ar['trend']} | Vars: {dep}, {', '.join(indep)}</p></div>""", unsafe_allow_html=True)
        st.markdown('<p class="sec-title">Coefficient Table</p>', unsafe_allow_html=True)
        ci=model.conf_int(); rows_html=""
        for vname in model.params.index:
            rows_html+=coef_row_html(vname,model.params[vname],model.bse[vname],model.tvalues[vname],model.pvalues[vname],ci.loc[vname,0],ci.loc[vname,1])
        st.markdown(coef_table_html(rows_html),unsafe_allow_html=True)
        st.markdown('<p class="sec-title">Long-Run Multipliers</p>', unsafe_allow_html=True)
        try:
            lr_params=model.params
            lag_sum=sum(lr_params.get(f"L{i}.{dep}",lr_params.get(f"{dep}.L{i}",0)) for i in range(1,p+1))
            denom=max(1e-10,abs(1-lag_sum)); lr_rows=""
            for v in indep:
                x_sum=sum(lr_params.get(f"L{j}.{v}",lr_params.get(v,0) if j==0 else 0) for j in range(0,best[indep.index(v)+1]+1 if len(best)>indep.index(v)+1 else 1))
                lr_c=x_sum/denom; lr_rows+=f"<tr><td>{v}</td><td>{fmt(lr_c)}</td><td>Long-run multiplier</td></tr>"
            st.markdown(f'<div class="coef-wrap"><table class="coef-tbl"><tr><th>Variable</th><th>Long-Run Coef.</th><th>Interpretation</th></tr>{lr_rows}</table></div>',unsafe_allow_html=True)
        except Exception as exc: st.warning(f"Long-run derivation: {exc}")
        ai_ardl=_interp_ardl(model,dep,indep,best,ic_crit)
        st.markdown(f'<div class="ai-box"><h4>⬡ AI INTERPRETATION · ARDL</h4>{ai_ardl}</div>',unsafe_allow_html=True)
        st.session_state.report_log["ardl"]={"dep":dep,"indep":indep,"order":list(best),"ic":ic_crit}
    with tab_bounds:
        ar=st.session_state.ardl_res
        if ar is None: st.info("Estimate ARDL first."); return
        model=ar["model"]
        st.markdown('<p class="sec-title">PSS Bounds Test for Cointegration</p>', unsafe_allow_html=True)
        st.markdown("""<div class="ai-box" style="margin-bottom:14px;"><h4>⬡ BOUNDS TEST THEORY</h4>
          <p>Pesaran, Shin & Smith (2001) tests H₀: no long-run relationship. The F-stat is compared to I(0) and I(1) critical bounds.</p>
          <ul><li><strong>F > I(1) bound</strong> → cointegration regardless of integration order</li>
              <li><strong>F &lt; I(0) bound</strong> → no cointegration</li>
              <li><strong>F between bounds</strong> → inconclusive</li></ul></div>""", unsafe_allow_html=True)
        try:
            bt=model.bounds_test(n_obs=len(ar["data"]),case=3,alpha=0.05); f_stat=bt.stat; conc=bt.conclusion
            badge_kind="pass" if conc=="cointegration" else "warn" if conc=="inconclusive" else "fail"
            conclusion_str={"cointegration":"COINTEGRATION CONFIRMED","inconclusive":"INCONCLUSIVE","no cointegration":"NO COINTEGRATION"}.get(conc,conc.upper())
            st.markdown(f"""<div class="brutalist-card-gold">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <p style="font-weight:700;font-size:0.95rem;margin:0;">PSS Bounds Test Result</p>
                {badge_html(conclusion_str,badge_kind)}</div>
              <p style="font-family:'Space Mono',monospace;font-size:1.1rem;margin-top:8px;">F-statistic: <strong>{fmt(f_stat)}</strong></p></div>""", unsafe_allow_html=True)
        except Exception as exc: st.warning(f"Bounds test: {exc}")
        st.markdown('<p class="sec-title">Error Correction Representation</p>', unsafe_allow_html=True)
        try:
            lr_params=model.params; p=ar["order"][0]
            lag_sum=sum(lr_params.get(f"L{i}.{ar['dep']}",lr_params.get(f"{ar['dep']}.L{i}",0)) for i in range(1,p+1))
            ect_coef=-(1-lag_sum)
            st.markdown(f"""<div class="eq-block"><p class="eq-muted">Error Correction Term (Speed of Adjustment):</p>
              <p>Δ{ar['dep']}(t) = <span class='eq-gold'>{fmt(ect_coef)}</span>·ECTₜ₋₁ + short-run dynamics + ε(t)</p>
              <p class="eq-muted">ECT = {ar['dep']}(t-1) − [long-run relationship]</p></div>""", unsafe_allow_html=True)
            st.markdown(f'<div class="ai-box"><h4>⬡ ECT INTERPRETATION</h4>{_interp_ect(ect_coef)}</div>',unsafe_allow_html=True)
        except Exception as exc: st.warning(f"ECM derivation: {exc}")

def _interp_ardl(model,dep,indep,order,ic):
    p=order[0]; sig=[n for n in model.params.index if model.pvalues.get(n,1)<0.05 and n!="const"]
    return (f"<p>The <strong>ARDL({', '.join(str(o) for o in order)})</strong> was selected via <strong>{ic.upper()}</strong>. "
            f"Includes {p} lag(s) of {dep} and distributed lags of {', '.join(indep)}.</p>"
            f"<p>Significant short-run terms (p&lt;0.05): {', '.join(sig[:6]) if sig else 'none at 5%'}.</p>"
            f"<p>Long-run multipliers (above) measure the total effect of each X once all adjustments complete. "
            f"Check the <strong>Bounds Test</strong> tab for long-run cointegration evidence.</p>")

def _interp_ect(ect):
    pct=abs(ect)*100; direction="toward" if ect<0 else "away from"
    speed="rapid" if pct>50 else "moderate" if pct>20 else "slow"; validity=ect<0
    lines=[f"<p>ECT = <span class='highlight'>{fmt(ect)}</span>: approximately <strong>{pct:.1f}%</strong> of any short-run "
           f"deviation from long-run equilibrium is corrected each period (moves <em>{direction}</em> equilibrium). Speed: <em>{speed}</em>.</p>"]
    if validity: lines.append("<p><span class='ok-text'>✅ Negative ECT confirms error-correcting behaviour — returns to long-run equilibrium after shock.</span></p>")
    else: lines.append("<p><span class='bad-text'>❌ Positive ECT indicates explosive dynamics — system diverges from equilibrium. Recheck cointegration.</span></p>")
    return "".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: VAR
# ─────────────────────────────────────────────────────────────────────────────
def page_var():
    st.markdown('<p class="page-title">🌐 VAR System</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Vector Autoregression · Lag selection · Granger causality · IRF · FEVD</p>', unsafe_allow_html=True)
    if st.session_state.clean_df is None: st.info("Load data first."); return
    df=st.session_state.clean_df; cols=df.select_dtypes(include=np.number).columns.tolist()
    tab_cfg,tab_lag,tab_res,tab_irf,tab_fevd,tab_granger=st.tabs(["① Config","② Lag Selection","③ Results","④ IRF","⑤ FEVD","⑥ Granger"])
    with tab_cfg:
        endog=st.multiselect("Endogenous Variables (2–8)",cols)
        c1,c2=st.columns(2)
        max_lags=c1.slider("Max lags to test",1,12,6); ic_var=c2.selectbox("Information Criterion",["aic","bic","hqic","fpe"])
        trend_v=st.selectbox("Trend",["c","ct","ctt","n"],index=0)
        if st.button("▶  ESTIMATE VAR",use_container_width=True):
            if len(endog)<2: st.error("Select at least 2 variables."); return
            try:
                data_v=df[endog].dropna(); var_m=VAR(data_v); ic_sel=var_m.select_order(max_lags)
                best_p=max(1,getattr(ic_sel,ic_var)); fitted=var_m.fit(best_p,trend=trend_v)
                st.session_state.var_res={"model":fitted,"endog":endog,"best_p":best_p,"ic":ic_var,"data":data_v}
                st.success(f"✓ VAR({best_p}) estimated via {ic_var.upper()}.")
            except Exception as exc: st.error(f"**VAR error:** {exc}")
    with tab_lag:
        if st.session_state.var_res is None: st.info("Estimate VAR first.")
        else:
            vr=st.session_state.var_res
            try:
                data_v=vr["data"]; var_m=VAR(data_v); ic_sel=var_m.select_order(8)
                ic_df=pd.DataFrame({"AIC":[getattr(ic_sel,"aic_by_lag",{}).get(i,np.nan) for i in range(9)],
                    "BIC":[getattr(ic_sel,"bic_by_lag",{}).get(i,np.nan) for i in range(9)],
                    "HQIC":[getattr(ic_sel,"hqic_by_lag",{}).get(i,np.nan) for i in range(9)]},index=range(9))
                st.markdown('<p class="sec-title">Information Criteria by Lag</p>', unsafe_allow_html=True)
                st.dataframe(ic_df.style.format("{:.4f}").highlight_min(color="#D1FAE5"),use_container_width=True)
                c1,c2,c3=st.columns(3)
                c1.metric("Optimal (AIC)",f"p = {ic_sel.aic}"); c2.metric("Optimal (BIC)",f"p = {ic_sel.bic}"); c3.metric("Optimal (HQIC)",f"p = {ic_sel.hqic}")
            except Exception as exc: st.warning(f"Lag table: {exc}")
    with tab_res:
        vr=st.session_state.var_res
        if vr is None: st.info("Estimate VAR first."); return
        fitted=vr["model"]; endog=vr["endog"]; best_p=vr["best_p"]
        c1,c2,c3=st.columns(3); c1.metric("VAR Order (p)",best_p); c2.metric("AIC",f"{fitted.aic:.4f}"); c3.metric("BIC",f"{fitted.bic:.4f}")
        st.markdown(f"""<div class="eq-block"><p class="eq-muted">VAR({best_p}) System — {len(endog)} variables:</p>
          <p>Y(t) = c + A₁·Y(t-1) + … + A{best_p}·Y(t-{best_p}) + ε(t)</p>
          <p class="eq-muted">Y(t) = [{', '.join(endog)}]ᵀ</p></div>""", unsafe_allow_html=True)
        for eq_var in endog:
            with st.expander(f"▸ Equation: {eq_var}"):
                try:
                    p_df=fitted.params[eq_var]; rows_h=""
                    for vname in p_df.index: rows_h+=f"<tr><td>{vname}</td><td>{fmt(p_df[vname])}</td></tr>"
                    st.markdown(f'<table class="coef-tbl"><tr><th>Regressor</th><th>Coef. ({eq_var})</th></tr>{rows_h}</table>',unsafe_allow_html=True)
                except Exception as exc: st.warning(f"Equation display: {exc}")
        st.session_state.report_log["var"]={"endog":endog,"best_p":best_p,"ic":vr["ic"]}
    with tab_irf:
        vr=st.session_state.var_res
        if vr is None: st.info("Estimate VAR first."); return
        fitted=vr["model"]; endog=vr["endog"]
        c1,c2,c3=st.columns(3)
        imp_var=c1.selectbox("Impulse (shock in)",endog,key="irf_imp"); resp_var=c2.selectbox("Response variable",endog,key="irf_resp"); periods=c3.slider("Periods",5,40,20)
        orth=st.checkbox("Orthogonalized IRF (Cholesky)",value=True)
        if st.button("▶  COMPUTE IRF",use_container_width=True):
            try:
                irf=fitted.irf(periods); imp_i=endog.index(imp_var); resp_i=endog.index(resp_var)
                irf_vals=irf.orth_irfs[:,resp_i,imp_i] if orth else irf.irfs[:,resp_i,imp_i]
                period_x=list(range(len(irf_vals))); ci_up=irf_vals+1.96*np.std(irf_vals); ci_dn=irf_vals-1.96*np.std(irf_vals)
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=period_x+period_x[::-1],y=list(ci_up)+list(ci_dn[::-1]),fill="toself",fillcolor="rgba(56,189,248,0.12)",line=dict(color="rgba(0,0,0,0)"),name="95% CI"))
                fig.add_trace(go.Scatter(x=period_x,y=irf_vals,line=dict(color=C["cyan"],width=2.5),name="IRF"))
                fig.add_hline(y=0,line=dict(color=C["navy"],dash="dash",width=1))
                fig=navy_fig(fig,380,f"IRF: {resp_var} → {imp_var} shock ({'Orth.' if orth else 'Raw'})"); st.plotly_chart(fig,use_container_width=True)
            except Exception as exc: st.error(f"IRF error: {exc}")
    with tab_fevd:
        vr=st.session_state.var_res
        if vr is None: st.info("Estimate VAR first."); return
        fitted=vr["model"]; endog=vr["endog"]
        periods_f=st.slider("FEVD horizon",5,40,20,key="fevd_h")
        target=st.selectbox("Variable to decompose",endog,key="fevd_var")
        if st.button("▶  COMPUTE FEVD",use_container_width=True):
            try:
                fevd=fitted.fevd(periods_f); fevd_df=fevd.decomp; t_idx=endog.index(target); decomp=fevd_df[:,t_idx,:]
                fig=go.Figure()
                for i,src in enumerate(endog):
                    fig.add_trace(go.Scatter(x=list(range(periods_f)),y=decomp[:,i]*100,name=src,mode="lines",stackgroup="one",line=dict(color=CHART_COLORS[i%len(CHART_COLORS)],width=1.5)))
                fig=navy_fig(fig,380,f"FEVD: {target} variance by shock (%)"); fig.update_layout(yaxis_title="% of Variance"); st.plotly_chart(fig,use_container_width=True)
            except Exception as exc: st.error(f"FEVD error: {exc}")
    with tab_granger:
        vr=st.session_state.var_res
        if vr is None: st.info("Estimate VAR first."); return
        fitted=vr["model"]; endog=vr["endog"]; data_v=vr["data"]
        max_gr_lag=st.slider("Max lags",1,8,vr.get("best_p",4)); alpha_gr=st.select_slider("α",[0.01,0.05,0.10],value=0.05,key="gr_a")
        if st.button("▶  GRANGER CAUSALITY MATRIX",use_container_width=True):
            p_matrix=np.ones((len(endog),len(endog)))
            try:
                for i,caused in enumerate(endog):
                    for j,causing in enumerate(endog):
                        if i==j: continue
                        try:
                            gc=grangercausalitytests(data_v[[caused,causing]].dropna(),maxlag=max_gr_lag,verbose=False)
                            p_matrix[i,j]=min(gc[lag][0]["ssr_ftest"][1] for lag in gc)
                        except: p_matrix[i,j]=np.nan
                fig=go.Figure(go.Heatmap(z=p_matrix,x=endog,y=endog,
                    colorscale=[[0,C["green"]],[0.05,"#FEF3C7"],[0.1,"#FEE2E2"],[1,"#FEE2E2"]],
                    text=np.round(p_matrix,3),texttemplate="%{text}",zmin=0,zmax=0.15))
                fig=navy_fig(fig,380,"Granger Causality p-value Matrix (row → caused by col)"); st.plotly_chart(fig,use_container_width=True)
                st.dataframe(pd.DataFrame(p_matrix,index=endog,columns=endog).round(4).style.background_gradient(cmap="RdYlGn_r",vmin=0,vmax=0.15).format("{:.4f}"),use_container_width=True)
            except Exception as exc: st.error(f"Granger error: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: JOHANSEN / VECM
# ─────────────────────────────────────────────────────────────────────────────
def page_vecm():
    st.markdown('<p class="page-title">⛓ Johansen Cointegration & VECM</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Johansen Trace & Max-Eigenvalue · Cointegrating vectors β · Loading matrix α · VECM dynamics</p>', unsafe_allow_html=True)
    if st.session_state.clean_df is None: st.info("Load data first."); return
    df=st.session_state.clean_df; cols=df.select_dtypes(include=np.number).columns.tolist()
    tab_joh,tab_vecm_est=st.tabs(["① Johansen Test","② VECM Estimation"])
    with tab_joh:
        endog_j=st.multiselect("I(1) Variables (min 2)",cols,key="joh_vars")
        c1,c2=st.columns(2)
        k_ar=c1.slider("Lags in VECM (k)",1,8,2); det=c2.selectbox("Deterministics",["ci","li","n"],index=0)
        alpha_j=st.select_slider("Significance α",[0.01,0.05,0.10],value=0.05,key="joh_a")
        if st.button("▶  RUN JOHANSEN TEST",use_container_width=True):
            if len(endog_j)<2: st.error("Select at least 2 variables."); return
            try:
                data_j=df[endog_j].dropna()
                rank_j=select_coint_rank(data_j,det=det,k_ar_diff=k_ar,method="trace",signif=alpha_j)
                st.session_state.johansen_res={"rank_test":rank_j,"endog":endog_j,"k_ar":k_ar,"det":det,"data":data_j}
                st.success(f"✓ Johansen test complete. Cointegration rank r = {rank_j.rank}")
            except Exception as exc: st.error(f"**Johansen error:** {exc}")
        jr=st.session_state.johansen_res
        if jr is None: return
        rank_test=jr["rank_test"]; endog_j=jr["endog"]
        c1,c2=st.columns(2); c1.metric("Estimated Rank (r)",rank_test.rank); c2.metric("# Variables (k)",len(endog_j))
        try:
            trace_stat=rank_test.test_stats; crit_vals=rank_test.crit_vals; rows_j=""
            for i,(ts,cv) in enumerate(zip(trace_stat,crit_vals)):
                ok_row=ts>cv; rows_j+=(f"<tr><td>r ≤ {i}</td><td>{fmt(ts)}</td><td>{fmt(cv)}</td>"
                    f"<td>{badge_html('Reject H₀','pass') if ok_row else badge_html('Fail to Reject','fail')}</td></tr>")
            st.markdown(f'<div class="coef-wrap"><table class="coef-tbl"><tr><th>H₀</th><th>Test Stat.</th><th>Critical Value ({int(alpha_j*100)}%)</th><th>Decision</th></tr>{rows_j}</table></div>',unsafe_allow_html=True)
        except Exception as exc: st.warning(f"Johansen table: {exc}")
        st.markdown(f'<div class="ai-box"><h4>⬡ AI INTERPRETATION · JOHANSEN</h4>{_interp_johansen(rank_test.rank,len(endog_j))}</div>',unsafe_allow_html=True)
    with tab_vecm_est:
        jr=st.session_state.johansen_res
        if jr is None: st.info("Run Johansen test first."); return
        c1,c2=st.columns(2)
        r_rank=c1.number_input("Cointegration Rank (r)",min_value=1,max_value=len(jr["endog"])-1,value=max(1,jr["rank_test"].rank))
        k_diff=c2.slider("Lags in differences (k)",1,8,jr["k_ar"])
        if st.button("▶  ESTIMATE VECM",use_container_width=True):
            try:
                data_v=jr["data"]
                vecm=VECM(data_v,k_ar_diff=k_diff,coint_rank=int(r_rank),deterministic=jr["det"]); fitted_v=vecm.fit()
                st.session_state.vecm_res={"model":fitted_v,"endog":jr["endog"],"rank":int(r_rank),"k":k_diff}
                st.success(f"✓ VECM estimated with r={r_rank} cointegrating vectors.")
            except Exception as exc: st.error(f"**VECM error:** {exc}")
        vr=st.session_state.vecm_res
        if vr is None: return
        fitted_v=vr["model"]; endog_v=vr["endog"]; r=vr["rank"]
        st.markdown('<p class="sec-title">Cointegrating Vectors β (normalized)</p>', unsafe_allow_html=True)
        try:
            beta=fitted_v.beta
            beta_df=pd.DataFrame(beta[:len(endog_v),:],index=endog_v[:beta.shape[0]],columns=[f"CI Vector {i+1}" for i in range(r)])
            st.dataframe(beta_df.style.format("{:.6f}"),use_container_width=True)
        except Exception as exc: st.warning(f"β matrix: {exc}")
        st.markdown('<p class="sec-title">Loading Matrix α</p>', unsafe_allow_html=True)
        try:
            alpha_m=fitted_v.alpha
            alpha_df=pd.DataFrame(alpha_m,index=endog_v[:alpha_m.shape[0]],columns=[f"CI Vector {i+1}" for i in range(r)])
            st.dataframe(alpha_df.style.format("{:.6f}"),use_container_width=True)
        except Exception as exc: st.warning(f"α matrix: {exc}")
        st.markdown(f"""<div class="eq-block"><p class="eq-muted">VECM({k_diff}) with r={r} cointegrating vector(s):</p>
          <p>ΔY(t) = <span class='eq-gold'>αβᵀ</span>·Y(t-1) + Γ₁·ΔY(t-1) + … + Γ{k_diff}·ΔY(t-{k_diff}) + ε(t)</p>
          <p class="eq-muted">α = loading matrix | β = cointegrating vectors | Variables: {', '.join(endog_v[:3])}…</p></div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="ai-box"><h4>⬡ AI INTERPRETATION · VECM</h4>{_interp_vecm(fitted_v,endog_v,r)}</div>',unsafe_allow_html=True)
        st.session_state.report_log["vecm"]={"endog":endog_v,"rank":r,"k":k_diff}

def _interp_johansen(rank,k):
    if rank==0: return (f"<p><span class='bad-text'>No cointegrating relationships detected</span> among {k} variables. "
                        "If all are I(1), they drift independently. Consider VAR in differences.</p>")
    elif rank==k: return (f"<p><span class='warn-text'>Rank = {k} (number of variables)</span> — implies all are stationary I(0). "
                          "OLS in levels is valid; VECM is not appropriate.</p>")
    else: return (f"<p><span class='ok-text'>✅ Johansen identifies r = {rank} cointegrating vector(s)</span> among {k} variables. "
                  f"VECM with r={rank} is the appropriate specification. The {rank} linear combination(s) are stationary, "
                  f"providing {k-rank} common stochastic trend(s).</p>"
                  "<p>Normalized β vectors define long-run equilibrium. Negative α loading coefficients indicate corrective (stable) behaviour.</p>")

def _interp_vecm(model,endog,r):
    try:
        alpha=model.alpha; adj_strs=[]
        for i,v in enumerate(endog[:alpha.shape[0]]):
            a=alpha[i,0] if r>0 else 0; speed=abs(a)*100; direction="corrects toward" if a<0 else "diverges from"
            adj_strs.append(f"<li><strong>{v}</strong>: α={fmt(a)} → {speed:.1f}%/period, {direction} equilibrium {'✅' if a<0 else '❌'}</li>")
        return ("<p>VECM decomposes long-run cointegration from short-run dynamics.</p>"
                f"<p><strong>Adjustment speeds (α):</strong></p><ul>{''.join(adj_strs)}</ul>"
                "<p>Negative α values confirm error-correcting behaviour. Non-significant α → variable is weakly exogenous.</p>")
    except: return "<p>VECM results available above.</p>"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: GARCH / VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────
def page_garch():
    st.markdown('<p class="page-title">🌊 GARCH / Volatility Models</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">GARCH(1,1) · EGARCH (asymmetric) · TGARCH (threshold) · Conditional volatility · News impact curve</p>', unsafe_allow_html=True)
    if st.session_state.clean_df is None: st.info("Load data first."); return
    df=st.session_state.clean_df; cols=df.select_dtypes(include=np.number).columns.tolist()
    c1,c2,c3,c4=st.columns(4)
    ret_var=c1.selectbox("Return Series",cols); mdl_type=c2.selectbox("Model",["GARCH(1,1)","EGARCH(1,1)","TGARCH(1,1)"])
    dist=c3.selectbox("Error Distribution",["normal","t","skewt","ged"]); mean_m=c4.selectbox("Mean Model",["Constant","AR(1)","Zero"])
    scale100=st.checkbox("Scale by 100 (% returns)",value=True)
    if st.button("▶  ESTIMATE VOLATILITY MODEL",use_container_width=True):
        try:
            s=df[ret_var].dropna()
            if scale100: s=s*100
            mean_map={"Constant":"Constant","AR(1)":"AR","Zero":"Zero"}; mean_lags=1 if mean_m=="AR(1)" else 0
            if mdl_type=="GARCH(1,1)":   am=arch_model(s,mean=mean_map[mean_m],lags=mean_lags,vol="Garch",p=1,q=1,dist=dist)
            elif mdl_type=="EGARCH(1,1)": am=arch_model(s,mean=mean_map[mean_m],lags=mean_lags,vol="EGARCH",p=1,q=1,dist=dist)
            elif mdl_type=="TGARCH(1,1)": am=arch_model(s,mean=mean_map[mean_m],lags=mean_lags,vol="GARCH",p=1,o=1,q=1,dist=dist)
            res=am.fit(disp="off",show_warning=False)
            st.session_state.garch_res={"model":res,"var":ret_var,"type":mdl_type,"dist":dist,"series":s}
            st.success(f"✓ {mdl_type} estimated. AIC={res.aic:.3f}")
        except Exception as exc: st.error(f"**GARCH error:** {exc}")
    gr=st.session_state.garch_res
    if gr is None: return
    res=gr["model"]; p=res.params; mdl=gr["type"]
    omega=p.get("omega",p.get("alpha[0]",0)); alpha1=p.get("alpha[1]",p.get("alpha",0))
    beta1=p.get("beta[1]",p.get("beta",0)); gamma1=p.get("gamma[1]",p.get("o[1]",0))
    persist=alpha1+beta1+(0.5*abs(gamma1) if "TGARCH" in mdl or "EGARCH" in mdl else 0)
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("ω (base vol.)",f"{omega:.6f}"); c2.metric("α₁ (ARCH)",f"{alpha1:.4f}")
    c3.metric("β₁ (GARCH)",f"{beta1:.4f}"); c4.metric("γ (Leverage)",f"{gamma1:.4f}")
    c5.metric("Persistence",f"{persist:.4f}",delta="Stationary" if persist<1 else "Explosive!",delta_color="normal" if persist<1 else "inverse")
    st.markdown(f"""<div class="eq-block"><p class="eq-muted">{mdl} specification:</p>
      <p>rₜ = μ + εₜ &nbsp;|&nbsp; εₜ = σₜ·zₜ &nbsp;|&nbsp; zₜ ~ {dist.upper()}</p>
      <p>σ²ₜ = <span class='eq-gold'>{fmt(omega)}</span> + <span class='eq-gold'>{fmt(alpha1)}</span>·ε²ₜ₋₁ +
         <span class='eq-gold'>{fmt(beta1)}</span>·σ²ₜ₋₁{'  +  <span class=\'eq-gold\'>'+fmt(gamma1)+'</span>·Iₜ₋₁·ε²ₜ₋₁' if abs(gamma1)>1e-8 else ''}</p>
      <p class="eq-muted">Persistence (α+β) = {persist:.4f}  |  {'Stationary ✅' if persist<1 else 'Near-Integrated ⚠'}</p></div>""", unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Parameter Estimates</p>', unsafe_allow_html=True)
    rows_g=""
    for nm,val,se,tv,pv in zip(res.params.index,res.params,res.std_err,res.tvalues,res.pvalues):
        stars=pstar(pv); cls="sig3" if pv<0.01 else "sig2" if pv<0.05 else "sig1" if pv<0.10 else "insig"
        rows_g+=f"<tr><td class='{cls}'>{nm} <span class='sig-star'>{stars}</span></td><td class='{cls}'>{fmt(val,6)}</td><td>{fmt(se,6)}</td><td>{fmt(tv)}</td><td>{fmt(pv)}</td></tr>"
    st.markdown(f'<div class="coef-wrap"><table class="coef-tbl"><tr><th>Parameter</th><th>Estimate</th><th>Std.Err.</th><th>t-Stat</th><th>p-Value</th></tr>{rows_g}</table></div>',unsafe_allow_html=True)
    tab_cv,tab_nic=st.tabs(["📈 Conditional Volatility","📐 News Impact Curve"])
    with tab_cv:
        cond_vol=res.conditional_volatility; s_series=gr["series"]
        fig=make_subplots(rows=2,cols=1,vertical_spacing=0.08,subplot_titles=[f"Returns: {ret_var}","Conditional Volatility σₜ"])
        fig.add_trace(go.Scatter(x=s_series.index,y=s_series.values,name="Returns",line=dict(color=C["navy"],width=1)),1,1)
        fig.add_trace(go.Scatter(x=s_series.index,y=cond_vol,name="Cond. Volatility",fill="tozeroy",fillcolor="rgba(56,189,248,0.13)",line=dict(color=C["cyan"],width=2)),2,1)
        fig=navy_fig(fig,500,f"{mdl} — {ret_var}"); st.plotly_chart(fig,use_container_width=True)
    with tab_nic:
        try:
            eps_range=np.linspace(-3*np.std(gr["series"]),3*np.std(gr["series"]),200)
            sig2_base=omega/(1-alpha1-beta1) if persist<1 else omega
            if mdl=="GARCH(1,1)": nic=omega+alpha1*eps_range**2+beta1*sig2_base
            elif mdl=="EGARCH(1,1)": nic=np.exp(np.log(sig2_base)+alpha1*np.abs(eps_range/np.sqrt(sig2_base))+gamma1*eps_range/np.sqrt(sig2_base))
            elif mdl=="TGARCH(1,1)": nic=omega+(alpha1+gamma1*(eps_range<0))*eps_range**2+beta1*sig2_base
            fig2=go.Figure()
            fig2.add_trace(go.Scatter(x=eps_range,y=np.sqrt(nic),name="News Impact",line=dict(color=C["cyan"],width=2.5)))
            fig2=navy_fig(fig2,350,"News Impact Curve — σₜ as function of εₜ₋₁")
            fig2.update_layout(xaxis_title="Shock (εₜ₋₁)",yaxis_title="Implied σₜ"); st.plotly_chart(fig2,use_container_width=True)
        except Exception as exc: st.warning(f"NIC: {exc}")
    st.markdown(f'<div class="ai-box"><h4>⬡ AI INTERPRETATION · VOLATILITY</h4>{_interp_garch(omega,alpha1,beta1,gamma1,persist,mdl,dist)}</div>',unsafe_allow_html=True)
    st.session_state.report_log["garch"]={"var":ret_var,"type":mdl,"alpha1":float(alpha1),"beta1":float(beta1),"persist":float(persist)}

def _interp_garch(omega,alpha1,beta1,gamma1,persist,mdl,dist):
    half_life=-np.log(2)/np.log(persist) if 0<persist<1 else float("inf")
    vol_type="highly persistent" if persist>0.95 else "moderately persistent" if persist>0.85 else "mean-reverting"
    lines=[f"<p>The <strong>{mdl}</strong> model characterizes volatility clustering. Key findings:</p>",
           f"<p>• <strong>ARCH effect (α₁={alpha1:.4f})</strong>: yesterday's surprise shock feeds into today's variance. "
           f"{'Significant ARCH effect present.' if alpha1>0.05 else 'Weak ARCH effect.'}</p>",
           f"<p>• <strong>GARCH effect (β₁={beta1:.4f})</strong>: volatility inertia/memory. "
           f"High β₁ means volatility decays slowly after spikes.</p>",
           f"<p>• <strong>Persistence (α₁+β₁={persist:.4f})</strong>: series exhibits <em>{vol_type}</em> volatility. "
           f"{'Half-life of a vol shock ≈ ' + f'{half_life:.1f} periods.' if persist<1 else 'IGARCH (integrated) process — shocks persist indefinitely.'}</p>"]
    if abs(gamma1)>0.01:
        lines.append(f"<p>• <strong>Leverage effect (γ={gamma1:.4f})</strong>: negative shocks {'amplify' if gamma1>0 else 'dampen'} volatility "
                     f"more than positive shocks of the same size — {'asymmetric volatility' if abs(gamma1)>0.05 else 'mild asymmetry'} confirmed.</p>")
    if dist in ("t","skewt"): lines.append("<p>• <strong>Fat tails (t distribution)</strong>: the Student-t error distribution appropriately captures extreme return events beyond the normal distribution's capability.</p>")
    lines.append("<p style='font-size:0.78rem;color:#94A3B8;'>Validate with Ljung-Box on squared residuals (ARCH-LM test) to confirm no remaining ARCH effects post-estimation.</p>")
    return "".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ARIMA / SARIMA
# ─────────────────────────────────────────────────────────────────────────────
def page_arima():
    st.markdown('<p class="page-title">🔭 ARIMA / SARIMA</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Auto-ARIMA · Manual specification · Fan chart forecasts with 80% & 95% CI · RMSE · MAE</p>', unsafe_allow_html=True)
    if st.session_state.clean_df is None: st.info("Load data first."); return
    df=st.session_state.clean_df; cols=df.select_dtypes(include=np.number).columns.tolist()
    tab_cfg,tab_res,tab_fc=st.tabs(["① Configuration","② Results & Diagnostics","③ Fan Chart Forecast"])
    with tab_cfg:
        c1,c2=st.columns(2)
        target=c1.selectbox("Target Series",cols); mode=c2.selectbox("Mode",["Auto-ARIMA (pmdarima)","Manual Specification"])
        if mode=="Manual Specification":
            c3,c4,c5=st.columns(3)
            p_v=c3.slider("p (AR)",0,8,1); d_v=c4.slider("d (I)",0,2,1); q_v=c5.slider("q (MA)",0,8,1)
            seasonal=st.checkbox("SARIMA seasonal component")
            if seasonal:
                c6,c7,c8,c9=st.columns(4)
                P_s=c6.slider("P (seasonal AR)",0,3,1); D_s=c7.slider("D (seasonal I)",0,2,1)
                Q_s=c8.slider("Q (seasonal MA)",0,3,1); m_s=c9.slider("m (season length)",2,52,12)
            else: P_s=D_s=Q_s=m_s=0
        fc_h=st.slider("Forecast horizon (periods)",4,60,16)
        train_pct=st.slider("Training split (%)",60,95,85)
        if st.button("▶  ESTIMATE ARIMA",use_container_width=True):
            try:
                s=df[target].dropna(); split=int(len(s)*train_pct/100)
                train=s.iloc[:split]; test=s.iloc[split:]
                if mode=="Auto-ARIMA (pmdarima)":
                    with st.spinner("Running auto_arima search…"):
                        auto=pm.auto_arima(train,seasonal=(st.session_state.freq in ("Monthly","Quarterly")),
                            m=12 if st.session_state.freq=="Monthly" else 4 if st.session_state.freq=="Quarterly" else 1,
                            stepwise=True,suppress_warnings=True,error_action="ignore",information_criterion="aic")
                    p_v,d_v,q_v=auto.order
                    if hasattr(auto,"seasonal_order"): P_s,D_s,Q_s,m_s=auto.seasonal_order
                    else: P_s=D_s=Q_s=m_s=0
                if m_s>1:
                    fitted_m=SARIMAX(train,order=(p_v,d_v,q_v),seasonal_order=(P_s,D_s,Q_s,m_s)).fit(disp=False)
                else:
                    fitted_m=ARIMA(train,order=(p_v,d_v,q_v)).fit()
                model_name=f"SARIMA({p_v},{d_v},{q_v})×({P_s},{D_s},{Q_s})[{m_s}]" if m_s>1 else f"ARIMA({p_v},{d_v},{q_v})"
                test_fc=fitted_m.forecast(steps=len(test))
                rmse=float(np.sqrt(np.mean((test.values-test_fc.values)**2))) if len(test)>0 else None
                mae=float(np.mean(np.abs(test.values-test_fc.values))) if len(test)>0 else None
                fc_result=fitted_m.get_forecast(steps=fc_h); fc_mean=fc_result.predicted_mean
                fc_ci=fc_result.conf_int(alpha=0.05); fc_ci80=fc_result.conf_int(alpha=0.20)
                st.session_state.arima_res={"model":fitted_m,"target":target,"name":model_name,
                    "order":(p_v,d_v,q_v),"seasonal_order":(P_s,D_s,Q_s,m_s),
                    "train":train,"test":test,"test_fc":test_fc,
                    "fc_mean":fc_mean,"fc_ci":fc_ci,"fc_ci80":fc_ci80,
                    "rmse":rmse,"mae":mae,"series":s}
                st.session_state.diag_model=fitted_m; st.session_state.diag_resid=fitted_m.resid.values
                st.session_state.diag_fitted=fitted_m.fittedvalues.values
                st.success(f"✓ {model_name} estimated. RMSE={rmse:.4f}" if rmse else f"✓ {model_name} estimated.")
                st.session_state.report_log["arima"]={"target":target,"name":model_name,"rmse":rmse,"mae":mae}
            except Exception as exc: st.error(f"**ARIMA error:** {exc}")
    with tab_res:
        ar=st.session_state.arima_res
        if ar is None: st.info("Estimate ARIMA first."); return
        model=ar["model"]; name=ar["name"]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Model",name); c2.metric("AIC",f"{model.aic:.3f}"); c3.metric("BIC",f"{model.bic:.3f}")
        c4.metric("RMSE (OOS)",f"{ar['rmse']:.4f}" if ar["rmse"] else "—")
        if ar["mae"]: st.metric("MAE (OOS)",f"{ar['mae']:.4f}")
        p_v,d_v,q_v=ar["order"]; P_s,D_s,Q_s,m_s=ar["seasonal_order"]
        st.markdown(f"""<div class="eq-block"><p class="eq-muted">Selected model: {name}</p>
          <p>φ(B)Φ(Bˢ)∇ᵈ∇ˢᴰyₜ = θ(B)Θ(Bˢ)εₜ &nbsp;|&nbsp; εₜ ~ WN(0,σ²)</p>
          <p class="eq-muted">p={p_v} d={d_v} q={q_v}{'  |  P='+str(P_s)+' D='+str(D_s)+' Q='+str(Q_s)+' m='+str(m_s) if m_s>1 else ''}</p></div>""", unsafe_allow_html=True)
        try:
            summ=model.summary(); rows_a=""
            for vname,coef,se,zstat,pval in zip(model.params.index,model.params,model.bse,model.tvalues,model.pvalues):
                stars=pstar(pval); cls="sig3" if pval<0.01 else "sig2" if pval<0.05 else "sig1" if pval<0.10 else "insig"
                rows_a+=f"<tr><td class='{cls}'>{vname} <span class='sig-star'>{stars}</span></td><td class='{cls}'>{fmt(coef)}</td><td>{fmt(se)}</td><td>{fmt(zstat)}</td><td>{fmt(pval)}</td></tr>"
            st.markdown(f'<div class="coef-wrap"><table class="coef-tbl"><tr><th>Parameter</th><th>Coef.</th><th>Std.Err.</th><th>z-Stat</th><th>p-Value</th></tr>{rows_a}</table></div>',unsafe_allow_html=True)
        except Exception as exc: st.warning(f"Parameter table: {exc}")
        try:
            resid=model.resid.values; lb=acorr_ljungbox(pd.Series(resid),lags=min(20,len(resid)//4),return_df=True)
            lb_ok=lb["lb_pvalue"].min()>0.05
            st.markdown(f'<div class="glass-card"><p class="sec-title" style="font-size:0.82rem;">Ljung-Box on Residuals</p>'
                        f'<p>Min p-value: <strong>{lb["lb_pvalue"].min():.4f}</strong> | '
                        f'{badge_html("Residuals White Noise ✅","pass") if lb_ok else badge_html("Residual Autocorrelation ❌","fail")}</p></div>',unsafe_allow_html=True)
        except: pass
        st.markdown(f'<div class="ai-box"><h4>⬡ AI INTERPRETATION · ARIMA</h4>{_interp_arima(ar)}</div>',unsafe_allow_html=True)
    with tab_fc:
        ar=st.session_state.arima_res
        if ar is None: st.info("Estimate ARIMA first."); return
        train=ar["train"]; test=ar["test"]; test_fc=ar["test_fc"]
        fc_mean=ar["fc_mean"]; fc_ci=ar["fc_ci"]; fc_ci80=ar["fc_ci80"]
        try:
            fc_idx=pd.date_range(train.index[-1],periods=len(fc_mean)+1,freq=pd.infer_freq(train.index) or "ME")[1:]
        except: fc_idx=range(len(train),len(train)+len(fc_mean))
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=train.index,y=train.values,name="Training",line=dict(color=C["navy"],width=1.8)))
        if len(test)>0:
            fig.add_trace(go.Scatter(x=test.index,y=test.values,name="Actual (test)",line=dict(color=C["green"],width=2,dash="dot")))
            fig.add_trace(go.Scatter(x=test.index,y=test_fc.values,name="In-sample forecast",line=dict(color=C["gold"],width=1.5,dash="dash")))
        fig.add_trace(go.Scatter(x=fc_idx,y=fc_mean.values,name="Forecast",line=dict(color=C["cyan"],width=2.5)))
        try:
            fig.add_trace(go.Scatter(x=list(fc_idx)+list(fc_idx[::-1]),
                y=list(fc_ci80.iloc[:,1])+list(fc_ci80.iloc[:,0][::-1]),
                fill="toself",fillcolor="rgba(56,189,248,0.25)",line=dict(color="rgba(0,0,0,0)"),name="80% CI"))
            fig.add_trace(go.Scatter(x=list(fc_idx)+list(fc_idx[::-1]),
                y=list(fc_ci.iloc[:,1])+list(fc_ci.iloc[:,0][::-1]),
                fill="toself",fillcolor="rgba(56,189,248,0.10)",line=dict(color="rgba(0,0,0,0)"),name="95% CI"))
        except Exception as exc: pass
        fig=navy_fig(fig,480,f"{ar['name']} Fan Chart — {ar['target']}")
        fig.update_layout(xaxis_title="Period",yaxis_title=ar["target"]); st.plotly_chart(fig,use_container_width=True)
        if ar["rmse"]:
            c1,c2=st.columns(2); c1.metric("Out-of-Sample RMSE",f"{ar['rmse']:.4f}"); c2.metric("Out-of-Sample MAE",f"{ar['mae']:.4f}")
        st.markdown('<p class="sec-title">Forecast Values</p>', unsafe_allow_html=True)
        fc_df=pd.DataFrame({"Forecast":fc_mean.values,"Lower 95%":fc_ci.iloc[:,0].values,"Upper 95%":fc_ci.iloc[:,1].values},index=fc_idx)
        st.dataframe(fc_df.style.format("{:.4f}"),use_container_width=True)

def _interp_arima(ar):
    p_v,d_v,q_v=ar["order"]; name=ar["name"]; rmse=ar["rmse"]
    d_interp={0:"already stationary in levels",1:"required first-differencing (I(1))",2:"required second-differencing (I(2))"}
    lines=[f"<p>The <strong>{name}</strong> model was selected/estimated. The series {d_interp.get(d_v,'required differencing')}.</p>",
           f"<p>• <strong>AR({p_v}) component</strong>: current value depends on {p_v} past observation(s), capturing autocorrelation/persistence.</p>",
           f"<p>• <strong>MA({q_v}) component</strong>: incorporates {q_v} past forecast error(s), correcting for moving-average dynamics.</p>"]
    if rmse: lines.append(f"<p>• <strong>Out-of-sample RMSE = {rmse:.4f}</strong>. Compare vs a naïve random-walk (RMSE = std of the series) to assess genuine predictive value.</p>")
    lines.append("<p>Check Ljung-Box on residuals to verify white-noise errors. If autocorrelation persists, increase p or q. For seasonal series, ensure seasonal differencing is adequate.</p>")
    return "".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DIAGNOSTIC SUITE
# ─────────────────────────────────────────────────────────────────────────────
def page_diagnostics():
    st.markdown('<p class="page-title">🔬 Diagnostic Suite</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Full residual analysis for any estimated model · DW · BG · Ljung-Box · White · BP · ARCH-LM · RESET</p>', unsafe_allow_html=True)
    model_choice=st.selectbox("Select model to diagnose",["OLS","ARDL","ARIMA","GARCH"])
    model_map={"OLS":"ols_res","ARDL":"ardl_res","ARIMA":"arima_res","GARCH":"garch_res"}
    container=st.session_state.get(model_map[model_choice])
    if model_choice in ("OLS","ARDL"):
        fitted_model=container
    elif model_choice=="ARIMA":
        fitted_model=container["model"] if container else None
    elif model_choice=="GARCH":
        fitted_model=container["model"] if container else None
    else:
        fitted_model=None
    if fitted_model is None: st.warning(f"No {model_choice} model found. Estimate it first."); return
    try:
        resid=fitted_model.resid.values if hasattr(fitted_model.resid,"values") else np.array(fitted_model.resid)
        resid=resid[~np.isnan(resid)]
        fvals=fitted_model.fittedvalues.values if hasattr(fitted_model,"fittedvalues") else np.zeros(len(resid))
    except: st.error("Could not extract residuals from this model."); return
    n=len(resid)
    st.markdown('<p class="sec-title">① Autocorrelation Tests</p>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    dw=durbin_watson(resid); dw_ok=1.5<dw<2.5
    c1.markdown(f'<div class="stat-box"><p class="stat-name">Durbin-Watson</p><p class="stat-val">{dw:.4f}</p>'
                f'<p class="stat-sub">{"✅ No autocorr." if dw_ok else "❌ Autocorr."}</p>'
                f'<p class="stat-sub">Rule: DW≈2 → no autocorr | →0 positive | →4 negative</p></div>',unsafe_allow_html=True)
    try:
        if model_choice in ("OLS","ARDL") and hasattr(fitted_model,"model"):
            bg=acorr_breusch_godfrey(fitted_model,nlags=min(4,n//4)); bg_ok=bg[1]>0.05
            c2.markdown(f'<div class="stat-box"><p class="stat-name">Breusch-Godfrey LM</p><p class="stat-val">{bg[1]:.4f}</p>'
                        f'<p class="stat-sub">{"✅ No serial corr." if bg_ok else "❌ Serial correlation"}</p></div>',unsafe_allow_html=True)
    except: pass
    try:
        lb=acorr_ljungbox(pd.Series(resid),lags=min(20,n//4),return_df=True); lb_ok=lb["lb_pvalue"].min()>0.05
        st.markdown(f'<div class="glass-card"><p class="sec-title" style="font-size:0.82rem;">Ljung-Box Q-Test on Residuals</p>'
                    f'<p>Min p-value across lags: <strong>{lb["lb_pvalue"].min():.4f}</strong> | '
                    f'{badge_html("White Noise","pass") if lb_ok else badge_html("Autocorrelation","fail")}'
                    f'</p></div>',unsafe_allow_html=True)
    except: pass
    st.markdown('<p class="sec-title">② Heteroskedasticity Tests</p>', unsafe_allow_html=True)
    if model_choice in ("OLS","ARDL") and hasattr(fitted_model,"model"):
        X_exog=fitted_model.model.exog
        c3,c4=st.columns(2)
        try:
            wh=het_white(resid,X_exog); wh_ok=wh[1]>0.05
            c3.markdown(f'<div class="stat-box"><p class="stat-name">White\'s Test (p)</p><p class="stat-val">{wh[1]:.4f}</p>'
                        f'<p class="stat-sub">{"✅ Homoskedastic" if wh_ok else "❌ Heteroskedastic"}</p></div>',unsafe_allow_html=True)
        except: pass
        try:
            bp=het_breuschpagan(resid,X_exog); bp_ok=bp[1]>0.05
            c4.markdown(f'<div class="stat-box"><p class="stat-name">Breusch-Pagan (p)</p><p class="stat-val">{bp[1]:.4f}</p>'
                        f'<p class="stat-sub">{"✅ Homoskedastic" if bp_ok else "❌ Heteroskedastic"}</p></div>',unsafe_allow_html=True)
        except: pass
    try:
        arch_r=het_arch(resid); arch_ok=arch_r[1]>0.05
        st.markdown(f'<div class="glass-card"><p class="sec-title" style="font-size:0.82rem;">ARCH-LM Test</p>'
                    f'<p>LM stat: <strong>{arch_r[0]:.4f}</strong> | p: <strong>{arch_r[1]:.4f}</strong> | '
                    f'{badge_html("No ARCH","pass") if arch_ok else badge_html("ARCH Effects","warn")}</p></div>',unsafe_allow_html=True)
    except: pass
    st.markdown('<p class="sec-title">③ Normality & Distribution</p>', unsafe_allow_html=True)
    jb_stat,jb_p=jarque_bera(resid); jb_ok=jb_p>0.05; skew_r=sci_stats.skew(resid); kurt_r=sci_stats.kurtosis(resid)
    c5,c6=st.columns(2)
    c5.markdown(f'<div class="stat-box"><p class="stat-name">Jarque-Bera</p><p class="stat-val">{jb_p:.4f}</p>'
                f'<p class="stat-sub">{"✅ Normality OK" if jb_ok else "⚠ Non-normal"}</p>'
                f'<p class="stat-sub">Skew={skew_r:.3f} | Ex.Kurt={kurt_r:.3f}</p></div>',unsafe_allow_html=True)
    x_r=np.linspace(min(resid),max(resid),200)
    fig=go.Figure()
    fig.add_trace(go.Histogram(x=resid,nbinsx=30,name="Residuals",marker_color=C["cyan"],opacity=0.65,histnorm="probability density"))
    fig.add_trace(go.Scatter(x=x_r,y=sci_stats.norm.pdf(x_r,np.mean(resid),np.std(resid)),name="Normal PDF",line=dict(color=C["gold"],width=2.5)))
    fig=navy_fig(fig,300,"Residual Distribution vs Normal"); c6.plotly_chart(fig,use_container_width=True)
    st.markdown('<p class="sec-title">④ Residual Plots</p>', unsafe_allow_html=True)
    fig2=make_subplots(rows=1,cols=2,subplot_titles=["Residuals vs Fitted","Normal Q-Q"])
    fig2.add_trace(go.Scatter(x=fvals,y=resid,mode="markers",marker=dict(color=C["cyan"],size=5,opacity=0.6),name="Residuals"),1,1)
    fig2.add_hline(y=0,line=dict(color=C["navy"],dash="dash"),row=1,col=1)
    osm,osr=sci_stats.probplot(resid,dist="norm")[:2]
    sl,ic,_,_,_=sci_stats.linregress(osm[0],osm[1])
    fig2.add_trace(go.Scatter(x=osm[0],y=osm[1],mode="markers",marker=dict(color=C["cyan"],size=4,opacity=0.65),name="Quantiles"),1,2)
    xl=[min(osm[0]),max(osm[0])]
    fig2.add_trace(go.Scatter(x=xl,y=[sl*x+ic for x in xl],line=dict(color=C["gold"],width=2,dash="dash"),name="Normal line"),1,2)
    fig2=navy_fig(fig2,340); st.plotly_chart(fig2,use_container_width=True)
    grade_map={True:"A",False:"B"}
    passes=[dw_ok,jb_ok]; grade="A" if all(passes) else "B" if sum(passes)>=1 else "C"
    g_color={"A":C["green"],"B":C["gold"],"C":C["red"]}.get(grade,C["slate"])
    st.markdown(f'<div class="ai-box"><h4>⬡ DIAGNOSTIC REPORT CARD</h4>'
                f'<p style="font-family:Orbitron,monospace;font-size:1.4rem;color:{g_color};">MODEL GRADE: {grade}</p>'
                f'<p>{"✅ DW test: no first-order autocorrelation detected." if dw_ok else "❌ DW test: autocorrelation present — consider Newey-West HAC standard errors."}</p>'
                f'<p>{"✅ JB test: residuals normally distributed." if jb_ok else "⚠ JB test: non-normal residuals (skew="+fmt(skew_r,3)+", excess kurtosis="+fmt(kurt_r,3)+") — OLS still BLUE in large samples but inference may be unreliable in small samples."}</p>'
                f'<p style="font-size:0.78rem;color:#94A3B8;">Economic theory and variable selection remain paramount. Diagnostics confirm statistical properties, not economic validity.</p>'
                f'</div>',unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: STABILITY TESTS
# ─────────────────────────────────────────────────────────────────────────────
def page_stability():
    st.markdown('<p class="page-title">📏 Stability Tests</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">CUSUM · CUSUM-of-Squares · Chow Breakpoint Test · Structural change detection</p>', unsafe_allow_html=True)
    model=st.session_state.ols_res
    if model is None: st.info("Estimate OLS first to run stability tests."); return
    resid=model.resid.values; n=len(resid)
    st.markdown('<p class="sec-title">CUSUM & CUSUM-of-Squares</p>', unsafe_allow_html=True)
    cusum=np.cumsum(resid); sigma=np.std(resid)
    conf_upper= 0.948*sigma*np.sqrt(n)*np.array([2*i/n-1 for i in range(1,n+1)])
    conf_lower=-0.948*sigma*np.sqrt(n)*np.array([2*i/n-1 for i in range(1,n+1)])
    t_ax=list(range(1,n+1))
    cusq=np.cumsum(resid**2)/(np.sum(resid**2)+1e-12)
    expected=np.linspace(0,1,n)
    cusq_band=1.36/np.sqrt(n)
    fig=make_subplots(rows=2,cols=1,vertical_spacing=0.1,subplot_titles=["CUSUM (with 5% bands)","CUSUM-of-Squares (with 5% bands)"])
    fig.add_trace(go.Scatter(x=t_ax,y=cusum,name="CUSUM",line=dict(color=C["cyan"],width=2)),1,1)
    fig.add_trace(go.Scatter(x=t_ax,y=conf_upper,name="5% Upper",line=dict(color=C["red"],dash="dash",width=1.2),showlegend=False),1,1)
    fig.add_trace(go.Scatter(x=t_ax,y=conf_lower,name="5% Lower",line=dict(color=C["red"],dash="dash",width=1.2),showlegend=False),1,1)
    fig.add_hline(y=0,line=dict(color=C["navy"],dash="solid",width=0.8),row=1,col=1)
    fig.add_trace(go.Scatter(x=t_ax,y=cusq,name="CUSUMsq",line=dict(color=C["teal"],width=2)),2,1)
    fig.add_trace(go.Scatter(x=t_ax,y=expected+cusq_band,name="5% Upper",line=dict(color=C["red"],dash="dash",width=1.2),showlegend=False),2,1)
    fig.add_trace(go.Scatter(x=t_ax,y=expected-cusq_band,name="5% Lower",line=dict(color=C["red"],dash="dash",width=1.2),showlegend=False),2,1)
    fig.add_trace(go.Scatter(x=t_ax,y=expected,name="Expected",line=dict(color=C["navy"],dash="dot",width=1)),2,1)
    fig=navy_fig(fig,520); st.plotly_chart(fig,use_container_width=True)
    breaches_cusum=sum(1 for c,u,l in zip(cusum,conf_upper,conf_lower) if c>u or c<l)
    breaches_cusq=sum(1 for c,u,l in zip(cusq,expected+cusq_band,expected-cusq_band) if c>u or c<l)
    stable_cusum=breaches_cusum<int(0.05*n); stable_cusq=breaches_cusq<int(0.05*n)
    st.markdown(f"""<div class="ai-box"><h4>⬡ STABILITY INTERPRETATION</h4>
      <p>CUSUM: {breaches_cusum} breach(es) of 5% bands → {badge_html('Structurally Stable','pass') if stable_cusum else badge_html('Potential Break','fail')}</p>
      <p>CUSUM²: {breaches_cusq} breach(es) → {badge_html('Stable Variance','pass') if stable_cusq else badge_html('Variance Instability','warn')}</p>
      <p>If either CUSUM or CUSUM² crosses the 5% bands, there is evidence of structural instability.
      Consider: (1) including structural break dummies, (2) splitting the sample, (3) Chow test at suspected break date.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Chow Breakpoint Test</p>', unsafe_allow_html=True)
    break_pct=st.slider("Break point (%through sample)",20,80,50)
    if st.button("▶  CHOW TEST",use_container_width=True):
        try:
            break_idx=int(n*break_pct/100); Y=model.model.endog; X=model.model.exog; k=X.shape[1]
            Y1=Y[:break_idx]; X1=X[:break_idx,:]; Y2=Y[break_idx:]; X2=X[break_idx:,:]
            ssr_full=model.ssr
            m1=OLS(Y1,X1).fit(); m2=OLS(Y2,X2).fit(); ssr_sub=m1.ssr+m2.ssr
            chow_f=((ssr_full-ssr_sub)/k)/(ssr_sub/(n-2*k))
            chow_p=1-sci_stats.f.cdf(chow_f,k,n-2*k)
            chow_ok=chow_p>0.05
            st.markdown(f"""<div class="brutalist-card-gold">
              <p style="font-weight:700;margin:0 0 8px;">Chow Test at observation {break_idx}/{n} ({break_pct}%)</p>
              <p>F-statistic: <strong>{chow_f:.4f}</strong> &nbsp;|&nbsp; p-value: <strong>{chow_p:.4f}</strong> &nbsp;|&nbsp;
                 {badge_html('No Structural Break','pass') if chow_ok else badge_html('Structural Break Detected','fail')}</p>
              <p style="font-size:0.75rem;color:#64748B;">H₀: No structural break at this point (equal coefficients in both sub-samples)</p>
            </div>""", unsafe_allow_html=True)
        except Exception as exc: st.error(f"Chow test error: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: NORMALITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def page_normality():
    st.markdown('<p class="page-title">🔔 Normality Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Jarque-Bera · Histogram with normal overlay · Q-Q plot · Descriptive statistics</p>', unsafe_allow_html=True)
    if st.session_state.clean_df is None: st.info("Load data first."); return
    df=st.session_state.clean_df; cols=df.select_dtypes(include=np.number).columns.tolist()
    c1,c2=st.columns(2)
    sel=c1.selectbox("Variable",cols); diff_n=c2.selectbox("Transform",["Level","1st Difference","2nd Difference"])
    if st.button("▶  TEST NORMALITY",use_container_width=True):
        s=df[sel].dropna()
        if diff_n=="1st Difference": s=s.diff().dropna(); lbl=f"Δ{sel}"
        elif diff_n=="2nd Difference": s=s.diff().diff().dropna(); lbl=f"Δ²{sel}"
        else: lbl=sel
        jb_stat,jb_p=jarque_bera(s.values); jb_ok=jb_p>0.05
        sk=sci_stats.skew(s.values); ku=sci_stats.kurtosis(s.values)
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("JB Stat",f"{jb_stat:.4f}"); c2.metric("JB p-value",f"{jb_p:.4f}")
        c3.metric("Skewness",f"{sk:.4f}"); c4.metric("Excess Kurt.",f"{ku:.4f}")
        c5.metric("Verdict","NORMAL" if jb_ok else "NON-NORMAL",delta="✅" if jb_ok else "⚠")
        fig=make_subplots(rows=1,cols=2,subplot_titles=[f"Distribution — {lbl}","Normal Q-Q Plot"])
        x_r=np.linspace(s.min(),s.max(),200)
        fig.add_trace(go.Histogram(x=s.values,nbinsx=30,name=lbl,marker_color=C["cyan"],opacity=0.65,histnorm="probability density"),1,1)
        fig.add_trace(go.Scatter(x=x_r,y=sci_stats.norm.pdf(x_r,s.mean(),s.std()),name="Normal PDF",line=dict(color=C["gold"],width=2.5)),1,1)
        osm,osr=sci_stats.probplot(s.values,dist="norm")[:2]
        sl,ic,_,_,_=sci_stats.linregress(osm[0],osm[1]); xl=[min(osm[0]),max(osm[0])]
        fig.add_trace(go.Scatter(x=osm[0],y=osm[1],mode="markers",marker=dict(color=C["cyan"],size=5,opacity=0.65),name="Quantiles"),1,2)
        fig.add_trace(go.Scatter(x=xl,y=[sl*x+ic for x in xl],line=dict(color=C["gold"],width=2,dash="dash"),name="45° line"),1,2)
        fig=navy_fig(fig,380); st.plotly_chart(fig,use_container_width=True)
        if jb_ok:
            ai_n="<p><span class='ok-text'>✅ Jarque-Bera fails to reject normality</span> at 5%. Residuals (or series) are approximately normally distributed. OLS inference is reliable.</p>"
        else:
            ai_n=(f"<p><span class='warn-text'>⚠ Jarque-Bera rejects normality</span> (p={jb_p:.4f}). "
                  f"Skewness = {sk:.3f} ({'positive/right skew' if sk>0 else 'negative/left skew'}), "
                  f"excess kurtosis = {ku:.3f} ({'leptokurtic/fat tails' if ku>0 else 'platykurtic/thin tails'}).</p>"
                  "<p>OLS is still unbiased and consistent (Gauss-Markov), but hypothesis tests may be unreliable "
                  "in small samples. Consider robust SEs or bootstrap inference. If this is a return series, consider GARCH.</p>")
        st.markdown(f'<div class="ai-box"><h4>⬡ NORMALITY INTERPRETATION</h4>{ai_n}</div>',unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FORECASTING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def page_forecast():
    st.markdown('<p class="page-title">🎯 Forecasting Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Multi-model comparison · Dynamic out-of-sample forecasts · Fan charts with 80% & 95% CI · RMSE · MAE · Naive benchmark</p>', unsafe_allow_html=True)
    if st.session_state.arima_res is not None:
        ar=st.session_state.arima_res
        st.markdown(f'<div class="brutalist-card-inv"><p style="font-weight:700;font-size:0.9rem;">Active ARIMA Forecast: {ar["name"]} — {ar["target"]}</p>'
                    f'<p style="font-size:0.8rem;color:#CBD5E1;">RMSE={ar["rmse"]:.4f} | MAE={ar["mae"]:.4f} | Navigate to ARIMA tab for full fan chart.</p></div>',unsafe_allow_html=True)
    if st.session_state.clean_df is None: st.info("Load data and estimate a model first."); return
    df=st.session_state.clean_df; cols=df.select_dtypes(include=np.number).columns.tolist()
    st.markdown('<p class="sec-title">Quick Forecast (ARIMA / Exponential Smoothing)</p>', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    target_f=c1.selectbox("Variable",cols,key="fc_target")
    fc_method=c2.selectbox("Method",["ARIMA (auto)","ETS (Holt-Winters)","Naïve (random walk)","Seasonal Naïve"])
    fc_h_f=c3.slider("Horizon",4,60,16,key="fc_h")
    if st.button("▶  QUICK FORECAST",use_container_width=True):
        try:
            s=df[target_f].dropna(); split=int(len(s)*0.85)
            train=s.iloc[:split]; test=s.iloc[split:]
            if fc_method=="ARIMA (auto)":
                with st.spinner("Auto-ARIMA…"):
                    auto=pm.auto_arima(train,stepwise=True,suppress_warnings=True,error_action="ignore",information_criterion="aic")
                fc_vals=auto.predict(n_periods=fc_h_f+len(test))
                test_preds=fc_vals[:len(test)]; future_preds=fc_vals[len(test):]
                model_lbl=f"ARIMA{auto.order}"
            elif fc_method=="ETS (Holt-Winters)":
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                ets=ExponentialSmoothing(train,trend="add",seasonal="add" if len(train)>24 else None,
                                         seasonal_periods=12 if st.session_state.freq=="Monthly" else 4).fit()
                fc_vals=ets.forecast(fc_h_f+len(test)); test_preds=fc_vals[:len(test)]; future_preds=fc_vals[len(test):]
                model_lbl="ETS (Holt-Winters)"
            elif fc_method=="Naïve (random walk)":
                test_preds=np.repeat(train.iloc[-1],len(test)); future_preds=np.repeat(train.iloc[-1],fc_h_f)
                model_lbl="Naïve RW"
            elif fc_method=="Seasonal Naïve":
                period=12 if st.session_state.freq=="Monthly" else 4
                test_preds=np.array([train.iloc[-period+i%period] for i in range(len(test))])
                future_preds=np.array([train.iloc[-period+i%period] for i in range(fc_h_f)])
                model_lbl="Seasonal Naïve"
            rmse_f=float(np.sqrt(np.mean((test.values-test_preds[:len(test)])**2)))
            mae_f=float(np.mean(np.abs(test.values-test_preds[:len(test)])))
            try:
                fc_idx=pd.date_range(train.index[-1],periods=fc_h_f+1,freq=pd.infer_freq(train.index) or "ME")[1:]
            except: fc_idx=range(len(s),len(s)+fc_h_f)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=train.index,y=train.values,name="Training",line=dict(color=C["navy"],width=1.8)))
            fig.add_trace(go.Scatter(x=test.index,y=test.values,name="Actual",line=dict(color=C["green"],width=2,dash="dot")))
            fig.add_trace(go.Scatter(x=test.index,y=test_preds[:len(test)],name="Test Forecast",line=dict(color=C["gold"],width=1.5,dash="dash")))
            fig.add_trace(go.Scatter(x=fc_idx,y=future_preds[:fc_h_f],name=f"{model_lbl} Forecast",line=dict(color=C["cyan"],width=2.5)))
            naive_rmse=float(np.std(test.values))
            fig=navy_fig(fig,420,f"Forecast: {target_f} — {model_lbl}")
            fig.update_layout(xaxis_title="Period",yaxis_title=target_f); st.plotly_chart(fig,use_container_width=True)
            c1,c2,c3=st.columns(3)
            c1.metric("RMSE (test)",f"{rmse_f:.4f}"); c2.metric("MAE (test)",f"{mae_f:.4f}")
            c3.metric("Naïve RMSE (benchmark)",f"{naive_rmse:.4f}",
                      delta=f"{'Better' if rmse_f<naive_rmse else 'Worse'} than naïve",
                      delta_color="normal" if rmse_f<naive_rmse else "inverse")
        except Exception as exc: st.error(f"Forecast error: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────
def page_decompose():
    st.markdown('<p class="page-title">🧩 Decomposition</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">STL · Classical additive/multiplicative · Trend · Seasonal · Irregular components</p>', unsafe_allow_html=True)
    if st.session_state.clean_df is None: st.info("Load data first."); return
    df=st.session_state.clean_df; cols=df.select_dtypes(include=np.number).columns.tolist()
    c1,c2,c3=st.columns(3)
    sel_d=c1.selectbox("Variable",cols); method_d=c2.selectbox("Method",["STL","Classical Additive","Classical Multiplicative"])
    dp2={"Monthly":12,"Quarterly":4,"Weekly":52,"Annual":2,"Daily":365}
    period_d=c3.number_input("Seasonal period",value=int(dp2.get(st.session_state.freq or "Monthly",12)),min_value=2)
    if st.button("▶  DECOMPOSE",use_container_width=True):
        try:
            s=df[sel_d].dropna()
            if method_d=="STL":
                stl=STL(s,period=int(period_d),robust=True); res=stl.fit()
                trend_c=res.trend; seasonal_c=res.seasonal; resid_c=res.resid
            else:
                from statsmodels.tsa.seasonal import seasonal_decompose
                model_type="additive" if method_d=="Classical Additive" else "multiplicative"
                res2=seasonal_decompose(s,model=model_type,period=int(period_d),extrapolate_trend="freq")
                trend_c=res2.trend; seasonal_c=res2.seasonal; resid_c=res2.resid
            fig=make_subplots(rows=4,cols=1,vertical_spacing=0.04,subplot_titles=["Original","Trend","Seasonal","Residual / Irregular"])
            for i,(y,color,name) in enumerate([(s.values,C["navy"],"Original"),(trend_c if hasattr(trend_c,"values") else trend_c,C["cyan"],"Trend"),
                (seasonal_c if hasattr(seasonal_c,"values") else seasonal_c,C["gold"],"Seasonal"),
                (resid_c if hasattr(resid_c,"values") else resid_c,C["red"],"Residual")],1):
                y_arr=y if isinstance(y,np.ndarray) else np.array(y)
                fig.add_trace(go.Scatter(x=s.index,y=y_arr,name=name,line=dict(color=color,width=1.6)),i,1)
            fig=navy_fig(fig,620,f"{method_d} Decomposition — {sel_d}"); st.plotly_chart(fig,use_container_width=True)
            try:
                resid_arr=np.array(resid_c); var_explained=1-np.var(resid_arr[~np.isnan(resid_arr)])/np.var(s.values)
                st.metric("Variance explained by trend+seasonal",f"{var_explained*100:.2f}%")
            except: pass
        except Exception as exc: st.error(f"Decomposition error: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def page_stats():
    st.markdown('<p class="page-title">📊 Summary Statistics</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Descriptive statistics · Distribution plots · Time-series overview · Correlation analysis</p>', unsafe_allow_html=True)
    if st.session_state.clean_df is None: st.info("Load data first."); return
    df=st.session_state.clean_df; cols=df.select_dtypes(include=np.number).columns.tolist()
    tab_desc,tab_plots,tab_ts=st.tabs(["📋 Descriptive Stats","📊 Distribution","📈 Time Series"])
    with tab_desc:
        desc=df[cols].describe().T.round(4)
        desc["skew"]=df[cols].skew().round(4); desc["kurt"]=df[cols].kurt().round(4)
        desc["missing"]=df[cols].isnull().sum(); desc["CV%"]=(df[cols].std()/df[cols].mean().abs()*100).round(2)
        st.dataframe(desc.style.format("{:.4f}",subset=[c for c in desc.columns if c!="missing"]),use_container_width=True)
        csv_buf=io.StringIO(); desc.to_csv(csv_buf)
        st.download_button("⬇ Download Stats CSV",csv_buf.getvalue().encode(),"nexus_stats.csv","text/csv",use_container_width=True)
    with tab_plots:
        sel_p=st.selectbox("Variable",cols,key="stats_sel")
        fig=make_subplots(rows=1,cols=2,subplot_titles=["Distribution","Box Plot"])
        fig.add_trace(go.Histogram(x=df[sel_p].dropna(),nbinsx=25,name=sel_p,marker_color=C["cyan"],opacity=0.7),1,1)
        x_r=np.linspace(df[sel_p].dropna().min(),df[sel_p].dropna().max(),200)
        fig.add_trace(go.Scatter(x=x_r,y=sci_stats.norm.pdf(x_r,df[sel_p].mean(),df[sel_p].std())*len(df[sel_p].dropna())*(df[sel_p].max()-df[sel_p].min())/25,name="Normal fit",line=dict(color=C["gold"],width=2)),1,1)
        fig.add_trace(go.Box(y=df[sel_p].dropna(),name=sel_p,marker_color=C["cyan"],boxpoints="outliers"),1,2)
        fig=navy_fig(fig,380,f"Distribution — {sel_p}"); st.plotly_chart(fig,use_container_width=True)
    with tab_ts:
        sel_ts=st.multiselect("Variables",cols,default=cols[:min(4,len(cols))])
        normalize=st.checkbox("Normalize to index (=100 at start)")
        if sel_ts:
            fig2=go.Figure()
            for i,v in enumerate(sel_ts):
                y=df[v].dropna()
                if normalize:
                    base=y.dropna().iloc[0]; y=(y/base*100) if base!=0 else y
                fig2.add_trace(go.Scatter(x=y.index,y=y.values,name=v,line=dict(color=CHART_COLORS[i%len(CHART_COLORS)],width=2)))
            fig2=navy_fig(fig2,420,"Time-Series Overview"); st.plotly_chart(fig2,use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AI ECONOMETRICIAN CHATBOT
# ─────────────────────────────────────────────────────────────────────────────
def page_chat():
    st.markdown('<p class="page-title">🤖 AI Econometrician</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Ask anything about your data, models, or econometric theory. '
                'The AI has full context of your active session.</p>', unsafe_allow_html=True)

    def _build_context():
        ctx=[]
        if st.session_state.clean_df is not None:
            df=st.session_state.clean_df
            ctx.append(f"Active dataset: {len(df):,} observations, {len(df.columns)} variables, "
                       f"frequency={st.session_state.freq or 'unknown'}. "
                       f"Variables: {', '.join(df.select_dtypes(include=np.number).columns.tolist()[:10])}.")
        if st.session_state.stat_results:
            sr=st.session_state.stat_results
            ctx.append(f"Last stationarity test: {sr.get('variable','')} ({sr.get('level','')}) — "
                       f"ADF p={sr.get('adf',{}).get('pval','—')}, KPSS p={sr.get('kpss',{}).get('pval','—')}.")
        if st.session_state.ols_res is not None:
            m=st.session_state.ols_res
            ctx.append(f"OLS model: dep={m.model.endog_names}, R²={m.rsquared:.4f}, "
                       f"Adj.R²={m.rsquared_adj:.4f}, F-pval={m.f_pvalue:.4f}, N={int(m.nobs)}.")
        if st.session_state.ardl_res is not None:
            ar=st.session_state.ardl_res
            ctx.append(f"ARDL model: dep={ar['dep']}, regressors={ar['indep']}, "
                       f"order={ar['order']}, IC={ar['ic']}.")
        if st.session_state.var_res is not None:
            vr=st.session_state.var_res
            ctx.append(f"VAR({vr['best_p']}) model with variables: {vr['endog']}.")
        if st.session_state.vecm_res is not None:
            vr=st.session_state.vecm_res
            ctx.append(f"VECM with r={vr['rank']} cointegrating vectors, variables: {vr['endog']}.")
        if st.session_state.garch_res is not None:
            gr=st.session_state.garch_res
            ctx.append(f"GARCH model: {gr['type']} on {gr['var']}, dist={gr['dist']}.")
        if st.session_state.arima_res is not None:
            ar=st.session_state.arima_res
            ctx.append(f"ARIMA model: {ar['name']} on {ar['target']}, RMSE={ar['rmse']:.4f}" if ar['rmse'] else f"ARIMA model: {ar['name']} on {ar['target']}.")
        return " | ".join(ctx) if ctx else "No data or models loaded yet."

    ECON_KB = {
        "ardl": ("ARDL bounds testing (Pesaran, Shin & Smith 2001) is appropriate when variables may be I(0) or I(1). "
                 "The PSS bounds test produces an F-statistic compared to I(0) and I(1) critical bounds. "
                 "If F > I(1) bound → cointegration confirmed regardless of integration order. "
                 "A significant negative ECT confirms error-correcting behaviour."),
        "vecm": ("VECM is appropriate when all variables are I(1) and cointegrated (Johansen 1988). "
                 "The Johansen trace test tests H₀: rank=r. The number of cointegrating vectors equals "
                 "the number of stable long-run relationships. The loading matrix α gives speed of adjustment."),
        "garch": ("GARCH(1,1) models time-varying volatility. α₁ measures ARCH (shock) effects; "
                  "β₁ measures GARCH (persistence). α₁+β₁ is persistence — must be <1 for stationarity. "
                  "EGARCH and TGARCH capture asymmetric leverage effects (bad news amplifies volatility more than good news)."),
        "stationarity": ("ADF tests H₀: unit root. KPSS tests H₀: stationarity. "
                         "When they conflict, consider Zivot-Andrews (allows one structural break). "
                         "I(1) series need differencing before OLS. ARDL accommodates mixed I(0)/I(1)."),
        "var": ("VAR models joint dynamics of multiple endogenous variables. "
                "Select optimal lag via AIC/BIC. Granger causality tests whether lagged values of X help predict Y. "
                "IRF traces response to a one-standard-deviation shock. FEVD shows what fraction of forecast error variance is attributable to each shock."),
        "ols": ("OLS is BLUE under Gauss-Markov assumptions. Use HC1 SEs for heteroskedasticity, "
                "HAC Newey-West for autocorrelation+heteroskedasticity. Ramsey RESET tests for functional misspecification."),
        "arima": ("ARIMA(p,d,q): p=AR lags, d=differencing order, q=MA lags. "
                  "Use ACF/PACF to identify order. Auto-ARIMA searches over information criteria. "
                  "Ljung-Box on residuals should show white noise. RMSE vs naïve RW assesses forecast value."),
        "cointegration": ("Two I(1) series are cointegrated if a linear combination is I(0). "
                          "Test with Johansen (multivariate) or Engle-Granger (bivariate). "
                          "Cointegration implies a stable long-run relationship and justifies VECM/ARDL-ECM."),
        "heteroskedasticity": ("White's test and Breusch-Pagan test H₀: homoskedastic errors. "
                               "If rejected, use HC1 or HC3 robust SEs. For financial data with volatility clustering, use GARCH."),
        "ecm": ("The Error Correction Term (ECT) measures speed of adjustment to long-run equilibrium. "
                "Must be negative and significant for stable error-correcting dynamics. "
                "ECT = -0.3 means 30% of disequilibrium corrected each period."),
        "hp filter": ("The Hodrick-Prescott filter decomposes a series into trend and cyclical components. "
                      "λ=1600 for quarterly, λ=100 for annual, λ=14400 for monthly. "
                      "Criticism: HP can generate spurious cycles (Hamilton 2018) — BK filter avoids endpoint problems."),
        "johansen": ("Johansen (1988) tests for cointegration in a multivariate system. "
                     "Trace statistic tests H₀: rank≤r. Max-eigenvalue tests H₀: rank=r vs r+1. "
                     "Rank=0 → no cointegration; rank=k → all variables I(0)."),
        "forecast": ("Forecast accuracy: RMSE penalizes large errors more than MAE. "
                     "Compare to naïve random-walk RMSE to assess genuine predictive power. "
                     "Fan charts show 80% and 95% prediction intervals — widen with horizon."),
    }

    def _answer(question, context):
        q_lower=question.lower()
        matched_answers=[]
        for keyword,answer in ECON_KB.items():
            if keyword in q_lower:
                matched_answers.append(answer)
        base=""
        if context: base=f"**Session context:** {context}\n\n"
        if matched_answers:
            return base+"**Econometric Answer:**\n\n"+"\n\n".join(matched_answers)
        # Generic responses for common questions
        if any(w in q_lower for w in ["i(1)","i(0)","integrated","unit root"]):
            return base+ECON_KB["stationarity"]
        if any(w in q_lower for w in ["cointegrat","long run","long-run"]):
            return base+ECON_KB["cointegration"]
        if any(w in q_lower for w in ["ect","error correction","speed of adjustment"]):
            return base+ECON_KB["ecm"]
        if any(w in q_lower for w in ["heterosked","white test","bp test"]):
            return base+ECON_KB["heteroskedasticity"]
        if "which model" in q_lower or "what model" in q_lower or "should i use" in q_lower:
            return (base+"**Model selection guidance:**\n\n"
                    "• All I(0) → OLS in levels\n"
                    "• Mix of I(0) and I(1) → ARDL Bounds Test (Pesaran et al. 2001)\n"
                    "• All I(1), cointegrated → VECM (Johansen 1988)\n"
                    "• All I(1), not cointegrated → VAR in differences\n"
                    "• Financial returns / volatility → GARCH family\n"
                    "• Univariate forecasting → ARIMA/SARIMA\n\n"
                    "Always test stationarity first (ADF, KPSS, Zivot-Andrews), then test for cointegration.")
        if "kpss" in q_lower and "adf" in q_lower:
            return (base+"**ADF vs KPSS:**\n\n"
                    "ADF tests H₀: unit root (non-stationary). KPSS tests H₀: stationary.\n"
                    "Use both together: if both agree → high confidence. If they conflict:\n"
                    "• ADF rejects, KPSS rejects → near-unit-root or structural break present\n"
                    "• Neither rejects → series is likely stationary\n"
                    "Run Zivot-Andrews to allow for a structural break before concluding.")
        if any(w in q_lower for w in ["interpret","what does","mean","explain"]):
            return (base+"**Interpretation guidance:**\n\n"
                    "I can explain specific model outputs if you describe what you're seeing. "
                    "Common questions I can answer:\n"
                    "• What does a negative ECT mean? (→ ask about 'ecm')\n"
                    "• How do I interpret the Johansen trace test? (→ ask about 'johansen')\n"
                    "• What does persistence mean in GARCH? (→ ask about 'garch')\n"
                    "• How do I read the Granger causality matrix? (→ ask about 'var')")
        return (base+"I'm the NEXUS KERNEL AI Econometrician. I can help with:\n\n"
                "• Model selection (ARDL vs VECM vs VAR, etc.)\n"
                "• Interpreting test results (ADF, Johansen, Bounds Test, CUSUM)\n"
                "• Explaining coefficients (ECT, loading matrix α, IRF)\n"
                "• Diagnostic advice (what to do when White's test fails, etc.)\n"
                "• GARCH volatility interpretation\n"
                "• Forecast accuracy assessment\n\n"
                "Try asking: *'My series is I(1), should I use ARDL or VECM?'* or "
                "*'What does a negative ECT mean?'* or *'My CUSUM test shows instability — what should I do?'*")

    # Chat interface
    chat_container=st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt:=st.chat_input("Ask the AI Econometrician anything…"):
        st.session_state.chat_history.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        ctx=_build_context()
        answer=_answer(prompt,ctx)
        st.session_state.chat_history.append({"role":"assistant","content":answer})
        with st.chat_message("assistant"): st.markdown(answer)

    if st.session_state.chat_history:
        if st.button("🗑 Clear Chat History",key="clear_chat"):
            st.session_state.chat_history=[]; st.rerun()

    if not st.session_state.chat_history:
        st.markdown("""<div class="ai-box"><h4>⬡ SUGGESTED QUESTIONS</h4>
          <ul>
            <li>My series is I(1) — should I use ARDL or VECM?</li>
            <li>What does a negative ECT mean?</li>
            <li>Which stationarity test is better: ADF or KPSS?</li>
            <li>How do I interpret the Johansen trace test?</li>
            <li>My CUSUM test shows instability — what should I do?</li>
            <li>What does GARCH persistence mean?</li>
            <li>How do I read the Granger causality matrix?</li>
            <li>My White's test is significant — what should I do?</li>
          </ul>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PDF REPORT
# ─────────────────────────────────────────────────────────────────────────────
def page_report():
    st.markdown('<p class="page-title">📄 PDF Report Generator</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Generate a professional econometric analysis report with all active results.</p>', unsafe_allow_html=True)
    user=st.session_state.user_name; email=st.session_state.user_email; occ=st.session_state.user_occ
    c1,c2=st.columns(2)
    report_title=c1.text_input("Report Title","NEXUS KERNEL Econometric Analysis Report")
    institution=c2.text_input("Institution / Organization","")
    abstract=st.text_area("Abstract / Author Notes",
        "This report was generated using NEXUS KERNEL v3.0 — a professional time-series econometrics platform.",height=100)
    include_ols=st.checkbox("Include OLS Results",value=st.session_state.ols_res is not None)
    include_ardl=st.checkbox("Include ARDL Results",value=st.session_state.ardl_res is not None)
    include_var=st.checkbox("Include VAR Results",value=st.session_state.var_res is not None)
    include_vecm=st.checkbox("Include VECM Results",value=st.session_state.vecm_res is not None)
    include_garch=st.checkbox("Include GARCH Results",value=st.session_state.garch_res is not None)
    include_arima=st.checkbox("Include ARIMA Results",value=st.session_state.arima_res is not None)
    if st.button("▶  GENERATE PDF REPORT",use_container_width=True):
        with st.spinner("Building PDF…"):
            try:
                pdf_bytes=_build_pdf(report_title,user,email,occ,institution,abstract,
                    include_ols,include_ardl,include_var,include_vecm,include_garch,include_arima)
                st.success("✓ PDF report generated successfully!")
                st.download_button("⬇ DOWNLOAD PDF REPORT",pdf_bytes,
                    f"NEXUS_KERNEL_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    "application/pdf",use_container_width=True)
            except Exception as exc: st.error(f"**PDF generation failed:** {exc}")

def _build_pdf(title,user,email,occ,institution,abstract,
               inc_ols,inc_ardl,inc_var,inc_vecm,inc_garch,inc_arima):
    buf=io.BytesIO()
    NAVY_RL=rl_colors.HexColor("#0F172A"); CYAN_RL=rl_colors.HexColor("#38BDF8")
    GOLD_RL=rl_colors.HexColor("#F59E0B"); LIGHT_RL=rl_colors.HexColor("#E8EEFF")
    GRAY_RL=rl_colors.HexColor("#64748B"); GREEN_RL=rl_colors.HexColor("#10B981")
    doc=SimpleDocTemplate(buf,pagesize=A4,rightMargin=2*cm,leftMargin=2*cm,topMargin=2.5*cm,bottomMargin=2.5*cm)
    styles=getSampleStyleSheet()
    title_s=ParagraphStyle("NKTitle",parent=styles["Title"],fontSize=20,fontName="Helvetica-Bold",textColor=NAVY_RL,spaceAfter=4,alignment=TA_CENTER)
    h1_s=ParagraphStyle("NKH1",parent=styles["Heading1"],fontSize=12,fontName="Helvetica-Bold",textColor=NAVY_RL,spaceBefore=12,spaceAfter=5)
    h2_s=ParagraphStyle("NKH2",parent=styles["Heading2"],fontSize=10,fontName="Helvetica-Bold",textColor=CYAN_RL,spaceBefore=8,spaceAfter=3)
    body_s=ParagraphStyle("NKBody",parent=styles["Normal"],fontSize=9,fontName="Helvetica",textColor=NAVY_RL,spaceAfter=5,leading=14,alignment=TA_JUSTIFY)
    mono_s=ParagraphStyle("NKMono",parent=styles["Normal"],fontSize=8,fontName="Courier",textColor=NAVY_RL,spaceAfter=4,leading=12)
    foot_s=ParagraphStyle("NKFoot",parent=styles["Normal"],fontSize=7,fontName="Helvetica",textColor=GRAY_RL,alignment=TA_CENTER)
    story=[]
    story.append(Spacer(1,1.5*cm))
    story.append(Paragraph("⬡ NEXUS KERNEL",ParagraphStyle("Brand",parent=styles["Normal"],fontSize=26,fontName="Helvetica-Bold",textColor=CYAN_RL,alignment=TA_CENTER,spaceAfter=3)))
    story.append(Paragraph("Professional Time-Series Econometrics Platform",ParagraphStyle("BrandSub",parent=styles["Normal"],fontSize=10,textColor=GRAY_RL,alignment=TA_CENTER,spaceAfter=3)))
    story.append(HRFlowable(width="100%",thickness=3,color=CYAN_RL,spaceAfter=10))
    story.append(Paragraph(title,title_s))
    story.append(Spacer(1,0.3*cm))
    meta=[["Prepared By:",user],["Email:",email],["Occupation:",occ],
          ["Institution:",institution or "—"],["Frequency:",st.session_state.freq or "N/A"],
          ["Generated:",datetime.now().strftime("%B %d, %Y — %H:%M")]]
    mt=Table(meta,colWidths=[4*cm,12*cm])
    mt.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),9),
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("TEXTCOLOR",(0,0),(0,-1),NAVY_RL),
        ("TEXTCOLOR",(1,0),(1,-1),GRAY_RL),("BOTTOMPADDING",(0,0),(-1,-1),4),("TOPPADDING",(0,0),(-1,-1),4),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[rl_colors.white,LIGHT_RL])]))
    story.append(mt); story.append(Spacer(1,0.4*cm))
    story.append(HRFlowable(width="100%",thickness=1,color=GRAY_RL,spaceAfter=8))
    story.append(Paragraph("Abstract",h1_s)); story.append(Paragraph(abstract,body_s))
    if st.session_state.clean_df is not None:
        story.append(PageBreak()); story.append(Paragraph("1. Dataset Overview",h1_s))
        df=st.session_state.clean_df
        story.append(Paragraph(f"Dataset: {len(df):,} observations × {len(df.columns)} variables. Frequency: {st.session_state.freq or 'N/A'}.",body_s))
        desc=df.select_dtypes(include=np.number).describe().T.round(4)
        if not desc.empty:
            tdata=[["Variable"]+list(desc.columns)]
            for var in desc.index[:15]: tdata.append([var]+[str(round(v,4)) for v in desc.loc[var]])
            colw=[4*cm]+[(14*cm/len(desc.columns))]*len(desc.columns)
            t=Table(tdata,colWidths=colw)
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY_RL),("TEXTCOLOR",(0,0),(-1,0),CYAN_RL),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.5),
                ("FONTNAME",(0,1),(-1,-1),"Courier"),("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.white,LIGHT_RL]),
                ("GRID",(0,0),(-1,-1),0.3,GRAY_RL),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
            story.append(t)
    sec_num=2
    if inc_ols and st.session_state.ols_res is not None:
        story.append(PageBreak()); m=st.session_state.ols_res
        story.append(Paragraph(f"{sec_num}. OLS Regression Results",h1_s)); sec_num+=1
        story.append(Paragraph(f"Dep. Var.: {m.model.endog_names} | N={int(m.nobs)} | R²={m.rsquared:.4f} | Adj.R²={m.rsquared_adj:.4f} | F-stat={m.fvalue:.4f} (p={m.f_pvalue:.4f})",body_s))
        ci=m.conf_int(); cd=[["Variable","Coef.","Std.Err.","t-Stat","p-Value","95% CI Lo","95% CI Hi","Sig."]]
        for vn in m.params.index:
            pv=m.pvalues[vn]; stars=pstar(pv)
            cd.append([vn,f"{m.params[vn]:.6f}",f"{m.bse[vn]:.6f}",f"{m.tvalues[vn]:.4f}",f"{pv:.4f}",f"{ci.loc[vn,0]:.4f}",f"{ci.loc[vn,1]:.4f}",stars])
        ct=Table(cd,colWidths=[4.5*cm,2.2*cm,2.2*cm,2*cm,2*cm,1.8*cm,1.8*cm,1*cm])
        ct.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY_RL),("TEXTCOLOR",(0,0),(-1,0),CYAN_RL),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTNAME",(0,1),(-1,-1),"Courier"),
            ("FONTSIZE",(0,0),(-1,-1),7.5),("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.white,LIGHT_RL]),
            ("GRID",(0,0),(-1,-1),0.3,GRAY_RL),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
        story.append(ct)
    if inc_ardl and st.session_state.ardl_res is not None:
        story.append(PageBreak()); ar=st.session_state.ardl_res; m=ar["model"]
        story.append(Paragraph(f"{sec_num}. ARDL Estimation Results",h1_s)); sec_num+=1
        story.append(Paragraph(f"ARDL({', '.join(str(o) for o in ar['order'])}) | Dep: {ar['dep']} | IC: {ar['ic'].upper()} | N={int(m.nobs)}",body_s))
        story.append(Paragraph(f"AIC={m.aic:.3f} | BIC={m.bic:.3f} | Log-Lik={m.llf:.3f}",body_s))
    if inc_var and st.session_state.var_res is not None:
        story.append(PageBreak()); vr=st.session_state.var_res; fv=vr["model"]
        story.append(Paragraph(f"{sec_num}. VAR System Results",h1_s)); sec_num+=1
        story.append(Paragraph(f"VAR({vr['best_p']}) | Variables: {', '.join(vr['endog'])} | IC: {vr['ic'].upper()}",body_s))
        story.append(Paragraph(f"AIC={fv.aic:.4f} | BIC={fv.bic:.4f}",body_s))
    if inc_vecm and st.session_state.vecm_res is not None:
        story.append(PageBreak()); vr=st.session_state.vecm_res
        story.append(Paragraph(f"{sec_num}. VECM Results",h1_s)); sec_num+=1
        story.append(Paragraph(f"VECM with r={vr['rank']} cointegrating vector(s) | Variables: {', '.join(vr['endog'])} | k={vr['k']}",body_s))
    if inc_garch and st.session_state.garch_res is not None:
        story.append(PageBreak()); gr=st.session_state.garch_res; res=gr["model"]
        story.append(Paragraph(f"{sec_num}. Volatility Model Results",h1_s)); sec_num+=1
        story.append(Paragraph(f"{gr['type']} | Variable: {gr['var']} | Distribution: {gr['dist'].upper()}",body_s))
        story.append(Paragraph(f"AIC={res.aic:.3f} | BIC={res.bic:.3f}",body_s))
        rows_g=[["Parameter","Estimate","Std.Err.","t-Stat","p-Value"]]
        for nm,val,se,tv,pv in zip(res.params.index,res.params,res.std_err,res.tvalues,res.pvalues):
            rows_g.append([nm,f"{val:.6f}",f"{se:.6f}",f"{tv:.4f}",f"{pv:.4f}"])
        gt=Table(rows_g,colWidths=[4.5*cm,3*cm,3*cm,3*cm,3*cm])
        gt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY_RL),("TEXTCOLOR",(0,0),(-1,0),CYAN_RL),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTNAME",(0,1),(-1,-1),"Courier"),
            ("FONTSIZE",(0,0),(-1,-1),8),("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.white,LIGHT_RL]),
            ("GRID",(0,0),(-1,-1),0.3,GRAY_RL),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
        story.append(gt)
    if inc_arima and st.session_state.arima_res is not None:
        story.append(PageBreak()); ar=st.session_state.arima_res; m=ar["model"]
        story.append(Paragraph(f"{sec_num}. ARIMA/SARIMA Forecast Results",h1_s)); sec_num+=1
        story.append(Paragraph(f"Model: {ar['name']} | Variable: {ar['target']} | AIC={m.aic:.3f} | BIC={m.bic:.3f}",body_s))
        if ar["rmse"]: story.append(Paragraph(f"Out-of-sample RMSE={ar['rmse']:.4f} | MAE={ar['mae']:.4f}",body_s))
    story.append(PageBreak())
    story.append(Paragraph("Disclaimer & Methodology",h1_s))
    story.append(Paragraph("This report was generated by NEXUS KERNEL v3.0 — a professional time-series econometrics platform "
                           "developed by Ahmed Hisham. All statistical tests conform to established econometric standards. "
                           "Results should be interpreted alongside economic theory and data quality assessment. "
                           "NEXUS KERNEL is not responsible for decisions made on the basis of this output.",body_s))
    story.append(Spacer(1,0.5*cm))
    story.append(HRFlowable(width="100%",thickness=1,color=GRAY_RL,spaceAfter=6))
    story.append(Paragraph(f"NEXUS KERNEL v3.0 © {datetime.now().year} · Research by Ahmed Hisham · Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",foot_s))
    doc.build(story); return buf.getvalue()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EXPORT & DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def page_export():
    st.markdown('<p class="page-title">💾 Export & Download</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Download cleaned datasets, model results, and coefficient tables.</p>', unsafe_allow_html=True)
    if st.session_state.clean_df is not None:
        st.markdown('<p class="sec-title">📂 Dataset Downloads</p>', unsafe_allow_html=True)
        df=st.session_state.clean_df
        csv_buf=io.StringIO(); df.to_csv(csv_buf)
        st.download_button("⬇ Download Cleaned Dataset (CSV)",csv_buf.getvalue().encode(),
            "nexus_clean_data.csv","text/csv",use_container_width=True)
        try:
            xl_buf=io.BytesIO()
            with pd.ExcelWriter(xl_buf,engine="openpyxl") as writer:
                df.to_excel(writer,sheet_name="Clean Data")
                df.describe().T.to_excel(writer,sheet_name="Summary Stats")
                if st.session_state.ols_res is not None:
                    m=st.session_state.ols_res
                    coef_df=pd.DataFrame({"Coefficient":m.params,"Std.Err":m.bse,"t-Stat":m.tvalues,"p-Value":m.pvalues})
                    coef_df.to_excel(writer,sheet_name="OLS Coefficients")
                if st.session_state.ardl_res is not None:
                    m=st.session_state.ardl_res["model"]
                    ardl_df=pd.DataFrame({"Coefficient":m.params,"Std.Err":m.bse,"t-Stat":m.tvalues,"p-Value":m.pvalues})
                    ardl_df.to_excel(writer,sheet_name="ARDL Coefficients")
                if st.session_state.garch_res is not None:
                    res=st.session_state.garch_res["model"]
                    garch_df=pd.DataFrame({"Estimate":res.params,"Std.Err":res.std_err,"t-Stat":res.tvalues,"p-Value":res.pvalues})
                    garch_df.to_excel(writer,sheet_name="GARCH Parameters")
            st.download_button("⬇ Download Full Results (Excel)",xl_buf.getvalue(),
                "nexus_results.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)
        except Exception as exc: st.warning(f"Excel export: {exc}")
    else:
        st.info("No data loaded. Load and clean a dataset first.")
    if st.session_state.stat_results:
        st.markdown('<p class="sec-title">📉 Stationarity Results</p>', unsafe_allow_html=True)
        sr=st.session_state.stat_results
        stat_data={"Test":["ADF","PP","KPSS","Zivot-Andrews"],
            "Statistic":[sr.get("adf",{}).get("stat","—"),sr.get("pp",{}).get("stat","—"),sr.get("kpss",{}).get("stat","—"),sr.get("za",{}).get("stat","—")],
            "p-value":[sr.get("adf",{}).get("pval","—"),sr.get("pp",{}).get("pval","—"),sr.get("kpss",{}).get("pval","—"),sr.get("za",{}).get("pval","—")],
            "Verdict":["Stationary" if sr.get("adf",{}).get("ok") else "Unit Root","Stationary" if sr.get("pp",{}).get("ok") else "Unit Root",
                       "Stationary" if sr.get("kpss",{}).get("ok") else "Non-Stationary","Stationary" if sr.get("za",{}).get("ok") else "Unit Root"]}
        st.dataframe(pd.DataFrame(stat_data),use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.authenticated:
        render_login(); return
    render_sidebar(); render_topbar()
    menu=st.session_state.menu
    pages={
        "home": page_home, "data": page_data, "transform": page_transform,
        "export": page_export, "unitroot": page_unitroot, "acf_pacf": page_acf_pacf,
        "correlogram": page_correlogram, "ols": page_ols, "ardl": page_ardl,
        "var": page_var, "vecm": page_vecm, "garch": page_garch, "arima": page_arima,
        "diagnostics": page_diagnostics, "stability": page_stability,
        "normality": page_normality, "forecast": page_forecast,
        "decompose": page_decompose, "stats": page_stats,
        "chat": page_chat, "report": page_report,
    }
    page_fn=pages.get(menu, page_home)
    try:
        page_fn()
    except Exception as exc:
        st.error(f"**Page error in '{menu}':** {exc}")
        with st.expander("Technical details (for debugging)"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
