# app.py — Flight Price Optimizer (MCT → LHR)
# Robust CSV loader with auto-fallback when ML file is missing.
# Works locally and on Streamlit Cloud.

import os, io, joblib
from itertools import count
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------- Page setup & palette ----------------
st.set_page_config(page_title="Flight Price Optimizer — MCT → LHR", layout="wide")

BURGUNDY_DARKEST = "#4B0E1E"
BURGUNDY         = "#6D0F1A"
BURGUNDY_SOFT    = "#8C1F28"
GRAY_900         = "#111827"
GRAY_700         = "#374151"
GRAY_500         = "#6B7280"
GRAY_300         = "#D1D5DB"
GRAY_200         = "#E5E7EB"
GRAY_100         = "#F3F4F6"
WHITE            = "#FFFFFF"

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [BURGUNDY, "#4B5563", BURGUNDY_SOFT, "#9CA3AF"]

_chart_counter = count()
def next_key(prefix: str) -> str:
    return f"{prefix}-{next(_chart_counter)}"

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
      .stApp {{ font-family:'Inter', ui-sans-serif, system-ui; background:{WHITE}; color:{GRAY_900}; }}
      section[data-testid="stSidebar"] {{ background:{BURGUNDY_DARKEST}; color:white; }}
      section[data-testid="stSidebar"] * {{ color:white !important; }}
      .hero {{
        display:grid; grid-template-columns: 90px 1fr; gap:16px; align-items:center;
        background:{BURGUNDY_DARKEST}; color:white; padding:18px 22px; border-radius:14px;
      }}
      .hero h1 {{ margin:0; font-size:26px; font-weight:800; }}
      .section-title {{ font-weight:800; color:{BURGUNDY_DARKEST}; margin: 2px 0 8px 0; font-size:18px; }}
      .hr {{ height:1px; background:{GRAY_200}; margin:12px 0; }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Sidebar: engine and loader ----------------
st.sidebar.subheader("Data")
engine_choice = st.sidebar.radio("Pricing engine", ["Rule-based", "ML-based"], index=0)

DEFAULT_CSV_REL = "final_optimized_flights.csv"        # put this next to app.py
ML_CSV_REL      = "final_optimized_flights_ML.csv"     # optional ML csv (if generated)

st.sidebar.markdown("**Optimized CSV source**")
uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
csv_url       = st.sidebar.text_input("Or paste CSV URL (optional)", value="")
preferred_rel = ML_CSV_REL if engine_choice == "ML-based" else DEFAULT_CSV_REL
typed_path    = st.sidebar.text_input("Or local relative path", value=preferred_rel,
                                      help="Example: final_optimized_flights.csv")

@st.cache_data(show_spinner=False)
def load_csv_resilient(engine_selected, uploaded, url, hint_rel, rule_rel, ml_rel):
    """
    Returns: (df | None, used_engine_label, source_message)
    Never raises FileNotFound; caller decides what to show.
    Try order:
      1) uploaded
      2) url
      3) hint_rel (if exists)
      4) preferred by engine
      5) auto-fallback to the other one
      else -> (None, engine_selected, reason)
    """
    # 1) uploaded
    if uploaded is not None:
        df = pd.read_csv(uploaded, parse_dates=["departure_time","arrival_time"],
                         infer_datetime_format=True, low_memory=False)
        return df, engine_selected, "Uploaded file"

    # 2) url
    if url.strip():
        try:
            df = pd.read_csv(url.strip(), parse_dates=["departure_time","arrival_time"],
                             infer_datetime_format=True, low_memory=False)
            return df, engine_selected, "URL"
        except Exception as e:
            # continue; don't fail hard
            pass

    # 3) user hint
    hint = (hint_rel or "").strip()
    if hint and os.path.exists(hint):
        df = pd.read_csv(hint, parse_dates=["departure_time","arrival_time"],
                         infer_datetime_format=True, low_memory=False)
        # pick engine by filename type
        used_engine = "ML-based" if os.path.basename(hint) == os.path.basename(ml_rel) else "Rule-based"
        return df, used_engine, f"Local path: {hint}"

    # 4) preferred
    preferred = ml_rel if engine_selected == "ML-based" else rule_rel
    if os.path.exists(preferred):
        df = pd.read_csv(preferred, parse_dates=["departure_time","arrival_time"],
                         infer_datetime_format=True, low_memory=False)
        return df, engine_selected, f"Default ({preferred})"

    # 5) fallback to the other file
    fallback = rule_rel if engine_selected == "ML-based" else ml_rel
    if os.path.exists(fallback):
        df = pd.read_csv(fallback, parse_dates=["departure_time","arrival_time"],
                         infer_datetime_format=True, low_memory=False)
        used_engine = "Rule-based" if engine_selected == "ML-based" else "ML-based"
        return df, used_engine, f"Fallback to ({fallback})"

    # nothing found
    return None, engine_selected, (
        f"Could not find '{preferred}'. Upload a file, paste a URL, or place "
        f"'{rule_rel}' (rule) and/or '{ml_rel}' (ml) next to app.py."
    )

df, engine_used, data_src_msg = load_csv_resilient(
    engine_choice, uploaded_file, csv_url, typed_path, DEFAULT_CSV_REL, ML_CSV_REL
)

# Inline friendly uploader if nothing found (no error)
if df is None:
    st.markdown(
        f"""
        <div class="hero">
          <div>✈️</div>
          <div>
            <h1>Flight Ticket Price Optimizer — Muscat → London (LHR)</h1>
            <p>Dynamic pricing · Revenue uplift · Explainable recommendations</p>
          </div>
        </div>
        """, unsafe_allow_html=True
    )
    st.info(
        "I couldn't locate a CSV to load.\n\n"
        "- **Easiest**: put `final_optimized_flights.csv` next to `app.py` in your repo.\n"
        "- Or use the **Upload CSV**/**URL** boxes in the sidebar.\n\n"
        f"Tip: You selected **{engine_choice}**. Preferred file would be "
        f"`{ML_CSV_REL if engine_choice=='ML-based' else DEFAULT_CSV_REL}`.",
        icon="ℹ️"
    )
    st.stop()

# Let the user know if we auto-fell back
if engine_used != engine_choice:
    st.info(
        f"Selected **{engine_choice}**, but preferred CSV wasn't found. "
        f"Loaded **{engine_used}** data instead ({data_src_msg})."
    )
else:
    st.caption(f"Data source: {data_src_msg}")

# ---------------- Derived columns / checks ----------------
required_cols = {"optimal_revenue","current_expected_revenue","optimal_price","price"}
if missing := [c for c in required_cols if c not in df.columns]:
    st.error(f"CSV missing required columns: {missing}")
    st.stop()

if "season_label" not in df.columns and "season_month" in df.columns:
    season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                  6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"}
    df["season_label"] = df["season_month"].map(season_map)

if "pct_change_vs_current_price" not in df.columns:
    df["pct_change_vs_current_price"] = 100.0 * (df["optimal_price"] - df["price"]) / df["price"]

if {"capacity","seat_sold"}.issubset(df.columns) and "residual_capacity" not in df.columns:
    df["residual_capacity"] = (df["capacity"] - df["seat_sold"]).clip(lower=0)

def normalize_stops(v):
    if pd.isna(v): return None
    s=str(v).lower().strip()
    if s in {"direct","nonstop","non-stop","0","0 stop","0 stops"}: return 0
    if s in {"1","1 stop","one","one stop"}: return 1
    try:
        n=int(s.split()[0]); return 2 if n>=2 else n
    except: return 2 if any(x in s for x in ["2","two","3","three"]) else None

df["stops_norm"] = df["stops"].apply(normalize_stops)
df["dep_time"]   = pd.to_datetime(df["departure_time"], errors="coerce")
df["dep_date"]   = df["dep_time"].dt.date
df["dep_hour"]   = df["dep_time"].dt.hour

def time_bucket(h):
    if pd.isna(h): return "Unknown"
    h=int(h)
    if 6<=h<12:  return "Morning"
    if 12<=h<17: return "Afternoon"
    if 17<=h<21: return "Evening"
    return "Night"

df["time_of_day"] = df["dep_hour"].apply(time_bucket)

# ---------------- Hero ----------------
st.markdown(
    f"""
    <div class="hero">
      <div>✈️</div>
      <div>
        <h1>Flight Ticket Price Optimizer — Muscat → London (LHR)</h1>
        <p>Engine in use: <b>{engine_used}</b></p>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- Filters ----------------
st.sidebar.subheader("Quick Filters")
tod  = st.sidebar.radio("Flight time", ["All","Morning","Afternoon","Evening","Night"], index=0)
airlines = sorted(df["airline"].dropna().unique())
sel_air = st.sidebar.multiselect("Airline", airlines, default=[])

ftype = st.sidebar.radio("Flight type", ["Any","Direct","1 Stop","2+ Stops"], index=0)

min_dt, max_dt = df["dep_date"].min(), df["dep_date"].max()
c1, c2, c3, c4 = st.columns([1.4, 1.2, 1.2, 1.2])
with c1:
    d_from, d_to = st.date_input("Departure date range", value=[min_dt, max_dt], min_value=min_dt, max_value=max_dt)
with c2:
    f_class = st.selectbox("Class", ["All","Economy","Business","First"], index=0)
with c3:
    seasons = ["All"] + (sorted(df["season_label"].dropna().unique()) if "season_label" in df.columns else [])
    f_season = st.selectbox("Season", seasons, index=0 if seasons else 0)
with c4:
    days_slider_max = int(max(1, df.get("days_left", pd.Series([1])).max()))
    f_daysmax = st.slider("Days left (max)", 1, days_slider_max, days_slider_max)

mask = pd.Series(True, index=df.index)
mask &= (df["dep_date"] >= d_from) & (df["dep_date"] <= d_to)
if tod != "All": mask &= (df["time_of_day"] == tod)
if sel_air: mask &= df["airline"].isin(sel_air)
if ftype == "Direct":   mask &= (df["stops_norm"] == 0)
elif ftype == "1 Stop": mask &= (df["stops_norm"] == 1)
elif ftype == "2+ Stops": mask &= (df["stops_norm"] >= 2)
if f_class != "All": mask &= (df["class"] == f_class)
if f_season != "All" and "season_label" in df.columns: mask &= (df["season_label"] == f_season)
mask &= (df.get("days_left", pd.Series([0]*len(df))) <= f_daysmax)

dff = df.loc[mask].copy()
if dff.empty:
    st.warning("No rows after filters. Adjust filters to see data.")
    st.stop()

# ---------------- KPIs ----------------
k1, k2, k3, k4 = st.columns(4)
cur_rev = dff["current_expected_revenue"].sum()
opt_rev = dff["optimal_revenue"].sum()
uplift  = opt_rev - cur_rev
pct     = (uplift/cur_rev*100.0) if cur_rev>0 else 0.0
k1.metric("Current Expected Revenue", f"OMR {cur_rev:,.0f}")
k2.metric("Optimized Revenue",        f"OMR {opt_rev:,.0f}")
k3.metric("Total Uplift",             f"OMR {uplift:,.0f}", f"{pct:.1f}%")
k4.metric("Rows (filtered)",          f"{len(dff):,}")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ---------------- Table ----------------
st.markdown('<div class="section-title">Price recommendations (filtered)</div>', unsafe_allow_html=True)
cols = ["airline","flight","stops","class","departure_time","arrival_time","days_left",
        "price","optimal_price","pct_change_vs_current_price",
        "seat_sold","capacity","residual_capacity",
        "current_expected_revenue","optimal_revenue","season_label","time_of_day"]
present = [c for c in cols if c in dff.columns]
table = dff[present].sort_values("optimal_revenue", ascending=False).head(1000).copy()
st.dataframe(
    table.style.format({
        "price":"OMR {:,.2f}",
        "optimal_price":"OMR {:,.2f}",
        "current_expected_revenue":"OMR {:,.0f}",
        "optimal_revenue":"OMR {:,.0f}",
        "pct_change_vs_current_price":"{:+.1f}%"
    }),
    use_container_width=True
)

# ---------------- Charts ----------------
cA, cB = st.columns(2)
with cA:
    st.markdown('<div class="section-title">Revenue by class — before vs after</div>', unsafe_allow_html=True)
    g = dff.groupby("class")[["current_expected_revenue","optimal_revenue"]].sum().reset_index()
    fig = go.Figure(data=[
        go.Bar(name="Before", x=g["class"], y=g["current_expected_revenue"], marker_color=BURGUNDY),
        go.Bar(name="After",  x=g["class"], y=g["optimal_revenue"],        marker_color="#9CA3AF")
    ])
    fig.update_layout(barmode="group", yaxis_title="Revenue (OMR)")
    st.plotly_chart(fig, use_container_width=True, key=next_key("rev-class"))

with cB:
    st.markdown('<div class="section-title">% change vs current price — distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(dff, x="pct_change_vs_current_price", nbins=40, color_discrete_sequence=[BURGUNDY])
    fig.update_layout(xaxis_title="% change", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True, key=next_key("hist-change"))

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.caption("© Dynamic Pricing Simulator — Cloud-friendly loader with ML fallback")
