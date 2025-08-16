# app.py — Flight Ticket Price Optimizer (MCT → LHR)
# Professional edition: large white SVG airplane + ML-first + robust loader
# Deep explanations with formulas & worked examples + Eid highlighting + What-If

import os, io, joblib
from itertools import count
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================
# PAGE SETUP & THEME
# =========================
st.set_page_config(page_title="Flight Price Optimizer — MCT → LHR", layout="wide")

# Palette (unchanged)
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
RED_SEP          = "#E11D48"

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [BURGUNDY, "#4B5563", BURGUNDY_SOFT, "#9CA3AF"]

_chart_counter = count()
def next_key(name: str) -> str:
    return f"{name}-{next(_chart_counter)}"

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
      .stApp {{ font-family:'Inter', ui-sans-serif, system-ui; background:{WHITE}; color:{GRAY_900}; }}

      /* Sidebar */
      section[data-testid="stSidebar"] {{
        background:{BURGUNDY_DARKEST}; color:white; border-right:1px solid rgba(255,255,255,.12);
      }}
      section[data-testid="stSidebar"] * {{ color:white !important; }}
      .side-title {{ font-weight:800; font-size:22px; margin:8px 0 18px 0; }}
      .hr-soft {{ border:none; height:1px; background:{GRAY_200}; margin:14px 0; }}

      /* Hero */
      .hero {{
        display:grid; grid-template-columns: 200px 1fr; gap:18px; align-items:center;
        background:{BURGUNDY_DARKEST}; color:white;
        padding: 22px 26px; border-radius: 16px; border:1px solid rgba(255,255,255,.12);
        box-shadow: 0 2px 10px rgba(0,0,0,.06) inset;
      }}
      .hero h1 {{ margin:0; font-size:30px; font-weight:800; letter-spacing:.2px; }}
      .plane-wrap {{ display:flex; align-items:center; justify-content:center; }}
      .plane-wrap svg {{ width:180px; height:auto; display:block; }}
      .section-title {{ font-weight:800; color:{BURGUNDY_DARKEST}; margin:0 0 8px 0; font-size:18px; }}
      .soft-card {{ background:{GRAY_100}; border:1px solid {GRAY_200}; border-radius:12px; padding:12px 14px; }}
      .flight-card {{ border:1px solid {GRAY_200}; border-radius:14px; padding:12px; background:{WHITE}; }}
      .flight-head {{ font-weight:800; color:{BURGUNDY_DARKEST}; font-size:16px; }}
      .red-sep {{ height:3px; background:{RED_SEP}; border-radius:999px; margin:10px 0; }}
      div[data-testid="stDataFrame"] table {{ border:1px solid {GRAY_200}; border-radius:8px; }}
      .micro {{ color:{GRAY_500}; font-size:12px; }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# SIDEBAR — CONTROLS
# =========================
st.sidebar.markdown('<div class="side-title">Flight Dashboard — Controls & Filters</div>', unsafe_allow_html=True)

st.sidebar.subheader("Data")
# ML default
engine_selected = st.sidebar.radio("Pricing engine", ["Rule-based", "ML-based"], index=1)

DEFAULT_CSV_REL = "final_optimized_flights.csv"
ML_CSV_REL      = "final_optimized_flights_ML.csv"

st.sidebar.markdown("**Optimized CSV source**")
uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
csv_url       = st.sidebar.text_input("Or paste CSV URL (optional)", value="")
preferred_rel = ML_CSV_REL if engine_selected == "ML-based" else DEFAULT_CSV_REL
csv_path_hint = st.sidebar.text_input("Or local relative path", value=preferred_rel,
                                      help="Example: final_optimized_flights.csv (file next to app.py)")

# Audience + Explain depth
st.sidebar.subheader("Audience lens")
audience = st.sidebar.selectbox(
    "Tailor explanations for:",
    ["General", "Finance", "Revenue Management", "HR", "Operations", "Marketing"],
    index=0
)
explain_depth = st.sidebar.radio("Explanation depth", ["Basic", "Advanced (with formulas)"], index=1)

# Eid highlighting
st.sidebar.subheader("Holiday (Eid) highlighting")
highlight_eid = st.sidebar.toggle("Highlight Eid impact", value=False, help="Flag rows and mark chart points during Eid.")
eid_fitr = st.sidebar.date_input("Eid al-Fitr (range)", value=None)
eid_adha = st.sidebar.date_input("Eid al-Adha (range)", value=None)

# Explain mode + glossary
EXPLAIN_MODE = st.sidebar.toggle("Explain mode (show hints)", value=True)
with st.sidebar.expander("Glossary — simple definitions", expanded=False):
    st.markdown(
        "- **Expected revenue**: `ticket price × expected seats sold`.\n"
        "- **Optimized revenue**: expected revenue using the recommended price.\n"
        "- **Uplift**: `Optimized − Current expected revenue`.\n"
        "- **Load factor**: `seats sold ÷ capacity`.\n"
        "- **Days left**: days until departure.\n"
        "- **Demand index**: ~1 normal, >1 high, <1 low.\n"
    )

# =========================
# LOADER (robust)
# =========================
@st.cache_data(show_spinner=False)
def load_csv_resilient(engine_choice, uploaded, url, hint_rel, rb_rel, ml_rel):
    """
    Try: upload → URL → typed path → preferred by engine → fallback to the other file.
    Returns: (df|None, engine_used_label, source_msg)
    """
    parse_cols = ["departure_time","arrival_time"]

    # 1) uploaded
    if uploaded is not None:
        df = pd.read_csv(uploaded, parse_dates=parse_cols, low_memory=False)
        return df, engine_choice, "Uploaded file"

    # 2) URL
    if url.strip():
        try:
            df = pd.read_csv(url.strip(), parse_dates=parse_cols, low_memory=False)
            return df, engine_choice, "URL"
        except Exception:
            pass

    # 3) user hint (relative)
    hint = (hint_rel or "").strip()
    if hint and os.path.exists(hint):
        df = pd.read_csv(hint, parse_dates=parse_cols, low_memory=False)
        used_engine = "ML-based" if os.path.basename(hint)==os.path.basename(ml_rel) else "Rule-based"
        return df, used_engine, f"Local path: {hint}"

    # 4) preferred by engine
    preferred = ml_rel if engine_choice=="ML-based" else rb_rel
    if os.path.exists(preferred):
        df = pd.read_csv(preferred, parse_dates=parse_cols, low_memory=False)
        return df, engine_choice, f"Default ({preferred})"

    # 5) fallback to the other file
    fallback = rb_rel if engine_choice=="ML-based" else ml_rel
    if os.path.exists(fallback):
        df = pd.read_csv(fallback, parse_dates=parse_cols, low_memory=False)
        used_engine = "Rule-based" if engine_choice=="ML-based" else "ML-based"
        return df, used_engine, f"Fallback to ({fallback})"

    # nothing found
    return None, engine_choice, (
        f"Could not locate a CSV. Upload a file, paste a URL, or place "
        f"'{rb_rel}' (rule) and/or '{ml_rel}' (ml) next to app.py."
    )

df, engine_used, data_src_msg = load_csv_resilient(
    engine_selected, uploaded_file, csv_url, csv_path_hint, DEFAULT_CSV_REL, ML_CSV_REL
)

# =========================
# HERO (large drawn white airplane)
# =========================
PLANE_SVG = """
<svg viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="airplane">
  <g fill="white">
    <!-- fuselage -->
    <path d="M30 272c0-10 8-18 18-18l180-8 120-90c12-9 28-7 37 5 9 12 7 28-5 37l-96 72 120 6c11 1 20 10 20 21v18c0 12-10 21-22 21l-152-10-54 40 74 56c5 4 8 10 8 17v16c0 12-10 22-22 22-4 0-8-1-12-3l-102-59-86-5-28 22c-4 3-8 5-13 5H48c-10 0-18-8-18-18 0-6 3-11 7-15l35-28-26-18c-6-4-9-11-9-18v-14c0-10 8-18 18-18h5z"/>
    <!-- tail -->
    <path d="M286 132l26-60c4-9 15-13 24-9l20 9c9 4 13 15 9 24l-26 60-53-24z"/>
    <!-- nose cone -->
    <circle cx="382" cy="168" r="10"/>
  </g>
</svg>
"""

if df is None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="plane-wrap">{PLANE_SVG}</div>
          <div>
            <h1>Flight Ticket Price Optimizer — Muscat → London (LHR)</h1>
            <p>Dynamic pricing · Revenue uplift · Explainable recommendations</p>
          </div>
        </div>
        """, unsafe_allow_html=True
    )
    st.info(
        "- Easiest: place `final_optimized_flights.csv` next to `app.py`.\n"
        "- Or use the Upload / URL boxes in the sidebar.\n\n"
        f"You selected **{engine_selected}**. Preferred file is "
        f"`{ML_CSV_REL if engine_selected=='ML-based' else DEFAULT_CSV_REL}`.",
        icon="ℹ️"
    )
    st.stop()

# feedback
st.markdown(
    f"""
    <div class="hero">
      <div class="plane-wrap">{PLANE_SVG}</div>
      <div>
        <h1>Flight Ticket Price Optimizer — Muscat → London (LHR)</h1>
        <p>Engine in use: <b>{engine_used}</b>{' • Holiday highlighting ON' if highlight_eid else ''}</p>
      </div>
    </div>
    """, unsafe_allow_html=True
)
st.caption(f"Data source: {data_src_msg}")

# =========================
# DERIVED COLUMNS & CHECKS
# =========================
required = {"optimal_revenue","current_expected_revenue","optimal_price","price"}
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

if "stops" not in df.columns:
    df["stops"] = np.nan

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

# Eid ranges → bool column
def _range_to_tuple(val):
    if isinstance(val, (list, tuple)):
        if len(val)==2 and val[0] and val[1]:
            return (val[0], val[1])
    return None

eid_ranges = []
r1 = _range_to_tuple(eid_fitr)
r2 = _range_to_tuple(eid_adha)
if r1: eid_ranges.append(r1)
if r2: eid_ranges.append(r2)

def is_holiday_func(d):
    if not eid_ranges or pd.isna(d): return False
    for s,e in eid_ranges:
        if s <= d <= e: return True
    return False

df["is_holiday"] = df["dep_time"].dt.date.map(is_holiday_func) if highlight_eid else False

# =========================
# TOP INFO CARD (role tips)
# =========================
ROLE_TIPS = {
    "General": "Compare current vs optimized revenue to see where price changes matter most.",
    "Finance": "Focus on total uplift and revenue/seat; export tables for budget/forecast alignment.",
    "Revenue Management": "Use cabin bars & heatmap to target seasons and carriers with the highest leverage.",
    "HR": "Seats/Capacity informs counters & lounge staffing; holiday tags indicate crowding risk.",
    "Operations": "Watch load factor and holiday-tagged flights to plan gates, crew, and turnaround buffers.",
    "Marketing": "Histogram reveals discount vs increase mix; promote weak pockets (esp. outside Eid)."
}
if EXPLAIN_MODE:
    st.markdown(
        f"""
        <div class="soft-card">
          <b>Guided view — {audience}:</b> {ROLE_TIPS.get(audience, ROLE_TIPS['General'])}
        </div>
        """, unsafe_allow_html=True
    )

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.subheader("Quick Filters")
tod  = st.sidebar.radio("Flight time", ["All","Morning","Afternoon","Evening","Night"], index=0)
airlines = sorted(df["airline"].dropna().unique()) if "airline" in df.columns else []
sel_air = st.sidebar.multiselect("Airline", airlines, default=[])
ftype = st.sidebar.radio("Flight type", ["Any","Direct","1 Stop","2+ Stops"], index=0)

# =========================
# MAIN FILTER BAR
# =========================
min_dt, max_dt = df["dep_date"].min(), df["dep_date"].max()
if pd.isna(min_dt) or pd.isna(max_dt):
    min_dt = date.today()
    max_dt = date.today() + timedelta(days=90)

c1, c2, c3, c4 = st.columns([1.4, 1.2, 1.2, 1.2])
with c1:
    d_from, d_to = st.date_input("Departure date range", value=[min_dt, max_dt], min_value=min_dt, max_value=max_dt)
with c2:
    f_class = st.selectbox("Class", ["All","Economy","Business","First"], index=0)
with c3:
    seasons = ["All"] + (sorted(df["season_label"].dropna().unique()) if "season_label" in df.columns else [])
    f_season = st.selectbox("Season", seasons, index=0 if seasons else 0)
with c4:
    days_slider_max = int(max(1, df.get("days_left", pd.Series([1]*len(df))).max()))
    f_daysmax = st.slider("Days left (max)", 1, days_slider_max, days_slider_max)

mask = pd.Series(True, index=df.index)
mask &= (df["dep_date"] >= d_from) & (df["dep_date"] <= d_to)
if tod != "All": mask &= (df["time_of_day"] == tod)
if sel_air: mask &= df["airline"].isin(sel_air) if "airline" in df.columns else False
if ftype == "Direct":   mask &= (df["stops_norm"] == 0)
elif ftype == "1 Stop": mask &= (df["stops_norm"] == 1)
elif ftype == "2+ Stops": mask &= (df["stops_norm"] >= 2)
if f_class != "All": mask &= (df["class"] == f_class) if "class" in df.columns else False
if f_season != "All" and "season_label" in df.columns: mask &= (df["season_label"] == f_season)
mask &= (df.get("days_left", pd.Series([0]*len(df))) <= f_daysmax)

dff = df.loc[mask].copy()
if dff.empty:
    st.warning("No rows after filters. Adjust filters to see data.")
    st.stop()

# =========================
# KPIs + FORMULAS
# =========================
k1, k2, k3, k4 = st.columns(4)
cur_rev = dff["current_expected_revenue"].sum()
opt_rev = dff["optimal_revenue"].sum()
uplift  = opt_rev - cur_rev
pct     = (uplift/cur_rev*100.0) if cur_rev>0 else 0.0
k1.metric("Current Expected Revenue", f"OMR {cur_rev:,.0f}")
k2.metric("Optimized Revenue",        f"OMR {opt_rev:,.0f}")
k3.metric("Total Uplift",             f"OMR {uplift:,.0f}", f"{pct:.1f}%")
k4.metric("Rows (filtered)",          f"{len(dff):,}")

# extra KPIs
lf = None
if {"seat_sold","capacity"}.issubset(dff.columns):
    cap_sum  = dff["capacity"].sum()
    sold_sum = dff["seat_sold"].sum()
    lf = (sold_sum / cap_sum * 100.0) if cap_sum else None

rps = dff["current_expected_revenue"].sum() / max(dff["capacity"].sum(), 1) if "capacity" in dff.columns else None
avg_delta = dff["pct_change_vs_current_price"].mean() if "pct_change_vs_current_price" in dff.columns else None

e1, e2, e3 = st.columns(3)
e1.metric("Load factor (filtered)", f"{lf:.1f}%" if lf is not None else "—")
e2.metric("Revenue / seat", f"OMR {rps:,.2f}" if rps is not None else "—")
e3.metric("Avg. price change", f"{avg_delta:+.1f}%" if avg_delta is not None else "—")

if EXPLAIN_MODE and explain_depth == "Advanced (with formulas)":
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown("**How these KPIs are computed**")
    st.latex(r"""
        \text{ExpectedRevenue} = \text{Price} \times \text{ExpectedSeats}
    """)
    st.latex(r"""
        \text{Uplift} = \sum (\text{OptimizedRevenue}) - \sum (\text{CurrentExpectedRevenue})
    """)
    st.markdown(
        f"""
        **Worked example (using your filtered data):**  
        - Current = **OMR {cur_rev:,.0f}**  
        - Optimized = **OMR {opt_rev:,.0f}**  
        - Uplift = **OMR {uplift:,.0f}** ({pct:.1f}%)
        """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)

# =========================
# RECOMMENDATIONS TABLE + EXPORT
# =========================
st.markdown('<div class="section-title">Price recommendations (filtered)</div>', unsafe_allow_html=True)
cols = ["airline","flight","stops","class","departure_time","arrival_time","days_left",
        "price","optimal_price","pct_change_vs_current_price",
        "seat_sold","capacity","residual_capacity",
        "current_expected_revenue","optimal_revenue","season_label","time_of_day","is_holiday"]
present = [c for c in cols if c in dff.columns]
table = dff[present].sort_values("optimal_revenue", ascending=False).head(1000).copy()
if "is_holiday" in table.columns:
    table["is_holiday"] = table["is_holiday"].map({True:"Holiday", False:"—"})
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

if EXPLAIN_MODE:
    st.markdown(
        """
        <div class="soft-card">
          <b>How to read:</b> Each row shows the current vs recommended price and the resulting revenue impact.
          <br><b>Use it to:</b> Sort by uplift and prioritize the largest opportunities. Holiday = rows inside your Eid ranges.
        </div>
        """, unsafe_allow_html=True
    )

# Export
buf_csv = table.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=buf_csv, file_name="recommendations_filtered.csv", mime="text/csv")

try:
    buf_xlsx = io.BytesIO()
    with pd.ExcelWriter(buf_xlsx, engine="openpyxl") as writer:
        table.to_excel(writer, index=False, sheet_name="recommendations")
    st.download_button("Download filtered Excel", data=buf_xlsx.getvalue(),
                       file_name="recommendations_filtered.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
except Exception:
    st.caption("Excel export unavailable (openpyxl not found). CSV download still works.")

st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)

# =========================
# CHARTS — ROW 1
# =========================
cA, cB = st.columns(2)

with cA:
    st.markdown('<div class="section-title">Revenue by class — before vs after</div>', unsafe_allow_html=True)
    g = dff.groupby("class")[["current_expected_revenue","optimal_revenue"]].sum().reset_index()
    fig = go.Figure(data=[
        go.Bar(name="Before", x=g["class"], y=g["current_expected_revenue"], marker_color=BURGUNDY,
               hovertemplate="<b>%{x}</b><br>Before: OMR %{y:,.0f}<extra></extra>"),
        go.Bar(name="After",  x=g["class"], y=g["optimal_revenue"], marker_color="#9CA3AF",
               hovertemplate="<b>%{x}</b><br>After: OMR %{y:,.0f}<extra></extra>")
    ])
    fig.update_layout(barmode="group", yaxis_title="Revenue (OMR)")
    st.plotly_chart(fig, use_container_width=True, key=next_key("rev-class"))

    if EXPLAIN_MODE:
        st.markdown(
            """
            <div class="soft-card">
              <b>How to read:</b> Grouped bars compare total revenue before vs after optimization for each cabin.
              <br><b>Use it to:</b> Focus on cabins with the strongest absolute improvement.
            </div>
            """, unsafe_allow_html=True
        )
        if explain_depth == "Advanced (with formulas)":
            st.latex(r"""
                \text{CabinUplift}(\mathrm{cabin}) =
                \sum_{\mathrm{rows\ in\ cabin}} \big(\text{Revenue}_{\mathrm{after}} - \text{Revenue}_{\mathrm{before}}\big)
            """)

with cB:
    st.markdown('<div class="section-title">% change vs current price — distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(dff, x="pct_change_vs_current_price", nbins=40, color_discrete_sequence=[BURGUNDY],
                       labels={"pct_change_vs_current_price":"% change"})
    fig.update_traces(hovertemplate="%{y} flights at %{x:.1f}% change<extra></extra>")
    fig.update_layout(xaxis_title="% change", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True, key=next_key("hist-change"))

    if EXPLAIN_MODE:
        st.markdown(
            """
            <div class="soft-card">
              <b>How to read:</b> Left tail = discounts; right tail = increases. A balanced distribution around small changes is expected.
              <br><b>Use it to:</b> Sanity-check that recommendations are realistic and not overly aggressive.
            </div>
            """, unsafe_allow_html=True
        )
        if explain_depth == "Advanced (with formulas)":
            st.latex(r"""
                \% \Delta \text{Price} = 100 \times \frac{\text{OptimalPrice} - \text{CurrentPrice}}{\text{CurrentPrice}}
            """)

# =========================
# CHARTS — ROW 2
# =========================
cC, cD = st.columns(2)

with cC:
    st.markdown('<div class="section-title">Days left vs optimal price</div>', unsafe_allow_html=True)
    sample = dff.sample(min(5000, len(dff)), random_state=42) if len(dff)>5000 else dff
    if highlight_eid and "is_holiday" in sample.columns:
        fig = px.scatter(
            sample, x="days_left", y="optimal_price", color="class", symbol="is_holiday",
            opacity=0.65,
            color_discrete_sequence=[BURGUNDY, "#6B7280", BURGUNDY_SOFT],
            labels={"days_left":"Days left", "optimal_price":"Optimal price (OMR)", "is_holiday":"Holiday"}
        )
    else:
        fig = px.scatter(
            sample, x="days_left", y="optimal_price", color="class",
            opacity=0.65,
            color_discrete_sequence=[BURGUNDY, "#6B7280", BURGUNDY_SOFT],
            labels={"days_left":"Days left", "optimal_price":"Optimal price (OMR)"}
        )
    fig.update_traces(hovertemplate="Days left: %{x}<br>Optimal price: OMR %{y:,.0f}<extra></extra>")
    fig.update_layout(xaxis_title="Days left", yaxis_title="Optimal price (OMR)")
    st.plotly_chart(fig, use_container_width=True, key=next_key("days-opt"))

    if EXPLAIN_MODE:
        st.markdown(
            """
            <div class="soft-card">
              <b>How to read:</b> Each point is a flight/cabin. Moving right means fewer days left.
              <br><b>Use it to:</b> Understand how recommended prices rise closer to departure (last-minute premium).
            </div>
            """, unsafe_allow_html=True
        )
        if explain_depth == "Advanced (with formulas)":
            st.latex(r"""
                \text{LatePremium}(d) \propto 1 - \tanh\!\left(\frac{d}{40}\right)
            """)
            st.markdown("<span class='micro'>We use a smooth time effect that increases as days-left decreases.</span>", unsafe_allow_html=True)

with cD:
    st.markdown('<div class="section-title">Uplift % — airline × season</div>', unsafe_allow_html=True)
    if "season_label" in dff.columns and not dff.empty:
        heat = (dff.groupby(["airline", "season_label"], as_index=False)
                  .agg(cur=("current_expected_revenue","sum"),
                       opt=("optimal_revenue","sum")))
        heat["uplift_pct"] = np.where(
            heat["cur"]>0, (heat["opt"]-heat["cur"])/heat["cur"]*100.0, 0.0
        )
        if not heat.empty:
            pivot = heat.pivot(index="airline", columns="season_label", values="uplift_pct").fillna(0)
            fig_h = px.imshow(
                pivot.values, x=pivot.columns, y=pivot.index,
                color_continuous_scale=["#F9FAFB", "#E5E7EB", "#D1D5DB", BURGUNDY], origin="lower"
            )
            fig_h.update_layout(coloraxis_colorbar_title="% Uplift")
            st.plotly_chart(fig_h, use_container_width=True, key=next_key("uplift-heat"))

            if EXPLAIN_MODE:
                st.markdown(
                    """
                    <div class="soft-card">
                      <b>How to read:</b> Darker cells = higher % uplift. Rows are airlines; columns are seasons.
                      <br><b>Use it to:</b> Target the darkest pockets first (best ROI).
                    </div>
                    """, unsafe_allow_html=True
                )
                if explain_depth == "Advanced (with formulas)":
                    st.latex(r"""
                        \text{Uplift\%} = 100 \times \frac{\text{Revenue}_{\mathrm{after}} - \text{Revenue}_{\mathrm{before}}}{\text{Revenue}_{\mathrm{before}} + \epsilon}
                    """)
                    st.markdown("<span class='micro'>We use a tiny ε in code to avoid division by zero.</span>", unsafe_allow_html=True)
        else:
            st.info("Not enough data after filters to draw the heatmap.")
    else:
        st.info("Season labels not available in this dataset.")

# =========================
# TOP ALERTS
# =========================
st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Top alerts — biggest revenue opportunities</div>', unsafe_allow_html=True)
if {"optimal_revenue","current_expected_revenue","flight","airline"}.issubset(dff.columns):
    alerts = dff.copy()
    alerts["uplift_abs"] = alerts["optimal_revenue"] - alerts["current_expected_revenue"]
    top5 = (alerts.sort_values("uplift_abs", ascending=False)
                 .loc[:, ["airline","flight","departure_time","class","price","optimal_price","uplift_abs"]]
                 .head(5))
    st.dataframe(
        top5.style.format({"price":"OMR {:,.2f}", "optimal_price":"OMR {:,.2f}", "uplift_abs":"OMR {:,.0f}"}),
        use_container_width=True
    )
    if EXPLAIN_MODE and explain_depth == "Advanced (with formulas)":
        st.latex(r"""\text{Uplift}_{\text{row}} = \text{OptimalRevenue}_{\text{row}} - \text{CurrentExpectedRevenue}_{\text{row}}""")

# =========================
# WHAT-IF ENGINE (RB/ML) + EXPLANATIONS
# =========================
def _safe_float(x, default):
    try:
        v = float(x)
        if np.isnan(v): return float(default)
        return v
    except Exception:
        return float(default)

def _safe_int(x, default):
    try:
        v = int(round(float(x)))
        return max(0, v)
    except Exception:
        return int(default)

# Rule-based expected seats with elasticities by cabin
def expected_units_rule(row, test_price):
    el_map = {"Economy":-1.4, "Business":-0.8, "First":-0.5}
    el = el_map.get(row.get("class","Economy"), -0.8)

    base_price = _safe_float(row.get("base_price"), row.get("price", 1.0))
    price      = _safe_float(test_price, base_price)
    demand     = _safe_float(row.get("demand_index"), 1.0)
    days_left  = _safe_int(row.get("days_left"), 30)

    cap  = _safe_int(row.get("capacity"), 0)
    sold = _safe_int(row.get("seat_sold"), 0)
    res_cap = max(0, cap - sold)

    if base_price <= 0 or res_cap <= 0:
        return 0.0

    # Price elasticity effect and smooth time effect
    price_effect = (price / base_price) ** el
    time_effect  = 1.0 + 1.2 * (1 - np.tanh(days_left / 40))
    units = demand * price_effect * time_effect * (res_cap / 10.0)
    return float(np.clip(units, 0.0, res_cap))

# ML helpers (optional)
NUMERIC_FEATS_WISHLIST = [
    "price","days_left","demand_index","capacity","weekday","hour",
    "season_month","class_code","season_base_mult","event_mult",
    "weekday_mult","period_mult","stop_mult",
]
CATEG_FEATS_WISHLIST = ["airline","source_city","destination_city","departure_period","stops"]

def _safe_str(x, default="Unknown"):
    try:
        s=str(x)
        if s.strip()=="" or s.lower()=="nan": return default
        return s
    except Exception:
        return default

def prepare_feature_row(row, price_value):
    r=row.copy()
    if ("season_month" not in r.index) or pd.isna(r.get("season_month")):
        try: r["season_month"] = pd.to_datetime(r.get("departure_time"), errors="coerce").month
        except Exception: r["season_month"] = np.nan
    data={}
    for c in NUMERIC_FEATS_WISHLIST:
        if c=="price": continue
        if c in r.index: data[c]=[r.get(c, np.nan)]
    for c in CATEG_FEATS_WISHLIST:
        data[c]=[_safe_str(r.get(c,"Unknown"))]
    data["price"]=[float(price_value)]
    return pd.DataFrame(data)

def load_ml_models():
    models={}
    base="models_seats_ml"
    for cabin in ["Economy","Business","First"]:
        path=os.path.join(base, f"seats_model_{cabin.lower()}.joblib")
        if os.path.exists(path):
            models[cabin]=joblib.load(path)
    return models

def expected_units_ml(models,row,test_price):
    if not models:
        return expected_units_rule(row,test_price)
    cabin=row.get("class","Economy")
    model=models.get(cabin) or next(iter(models.values()))
    X=prepare_feature_row(row,test_price)
    pred=float(model.predict(X)[0])
    cap=_safe_int(row.get("capacity"),0); sold=_safe_int(row.get("seat_sold"),0)
    res=max(0, cap - sold)
    if res<=0: res = cap
    return max(0.0, min(pred, res))

def expected_units_unified(engine, models, row, test_price):
    if engine=="ML-based" and models:
        return expected_units_ml(models,row,test_price)
    return expected_units_rule(row,test_price)

ml_models = load_ml_models() if engine_used=="ML-based" else None
if engine_used=="ML-based" and not ml_models:
    st.warning("ML models not found. What-If will fallback to rule-based (place joblib models under models_seats_ml/).")

# =========================
# FLIGHT DETAILS + WHAT-IF
# =========================
st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Flight details</div>', unsafe_allow_html=True)

def format_key(r):
    dt = pd.to_datetime(r["departure_time"], errors="coerce")
    dt_text = dt.strftime("%Y-%m-%d %H:%M") if pd.notna(dt) else "NA"
    return f"{r.get('airline','?')} • {r.get('flight','?')} • {dt_text}"

sel_df = dff.copy()
sel_df["flight_key"] = sel_df.apply(format_key, axis=1)
choices = ["— Select a flight —"] + sel_df["flight_key"].unique().tolist()
selected = st.selectbox("Select a flight to view full details", options=choices, index=0)

if selected != "— Select a flight —":
    st.markdown('<div class="red-sep"></div>', unsafe_allow_html=True)
    sel = sel_df[sel_df["flight_key"] == selected].copy()

    row0 = sel.iloc[0]
    airline = row0.get("airline", "?")
    flight  = row0.get("flight", "?")
    dep     = pd.to_datetime(row0.get("departure_time"), errors="coerce")
    arr     = pd.to_datetime(row0.get("arrival_time"), errors="coerce")
    stops   = row0.get("stops", "—")
    season  = row0.get("season_label", "—")
    tod     = row0.get("time_of_day", "—")

    dur_h = ((arr - dep).total_seconds()/3600.0) if (pd.notna(dep) and pd.notna(arr)) else np.nan
    cap  = int(sel["capacity"].sum()) if "capacity" in sel.columns else None
    sold = int(sel["seat_sold"].sum()) if "seat_sold" in sel.columns else None
    util = (sold / cap * 100.0) if (cap and cap>0 and sold is not None) else None

    c_rev = sel["current_expected_revenue"].sum()
    o_rev = sel["optimal_revenue"].sum()
    uplift_abs = o_rev - c_rev
    uplift_pct = (uplift_abs / c_rev * 100.0) if c_rev>0 else 0.0

    st.markdown('<div class="flight-card">', unsafe_allow_html=True)
    a,b,c = st.columns([1.6, 1.2, 1.2])

    with a:
        st.markdown(f'<div class="flight-head">{airline} • {flight}</div>', unsafe_allow_html=True)
        st.write(f"**Departure:** {dep.strftime('%Y-%m-%d %H:%M') if pd.notna(dep) else '—'}")
        st.write(f"**Arrival:** {arr.strftime('%Y-%m-%d %H:%M') if pd.notna(arr) else '—'}")
        st.write(f"**Stops:** {stops}  |  **Season:** {season}  |  **Time of day:** {tod}")
        if not np.isnan(dur_h): st.write(f"**Duration:** {dur_h:.2f} hrs")
        if cap is not None or sold is not None:
            util_text = f"{util:.1f}%" if util is not None else "—"
            st.write(f"**Seats:** {sold if sold is not None else '—'} / {cap if cap is not None else '—'}  (Util: {util_text})")
        st.write(f"**Current revenue:** OMR {c_rev:,.0f}")
        st.write(f"**Optimized revenue:** OMR {o_rev:,.0f}")
        st.write(f"**Uplift:** OMR {uplift_abs:,.0f}  ({uplift_pct:.1f}%)")
        st.caption(f"Pricing engine: **{engine_used}**")

    with b:
        if cap is not None and sold is not None:
            mini = pd.DataFrame({"Metric":["Seats sold","Capacity"], "Value":[sold, cap]})
            fig_c = px.bar(mini, x="Metric", y="Value", color_discrete_sequence=[BURGUNDY, "#9CA3AF"])
            fig_c.update_layout(margin=dict(l=6,r=6,t=10,b=6), height=220,
                                yaxis_title=None, xaxis_title=None)
            st.plotly_chart(fig_c, use_container_width=True, key=next_key("seats-vs-capacity"))
        else:
            st.write("—")
        st.markdown("<div class='soft-card'><b>How to read:</b> Sold seats vs capacity for this flight.<br><b>Use it to:</b> spot underfilled flights.</div>", unsafe_allow_html=True)

    with c:
        if "class" in sel.columns:
            cab = (
                sel.groupby("class")
                   .agg(seats_sold=("seat_sold","sum") if "seat_sold" in sel.columns else ("price","count"),
                        capacity=("capacity","sum") if "capacity" in sel.columns else ("price","count"),
                        current_rev=("current_expected_revenue","sum"),
                        optimal_rev=("optimal_revenue","sum"))
                   .reset_index()
            )
            fig_b = go.Figure()
            fig_b.add_bar(name="Seats sold", x=cab["class"], y=cab["seats_sold"], marker_color=BURGUNDY)
            if "capacity" in cab.columns:
                fig_b.add_bar(name="Capacity", x=cab["class"], y=cab["capacity"], marker_color="#9CA3AF")
            fig_b.update_layout(barmode="group", height=220, margin=dict(l=6,r=6,t=10,b=6),
                                yaxis_title=None, xaxis_title=None)
            st.plotly_chart(fig_b, use_container_width=True, key=next_key("cabin-breakdown"))
        else:
            st.write("—")

    # Why this price? (clear bullet logic + formulas)
    if EXPLAIN_MODE:
        _cap = int(row0.get("capacity", 0) or 0)
        _sold = int(row0.get("seat_sold", 0) or 0)
        _lf = (_sold/_cap*100) if _cap else None
        _days = int(row0.get("days_left", 0) or 0)
        _demand = float(row0.get("demand_index", 1.0) or 1.0)
        _stops = str(row0.get("stops","")).strip().lower()
        _class = str(row0.get("class",""))
        bullets = []
        if _demand >= 1.2: bullets.append("Demand is **high** → small price lift is reasonable.")
        elif _demand <= 0.8: bullets.append("Demand is **low** → discounts can stimulate pickup.")
        else: bullets.append("Demand is **normal** → keep changes modest near base price.")
        if _days <= 7: bullets.append("Departure is **soon** → late-booking premium is typical.")
        elif _days >= 60: bullets.append("Departure is **far** → early buyers are price-sensitive.")
        if _lf is not None:
            if _lf >= 85: bullets.append("Load factor is **high** → room to raise price.")
            elif _lf <= 40: bullets.append("Load factor is **low** → consider more attractive pricing.")
            else: bullets.append("Load factor is **medium** → moderate adjustments.")
        if "direct" in _stops or _stops == "0": bullets.append("**Direct** flight → usually priced higher.")
        elif "1" in _stops: bullets.append("**1 stop** → slightly lower than direct.")
        else: bullets.append("**Multi-stop** → needs competitive price.")
        if _class == "Economy": bullets.append("**Economy**: highest price sensitivity.")
        elif _class == "Business": bullets.append("**Business**: moderate sensitivity.")
        elif _class == "First": bullets.append("**First**: lowest sensitivity.")
        st.markdown("#### 🧠 Why this price?")
        st.markdown("\n".join([f"- {b}" for b in bullets]))
        if explain_depth == "Advanced (with formulas)":
            st.latex(r"""
                \text{ExpectedSeats} \;\propto\; \underbrace{\left(\frac{P}{P_0}\right)^{\varepsilon}}_{\text{price effect}}
                \times \underbrace{\big(1 + 1.2\,[1-\tanh(d/40)]\big)}_{\text{time effect}} \times \underbrace{\text{DemandIndex}}_{\text{demand}}
            """)

    # What-If
    st.markdown('<div class="red-sep"></div>', unsafe_allow_html=True)
    st.subheader("Quick What-If: try a manual price")
    if (cap is not None) and (sold is not None):
        cur_price_for_sim = float(row0.get("price", np.nan))
        base_price = float(sel["price"].mean()) if np.isnan(cur_price_for_sim) else cur_price_for_sim
        default_price = float(row0.get("optimal_price", base_price))
        new_price = st.slider("Try a new ticket price (OMR)",
                              min_value=max(1.0, base_price*0.5),
                              max_value=base_price*1.5,
                              value=default_price, step=1.0)

        base_rev_sim = 0.0
        new_rev_sim  = 0.0
        for _, r in sel.iterrows():
            u_base = expected_units_unified(engine_used, ml_models, r, cur_price_for_sim)
            u_new  = expected_units_unified(engine_used, ml_models, r, new_price)
            base_rev_sim += cur_price_for_sim * u_base
            new_rev_sim  += new_price         * u_new

        delta_abs = new_rev_sim - base_rev_sim
        delta_pct = (delta_abs / base_rev_sim * 100.0) if base_rev_sim > 0 else 0.0

        w1, w2, w3 = st.columns(3)
        w1.metric("Base revenue (simulated)", f"OMR {base_rev_sim:,.0f}")
        w2.metric("What-If revenue",         f"OMR {new_rev_sim:,.0f}")
        w3.metric("Δ vs base",               f"OMR {delta_abs:,.0f}", f"{delta_pct:+.1f}%")

        if EXPLAIN_MODE and explain_depth == "Advanced (with formulas)":
            st.latex(r"""
                \Delta \text{Revenue} = P_{\text{new}}\cdot Q_{\text{new}} - P_{\text{base}}\cdot Q_{\text{base}}
            """)
            st.markdown("<span class='micro'>Where \(Q\) is simulated expected seats from the selected engine (ML if available, otherwise rule-based).</span>", unsafe_allow_html=True)
    else:
        st.info("Seats and capacity not available to simulate a Quick What-If for this flight.")

    st.markdown('<div class="red-sep"></div>', unsafe_allow_html=True)
    show_cols = [c for c in [
        "airline","flight","stops","class","departure_time","arrival_time","days_left",
        "price","optimal_price","pct_change_vs_current_price","seat_sold","capacity",
        "residual_capacity","current_expected_revenue","optimal_revenue","season_label","time_of_day","is_holiday"
    ] if c in sel.columns]
    st.dataframe(sel[show_cols].sort_values("optimal_revenue", ascending=False), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)
st.caption("© Dynamic Pricing Simulator — Streamlit + Plotly | ML-first with rule fallback | Large white SVG airplane | Deep explanations with formulas")
