# app.py — Flight Ticket Price Optimizer (Muscat → London)
# Cloud-friendly CSV loader (upload/URL/relative path)
# Clean English UI + Explain Mode + Glossary + Visual hints
# Supports Rule-based & ML-based CSVs

import os, io, joblib
from itertools import count
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================
# PAGE & THEME
# =========================
st.set_page_config(page_title="Flight Price Optimizer — MCT → LHR", layout="wide")

# Palette
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

# =========================
# GLOBAL STYLE
# =========================
st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
      .stApp {{ font-family: 'Inter', ui-sans-serif, system-ui; background:{WHITE}; color:{GRAY_900}; }}

      /* Sidebar */
      section[data-testid="stSidebar"] {{
        background:{BURGUNDY_DARKEST}; color:{WHITE};
        border-right:1px solid rgba(255,255,255,.12);
      }}
      section[data-testid="stSidebar"] * {{ color:{WHITE} !important; }}
      .side-title {{ font-weight:800; font-size:22px; margin:8px 0 18px 0; }}
      .hr-soft {{ border:none; height:1px; background:{GRAY_200}; margin: 14px 0; }}

      /* Hero */
      .hero {{
        display:grid; grid-template-columns: 108px 1fr; gap:18px; align-items:center;
        background:{BURGUNDY_DARKEST}; color:{WHITE};
        padding: 20px 24px; border-radius: 16px;
        border: 1px solid rgba(255,255,255,.12);
        box-shadow: 0 2px 10px rgba(0,0,0,.06) inset;
      }}
      .hero h1 {{ margin:0; font-size:28px; font-weight:800; color:{WHITE}; letter-spacing:.2px; }}

      .section-title {{ font-weight:800; color:{BURGUNDY_DARKEST}; margin: 0 0 8px 0; font-size:18px; }}
      .soft-card {{ background:{GRAY_100}; border:1px solid {GRAY_200}; border-radius:12px; padding:10px 12px; color:{GRAY_900}; }}

      .flight-card {{ border:1px solid {GRAY_200}; border-radius:14px; padding:12px; background:{WHITE}; }}
      .flight-head {{ font-weight:800; color:{BURGUNDY_DARKEST}; font-size:16px; }}
      .red-sep {{ height:3px; background:{RED_SEP}; border-radius:999px; margin:10px 0; }}

      div[data-testid="stDataFrame"] table {{ border:1px solid {GRAY_200}; border-radius:8px; }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# SIDEBAR — CONTROLS & HELP
# =========================
st.sidebar.markdown('<div class="side-title">Flight Dashboard — Controls & Filters</div>', unsafe_allow_html=True)

st.sidebar.subheader("Data")
engine = st.sidebar.radio("Pricing engine", ["Rule-based", "ML-based"], index=0)

# ===== Cloud-friendly loader =====
DEFAULT_CSV_REL = "final_optimized_flights.csv"        # rule-based CSV shipped with app
ML_CSV_REL      = "final_optimized_flights_ML.csv"     # ML-based CSV (if available)

st.sidebar.markdown("**Optimized CSV source**")
uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
csv_url = st.sidebar.text_input("Or paste CSV URL (optional)", value="")
fallback_rel = ML_CSV_REL if engine == "ML-based" else DEFAULT_CSV_REL
csv_path_hint = st.sidebar.text_input("Or local relative path", value=fallback_rel,
                                      help="Example: final_optimized_flights.csv (file in repo root)")

@st.cache_data(show_spinner=False)
def load_csv_from_any(uploaded, url, local_hint):
    # 1) uploaded file
    if uploaded is not None:
        return pd.read_csv(uploaded, parse_dates=["departure_time","arrival_time"],
                           infer_datetime_format=True, low_memory=False)
    # 2) URL
    if url.strip():
        try:
            return pd.read_csv(url.strip(), parse_dates=["departure_time","arrival_time"],
                               infer_datetime_format=True, low_memory=False)
        except Exception as e:
            st.warning(f"Could not read URL: {e}")
    # 3) relative path (hint)
    hint = (local_hint or "").strip()
    if hint and os.path.exists(hint):
        return pd.read_csv(hint, parse_dates=["departure_time","arrival_time"],
                           infer_datetime_format=True, low_memory=False)
    # 4) fallback by engine
    fallback = ML_CSV_REL if engine == "ML-based" else DEFAULT_CSV_REL
    if os.path.exists(fallback):
        return pd.read_csv(fallback, parse_dates=["departure_time","arrival_time"],
                           infer_datetime_format=True, low_memory=False)
    raise FileNotFoundError(
        f"Could not locate a CSV. Upload a file, paste a URL, or place '{fallback}' next to app.py."
    )

try:
    df = load_csv_from_any(uploaded_file, csv_url, csv_path_hint)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.sidebar.markdown("---")
EXPLAIN_MODE = st.sidebar.toggle("Explain mode (show hints)", value=True)
with st.sidebar.expander("Glossary — simple definitions", expanded=False):
    st.markdown(
        "- **Expected revenue**: ticket price × **expected** seats sold (a forecast).\n"
        "- **Optimized revenue**: same but after applying the recommended price.\n"
        "- **Uplift**: Optimized revenue − Current revenue.\n"
        "- **Load factor**: seats sold ÷ capacity (how full the cabin is).\n"
        "- **Days left**: days until departure.\n"
        "- **Demand index**: ~1 = normal, >1 = higher demand, <1 = lower demand.\n"
    )

st.sidebar.subheader("Quick Filters")
sel_tod = st.sidebar.radio("Flight time", ["All","Morning","Afternoon","Evening","Night"], index=0)

# =========================
# DERIVATIONS & CHECKS
# =========================
required = {"optimal_revenue","current_expected_revenue","optimal_price","price"}
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

if "season_label" not in df.columns and "season_month" in df.columns:
    season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                  6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"}
    df["season_label"] = df["season_month"].map(season_map)

if "pct_change_vs_current_price" not in df.columns:
    df["pct_change_vs_current_price"] = 100.0 * (df["optimal_price"] - df["price"]) / df["price"]

if "residual_capacity" not in df.columns and {"capacity","seat_sold"}.issubset(df.columns):
    df["residual_capacity"] = (df["capacity"] - df["seat_sold"]).clip(lower=0)

def normalize_stops(val):
    if pd.isna(val): return None
    s = str(val).strip().lower()
    if s in {"direct","nonstop","non-stop","0","0 stop","0 stops"}: return 0
    if s in {"1","1 stop","one","one stop"}: return 1
    try:
        n = int(s.split()[0]); return 2 if n >= 2 else n
    except Exception:
        return 2 if any(x in s for x in ["2","two","3","three"]) else None
df["stops_norm"] = df["stops"].apply(normalize_stops)

df["dep_date"] = pd.to_datetime(df["departure_time"]).dt.date
df["dep_hour"] = pd.to_datetime(df["departure_time"]).dt.hour
def time_bucket(h):
    if h is None or pd.isna(h): return "Unknown"
    h = int(h)
    if 6 <= h < 12:  return "Morning"
    if 12 <= h < 17: return "Afternoon"
    if 17 <= h < 21: return "Evening"
    return "Night"
df["time_of_day"] = df["dep_hour"].apply(time_bucket)

# =========================
# HERO
# =========================
AIRPLANE_SVG = f"""
<svg width="108" height="108" viewBox="0 0 108 108" xmlns="http://www.w3.org/2000/svg">
  <defs><linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#FFFFFF" stop-opacity=".98"/>
    <stop offset="100%" stop-color="#F9FAFB" stop-opacity=".85"/></linearGradient></defs>
  <path d="M98 50c0 3-2.4 5.4-5.4 5.4H69L45 82l-9-9 13.5-12.5H29l-9 7-6-6 9.5-12.5h30L69 37h23.6c3 0 5.4 2.4 5.4 5.4V50z"
        fill="url(#g1)" stroke="{WHITE}" stroke-opacity=".25" stroke-width="1.3" />
  <circle cx="83" cy="32" r="3.5" fill="{WHITE}" />
</svg>
"""
st.markdown(
    f"""
    <div class="hero">
      <div>{AIRPLANE_SVG}</div>
      <div>
        <h1>Flight Ticket Price Optimizer — Muscat → London (LHR)</h1>
        <p>Dynamic pricing · Revenue uplift · Explainable recommendations</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
if EXPLAIN_MODE:
    st.info(
        "How to read this page:\n"
        "- KPIs at the top show current revenue, optimized revenue, and the uplift.\n"
        "- The table lists current vs recommended price per flight/cabin.\n"
        "- Charts summarize where improvements come from.\n",
        icon="ℹ️"
    )
st.markdown("")

# =========================
# MAIN FILTER BAR
# =========================
airlines_all = sorted(df["airline"].dropna().unique().tolist())
sel_airlines = st.sidebar.multiselect("Airline", airlines_all, default=[])
flight_type_sidebar = st.sidebar.radio("Flight type", ["Any","Direct","1 Stop","2+ Stops"], index=0)

min_dt, max_dt = df["dep_date"].min(), df["dep_date"].max()
cfa, cfb, cfc, cfd = st.columns([1.4, 1.2, 1.2, 1.2])
with cfa:
    date_from, date_to = st.date_input("Departure date range",
                                       value=[min_dt, max_dt],
                                       min_value=min_dt, max_value=max_dt)
with cfb:
    f_class = st.selectbox("Class", ["All","Economy","Business","First"], index=0)
with cfc:
    seasons = ["All"] + (sorted(df["season_label"].dropna().unique()) if "season_label" in df.columns else [])
    f_season = st.selectbox("Season", seasons, index=0 if seasons else 0)
with cfd:
    days_slider_max = int(max(1, df["days_left"].max()))
    f_daysmax = st.slider("Days left (max)", 1, days_slider_max, days_slider_max)

# Global mask
mask = pd.Series(True, index=df.index)
mask &= (df["dep_date"] >= date_from) & (df["dep_date"] <= date_to)
if sel_tod != "All": mask &= (df["time_of_day"] == sel_tod)
if sel_airlines: mask &= df["airline"].isin(sel_airlines)
if flight_type_sidebar == "Direct":      mask &= (df["stops_norm"] == 0)
elif flight_type_sidebar == "1 Stop":    mask &= (df["stops_norm"] == 1)
elif flight_type_sidebar == "2+ Stops":  mask &= (df["stops_norm"] >= 2)
mask &= (df["days_left"] <= f_daysmax)
if f_class != "All":  mask &= (df["class"] == f_class)
if f_season != "All" and "season_label" in df.columns: mask &= (df["season_label"] == f_season)

dff = df.loc[mask].copy()
if dff.empty:
    st.warning("No rows after filters. Adjust filters to see data.")
    st.stop()

# =========================
# KPIs
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

# Extra KPIs
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

if EXPLAIN_MODE:
    st.success(f"Summary: optimized revenue is **{'higher' if uplift>0 else 'lower'}** by OMR {abs(uplift):,.0f} ({pct:.1f}%).")

st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)

# =========================
# TABLE — RECOMMENDATIONS
# =========================
st.markdown('<div class="section-title">Price recommendations (filtered)</div>', unsafe_allow_html=True)
cols = [
    "airline","flight","stops","class","departure_time","arrival_time","days_left",
    "price","optimal_price","pct_change_vs_current_price",
    "seat_sold","capacity","residual_capacity",
    "current_expected_revenue","optimal_revenue","season_label","time_of_day"
]
present = [c for c in cols if c in dff.columns]
show_df = dff[present].sort_values("optimal_revenue", ascending=False).head(1000).copy()

if "pct_change_vs_current_price" in show_df.columns:
    def tag_action(v):
        try:
            v = float(v)
            if v > 1:  return "Increase"
            if v < -1: return "Decrease"
            return "Keep"
        except: return "—"
    show_df["Action"] = show_df["pct_change_vs_current_price"].apply(tag_action)

def cell_color(v):
    try:
        v = float(v)
        if v > 0:  return "background-color: rgba(109,15,26,.12); color: #3D0A10;"
        if v < 0:  return "background-color: rgba(225,29,72,.15); color: #7F1D1D;"
    except:
        pass
    return ""

styler = (show_df.style
          .format({
              "price": "OMR {:,.2f}",
              "optimal_price": "OMR {:,.2f}",
              "current_expected_revenue": "OMR {:,.0f}",
              "optimal_revenue": "OMR {:,.0f}",
              "pct_change_vs_current_price": "{:+.1f}%"
          })
          .applymap(cell_color, subset=["pct_change_vs_current_price"]))
st.dataframe(styler, use_container_width=True)

# Export buttons
buf_csv = show_df.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=buf_csv, file_name="recommendations_filtered.csv", mime="text/csv")
buf_xlsx = io.BytesIO()
with pd.ExcelWriter(buf_xlsx, engine="openpyxl") as writer:
    show_df.to_excel(writer, index=False, sheet_name="recommendations")
st.download_button("Download filtered Excel", data=buf_xlsx.getvalue(),
                   file_name="recommendations_filtered.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)

# =========================
# CHARTS — ROW 1
# =========================
c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="section-title">Revenue by class — before vs after</div>', unsafe_allow_html=True)
    class_rev = dff.groupby("class")[["current_expected_revenue","optimal_revenue"]].sum().reset_index()
    fig = go.Figure(data=[
        go.Bar(name="Before", x=class_rev["class"], y=class_rev["current_expected_revenue"], marker_color=BURGUNDY,
               hovertemplate="<b>%{x}</b><br>Before: OMR %{y:,.0f}<extra></extra>"),
        go.Bar(name="After",  x=class_rev["class"], y=class_rev["optimal_revenue"], marker_color="#9CA3AF",
               hovertemplate="<b>%{x}</b><br>After: OMR %{y:,.0f}<extra></extra>")
    ])
    fig.update_layout(barmode="group", yaxis_title="Revenue (OMR)", paper_bgcolor=WHITE, plot_bgcolor=WHITE)
    st.plotly_chart(fig, use_container_width=True, key="rev_by_class")
    if EXPLAIN_MODE:
        st.caption("If the ‘After’ bar is higher than ‘Before’, the recommendation improved revenue for that cabin.")

with c2:
    st.markdown('<div class="section-title">% change vs current price — distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(dff, x="pct_change_vs_current_price", nbins=40,
                       color_discrete_sequence=[BURGUNDY],
                       labels={"pct_change_vs_current_price":"% change"})
    fig.update_traces(hovertemplate="%{y} flights at %{x:.1f}% change<extra></extra>")
    fig.update_layout(xaxis_title="% change", yaxis_title="Count", paper_bgcolor=WHITE, plot_bgcolor=WHITE)
    st.plotly_chart(fig, use_container_width=True, key="pct_change_hist")
    if EXPLAIN_MODE:
        st.caption("Right = price increases; left = discounts. The overall balance should look reasonable.")

# =========================
# CHARTS — ROW 2
# =========================
c3, c4 = st.columns(2)

with c3:
    st.markdown('<div class="section-title">Days left vs optimal price</div>', unsafe_allow_html=True)
    sample = dff.sample(min(5000, len(dff)), random_state=42) if len(dff)>5000 else dff
    fig = px.scatter(sample, x="days_left", y="optimal_price", color="class",
                     opacity=0.6, color_discrete_sequence=[BURGUNDY, "#6B7280", BURGUNDY_SOFT])
    fig.update_traces(hovertemplate="Days left: %{x}<br>Optimal price: OMR %{y:,.0f}<extra></extra>")
    fig.update_layout(xaxis_title="Days left", yaxis_title="Optimal price (OMR)", paper_bgcolor=WHITE, plot_bgcolor=WHITE)
    st.plotly_chart(fig, use_container_width=True, key="days_vs_opt_price")
    if EXPLAIN_MODE:
        st.caption("Closer to departure usually allows higher prices; far out tends to be more price-sensitive.")

with c4:
    st.markdown('<div class="section-title">Uplift % — airline × season</div>', unsafe_allow_html=True)
    if "season_label" in dff.columns:
        heat = (dff.groupby(["airline", "season_label"], as_index=False)
                  .agg(current_rev=("current_expected_revenue", "sum"),
                       optimal_rev=("optimal_revenue", "sum")))
        heat["uplift_pct"] = np.where(
            heat["current_rev"] > 0,
            (heat["optimal_rev"] - heat["current_rev"]) / heat["current_rev"] * 100.0,
            0.0
        )
        if not heat.empty:
            pivot = heat.pivot(index="airline", columns="season_label", values="uplift_pct").fillna(0)
            fig_h = px.imshow(pivot.values, x=pivot.columns, y=pivot.index,
                              color_continuous_scale=["#F9FAFB", "#E5E7EB", "#D1D5DB", BURGUNDY], origin="lower")
            fig_h.update_layout(coloraxis_colorbar_title="% Uplift", paper_bgcolor=WHITE, plot_bgcolor=WHITE)
            st.plotly_chart(fig_h, use_container_width=True, key="uplift_heatmap")
        else:
            st.info("Not enough data after filters to draw the heatmap.")
    else:
        st.info("Season labels not available in this dataset.")
    if EXPLAIN_MODE:
        st.caption("Darker cells = bigger improvement. Use this to spot where strategy works best.")

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
    if EXPLAIN_MODE:
        st.caption("Start with these flights — they unlock the largest immediate gains.")

# =========================
# WHAT-IF HELPERS
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

def expected_units_rule_based(row, test_price):
    el_map = {"Economy":-1.4, "Business":-0.8, "First":-0.5}
    el = el_map.get(row.get("class","Economy"), -0.8)

    base_price = _safe_float(row.get("base_price"), row.get("price"))
    price      = _safe_float(test_price, base_price)
    demand     = _safe_float(row.get("demand_index"), 1.0)
    days_left  = _safe_int(row.get("days_left"), 30)

    res_cap = row.get("residual_capacity")
    if res_cap is None or (isinstance(res_cap, float) and np.isnan(res_cap)):
        cap=_safe_int(row.get("capacity"),0); sold=_safe_int(row.get("seat_sold"),0)
        res_cap = max(0, cap - sold)
    res_cap = int(res_cap)

    if base_price <= 0 or res_cap <= 0:
        return 0.0

    price_effect = (price / base_price) ** el
    time_effect  = 1.0 + 1.2 * (1 - np.tanh(days_left / 40))
    units = demand * price_effect * time_effect * (res_cap / 10.0)
    return float(np.clip(units, 0.0, res_cap))

# Optional ML What-If (if models exist)
def load_ml_models():
    models={}
    for cabin in ["Economy","Business","First"]:
        path = os.path.join("models_seats_ml", f"seats_model_{cabin.lower()}.joblib")
        if os.path.exists(path):
            models[cabin]=joblib.load(path)
    return models

def _safe_str(x, default="Unknown"):
    try:
        s=str(x)
        if s.strip()=="" or s.lower()=="nan": return default
        return s
    except Exception:
        return default

NUMERIC_FEATS_WISHLIST = [
    "price","days_left","demand_index","capacity","weekday","hour",
    "season_month","class_code","season_base_mult","event_mult",
    "weekday_mult","period_mult","stop_mult",
]
CATEG_FEATS_WISHLIST = ["airline","source_city","destination_city","departure_period","stops"]

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

def expected_units_ml(models,row,test_price):
    if not models:
        return expected_units_rule_based(row,test_price)
    cabin=row.get("class","Economy")
    model=models.get(cabin) or next(iter(models.values()))
    X=prepare_feature_row(row,test_price)
    pred=float(model.predict(X)[0])
    cap=_safe_int(row.get("capacity"),0); sold=_safe_int(row.get("seat_sold"),0)
    res=max(0, cap - sold)
    return max(0.0, min(pred, res if res>0 else cap))

def expected_units_unified(engine, models, row, test_price):
    if engine=="ML-based" and models:
        return expected_units_ml(models,row,test_price)
    return expected_units_rule_based(row,test_price)

ml_models = load_ml_models() if engine=="ML-based" else None
if engine=="ML-based" and not ml_models:
    st.warning("ML models not found. What-If will fallback to rule-based (run train_models.py to enable ML).")

# =========================
# FLIGHT DETAILS + WHAT-IF
# =========================
st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Flight details</div>', unsafe_allow_html=True)

def format_key(r):
    dt = r["departure_time"]
    dt_text = pd.to_datetime(dt).strftime("%Y-%m-%d %H:%M") if pd.notna(dt) else "NA"
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
    dep     = row0.get("departure_time", pd.NaT)
    arr     = row0.get("arrival_time", pd.NaT)
    stops   = row0.get("stops", "—")
    season  = row0.get("season_label", "—")
    tod     = row0.get("time_of_day", "—")

    dep = pd.to_datetime(dep) if pd.notna(dep) else pd.NaT
    arr = pd.to_datetime(arr) if pd.notna(arr) else pd.NaT
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
        if not np.isnan(dur_h):
            st.write(f"**Duration:** {dur_h:.2f} hrs")
        if cap is not None or sold is not None:
            util_text = f"{util:.1f}%" if util is not None else "—"
            st.write(f"**Seats:** {sold if sold is not None else '—'} / {cap if cap is not None else '—'}  (Util: {util_text})")
        st.write(f"**Current revenue:** OMR {c_rev:,.0f}")
        st.write(f"**Optimized revenue:** OMR {o_rev:,.0f}")
        st.write(f"**Uplift:** OMR {uplift_abs:,.0f}  ({uplift_pct:.1f}%)")
        st.caption(f"Pricing engine: **{engine}**")

    with b:
        if cap is not None and sold is not None:
            mini = pd.DataFrame({"Metric":["Seats sold","Capacity"], "Value":[sold, cap]})
            fig_c = px.bar(mini, x="Metric", y="Value", color_discrete_sequence=[BURGUNDY, "#9CA3AF"])
            fig_c.update_layout(margin=dict(l=6,r=6,t=10,b=6), height=220,
                                yaxis_title=None, xaxis_title=None,
                                paper_bgcolor=WHITE, plot_bgcolor=WHITE)
            st.plotly_chart(fig_c, use_container_width=True, key=next_key("seats-vs-capacity"))
        else:
            st.write("—")
        st.markdown("<div class='soft-card'><b>How to read:</b> compare sold seats vs capacity.</div>", unsafe_allow_html=True)

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
                                yaxis_title=None, xaxis_title=None,
                                paper_bgcolor=WHITE, plot_bgcolor=WHITE)
            st.plotly_chart(fig_b, use_container_width=True, key=next_key("cabin-breakdown"))
        else:
            st.write("—")

    # ---------- Why this price? (simple explainer)
    if EXPLAIN_MODE:
        _cap = int(row0.get("capacity", 0) or 0)
        _sold = int(row0.get("seat_sold", 0) or 0)
        _lf = (_sold/_cap*100) if _cap else None
        _days = int(row0.get("days_left", 0) or 0)
        _demand = float(row0.get("demand_index", 1.0) or 1.0)
        _stops = str(row0.get("stops","")).strip().lower()
        _class = str(row0.get("class",""))
        bullets = []
        if _demand >= 1.2: bullets.append("Demand is **high** → we can lift price slightly (with caution).")
        elif _demand <= 0.8: bullets.append("Demand is **low** → small discounts help stimulate bookings.")
        else: bullets.append("Demand is **normal** → small moves around the base price.")
        if _days <= 7: bullets.append("Departure is **soon** → last-minute premium is typical.")
        elif _days >= 60: bullets.append("Departure is **far** → more price sensitivity; lighter prices help.")
        if _lf is not None:
            if _lf >= 85: bullets.append("Load factor is **high** → limited but possible upside on price.")
            elif _lf <= 40: bullets.append("Load factor is **low** → be attractive to boost pickup.")
            else: bullets.append("Load factor is **medium** → moderate adjustments.")
        if "direct" in _stops or _stops == "0": bullets.append("Flight is **direct** → usually priced higher.")
        elif "1" in _stops: bullets.append("**1 stop** → a bit lower than direct.")
        else: bullets.append("**Multiple stops** → needs more competitive price.")
        if _class == "Economy": bullets.append("**Economy**: highest price sensitivity.")
        elif _class == "Business": bullets.append("**Business**: moderate sensitivity.")
        elif _class == "First": bullets.append("**First**: lowest sensitivity.")
        st.markdown("#### 🧠 Why this price?")
        st.markdown("\n".join([f"- {b}" for b in bullets]))

    # ---------- Quick What-If (engine-aware)
    st.markdown('<div class="red-sep"></div>', unsafe_allow_html=True)
    st.subheader("Quick What-If: try a manual price")
    if cap is not None and sold is not None:
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
            u_base = expected_units_unified(engine, ml_models, r, cur_price_for_sim)
            u_new  = expected_units_unified(engine, ml_models, r, new_price)
            base_rev_sim += cur_price_for_sim * u_base
            new_rev_sim  += new_price         * u_new

        delta_abs = new_rev_sim - base_rev_sim
        delta_pct = (delta_abs / base_rev_sim * 100.0) if base_rev_sim > 0 else 0.0

        w1, w2, w3 = st.columns(3)
        w1.metric("Base revenue (simulated)", f"OMR {base_rev_sim:,.0f}")
        w2.metric("What-If revenue",         f"OMR {new_rev_sim:,.0f}")
        w3.metric("Δ vs base",               f"OMR {delta_abs:,.0f}", f"{delta_pct:+.1f}%")
    else:
        st.info("Seats and capacity not available to simulate a quick What-If for this flight.")

    st.markdown('<div class="red-sep"></div>', unsafe_allow_html=True)
    show_cols = [c for c in [
        "airline","flight","stops","class","departure_time","arrival_time","days_left",
        "price","optimal_price","pct_change_vs_current_price","seat_sold","capacity",
        "residual_capacity","current_expected_revenue","optimal_revenue","season_label","time_of_day"
    ] if c in sel.columns]
    st.dataframe(sel[show_cols].sort_values("optimal_revenue", ascending=False), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)
st.caption("© Dynamic Pricing Simulator — Muscat → London | Streamlit + Plotly | Cloud-friendly loader + Explain Mode")
