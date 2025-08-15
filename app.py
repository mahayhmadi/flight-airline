# app.py — MCT → LHR Pricing Dashboard (Darker Burgundy Theme)
# English-only • Header/Sidebar #4B0E1E • White content area • Unique Plotly keys
# Seasonal View removed

import os
import io
from itertools import count
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============== CONFIG ==============
st.set_page_config(page_title="MCT → LHR Pricing Optimizer", layout="wide")

# ----- Brand palette (Darker Burgundy) -----
BURGUNDY_DARKEST = "#4B0E1E"  # header & sidebar background (very dark burgundy)
BURGUNDY         = "#6D0F1A"  # primary accent (bars, highlights)
BURGUNDY_SOFT    = "#8C1F28"  # lighter accent
GRAY_900         = "#111827"
GRAY_700         = "#374151"
GRAY_500         = "#6B7280"
GRAY_300         = "#D1D5DB"
GRAY_200         = "#E5E7EB"
GRAY_100         = "#F3F4F6"
WHITE            = "#FFFFFF"
RED_SEP          = "#E11D48"  # separator inside flight details

# Plotly default palette (burgundy + neutrals)
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [BURGUNDY, "#4B5563", BURGUNDY_SOFT, "#9CA3AF"]

# Unique keys for Plotly charts (avoid duplicate IDs)
_chart_counter = count()
def next_key(name: str) -> str:
    return f"{name}-{next(_chart_counter)}"

# ============== GLOBAL STYLE ==============
st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

      .stApp {{
        font-family: 'Inter', ui-sans-serif, system-ui;
        background: {WHITE};
        color: {GRAY_900};
      }}

      /* Header / hero */
      .hero {{
        display:grid; grid-template-columns: 108px 1fr; gap:18px; align-items:center;
        background: {BURGUNDY_DARKEST}; color: {WHITE};
        padding: 20px 24px; border-radius: 16px;
        border: 1px solid rgba(255,255,255,.12);
        box-shadow: 0 2px 10px rgba(0,0,0,.06) inset;
      }}
      .hero h1 {{ margin:0; font-size:28px; font-weight:800; color:{WHITE}; letter-spacing:.2px; }}
      .hero p  {{ margin:0; opacity:.92; font-size:14px; color:{WHITE}; }}

      .section-title {{ font-weight:800; color:{BURGUNDY_DARKEST}; margin: 0 0 8px 0; font-size:18px; }}

      .soft-card {{
        background: {GRAY_100};
        border: 1px solid {GRAY_200};
        border-radius: 12px;
        padding: 10px 12px;
        color: {GRAY_900};
      }}

      /* Tables */
      div[data-testid="stDataFrame"] table {{
        border:1px solid {GRAY_200};
        border-radius:8px;
      }}

      /* Flight details card */
      .flight-card {{
        border:1px solid {GRAY_200};
        border-radius:14px;
        padding:12px;
        background:{WHITE};
      }}
      .flight-head {{ font-weight:800; color:{BURGUNDY_DARKEST}; font-size:16px; }}
      .red-sep {{ height:3px; background:{RED_SEP}; border-radius:999px; margin:10px 0; }}

      /* Sidebar: darker burgundy with white text */
      section[data-testid="stSidebar"] {{
        background: {BURGUNDY_DARKEST};
        color: {WHITE};
        border-right: 1px solid rgba(255,255,255,.12);
      }}
      section[data-testid="stSidebar"] * {{ color: {WHITE} !important; }}
      [data-baseweb="radio"] div, [data-baseweb="checkbox"] div {{ color: {WHITE} !important; }}

      .side-label {{
        display:flex; align-items:center; gap:8px; font-weight:700; color:{WHITE};
        margin: 10px 0 2px 0;
      }}
      .side-icon svg {{ display:block; }}

      /* Inputs on white background remain readable */
      .hr-soft {{ border:none; height:1px; background:{GRAY_200}; margin: 14px 0; }}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar icons (white)
ICON_CLOCK = """
<span class="side-icon">
<svg width="16" height="16" viewBox="0 0 24 24" fill="#FFFFFF" xmlns="http://www.w3.org/2000/svg">
  <circle cx="12" cy="12" r="9" stroke="#FFFFFF" stroke-width="2" fill="none"/>
  <path d="M12 7v6l4 2" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
</span>
"""
ICON_AIRLINE = """
<span class="side-icon">
<svg width="16" height="16" viewBox="0 0 64 64" fill="#FFFFFF" xmlns="http://www.w3.org/2000/svg">
  <path d="M60 30c0 2-1.6 3.6-3.6 3.6H39.2L26 47l-5.6-5.6 8.4-7.8H14.2l-5.8 4.6-4-4 6.2-8.2h18.2l12.2-6.6h15.4C58.4 19.4 60 21 60 23v7z"/>
</svg>
</span>
"""
ICON_STOPS = """
<span class="side-icon">
<svg width="16" height="16" viewBox="0 0 24 24" fill="#FFFFFF" xmlns="http://www.w3.org/2000/svg">
  <circle cx="4" cy="12" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="20" cy="12" r="2"/>
</svg>
</span>
"""

# ============== DATA LOAD ==============
DEFAULT_CSV = r"C:\Users\bbuser\Desktop\flight_dashbord\final_optimized_flights.csv"
st.sidebar.header("Data")
csv_path = st.sidebar.text_input("Optimized CSV path", value=DEFAULT_CSV)

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["departure_time","arrival_time"], infer_datetime_format=True)
    return df

if not os.path.exists(csv_path):
    st.error(f"File not found: {csv_path}")
    st.stop()

df = load_data(csv_path)

# ============== DERIVED / CHECKS ==============
required = {"optimal_revenue","current_expected_revenue","optimal_price","price"}
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

if "season_label" not in df.columns and "season_month" in df.columns:
    season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"}
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

df["dep_date"] = df["departure_time"].dt.date
df["dep_hour"] = df["departure_time"].dt.hour
def time_bucket(h):
    if h is None or pd.isna(h): return "Unknown"
    h = int(h)
    if 6 <= h < 12:  return "Morning"
    if 12 <= h < 17: return "Afternoon"
    if 17 <= h < 21: return "Evening"
    return "Night"
df["time_of_day"] = df["dep_hour"].apply(time_bucket)

# ============== HERO (bigger airplane) ==============
AIRPLANE_SVG = f"""
<svg width="108" height="108" viewBox="0 0 108 108" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#FFFFFF" stop-opacity=".98"/>
      <stop offset="100%" stop-color="#F9FAFB" stop-opacity=".85"/>
    </linearGradient>
  </defs>
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
        <p>Dynamic pricing • Revenue uplift • Explainable recommendations</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("")

# ============== SIDEBAR FILTERS ==============
st.sidebar.header("Quick Filters")
st.sidebar.markdown(f'<div class="side-label">{ICON_CLOCK} Flight time</div>', unsafe_allow_html=True)
sel_tod = st.sidebar.radio(label="", options=["All","Morning","Afternoon","Evening","Night"], index=0, key="tod_radio")

st.sidebar.markdown(f'<div class="side-label">{ICON_AIRLINE} Airline</div>', unsafe_allow_html=True)
airlines_all = sorted(df["airline"].dropna().unique().tolist())
sel_airlines = st.sidebar.multiselect(label="", options=airlines_all, default=[], key="airline_multi")

st.sidebar.markdown(f'<div class="side-label">{ICON_STOPS} Flight type</div>', unsafe_allow_html=True)
flight_type_sidebar = st.sidebar.radio(label="", options=["Any","Direct","1 Stop","2+ Stops"], index=0, key="type_radio")

# ============== MAIN FILTER BAR ==============
min_dt, max_dt = df["dep_date"].min(), df["dep_date"].max()
cfa, cfb, cfc, cfd = st.columns([1.4, 1.2, 1.2, 1.2])
with cfa:
    date_from, date_to = st.date_input("Departure date range", value=[min_dt, max_dt], min_value=min_dt, max_value=max_dt, key="date_range")
with cfb:
    f_class = st.selectbox("Class", ["All","Economy","Business","First"], index=0, key="class_select")
with cfc:
    seasons = ["All"] + (sorted(df["season_label"].dropna().unique()) if "season_label" in df.columns else [])
    f_season = st.selectbox("Season", seasons, index=0 if seasons else 0, key="season_select")
with cfd:
    days_slider_max = int(max(1, df["days_left"].max()))
    f_daysmax = st.slider("Days Left (max)", 1, days_slider_max, days_slider_max, key="days_slider")

# ============== GLOBAL MASK ==============
mask = pd.Series(True, index=df.index)
mask &= (df["dep_date"] >= date_from) & (df["dep_date"] <= date_to)
if sel_tod != "All": mask &= (df["time_of_day"] == sel_tod)
if sel_airlines: mask &= df["airline"].isin(sel_airlines)
if flight_type_sidebar == "Direct":
    mask &= (df["stops_norm"] == 0)
elif flight_type_sidebar == "1 Stop":
    mask &= (df["stops_norm"] == 1)
elif flight_type_sidebar == "2+ Stops":
    mask &= (df["stops_norm"] >= 2)
mask &= (df["days_left"] <= f_daysmax)
if f_class != "All":  mask &= (df["class"] == f_class)
if f_season != "All" and "season_label" in df.columns: mask &= (df["season_label"] == f_season)

dff = df.loc[mask].copy()
if dff.empty:
    st.warning("No rows after filters. Adjust filters to see data.")
    st.stop()

# ============== KPIs ==============
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

rps = None
if "current_expected_revenue" in dff.columns and "capacity" in dff.columns:
    rps = dff["current_expected_revenue"].sum() / max(dff["capacity"].sum(), 1)

avg_delta = dff["pct_change_vs_current_price"].mean() if "pct_change_vs_current_price" in dff.columns else None

e1, e2, e3 = st.columns(3)
e1.metric("Load Factor (filtered)", f"{lf:.1f}%" if lf is not None else "—")
e2.metric("Revenue / Seat", f"OMR {rps:,.2f}" if rps is not None else "—")
e3.metric("Avg. Price Change", f"{avg_delta:+.1f}%" if avg_delta is not None else "—")

st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)

# ============== RECOMMENDATIONS TABLE + EXPORT ==============
st.markdown('<div class="section-title">Price Recommendations (filtered)</div>', unsafe_allow_html=True)
cols = [
    "airline","flight","stops","class","departure_time","arrival_time","days_left",
    "price","optimal_price","pct_change_vs_current_price",
    "seat_sold","capacity","residual_capacity",
    "current_expected_revenue","optimal_revenue","season_label","time_of_day"
]
present = [c for c in cols if c in dff.columns]
show_df = dff[present].sort_values("optimal_revenue", ascending=False).head(1000).copy()

def color_price_change(v):
    try:
        v = float(v)
        if v > 0:  return "background-color: rgba(109,15,26,.12); color: #3D0A10;"   # burgundy tint for increase
        if v < 0:  return "background-color: rgba(225,29,72,.15); color: #7F1D1D;"   # red tint for discount
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
          .applymap(color_price_change, subset=["pct_change_vs_current_price"])
)
st.dataframe(styler, use_container_width=True)

# Export
buf_csv = show_df.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=buf_csv, file_name="recommendations_filtered.csv", mime="text/csv")
buf_xlsx = io.BytesIO()
with pd.ExcelWriter(buf_xlsx, engine="openpyxl") as writer:
    show_df.to_excel(writer, index=False, sheet_name="recommendations")
st.download_button("Download filtered Excel", data=buf_xlsx.getvalue(),
                   file_name="recommendations_filtered.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)

# ============== CHARTS ROW 1 ==============
c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="section-title">Revenue by Class — Before vs After</div>', unsafe_allow_html=True)
    class_rev = dff.groupby("class")[["current_expected_revenue","optimal_revenue"]].sum().reset_index()
    fig = go.Figure(data=[
        go.Bar(name="Before", x=class_rev["class"], y=class_rev["current_expected_revenue"], marker_color=BURGUNDY),
        go.Bar(name="After",  x=class_rev["class"], y=class_rev["optimal_revenue"], marker_color="#9CA3AF")
    ])
    fig.update_layout(barmode="group", yaxis_title="Revenue (OMR)", paper_bgcolor=WHITE, plot_bgcolor=WHITE)
    st.plotly_chart(fig, use_container_width=True, key="rev_by_class")
    st.markdown(
        """
        <div class="soft-card">
          <b>How to read:</b> compares revenue before vs after optimization by cabin.
          <br><b>Use it to:</b> focus on cabins where improvement is strongest.
        </div>
        """, unsafe_allow_html=True
    )

with c2:
    st.markdown('<div class="section-title">% Change vs Current Price — Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(dff, x="pct_change_vs_current_price", nbins=40, color_discrete_sequence=[BURGUNDY])
    fig.update_layout(xaxis_title="% change", yaxis_title="Count", paper_bgcolor=WHITE, plot_bgcolor=WHITE)
    st.plotly_chart(fig, use_container_width=True, key="pct_change_hist")
    st.markdown(
        """
        <div class="soft-card">
          <b>How to read:</b> distribution of recommended price shifts (left = discounts, right = increases).
          <br><b>Use it to:</b> sanity-check the balance of recommendations.
        </div>
        """, unsafe_allow_html=True
    )

# ============== CHARTS ROW 2 ==============
c3, c4 = st.columns(2)
with c3:
    st.markdown('<div class="section-title">Days Left vs Optimal Price</div>', unsafe_allow_html=True)
    sample = dff.sample(min(5000, len(dff)), random_state=42) if len(dff)>5000 else dff
    fig = px.scatter(sample, x="days_left", y="optimal_price", color="class",
                     opacity=0.6, color_discrete_sequence=[BURGUNDY, "#6B7280", BURGUNDY_SOFT])
    fig.update_layout(xaxis_title="Days Left", yaxis_title="Optimal Price (OMR)",
                      paper_bgcolor=WHITE, plot_bgcolor=WHITE)
    st.plotly_chart(fig, use_container_width=True, key="days_vs_opt_price")
    st.markdown(
        """
        <div class="soft-card">
          <b>How to read:</b> shows how prices evolve as departure nears.
          <br><b>Use it to:</b> tune last-minute premiums or early discounts.
        </div>
        """, unsafe_allow_html=True
    )

with c4:
    st.markdown('<div class="section-title">Uplift % — Airline × Season</div>', unsafe_allow_html=True)
    if "season_label" in dff.columns:
        heat = (
            dff.groupby(["airline", "season_label"], as_index=False)
               .agg(current_rev=("current_expected_revenue", "sum"),
                    optimal_rev=("optimal_revenue", "sum"))
        )
        heat["uplift_pct"] = np.where(
            heat["current_rev"] > 0,
            (heat["optimal_rev"] - heat["current_rev"]) / heat["current_rev"] * 100.0,
            0.0
        )
        if not heat.empty:
            pivot = heat.pivot(index="airline", columns="season_label", values="uplift_pct").fillna(0)
            fig_h = px.imshow(
                pivot.values, x=pivot.columns, y=pivot.index,
                color_continuous_scale=["#F9FAFB", "#E5E7EB", "#D1D5DB", BURGUNDY],
                origin="lower"
            )
            fig_h.update_layout(coloraxis_colorbar_title="% Uplift", paper_bgcolor=WHITE, plot_bgcolor=WHITE)
            st.plotly_chart(fig_h, use_container_width=True, key="uplift_heatmap")
        else:
            st.info("Not enough data after filters to draw the heatmap.")
    else:
        st.info("Season labels not available in this dataset.")
    st.markdown(
        """
        <div class="soft-card">
          <b>How to read:</b> darker burgundy = stronger uplift.
          <br><b>Use it to:</b> plan airline/season strategies where upside is highest.
        </div>
        """, unsafe_allow_html=True
    )

# ============== TOP ALERTS ==============
st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Top Alerts — biggest revenue opportunities</div>', unsafe_allow_html=True)
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
    st.markdown(
        """
        <div class="soft-card">
          <b>Tip:</b> start with these flights — they unlock the largest immediate gains.
        </div>
        """, unsafe_allow_html=True
    )

# ============== FLIGHT DETAILS (only after selection) ==============
st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Flight Details</div>', unsafe_allow_html=True)

def format_key(r):
    dt = r["departure_time"]
    dt_text = dt.strftime("%Y-%m-%d %H:%M") if pd.notna(dt) else "NA"
    return f"{r.get('airline','?')} • {r.get('flight','?')} • {dt_text}"

sel_df = dff.copy()
sel_df["flight_key"] = sel_df.apply(format_key, axis=1)
choices = ["— Select a flight —"] + sel_df["flight_key"].unique().tolist()
selected = st.selectbox("Select a flight to view full details", options=choices, index=0, key="flight_select")

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

    dur_h = ((arr - dep).total_seconds()/3600.0) if pd.notna(dep) and pd.notna(arr) else np.nan
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
        st.write(f"**Current Revenue:** OMR {c_rev:,.0f}")
        st.write(f"**Optimized Revenue:** OMR {o_rev:,.0f}")
        st.write(f"**Uplift:** OMR {uplift_abs:,.0f}  ({uplift_pct:.1f}%)")

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
        st.markdown(
            """
            <div class="soft-card">
              <b>How to read:</b> sold seats vs capacity for this flight.
              <br><b>Use it to:</b> spot underfilled flights and decide if discounts are warranted.
            </div>
            """, unsafe_allow_html=True
        )

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
        st.markdown(
            """
            <div class="soft-card">
              <b>How to read:</b> cabin breakdown of seats and capacity.
              <br><b>Use it to:</b> target actions where load is weak.
            </div>
            """, unsafe_allow_html=True
        )

    # Quick What-If
    st.markdown('<div class="red-sep"></div>', unsafe_allow_html=True)
    st.subheader("Quick What-If: manual price tweak")
    if cap is not None and sold is not None:
        base_price = float(sel["price"].mean())
        default_price = float(row0["optimal_price"]) if "optimal_price" in sel.columns else base_price
        new_price = st.slider("Try a new ticket price (OMR)",
                              min_value=max(1.0, base_price*0.5),
                              max_value=base_price*1.5,
                              value=default_price,
                              step=1.0,
                              key="whatif_price_slider")
        whatif_rev = new_price * sold
        base_rev   = c_rev
        delta_abs  = whatif_rev - base_rev
        delta_pct  = (delta_abs / base_rev * 100.0) if base_rev > 0 else 0.0

        w1, w2, w3 = st.columns(3)
        w1.metric("Base Revenue (current)", f"OMR {base_rev:,.0f}")
        w2.metric("What-If Revenue", f"OMR {whatif_rev:,.0f}")
        w3.metric("Δ vs Base", f"OMR {delta_abs:,.0f}", f"{delta_pct:+.1f}%")
    else:
        st.info("Seats and capacity not available to simulate a quick What-If for this flight.")

    # Full rows
    st.markdown('<div class="red-sep"></div>', unsafe_allow_html=True)
    show_cols = [c for c in [
        "airline","flight","stops","class","departure_time","arrival_time","days_left",
        "price","optimal_price","pct_change_vs_current_price","seat_sold","capacity",
        "residual_capacity","current_expected_revenue","optimal_revenue","season_label","time_of_day"
    ] if c in sel.columns]
    st.dataframe(sel[show_cols].sort_values("optimal_revenue", ascending=False), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============== FOOTER ==============
st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)
st.caption("© Dynamic Pricing Simulator — Muscat → London | Streamlit + Plotly | Dark Burgundy theme")
