# optimize_prices_ml.py — ML/RB optimizer without absolute paths
# Usage (ML):
#   python optimize_prices_ml.py --input final_optimized_flights.csv --models models_seats_ml --output final_optimized_flights_ML.csv
# If models not found → falls back to rule-based logic automatically.

import os
import argparse
import numpy as np
import pandas as pd
import joblib

# --------- CLI ---------
def parse_args():
    p = argparse.ArgumentParser(description="Optimize prices using ML seats models (auto-fallback to rule-based).")
    p.add_argument("--input", "-i", default="final_optimized_flights.csv", help="Input CSV (relative path).")
    p.add_argument("--models", "-m", default="models_seats_ml", help="Directory containing joblib models.")
    p.add_argument("--output", "-o", default="final_optimized_flights_ML.csv", help="Output CSV filename.")
    p.add_argument("--steps", type=int, default=50, help="Grid search steps per row.")
    return p.parse_args()

# --------- Utilities ---------
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

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["departure_time", "arrival_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "season_month" not in df.columns:
        if "departure_time" in df.columns:
            df["season_month"] = df["departure_time"].dt.month
        else:
            df["season_month"] = np.nan

    if "pct_change_vs_current_price" not in df.columns and {"price","optimal_price"}.issubset(df.columns):
        df["pct_change_vs_current_price"] = 100.0 * (df["optimal_price"] - df["price"]) / df["price"]

    return df

# ---- Rule-based expected units (fallback) ----
def expected_units_rule(row, test_price):
    el_map = {"Economy":-1.4, "Business":-0.8, "First":-0.5}
    el = el_map.get(row.get("class","Economy"), -0.8)

    base_price = _safe_float(row.get("base_price"), row.get("price", 1.0))
    price      = _safe_float(test_price, base_price)
    demand     = _safe_float(row.get("demand_index"), 1.0)
    days_left  = _safe_int(row.get("days_left"), 30)

    cap = _safe_int(row.get("capacity"), 0)
    sold = _safe_int(row.get("seat_sold"), 0)
    res_cap = max(0, cap - sold)

    if base_price <= 0 or res_cap <= 0:
        return 0.0

    price_effect = (price / base_price) ** el
    time_effect  = 1.0 + 1.2 * (1 - np.tanh(days_left / 40))
    units = demand * price_effect * time_effect * (res_cap / 10.0)
    return float(np.clip(units, 0.0, res_cap))

# ---- ML helpers ----
NUMERIC_FEATS = [
    "price","days_left","demand_index","capacity","weekday","hour",
    "season_month","class_code","season_base_mult","event_mult",
    "weekday_mult","period_mult","stop_mult",
]
CATEG_FEATS = ["airline","source_city","destination_city","departure_period","stops"]

def _safe_str(x, default="Unknown"):
    try:
        s = str(x)
        if s.strip()=="" or s.lower()=="nan": return default
        return s
    except Exception:
        return default

def prepare_feature_row(row, price_value):
    r = row.copy()
    if ("season_month" not in r.index) or pd.isna(r.get("season_month")):
        try: r["season_month"] = pd.to_datetime(r.get("departure_time"), errors="coerce").month
        except Exception: r["season_month"] = np.nan

    data = {}
    for c in NUMERIC_FEATS:
        if c == "price": continue
        if c in r.index:
            data[c] = [r.get(c, np.nan)]
    for c in CATEG_FEATS:
        data[c] = [_safe_str(r.get(c, "Unknown"))]
    data["price"] = [float(price_value)]
    return pd.DataFrame(data)

def load_models(models_dir):
    models = {}
    for cabin in ["Economy","Business","First"]:
        path = os.path.join(models_dir, f"seats_model_{cabin.lower()}.joblib")
        if os.path.exists(path):
            models[cabin] = joblib.load(path)
    return models

def expected_units_ml(models, row, test_price):
    if not models:
        return expected_units_rule(row, test_price)
    cabin = row.get("class", "Economy")
    model = models.get(cabin) or next(iter(models.values()))
    X = prepare_feature_row(row, test_price)
    pred = float(model.predict(X)[0])

    cap = _safe_int(row.get("capacity"), 0)
    sold = _safe_int(row.get("seat_sold"), 0)
    res_cap = max(0, cap - sold)
    if res_cap <= 0:
        res_cap = cap
    return max(0.0, min(pred, res_cap))

def expected_units_unified(models, row, test_price):
    # use ML if available, else rule-based
    if models:
        return expected_units_ml(models, row, test_price)
    return expected_units_rule(row, test_price)

# ---- Price band rules (around base_price) ----
CLASS_BANDS = {
    "Economy":  {"elasticity": -1.4, "min_mult": 0.90, "max_mult": 1.10},
    "Business": {"elasticity": -0.8,  "min_mult": 0.75, "max_mult": 1.25},
    "First":    {"elasticity": -0.5,  "min_mult": 0.80, "max_mult": 1.20},
}

def price_bounds(row):
    clz = row.get("class", "Economy")
    p = CLASS_BANDS.get(clz, CLASS_BANDS["Business"])

    base_price = _safe_float(row.get("base_price"), row.get("price", 1.0))
    lo_mult, hi_mult = p["min_mult"], p["max_mult"]

    # Load-factor-based tightening
    cap_full = _safe_int(row.get("capacity"), 1)
    seat_sold = _safe_int(row.get("seat_sold"), 0)
    load_factor = seat_sold / max(1, cap_full)
    days_left = _safe_int(row.get("days_left"), 30)

    if clz == "Economy":
        if load_factor < 0.20:
            lo_mult = min(lo_mult, 0.88)
            hi_mult = min(hi_mult, 1.05)
        elif load_factor < 0.40:
            hi_mult = min(hi_mult, 1.05)
        elif load_factor > 0.85:
            hi_mult = min(1.12, hi_mult)

    if clz in ("Business","First"):
        if load_factor > 0.85 or days_left <= 10:
            hi_mult = min(hi_mult * 1.05, 1.30)
        if load_factor < 0.30 and days_left > 45:
            hi_mult = min(hi_mult, 1.10)

    lo = max(0.01, base_price * lo_mult)
    hi = max(lo,    base_price * hi_mult)
    return lo, hi

def optimize_row(row, models, steps=50):
    cur_price  = _safe_float(row.get("price"), row.get("base_price", 1.0))
    base_price = _safe_float(row.get("base_price"), cur_price)

    lo, hi = price_bounds(row)
    grid = np.linspace(lo, hi, steps)

    # current baseline
    cur_units = expected_units_unified(models, row, cur_price)
    cur_rev = cur_price * cur_units

    best_price, best_rev, best_units = cur_price, cur_rev, cur_units
    for test_price in grid:
        u = expected_units_unified(models, row, test_price)
        rev = test_price * u
        if rev > best_rev:
            best_price, best_rev, best_units = float(test_price), float(rev), float(u)

    kept = best_rev <= cur_rev + 1e-9
    if kept:
        best_price, best_rev, best_units = cur_price, cur_rev, cur_units

    return pd.Series({
        "optimal_price": best_price,
        "optimal_revenue": best_rev,
        "expected_units": best_units,
        "kept_current_price": kept
    })

# --------- Main ---------
def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input, low_memory=False)
    df = basic_clean(df)

    models = load_models(args.models)
    using_ml = bool(models)
    if using_ml:
        print(f"[INFO] Loaded ML models from '{args.models}'.")
    else:
        print("[WARN] No ML models found. Falling back to rule-based expectations.")

    # Optimize
    opt_cols = df.apply(lambda r: optimize_row(r, models, steps=args.steps), axis=1)
    df_opt = pd.concat([df.reset_index(drop=True), opt_cols], axis=1)

    # Current expected revenue (with the same engine used for opt)
    def current_rev_row(row):
        units = expected_units_unified(models, row, _safe_float(row.get("price"), row.get("base_price", 1.0)))
        return _safe_float(row.get("price"), row.get("base_price", 1.0)) * units

    df_opt["current_expected_revenue"] = df_opt.apply(current_rev_row, axis=1)
    if "pct_change_vs_current_price" not in df_opt.columns:
        df_opt["pct_change_vs_current_price"] = 100.0 * (df_opt["optimal_price"] - df_opt["price"]) / df_opt["price"]

    # Save
    df_opt.to_csv(args.output, index=False)
    tot_cur = df_opt["current_expected_revenue"].sum()
    tot_opt = df_opt["optimal_revenue"].sum()
    uplift  = tot_opt - tot_cur
    pct     = (uplift / tot_cur * 100.0) if tot_cur > 0 else 0.0

    print(f"[OK] Wrote: {args.output}")
    print(f"Current Expected Revenue: {tot_cur:,.0f}")
    print(f"Optimized Revenue:       {tot_opt:,.0f}")
    print(f"Uplift:                  {uplift:,.0f} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
