# optimize_prices_ml.py  (robust version)
import os, joblib, numpy as np, pandas as pd

INPUT_CSV  = r"C:\Users\bbuser\Desktop\flight_dashbord\final_optimized_flights.csv"
OUTPUT_CSV = r"C:\Users\bbuser\Desktop\flight_dashbord\final_optimized_flights_ML.csv"
MODELS_DIR = "models_seats_ml"

CLASS_BANDS = {
    "Economy":  {"min_mult": 0.90, "max_mult": 1.10},
    "Business": {"min_mult": 0.75, "max_mult": 1.25},
    "First":    {"min_mult": 0.80, "max_mult": 1.20},
}
DEFAULT_BAND = {"min_mult": 0.85, "max_mult": 1.15}

NUMERIC_FEATS_WISHLIST = [
    "price","days_left","demand_index","capacity","weekday","hour",
    "season_month","class_code","season_base_mult","event_mult",
    "weekday_mult","period_mult","stop_mult",
]
CATEG_FEATS_WISHLIST = ["airline","source_city","destination_city","departure_period","stops"]
ALL_CLASSES = ["Economy","Business","First"]

def load_models():
    models={}
    for cabin in ALL_CLASSES:
        path=os.path.join(MODELS_DIR, f"seats_model_{cabin.lower()}.joblib")
        if os.path.exists(path):
            models[cabin]=joblib.load(path)
    if not models:
        raise FileNotFoundError("No models found in models_seats_ml. Run train_models.py first.")
    return models

def ensure_season_month_series(df):
    if "season_month" not in df.columns and "departure_time" in df.columns:
        try:
            df["season_month"] = pd.to_datetime(df["departure_time"], errors="coerce").dt.month
        except Exception:
            df["season_month"] = np.nan
    return df

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def capacity_guard(row):
    cap  = float(row.get("capacity", 0) or 0)
    sold = float(row.get("seat_sold", 0) or 0)
    residual = max(0.0, cap - sold)
    return cap, sold, residual

def price_band(row):
    base = float(row.get("base_price", np.nan))
    if not np.isfinite(base) or base <= 0:
        base = float(row.get("price", 1.0) or 1.0)
    band = CLASS_BANDS.get(row.get("class",""), DEFAULT_BAND).copy()
    cap, sold, res = capacity_guard(row)
    lf = (sold/cap) if cap>0 else 0.0
    days = float(row.get("days_left",30) or 30)

    if row.get("class","")=="Economy":
        if lf<0.20: band["min_mult"]=min(band["min_mult"],0.88); band["max_mult"]=min(band["max_mult"],1.05)
        elif lf<0.40: band["max_mult"]=min(band["max_mult"],1.05)
        elif lf>0.85: band["max_mult"]=min(1.12,band["max_mult"])
    else:
        if lf>0.85 or days<=10: band["max_mult"]=min(band["max_mult"]*1.05,1.30)
        if lf<0.30 and days>45: band["max_mult"]=min(band["max_mult"],1.10)

    lo=max(1.0, base*band["min_mult"]); hi=max(lo, base*band["max_mult"])
    return lo,hi

def _safe_str(x, default="Unknown"):
    try:
        s=str(x)
        if s.strip()=="" or s.lower()=="nan":
            return default
        return s
    except Exception:
        return default

def prepare_feature_row(row, price_value):
    r=row.copy()
    if ("season_month" not in r.index) or pd.isna(r.get("season_month")):
        try:
            r["season_month"] = pd.to_datetime(r.get("departure_time"), errors="coerce").month
        except Exception:
            r["season_month"] = np.nan

    data={}
    for c in NUMERIC_FEATS_WISHLIST:
        if c=="price": continue
        if c in r.index:
            data[c]=[r.get(c, np.nan)]
    for c in CATEG_FEATS_WISHLIST:
        data[c]=[_safe_str(r.get(c,"Unknown"))]
    data["price"]=[float(price_value)]
    return pd.DataFrame(data)

def predict_units(models,row,test_price):
    cabin=row.get("class","Economy")
    model=models.get(cabin) or next(iter(models.values()))
    X=prepare_feature_row(row,test_price)
    pred=float(model.predict(X)[0])
    cap,sold,res=capacity_guard(row)
    return max(0.0, min(pred, res if res>0 else cap))

def main():
    models=load_models()
    df=pd.read_csv(INPUT_CSV, parse_dates=["departure_time","arrival_time"], low_memory=False)

    df = ensure_season_month_series(df)
    df = ensure_numeric(df, [
        "price","base_price","days_left","capacity","seat_sold","demand_index",
        "weekday","hour","season_month","class_code",
        "season_base_mult","event_mult","weekday_mult","period_mult","stop_mult",
    ])

    if "demand_index" not in df.columns:
        df["demand_index"] = 1.0
    df["demand_index"] = df["demand_index"].fillna(1.0).clip(0.3,2.0)
    if "seat_sold" not in df.columns: df["seat_sold"]=0
    if "capacity"  not in df.columns: raise ValueError("Missing 'capacity' column in input CSV.")

    cur_rev_list=[]; opt_price_list=[]; opt_rev_list=[]; opt_units_list=[]; kept_list=[]
    for _,row in df.iterrows():
        cur_price = float(row.get("price", np.nan))
        if not np.isfinite(cur_price) or cur_price<=0:
            base=float(row.get("base_price",1.0) or 1.0); cur_price=max(1.0, base)

        cur_units = predict_units(models,row,cur_price)
        cur_rev   = cur_price * cur_units

        lo,hi = price_band(row); grid = np.linspace(lo,hi,50)
        best_price,best_rev,best_units = cur_price,cur_rev,cur_units
        for p in grid:
            u=predict_units(models,row,p); r=p*u
            if r>best_rev: best_price,best_rev,best_units=float(p),float(r),float(u)

        kept = best_rev <= cur_rev + 1e-9
        if kept: best_price,best_rev,best_units = cur_price,cur_rev,cur_units

        cur_rev_list.append(cur_rev); opt_price_list.append(best_price)
        opt_rev_list.append(best_rev); opt_units_list.append(best_units); kept_list.append(kept)

    df["current_expected_revenue"]=np.array(cur_rev_list,float)
    df["optimal_price"]=np.array(opt_price_list,float)
    df["optimal_revenue"]=np.array(opt_rev_list,float)
    df["expected_units"]=np.array(opt_units_list,float)
    df["kept_current_price"]=np.array(kept_list,bool)
    df["pct_change_vs_current_price"]=100.0*(df["optimal_price"]-df["price"])/df["price"]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Wrote ML-optimized CSV → {OUTPUT_CSV}")

if __name__=="__main__":
    main()
