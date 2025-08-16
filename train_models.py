# train_models.py (robust)
import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

INPUT_CSV = r"C:\Users\bbuser\Desktop\flight_dashbord\final_optimized_flights.csv"
MODELS_DIR = "models_seats_ml"
os.makedirs(MODELS_DIR, exist_ok=True)

# ميزات "مقترحة" (قد لا تتوفر كلها وسنتعامل مع ذلك)
NUMERIC_FEATS_WISHLIST = [
    "price","days_left","demand_index","capacity","weekday","hour",
    "season_month","class_code","season_base_mult","event_mult",
    "weekday_mult","period_mult","stop_mult",
]
CATEG_FEATS_WISHLIST = ["airline","source_city","destination_city","departure_period","stops"]
ALL_CLASSES = ["Economy","Business","First"]

def load_data(path):
    return pd.read_csv(path, parse_dates=["departure_time","arrival_time"], low_memory=False)

def ensure_season_month(df):
    if "season_month" not in df.columns and "departure_time" in df.columns:
        try:
            df["season_month"] = pd.to_datetime(df["departure_time"], errors="coerce").dt.month
        except Exception:
            df["season_month"] = np.nan
    return df

def basic_clean(df):
    MUST_HAVE = ["class","seat_sold","price","days_left","capacity"]
    miss = [c for c in MUST_HAVE if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    df = ensure_season_month(df)

    for c in ["seat_sold","price","days_left","capacity","demand_index","weekday","hour","class_code",
              "season_month","season_base_mult","event_mult","weekday_mult","period_mult","stop_mult"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["seat_sold","price","days_left","capacity"])
    df["seat_sold"] = df[["seat_sold","capacity"]].min(axis=1).clip(lower=0)

    if "demand_index" in df.columns:
        df["demand_index"] = df["demand_index"].fillna(1.0).clip(0.3, 2.0)

    for c in NUMERIC_FEATS_WISHLIST:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    for c in CATEG_FEATS_WISHLIST:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("Unknown")

    return df

def select_present_features(df):
    num = [c for c in NUMERIC_FEATS_WISHLIST if c in df.columns]
    cat = [c for c in CATEG_FEATS_WISHLIST if c in df.columns]
    return num, cat

def train_one(df_class, cabin, num_feats, cat_feats):
    X = df_class[num_feats + cat_feats].copy()
    y = df_class["seat_sold"].astype(float)

    pre = ColumnTransformer(
        [("num","passthrough", num_feats),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)
    pipe = Pipeline([("pre", pre), ("rf", model)])

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xva)
    mae = mean_absolute_error(yva, pred); r2 = r2_score(yva, pred)

    out = os.path.join(MODELS_DIR, f"seats_model_{cabin.lower()}.joblib")
    joblib.dump(pipe, out)
    print(f"[OK] {cabin} → {out} | MAE={mae:.2f} R²={r2:.3f} | feats: num={len(num_feats)} cat={len(cat_feats)}")
    return out

def main():
    df = basic_clean(load_data(INPUT_CSV))
    num_feats, cat_feats = select_present_features(df)
    if not num_feats and not cat_feats:
        raise ValueError("No usable features found for training.")

    for cabin in ALL_CLASSES:
        sub = df[df["class"]==cabin].copy()
        if len(sub) < 200:
            print(f"[WARN] Not enough rows for {cabin} ({len(sub)}). Skipping.")
            continue
        train_one(sub, cabin, num_feats, cat_feats)

if __name__ == "__main__":
    main()
