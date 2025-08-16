# train_models.py — trains per-class seat-demand models without absolute paths
# Usage:
#   python train_models.py --input final_optimized_flights.csv --outdir models_seats_ml
# (Defaults: input='final_optimized_flights.csv', outdir='models_seats_ml')

import os
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# --------- CLI ---------
def parse_args():
    p = argparse.ArgumentParser(description="Train per-class seat pickup models (cloud-friendly).")
    p.add_argument("--input", "-i", default="final_optimized_flights.csv",
                   help="Input CSV file (relative path). Must contain price, seat_sold/capacity, features.")
    p.add_argument("--outdir", "-o", default="models_seats_ml",
                   help="Directory to save joblib models (relative path).")
    return p.parse_args()

# --------- Utilities ---------
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Parse datetimes if columns exist
    for c in ["departure_time", "arrival_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Derive season_month if missing
    if "season_month" not in df.columns:
        if "departure_time" in df.columns:
            df["season_month"] = df["departure_time"].dt.month
        else:
            raise ValueError("Missing columns: ['season_month'] and no 'departure_time' to derive from.")

    # Safe numeric casts where present
    num_cols = [
        "price","days_left","demand_index","capacity","seat_sold","weekday","hour",
        "season_month","class_code","season_base_mult","event_mult","weekday_mult",
        "period_mult","stop_mult","residual_capacity"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill a few basics with sensible defaults if present
    fill_defaults = {
        "demand_index": 1.0,
        "days_left": 30,
        "weekday": 3,
        "hour": 12,
        "season_base_mult": 1.0,
        "event_mult": 1.0,
        "weekday_mult": 1.0,
        "period_mult": 1.0,
        "stop_mult": 1.0,
        "class_code": 0,
    }
    for k, v in fill_defaults.items():
        if k in df.columns:
            df[k] = df[k].fillna(v)

    # Target and guards
    if "seat_sold" not in df.columns and "residual_capacity" not in df.columns:
        raise ValueError("CSV must have 'seat_sold' or 'residual_capacity' (preferably both).")

    if "seat_sold" not in df.columns:
        df["seat_sold"] = 0  # minimal fallback

    if "capacity" not in df.columns:
        df["capacity"] = df["seat_sold"]  # fallback (not ideal, but avoids crash)

    # Keep typical columns (others kept if present)
    return df

def build_pipeline(categorical_features, numeric_features):
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
    )
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    pipe = Pipeline([("pre", pre), ("rf", model)])
    return pipe

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    return mae, r2

# --------- Main ---------
def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    df = basic_clean(df)

    # Feature wishlists (use what exists)
    numeric_feats = [
        "price","days_left","demand_index","capacity","weekday","hour",
        "season_month","class_code","season_base_mult","event_mult",
        "weekday_mult","period_mult","stop_mult",
    ]
    categorical_feats = ["airline","source_city","destination_city","departure_period","stops"]

    numeric_feats = [c for c in numeric_feats if c in df.columns]
    categorical_feats = [c for c in categorical_feats if c in df.columns]

    classes = ["Economy","Business","First"]
    for cabin in classes:
        dff = df[df["class"] == cabin].copy()
        if dff.empty:
            print(f"[SKIP] {cabin}: no rows in input.")
            continue

        X = dff[categorical_feats + numeric_feats].copy()
        y = dff["seat_sold"].astype(float)

        pipe = build_pipeline(categorical_feats, numeric_feats)
        pipe.fit(X, y)

        preds = pipe.predict(X)
        mae, r2 = evaluate(y, preds)

        out_path = os.path.join(args.outdir, f"seats_model_{cabin.lower()}.joblib")
        joblib.dump(pipe, out_path)

        print(f"[OK] {cabin} → {out_path} | MAE={mae:.2f} R²={r2:.3f} | feats: num={len(numeric_feats)} cat={len(categorical_feats)}")

if __name__ == "__main__":
    main()
