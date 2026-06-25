"""STEP 4: Chronological split + leakage-free normalization (Bab III §3.1.4)."""
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from config import (DATA_PROCESSED, RESULTS_MODELS, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)

np.random.seed(RANDOM_SEED)


def split_data(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    n = len(df)
    tr, vl = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    train, val, test = df.iloc[:tr].copy(), df.iloc[tr:vl].copy(), df.iloc[vl:].copy()
    for name, d in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"{name}: {len(d):3d} | {d.index.min().date()} -> {d.index.max().date()}")
    return train, val, test


def main():
    RESULTS_MODELS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PROCESSED / "weekly_features.csv", index_col=0, parse_dates=True)
    train, val, test = split_data(df)

    feature_cols = [c for c in df.columns if c != "sales"]
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    scaler_X.fit(train[feature_cols])      # FIT ON TRAIN ONLY
    scaler_y.fit(train[["sales"]])

    for name, d in [("train", train), ("val", val), ("test", test)]:
        out = d.copy()
        out[feature_cols] = scaler_X.transform(d[feature_cols])
        out["sales"] = scaler_y.transform(d[["sales"]])
        out.to_csv(DATA_PROCESSED / f"{name}_scaled.csv")
        d.to_csv(DATA_PROCESSED / f"{name}.csv")     # raw (unscaled) copies too

    joblib.dump(scaler_X, RESULTS_MODELS / "scaler_X.pkl")
    joblib.dump(scaler_y, RESULTS_MODELS / "scaler_y.pkl")
    print("Split + scaling done (scalers saved).")


if __name__ == "__main__":
    main()
