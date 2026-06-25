"""STEP 3: Preprocessing & Feature Engineering (Bab III Tahap 3)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import (
    DATA_TRENDS, DATA_PROCESSED, RESULTS_FIGURES, RANDOM_SEED,
    RAW_DATA_FILE, STATUS_VALID, CATEGORY_FILTER,
    DATE_COL_PRIMARY, DATE_COL_FALLBACK, DEDUPE_KEYS, EXCLUDE_NEGATIVE_QTY,
)

np.random.seed(RANDOM_SEED)


def load_data():
    """Load + clean real POS transactions (Bab III §3.1.3 sub-step 1)."""
    df = pd.read_csv(RAW_DATA_FILE, encoding="utf-8-sig",
                     parse_dates=["created_at", "paid_at"])
    n_raw = len(df)

    df[DATE_COL_PRIMARY] = df[DATE_COL_PRIMARY].fillna(df[DATE_COL_FALLBACK])

    before = len(df)
    df = df[df["status"].isin(STATUS_VALID)].copy()
    print(f"Valid-status filter: {before} -> {len(df)}")

    if CATEGORY_FILTER:
        before = len(df)
        df = df[df["category"].isin(CATEGORY_FILTER)].copy()
        print(f"Category filter: {before} -> {len(df)}")

    before = len(df)
    df = df.drop_duplicates(subset=DEDUPE_KEYS, keep="first").copy()
    print(f"Dedupe on {DEDUPE_KEYS}: {before} -> {len(df)}")

    n_neg = (df["qty"] < 0).sum()
    if EXCLUDE_NEGATIVE_QTY and n_neg:
        df = df[df["qty"] >= 0].copy()
        print(f"Excluded {n_neg} negative-qty rows")

    print(f"Clean rows: {len(df)} (from {n_raw} raw)")

    trends = pd.read_csv(DATA_TRENDS / "trends_combined.csv",
                         index_col=0, parse_dates=True)
    return df, trends


def aggregate_to_weekly(df):
    """Weekly reconstruction (Bab III §3.1.3 sub-step 4).
    Output column kept as 'sales' so downstream code is dataset-agnostic.
    """
    daily = (df.groupby(df[DATE_COL_PRIMARY].dt.date)["qty"].sum().reset_index())
    daily.columns = ["date", "sales"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily.set_index("date", inplace=True)
    weekly = daily.resample("W").sum()
    weekly.columns = ["sales"]
    print(f"Daily obs: {len(daily)} | Weekly obs: {len(weekly)}")
    return weekly


def detect_outliers_iqr(series, multiplier=1.5):
    """IQR outlier flagging (Bab III §3.1.3 sub-step 3). Flag, don't auto-remove."""
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - multiplier * IQR, Q3 + multiplier * IQR
    return (series < lower) | (series > upper), lower, upper


def integrate_google_trends(weekly_sales, trends_df):
    """Merge composite Trends index onto weekly sales (Bab III §3.2.2 group 5)."""
    tw = trends_df.copy()
    tw["composite_index"] = tw.mean(axis=1)
    tw = tw.resample("W").mean()
    tw.ffill(inplace=True)
    tw.bfill(inplace=True)
    merged = weekly_sales.merge(tw[["composite_index"]],
                                left_index=True, right_index=True, how="left")
    merged["composite_index"] = merged["composite_index"].ffill()
    merged["composite_index"] = merged["composite_index"].fillna(merged["composite_index"].mean())
    return merged


def create_features(df):
    """Feature engineering (Bab III §3.1.3 sub-step 5 / §3.2.2).

    Leakage-free: every series-derived feature uses only information available
    *before* the week being predicted. Rolling windows are shifted by 1 so they
    never include sales[t], and sales_diff is the last *observed* change
    (sales[t-1] - sales[t-2]). Lag features are already shifted correctly.
    """
    df = df.copy()
    for lag in [1, 2, 4, 8, 12]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)
    df["rolling_mean_4w"] = df["sales"].rolling(4).mean().shift(1)
    df["rolling_std_4w"] = df["sales"].rolling(4).std().shift(1)
    df["rolling_mean_12w"] = df["sales"].rolling(12).mean().shift(1)
    df["rolling_std_12w"] = df["sales"].rolling(12).std().shift(1)
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year
    df["sales_diff"] = df["sales"].diff().shift(1)
    df.dropna(inplace=True)
    return df


def save_diagnostic_plots(weekly, df_processed):
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(weekly.index, weekly["sales"], color="black", linewidth=0.8)
    ax.set_title("Weekly Smartphone Units Sold (2022-2025)")
    ax.set_xlabel("Date"); ax.set_ylabel("Units"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(RESULTS_FIGURES / "fig_4_1_weekly_sales.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_processed.index, df_processed["composite_index"], color="black", linewidth=0.8)
    ax.set_title("Google Trends Indonesia - Composite Index")
    ax.set_xlabel("Date"); ax.set_ylabel("Search Interest"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(RESULTS_FIGURES / "fig_4_2_google_trends.png", dpi=150); plt.close()

    is_o, lo, hi = detect_outliers_iqr(weekly["sales"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(weekly["sales"].values, vert=False)
    ax.axvline(lo, color="gray", ls="--", label=f"Lower: {lo:.0f}")
    ax.axvline(hi, color="gray", ls="--", label=f"Upper: {hi:.0f}")
    ax.set_title(f"Outlier Detection - IQR ({is_o.sum()} flagged)"); ax.legend()
    plt.tight_layout(); plt.savefig(RESULTS_FIGURES / "fig_4_3_outliers.png", dpi=150); plt.close()

    feat = [c for c in df_processed.columns if c not in ["week_of_year", "month", "quarter", "year"]]
    corr = df_processed[feat].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="gray_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    plt.colorbar(im, ax=ax); ax.set_title("Feature Correlation Matrix")
    plt.tight_layout(); plt.savefig(RESULTS_FIGURES / "fig_4_4_correlation.png", dpi=150); plt.close()
    print(f"Plots saved to {RESULTS_FIGURES}")


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df, trends = load_data()
    weekly = aggregate_to_weekly(df)
    is_o, lo, hi = detect_outliers_iqr(weekly["sales"])
    print(f"Outliers flagged: {is_o.sum()} weeks (bounds {lo:.0f}-{hi:.0f})")
    merged = integrate_google_trends(weekly[["sales"]], trends)
    final = create_features(merged)
    print(f"Final shape: {final.shape}")
    final.to_csv(DATA_PROCESSED / "weekly_features.csv")
    save_diagnostic_plots(weekly, final)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
