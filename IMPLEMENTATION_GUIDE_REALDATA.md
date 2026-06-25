# Implementation Guide — Smartphone Demand Forecasting (Real POS Data Edition)

> **Project**: Comparative evaluation of LSTM, Random Forest, and SARIMA for
> smartphone demand forecasting, augmented with Google Trends Indonesia.
> **Author**: Valentinus Ferdian Filan Gunawan Eka Putra (NPM 221711713)
> **Institution**: Universitas Atma Jaya Yogyakarta — Information Systems
> **Data object**: Real point-of-sale (POS) transaction history from a single
> smartphone retail counter ("Counter HP Berkah Cell - Pusat"), Jan 2022 – Jun 2025.

---

## How This Guide Relates to the Thesis (Bab I–III)

This document is the **executable counterpart** to the methodology described in
Chapters I–III of the thesis. Every step below maps directly onto a stage the
thesis already commits to in writing, so the code and the document stay in sync.
The one deliberate change versus the thesis text: **the data object is now real
store POS transactions, not the public Kaggle "Store Item Demand Forecasting
Challenge" dataset.** Everything else — the eight research stages (Tahap 1–8),
the weekly aggregation, Google Trends augmentation, TimeSeriesSplit validation,
the three models, Diebold-Mariano testing, and the order-up-to-level inventory
layer — is preserved exactly as Bab III specifies.

| Thesis stage (Bab III) | Step in this guide |
|---|---|
| Tahap 2 — Pengumpulan Data | STEP 1 (load real POS export) + STEP 2 (Google Trends) |
| Tahap 3 — Praproses Data | STEP 3 (cleaning, weekly reconstruction, feature engineering) |
| Tahap 4 — Normalisasi & Pembagian Data | STEP 4 (MinMaxScaler fit-on-train, 70/15/15 chronological split) |
| Tahap 5 — Pembangunan & Pelatihan Model | STEP 5 (SARIMA), STEP 6 (Random Forest), STEP 7 (LSTM) |
| Tahap 6 — Evaluasi & Perbandingan | STEP 8 (MAPE/RMSE/MAE + Diebold-Mariano) |
| Tahap 7 — Integrasi Optimasi Inventori | STEP 9 (Safety Stock, ROP, OUL) |
| Tahap 8 — Analisis Dampak | STEP 10 (holding cost + stockout cost simulation) |
| Tujuan Khusus 5 — kerangka reprodusibel | STEP 11 (compiled Bab IV report) |

### What changes in the thesis text because of the real data

These are prose edits you (or your advisor) will make in the document — they are
**not** code, but they must stay consistent with what this pipeline produces:

1. **Bab I, Latar Belakang** — remove the paragraph justifying the public Kaggle
   dataset (the "skor kemiripan 83,5/100" and "*open access*" argument). Replace
   with a statement that the study now uses real local POS transaction data,
   which strengthens rather than weakens the contribution.
2. **Bab III §3.1.2 & §3.2.1 (Pengumpulan Data)** — replace the Kaggle profile
   ("913.000 baris, 50 item, 10 toko, 2013–2017") with the real profile:
   **63,674 transaction line items, single store, 15 smartphone SKUs,
   2 Jan 2022 – 30 Jun 2025 (≈3.49 years / 184 weeks)**.
3. **Bab III §1.5 Batasan Masalah** — the "Dataset" bullet changes from Kaggle
   to the store POS export. The 3-year minimum criterion is still satisfied.
4. **New limitation to add** — data comes from one store, so representativeness
   of the national market is limited; and the data is proprietary (not open
   access), so reproducibility now depends on a data-use statement / store
   permission rather than a public download link. Confirm with your advisor
   whether a signed data-use letter must be attached as a lampiran.
5. **Google Trends timeframe** changes from `2013-01-01 2017-12-31` to
   `2022-01-02 2025-06-30`.

---

## Dataset Profile (already validated)

The raw export has been consolidated from 42 monthly sheets into a single file,
`data/raw/pos_transactions_raw.csv`, with this verified profile:

- **63,674** transaction line items, **23 columns**, identical schema across all months
- Date span **2 Jan 2022 → 30 Jun 2025 = 3.49 years** (clears the 3-year / 3-seasonal-cycle requirement)
- `status`: **62,775 Paid**, **899 Void** (every Void has null `paid_at`, qty 0 — internally consistent)
- Single store, single category ("Handphone") — **no store/category filter needed**
- **15 smartphone SKUs** over the full period (catalogue grew from 8 in 2022 as new models launched)
- Brand mix: Xiaomi 15,432 · OPPO 15,109 · Samsung 13,277 · Realme 11,704 · Vivo 8,152
- **257 duplicate rows** on `sales_no`+`sku`+`qty` (~0.4%) — handled as an explicit cleaning step in STEP 3

Column schema:
```
sales_no, created_at, paid_at, status, order_type, store_name, customer_name,
product_name, variant_name, sku, category, qty, price, discount, subtotal, tax,
service_charge, rounding, grand_total, payment_method, staff_name, printer_id, note
```

> **Preview of what the models will see** (already computed): after filtering to
> Paid + dedupe, weekly aggregation yields **184 weekly observations**, mean ≈ 377
> units/week, weekly coefficient of variation ≈ 0.23, with **0 missing calendar
> weeks** and 1 near-empty week (2022-05-08, around Idul Fitri — a real low-traffic
> week, not a data gap). This is a clean, well-behaved series for SARIMA and a
> workable length for LSTM.

---

## Using This Guide with Claude Code

This guide is meant to be executed sequentially. Because the real data's exact
filter values were already discovered during exploration, the early steps are
fully specified — no guessing required. Run it in a **fresh project folder**.

```bash
mkdir -p ~/TA_Filan_RealData && cd ~/TA_Filan_RealData
claude
```

Then, step by step:
```
> "Read IMPLEMENTATION_GUIDE_REALDATA.md and do STEP 0 to STEP 3.
>  Verify each step's acceptance criteria before moving on. Stop after STEP 3
>  and show me the weekly series summary."
```

Recommended session breakdown (mirrors the thesis's three-notebook structure for
Tujuan Khusus 5):
- **Session 1**: STEP 0–4 (setup → data → preprocessing → split)
- **Session 2**: STEP 5 (SARIMA, longest-running)
- **Session 3**: STEP 6–7 (Random Forest + LSTM)
- **Session 4**: STEP 8–11 (evaluation → inventory → impact → report)

Commit after each session: `git add -A && git commit -m "Step N: <desc>"`.

---

## STEP 0 — Project Setup

### Goal
Create the project structure, virtual environment, and install dependencies.

### Implementation

**0.1 Folder structure:**
```bash
mkdir -p ~/TA_Filan_RealData/{data/{raw,trends,processed},notebooks,src,results/{figures,tables,models},docs}
cd ~/TA_Filan_RealData
```

**0.2 Virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

**0.3 `requirements.txt`:**
```txt
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tensorflow>=2.15.0
statsmodels>=0.14.0
pmdarima>=2.0.4
pytrends>=4.9.2
openpyxl>=3.1.0
jupyterlab>=4.0.0
ipykernel>=6.25.0
tqdm>=4.66.0
joblib>=1.3.0
```
Note: `kaggle` is intentionally removed — no longer downloading from Kaggle.
`openpyxl` is added in case you re-consolidate from the original Excel workbook.

**0.4 Install:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=ta_filan --display-name="TA Filan RealData"
```

**0.5 `.gitignore`:**
```
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/raw/*
data/trends/*
data/processed/*
results/models/*
.env
```

**0.6 `src/config.py`** — global constants, calibrated to the real data:
```python
"""Global configuration — Real POS Data edition."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_TRENDS = PROJECT_ROOT / "data" / "trends"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_MODELS = PROJECT_ROOT / "results" / "models"

RANDOM_SEED = 42

# === Real POS data source ===
RAW_DATA_FILE = DATA_RAW / "pos_transactions_raw.csv"

# Filter values — VERIFIED from data exploration (not guesses)
STATUS_VALID = ["Paid"]            # 62,775 Paid rows; "Void" excluded
CATEGORY_FILTER = None             # single category ("Handphone") — no filter needed
DATE_COL_PRIMARY = "paid_at"       # demand counted at payment, not order creation
DATE_COL_FALLBACK = "created_at"   # used only if paid_at missing (Void rows, already excluded)
DEDUPE_KEYS = ["sales_no", "sku", "qty"]   # drop the ~257 double-scan duplicates
EXCLUDE_NEGATIVE_QTY = False       # no negative qty present; net-demand policy if it ever appears

# Date range — VERIFIED from data (paid_at min/max)
DATE_START = "2022-01-02"
DATE_END = "2025-06-30"

# Google Trends
TRENDS_KEYWORDS = ["smartphone", "beli hp", "Samsung", "Xiaomi", "OPPO"]
TRENDS_GEO = "ID"
TRENDS_TIMEFRAME = f"{DATE_START} {DATE_END}"

# Train/Val/Test split (chronological) — Bab III §3.1.4
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# LSTM hyperparameters — Bab III §3.1.5
LSTM_LOOKBACK = 12
LSTM_UNITS_LAYER1 = 64
LSTM_UNITS_LAYER2 = 32
LSTM_DROPOUT = 0.2
LSTM_LEARNING_RATE = 0.001
LSTM_BATCH_SIZE = 32
LSTM_MAX_EPOCHS = 200
LSTM_PATIENCE = 20

# Random Forest search — Bab III §3.1.5
RF_N_ITER = 50
RF_CV_FOLDS = 5

# SARIMA — weekly seasonality
SARIMA_SEASONAL_PERIOD = 52
SARIMA_MAX_P = 3
SARIMA_MAX_Q = 3

# Inventory parameters — Bab III §3.1.7
SERVICE_LEVEL = 0.95
Z_SCORE = 1.645
LEAD_TIME_WEEKS = 2
REVIEW_PERIOD_WEEKS = 4

# Cost simulation — Bab III §3.1.8
HOLDING_COST_RATE = 0.10
STOCKOUT_COST_RATE = 1.00

# Target performance — Bab III §3.1.6
TARGET_MAPE = 15.0
```

### Acceptance Criteria
```bash
which python   # -> venv/bin/python
python -c "import pandas, numpy, tensorflow, sklearn, statsmodels, pmdarima, pytrends, openpyxl; print('All imports OK')"
```

### Common Issues
- **`pmdarima` build fails** — `pip install Cython numpy && pip install pmdarima --no-build-isolation`
- **TensorFlow GPU warning** — harmless on CPU; the dataset size doesn't need a GPU.

---

## STEP 1 — Load & Consolidate the Real POS Data

### Goal
Place the consolidated raw transaction file and confirm it reads correctly.
(If you only have the original 42-sheet Excel workbook, this step also shows how
to rebuild the consolidated CSV.)

### Inputs
- `data/raw/pos_transactions_raw.csv` (already consolidated), **or**
- the original `dat_tokoxyzsasas.xlsx` workbook with 42 `Penjualan_YYYY-MM` sheets

### Implementation

**1A — If you already have the consolidated CSV**, just drop it into
`data/raw/pos_transactions_raw.csv` and skip to 1.2.

**1B — If you only have the Excel workbook**, create `src/consolidate_raw.py`:
```python
"""STEP 1B: Merge 42 monthly POS sheets into one consolidated raw CSV."""
import pandas as pd
from config import DATA_RAW

WORKBOOK = DATA_RAW / "dat_tokoxyzsasas.xlsx"   # put the original workbook here


def main():
    xl = pd.ExcelFile(WORKBOOK, engine="openpyxl")
    monthly = [s for s in xl.sheet_names if s.startswith("Penjualan_")]
    print(f"Found {len(monthly)} monthly sheets (ignoring '{[s for s in xl.sheet_names if s not in monthly]}')")

    # Verify identical column schema across every sheet before merging
    ref = None
    for s in monthly:
        cols = list(pd.read_excel(xl, s, engine="openpyxl", nrows=1).columns)
        if ref is None:
            ref = cols
        elif cols != ref:
            raise ValueError(f"Sheet {s} has a different column schema: {cols}")

    frames = [pd.read_excel(xl, s, engine="openpyxl") for s in monthly]
    df = pd.concat(frames, ignore_index=True)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["paid_at"] = pd.to_datetime(df["paid_at"], errors="coerce")
    df = df.sort_values("paid_at", na_position="last").reset_index(drop=True)

    out = DATA_RAW / "pos_transactions_raw.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Consolidated {len(df):,} rows -> {out}")


if __name__ == "__main__":
    main()
```
Run: `python src/consolidate_raw.py`

**1.2 Quick exploration** — create `src/explore_raw_data.py`:
```python
"""STEP 1.2: Sanity-check the consolidated raw POS data."""
import pandas as pd
from config import RAW_DATA_FILE


def main():
    df = pd.read_csv(RAW_DATA_FILE, encoding="utf-8-sig",
                     parse_dates=["created_at", "paid_at"])
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}\n")

    span_years = (df['paid_at'].max() - df['paid_at'].min()).days / 365.25
    print(f"Date range (paid_at): {df['paid_at'].min()} -> {df['paid_at'].max()}")
    print(f"Span: {span_years:.2f} years (need >= 3)\n")

    print("status:\n", df['status'].value_counts(dropna=False), "\n")
    print("category:\n", df['category'].value_counts(dropna=False), "\n")
    print("unique SKUs:", df['sku'].nunique())
    print(df['sku'].value_counts(), "\n")
    print("duplicate rows on sales_no+sku+qty:",
          df.duplicated(subset=['sales_no', 'sku', 'qty']).sum())


if __name__ == "__main__":
    main()
```
Run: `python src/explore_raw_data.py | tee results/tables/raw_data_exploration.txt`

### Acceptance Criteria
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/raw/pos_transactions_raw.csv', encoding='utf-8-sig', parse_dates=['paid_at'])
span = (df['paid_at'].max() - df['paid_at'].min()).days / 365.25
assert span >= 3, f'Span {span:.1f}y < 3y requirement'
assert len(df) > 50000, f'Only {len(df)} rows'
print(f'OK: {len(df):,} rows, {span:.2f} years')
"
```

---

## STEP 2 — Google Trends Indonesia

### Goal
Fetch Google Trends Indonesia for the five smartphone keywords, over the **real
data's date range** (2022-01-02 → 2025-06-30), as the exogenous variable.

### Implementation
Create `src/data_collection.py`:
```python
"""STEP 2: Fetch Google Trends Indonesia (exogenous variable)."""
import time
import pandas as pd
from pytrends.request import TrendReq
from config import DATA_TRENDS, TRENDS_KEYWORDS, TRENDS_GEO, TRENDS_TIMEFRAME


def fetch_google_trends():
    print(f"Fetching Google Trends ({TRENDS_TIMEFRAME})...")
    DATA_TRENDS.mkdir(parents=True, exist_ok=True)
    pytrends = TrendReq(hl="id-ID", tz=420)  # WIB

    all_trends = {}
    for kw in TRENDS_KEYWORDS:
        print(f"  {kw}")
        try:
            pytrends.build_payload([kw], cat=0, timeframe=TRENDS_TIMEFRAME, geo=TRENDS_GEO)
            data = pytrends.interest_over_time()
            if not data.empty:
                data = data.drop(columns=["isPartial"], errors="ignore")
                all_trends[kw] = data
                data.to_csv(DATA_TRENDS / f"trends_{kw.replace(' ', '_')}.csv")
            else:
                print(f"    WARNING: no data for '{kw}'")
            time.sleep(60)
        except Exception as e:
            print(f"    ERROR '{kw}': {e}")
            time.sleep(120)

    combined = pd.concat(all_trends.values(), axis=1)
    combined.columns = list(all_trends.keys())
    combined.to_csv(DATA_TRENDS / "trends_combined.csv")
    print(f"Combined trends shape: {combined.shape}")
    return combined


if __name__ == "__main__":
    fetch_google_trends()
```
Run: `python src/data_collection.py`

### Acceptance Criteria
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/trends/trends_combined.csv', index_col=0, parse_dates=True)
assert df.shape[1] == 5, f'Expected 5 keywords, got {df.shape[1]}'
assert len(df) >= 150, f'Expected >=150 weeks, got {len(df)}'
print(f'OK: trends {df.shape}')
"
```

### Common Issues
- **pytrends 429 rate limit** — the `time.sleep(60)` handles most cases; raise to 120 if needed.
- **Empty DataFrame** — try a VPN with an Indonesian exit, or set `tz=0`.
- **Trends weekly grid doesn't perfectly align with sales weeks** — handled in STEP 3 by resampling both to week-ending-Sunday before merge.

---

## STEP 3 — Preprocessing & Feature Engineering

### Goal
Clean the data, reconstruct a weekly time series, integrate Google Trends, and
engineer features — exactly the five sub-steps of Bab III §3.1.3.

### Implementation
Create `src/preprocessing.py`:
```python
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
    """Feature engineering (Bab III §3.1.3 sub-step 5 / §3.2.2)."""
    df = df.copy()
    for lag in [1, 2, 4, 8, 12]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)
    df["rolling_mean_4w"] = df["sales"].rolling(4).mean()
    df["rolling_std_4w"] = df["sales"].rolling(4).std()
    df["rolling_mean_12w"] = df["sales"].rolling(12).mean()
    df["rolling_std_12w"] = df["sales"].rolling(12).std()
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year
    df["sales_diff"] = df["sales"].diff()
    df.dropna(inplace=True)
    return df


def save_diagnostic_plots(weekly, df_processed):
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(weekly.index, weekly["sales"], color="black", linewidth=0.8)
    ax.set_title("Weekly Smartphone Units Sold (2022–2025)")
    ax.set_xlabel("Date"); ax.set_ylabel("Units"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(RESULTS_FIGURES / "fig_4_1_weekly_sales.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_processed.index, df_processed["composite_index"], color="black", linewidth=0.8)
    ax.set_title("Google Trends Indonesia — Composite Index")
    ax.set_xlabel("Date"); ax.set_ylabel("Search Interest"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(RESULTS_FIGURES / "fig_4_2_google_trends.png", dpi=150); plt.close()

    is_o, lo, hi = detect_outliers_iqr(weekly["sales"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(weekly["sales"].values, vert=False)
    ax.axvline(lo, color="gray", ls="--", label=f"Lower: {lo:.0f}")
    ax.axvline(hi, color="gray", ls="--", label=f"Upper: {hi:.0f}")
    ax.set_title(f"Outlier Detection — IQR ({is_o.sum()} flagged)"); ax.legend()
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
    print(f"Outliers flagged: {is_o.sum()} weeks (bounds {lo:.0f}–{hi:.0f})")
    merged = integrate_google_trends(weekly[["sales"]], trends)
    final = create_features(merged)
    print(f"Final shape: {final.shape}")
    final.to_csv(DATA_PROCESSED / "weekly_features.csv")
    save_diagnostic_plots(weekly, final)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
```
Run: `python src/preprocessing.py`

### Acceptance Criteria
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/processed/weekly_features.csv', index_col=0, parse_dates=True)
need = ['sales','composite_index','lag_1','lag_12','rolling_mean_4w','week_of_year','sales_diff']
assert all(c in df.columns for c in need), 'Missing columns'
assert df.isnull().sum().sum() == 0, 'NaN remain'
assert len(df) > 100, f'Too few weeks: {len(df)}'
print(f'OK: {df.shape}, weeks={len(df)}')
"
```

> **Expect real findings here** unlike the Kaggle run: real outliers around
> Lebaran/Harbolnas peaks and the quiet Idul Fitri week, and genuine feature
> correlations. This is good — the preprocessing chapter now has substance.

---

## STEP 4 — Train/Val/Test Split & Normalization

### Goal
Chronological 70/15/15 split with MinMaxScaler fit on train only — Bab III §3.1.4.

### Implementation
Create `src/data_split.py`:
```python
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
```
Run: `python src/data_split.py`

### Acceptance Criteria
```bash
python -c "
import pandas as pd
for s in ['train','val','test']:
    d = pd.read_csv(f'data/processed/{s}_scaled.csv', index_col=0)
    assert d['sales'].between(-0.01, 1.01).all() or s != 'train', 'train sales not in [0,1]'
print('OK: split + scaling')
"
```

---

## STEP 5 — SARIMA (statistical baseline)

### Goal
Fit SARIMA via auto-ARIMA on the weekly series — Bab III §3.1.5, §3.2.2 Tahap 1.

### Implementation
Create `src/model_sarima.py`:
```python
"""STEP 5: SARIMA baseline (Bab III)."""
import warnings, json
import numpy as np
import pandas as pd
import joblib
import pmdarima as pm
import matplotlib.pyplot as plt
from config import (DATA_PROCESSED, RESULTS_TABLES, RESULTS_FIGURES, RESULTS_MODELS,
                    SARIMA_SEASONAL_PERIOD, SARIMA_MAX_P, SARIMA_MAX_Q)
warnings.filterwarnings("ignore")


def main():
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_PROCESSED / "train.csv", index_col=0, parse_dates=True)
    val = pd.read_csv(DATA_PROCESSED / "val.csv", index_col=0, parse_dates=True)
    test = pd.read_csv(DATA_PROCESSED / "test.csv", index_col=0, parse_dates=True)

    y_train = pd.concat([train["sales"], val["sales"]])   # train+val for final fit
    exog_cols = ["composite_index"]
    X_train = pd.concat([train[exog_cols], val[exog_cols]])
    X_test = test[exog_cols]

    m = min(SARIMA_SEASONAL_PERIOD, len(y_train) // 2)
    print(f"auto_arima (seasonal m={m})...")
    model = pm.auto_arima(
        y_train, X=X_train, seasonal=True, m=m,
        max_p=SARIMA_MAX_P, max_q=SARIMA_MAX_Q,
        stepwise=True, suppress_warnings=True, error_action="ignore", trace=True,
    )

    with open(RESULTS_TABLES / "sarima_summary.txt", "w") as f:
        f.write(str(model.summary()))

    preds = model.predict(n_periods=len(test), X=X_test)
    pd.DataFrame({"actual": test["sales"].values, "pred": np.asarray(preds)},
                 index=test.index).to_csv(RESULTS_TABLES / "sarima_predictions.csv")
    joblib.dump(model, RESULTS_MODELS / "sarima_model.pkl")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test.index, test["sales"].values, "k-", label="Actual")
    ax.plot(test.index, np.asarray(preds), "k--", label="SARIMA")
    ax.legend(); ax.set_title("SARIMA — Test Forecast"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(RESULTS_FIGURES / "fig_4_6_sarima_diagnostics.png", dpi=150); plt.close()
    print(f"SARIMA order: {model.order} x {model.seasonal_order}")


if __name__ == "__main__":
    main()
```
Run: `python src/model_sarima.py`  *(longest step — minutes to ~1 hr)*

> Note: SARIMA uses the **unscaled** series (it's scale-invariant and the
> inventory layer needs real units). RF/LSTM use the scaled series.

---

## STEP 6 — Random Forest

### Implementation
Create `src/model_random_forest.py`:
```python
"""STEP 6: Random Forest (Bab III §3.1.5)."""
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from config import (DATA_PROCESSED, RESULTS_TABLES, RESULTS_FIGURES, RESULTS_MODELS,
                    RF_N_ITER, RF_CV_FOLDS, RANDOM_SEED)


def main():
    train = pd.read_csv(DATA_PROCESSED / "train_scaled.csv", index_col=0, parse_dates=True)
    val = pd.read_csv(DATA_PROCESSED / "val_scaled.csv", index_col=0, parse_dates=True)
    test = pd.read_csv(DATA_PROCESSED / "test_scaled.csv", index_col=0, parse_dates=True)

    feats = [c for c in train.columns if c != "sales"]
    Xtr = pd.concat([train[feats], val[feats]]); ytr = pd.concat([train["sales"], val["sales"]])
    Xte, yte = test[feats], test["sales"]

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 1.0],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(bootstrap=True, random_state=RANDOM_SEED),
        param_dist, n_iter=RF_N_ITER, cv=TimeSeriesSplit(n_splits=RF_CV_FOLDS),
        scoring="neg_mean_absolute_percentage_error", random_state=RANDOM_SEED, n_jobs=-1,
    )
    search.fit(Xtr, ytr)
    best = search.best_estimator_
    with open(RESULTS_TABLES / "rf_best_params.json", "w") as f:
        json.dump(search.best_params_, f, indent=2)

    preds = best.predict(Xte)
    pd.DataFrame({"actual": yte.values, "pred": preds}, index=test.index).to_csv(
        RESULTS_TABLES / "rf_predictions.csv")
    joblib.dump(best, RESULTS_MODELS / "rf_model.pkl")

    imp = pd.Series(best.feature_importances_, index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot.barh(ax=ax, color="gray"); ax.set_title("RF Feature Importance (MDI)")
    plt.tight_layout(); plt.savefig(RESULTS_FIGURES / "fig_4_7_rf_feature_importance.png", dpi=150); plt.close()
    print("Best RF params:", search.best_params_)


if __name__ == "__main__":
    main()
```
Run: `python src/model_random_forest.py`

---

## STEP 7 — LSTM

### Implementation
Create `src/model_lstm.py`:
```python
"""STEP 7: Stacked LSTM (Bab III §3.1.5 — 64/32 units, dropout 0.2, lookback 12)."""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from config import (DATA_PROCESSED, RESULTS_TABLES, RESULTS_FIGURES, RESULTS_MODELS,
                    LSTM_LOOKBACK, LSTM_UNITS_LAYER1, LSTM_UNITS_LAYER2, LSTM_DROPOUT,
                    LSTM_LEARNING_RATE, LSTM_BATCH_SIZE, LSTM_MAX_EPOCHS, LSTM_PATIENCE,
                    RANDOM_SEED)

tf.random.set_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)


def make_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i]); ys.append(y[i])
    return np.array(Xs), np.array(ys)


def main():
    train = pd.read_csv(DATA_PROCESSED / "train_scaled.csv", index_col=0, parse_dates=True)
    val = pd.read_csv(DATA_PROCESSED / "val_scaled.csv", index_col=0, parse_dates=True)
    test = pd.read_csv(DATA_PROCESSED / "test_scaled.csv", index_col=0, parse_dates=True)
    feats = [c for c in train.columns if c != "sales"]

    trv = pd.concat([train, val])
    Xtr, ytr = make_sequences(trv[feats].values, trv["sales"].values, LSTM_LOOKBACK)
    # build test sequences using tail of train+val for the lookback context
    ctx = pd.concat([trv.tail(LSTM_LOOKBACK), test])
    Xte, yte = make_sequences(ctx[feats].values, ctx["sales"].values, LSTM_LOOKBACK)

    model = Sequential([
        LSTM(LSTM_UNITS_LAYER1, return_sequences=True, input_shape=(LSTM_LOOKBACK, len(feats))),
        Dropout(LSTM_DROPOUT),
        LSTM(LSTM_UNITS_LAYER2),
        Dropout(LSTM_DROPOUT),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(LSTM_LEARNING_RATE), loss="mse")
    with open(RESULTS_TABLES / "lstm_architecture.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    hist = model.fit(
        Xtr, ytr, validation_split=0.15, epochs=LSTM_MAX_EPOCHS, batch_size=LSTM_BATCH_SIZE,
        callbacks=[EarlyStopping(patience=LSTM_PATIENCE, restore_best_weights=True),
                   ReduceLROnPlateau(patience=10, factor=0.5)],
        verbose=1,
    )
    preds = model.predict(Xte).flatten()
    pd.DataFrame({"actual": yte, "pred": preds}, index=test.index[:len(yte)]).to_csv(
        RESULTS_TABLES / "lstm_predictions.csv")
    model.save(RESULTS_MODELS / "lstm_model.keras")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist.history["loss"], "k-", label="train loss")
    ax.plot(hist.history["val_loss"], "k--", label="val loss")
    ax.legend(); ax.set_title("LSTM Loss"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(RESULTS_FIGURES / "fig_4_8_lstm_loss_curve.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
```
Run: `python src/model_lstm.py`

> All three prediction CSVs are saved in the same `actual`/`pred` format so the
> evaluation step treats them uniformly. RF and LSTM predictions are in scaled
> units; STEP 8 inverse-transforms them with the saved `scaler_y` before
> computing metrics and feeding the inventory layer, so SARIMA (already unscaled)
> and the ML models are compared on identical real-unit footing.

---

## STEP 8 — Evaluation & Diebold-Mariano

### Implementation
Create `src/evaluation.py`:
```python
"""STEP 8: MAPE/RMSE/MAE + Diebold-Mariano (Bab III §3.1.6, §3.2.3)."""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf
from config import DATA_PROCESSED, RESULTS_TABLES, RESULTS_FIGURES, RESULTS_MODELS, TARGET_MAPE


def metrics(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mae = np.mean(np.abs(actual - pred))
    return mape, rmse, mae


def dm_test(e1, e2, h=1):
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    gamma = acovf(d, fft=True, nlag=h - 1)
    var_d = gamma[0] + 2 * np.sum(gamma[1:h]) if h > 1 else gamma[0]
    dm = np.mean(d) / np.sqrt(var_d / n)
    from scipy.stats import t
    p = 2 * (1 - t.cdf(abs(dm), df=n - 1))
    return dm, p


def main():
    scaler_y = joblib.load(RESULTS_MODELS / "scaler_y.pkl")

    def load(name, scaled):
        df = pd.read_csv(RESULTS_TABLES / f"{name}_predictions.csv", index_col=0, parse_dates=True)
        if scaled:   # inverse-transform RF/LSTM back to real units
            df["actual"] = scaler_y.inverse_transform(df[["actual"]])
            df["pred"] = scaler_y.inverse_transform(df[["pred"]])
        return df

    sarima = load("sarima", scaled=False)
    rf = load("rf", scaled=True)
    lstm = load("lstm", scaled=True)

    rows = {}
    for name, df in [("SARIMA", sarima), ("RandomForest", rf), ("LSTM", lstm)]:
        rows[name] = dict(zip(["MAPE", "RMSE", "MAE"], metrics(df["actual"], df["pred"])))
    res = pd.DataFrame(rows).T
    res["meets_target"] = res["MAPE"] < TARGET_MAPE
    res.to_csv(RESULTS_TABLES / "evaluation_metrics.csv")
    print(res)

    # Pairwise DM on the common overlapping index
    idx = sarima.index.intersection(rf.index).intersection(lstm.index)
    err = {"SARIMA": (sarima.loc[idx, "actual"] - sarima.loc[idx, "pred"]).values,
           "RandomForest": (rf.loc[idx, "actual"] - rf.loc[idx, "pred"]).values,
           "LSTM": (lstm.loc[idx, "actual"] - lstm.loc[idx, "pred"]).values}
    dm_rows = []
    for a, b in [("SARIMA", "RandomForest"), ("SARIMA", "LSTM"), ("RandomForest", "LSTM")]:
        dm, p = dm_test(err[a], err[b])
        dm_rows.append({"pair": f"{a} vs {b}", "DM": round(dm, 3), "p_value": round(p, 4),
                        "significant_5pct": p < 0.05})
    pd.DataFrame(dm_rows).to_csv(RESULTS_TABLES / "diebold_mariano.csv", index=False)
    print(pd.DataFrame(dm_rows))


if __name__ == "__main__":
    main()
```
Run: `python src/evaluation.py`

### Acceptance Criteria
```bash
python -c "
import pandas as pd
m = pd.read_csv('results/tables/evaluation_metrics.csv', index_col=0)
assert {'MAPE','RMSE','MAE'}.issubset(m.columns)
print(m); print('OK')
"
```

---

## STEP 9 — Inventory Integration (Order-Up-To-Level)

### Implementation
Create `src/inventory.py`:
```python
"""STEP 9: Safety Stock / ROP / OUL (Bab III §3.1.7)."""
import numpy as np
import pandas as pd
import joblib
from config import (RESULTS_TABLES, RESULTS_MODELS, Z_SCORE, LEAD_TIME_WEEKS, REVIEW_PERIOD_WEEKS)


def main():
    scaler_y = joblib.load(RESULTS_MODELS / "scaler_y.pkl")

    def load(name, scaled):
        df = pd.read_csv(RESULTS_TABLES / f"{name}_predictions.csv", index_col=0, parse_dates=True)
        if scaled:
            df["actual"] = scaler_y.inverse_transform(df[["actual"]])
            df["pred"] = scaler_y.inverse_transform(df[["pred"]])
        return df

    models = {"SARIMA": load("sarima", False), "RandomForest": load("rf", True), "LSTM": load("lstm", True)}
    rows = {}
    for name, df in models.items():
        resid = df["actual"] - df["pred"]
        sigma = resid.std()
        ss = Z_SCORE * sigma * np.sqrt(LEAD_TIME_WEEKS)
        dbar_lt = df["pred"].mean() * LEAD_TIME_WEEKS
        rop = dbar_lt + ss
        oul = df["pred"].mean() * (REVIEW_PERIOD_WEEKS + LEAD_TIME_WEEKS) + ss
        rows[name] = {"sigma_resid": round(sigma, 2), "safety_stock": round(ss, 2),
                      "reorder_point": round(rop, 2), "order_up_to_level": round(oul, 2)}
    pd.DataFrame(rows).T.to_csv(RESULTS_TABLES / "inventory_parameters.csv")
    print(pd.DataFrame(rows).T)


if __name__ == "__main__":
    main()
```
Run: `python src/inventory.py`

---

## STEP 10 — Impact Simulation (cost)

### Implementation
Create `src/impact_simulation.py`:
```python
"""STEP 10: Holding + stockout cost simulation vs SARIMA baseline (Bab III §3.1.8)."""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from config import (RESULTS_TABLES, RESULTS_FIGURES, RESULTS_MODELS,
                    HOLDING_COST_RATE, STOCKOUT_COST_RATE, Z_SCORE, LEAD_TIME_WEEKS)


def simulate(actual, pred, holding_rate, stockout_rate):
    resid_sigma = np.std(actual - pred)
    ss = Z_SCORE * resid_sigma * np.sqrt(LEAD_TIME_WEEKS)
    inv = pred + ss                       # stock positioned to forecast + safety stock
    holding = np.maximum(inv - actual, 0).sum() * holding_rate
    stockout = np.maximum(actual - inv, 0).sum() * stockout_rate
    n_stockout = int((actual > inv).sum())
    return holding + stockout, holding, stockout, n_stockout


def main():
    scaler_y = joblib.load(RESULTS_MODELS / "scaler_y.pkl")

    def load(name, scaled):
        df = pd.read_csv(RESULTS_TABLES / f"{name}_predictions.csv", index_col=0, parse_dates=True)
        if scaled:
            df["actual"] = scaler_y.inverse_transform(df[["actual"]])
            df["pred"] = scaler_y.inverse_transform(df[["pred"]])
        return df

    models = {"SARIMA": load("sarima", False), "RandomForest": load("rf", True), "LSTM": load("lstm", True)}
    rows = {}
    for name, df in models.items():
        total, hold, stock, nso = simulate(df["actual"].values, df["pred"].values,
                                           HOLDING_COST_RATE, STOCKOUT_COST_RATE)
        rows[name] = {"total_cost": round(total, 2), "holding_cost": round(hold, 2),
                      "stockout_cost": round(stock, 2), "n_stockout_weeks": nso}
    cost = pd.DataFrame(rows).T
    cost.to_csv(RESULTS_TABLES / "cost_simulation.csv")

    base = cost.loc["SARIMA", "total_cost"]
    sav = ((base - cost["total_cost"]) / base * 100).round(2)
    sav.to_frame("savings_vs_sarima_pct").to_csv(RESULTS_TABLES / "savings_vs_sarima.csv")
    print(cost); print("\nSavings vs SARIMA (%):\n", sav)

    fig, ax = plt.subplots(figsize=(8, 5))
    cost[["holding_cost", "stockout_cost"]].plot.bar(stacked=True, ax=ax, color=["gray", "black"])
    ax.set_title("Inventory Cost by Model"); ax.set_ylabel("Cost (units x rate)")
    plt.tight_layout(); plt.savefig(RESULTS_FIGURES / "fig_4_13_cost_comparison.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
```
Run: `python src/impact_simulation.py`

---

## STEP 11 — Compile Bab IV Report

### Implementation
Create `src/generate_report.py`:
```python
"""STEP 11: Compile results into a Bab IV markdown report (real POS data)."""
import json
import pandas as pd
from datetime import datetime
from config import RESULTS_TABLES, RESULTS_FIGURES, PROJECT_ROOT


def main():
    o = ["# BAB IV — HASIL DAN PEMBAHASAN",
         f"\n*Auto-generated {datetime.now():%Y-%m-%d %H:%M}*\n",
         "## 4.1 Hasil Pengumpulan Data\n",
         ("Data transaksi penjualan smartphone diperoleh langsung dari sistem POS "
          "satu gerai ritel (Counter HP Berkah Cell), mencakup seluruh transaksi "
          "berstatus lunas pada kategori smartphone periode 2 Januari 2022 hingga "
          "30 Juni 2025 (\u2248 3,49 tahun / 184 minggu, 15 SKU). Data Google Trends "
          "Indonesia ditarik untuk rentang waktu yang identik sebagai variabel "
          "eksogen.\n"),
         f"![Weekly Sales]({RESULTS_FIGURES}/fig_4_1_weekly_sales.png)\n",
         f"![Google Trends]({RESULTS_FIGURES}/fig_4_2_google_trends.png)\n",
         "## 4.2 Praproses & Feature Engineering\n",
         f"![Outliers]({RESULTS_FIGURES}/fig_4_3_outliers.png)\n",
         f"![Correlation]({RESULTS_FIGURES}/fig_4_4_correlation.png)\n",
         "## 4.4 SARIMA\n"]
    if (RESULTS_TABLES / "sarima_summary.txt").exists():
        fence = chr(96) * 3
        o.append(fence + "\n" + open(RESULTS_TABLES / "sarima_summary.txt").read() + "\n" + fence + "\n")
    o.append(f"![SARIMA]({RESULTS_FIGURES}/fig_4_6_sarima_diagnostics.png)\n")
    o.append("## 4.5 Random Forest\n")
    if (RESULTS_TABLES / "rf_best_params.json").exists():
        p = json.load(open(RESULTS_TABLES / "rf_best_params.json"))
        o += ["**Best hyperparameters:**\n"] + [f"- `{k}`: {v}" for k, v in p.items()] + [""]
    o.append(f"![RF Importance]({RESULTS_FIGURES}/fig_4_7_rf_feature_importance.png)\n")
    o.append("## 4.6 LSTM\n")
    o.append(f"![LSTM Loss]({RESULTS_FIGURES}/fig_4_8_lstm_loss_curve.png)\n")
    o.append("## 4.7 Komparasi Performa\n")
    if (RESULTS_TABLES / "evaluation_metrics.csv").exists():
        o.append(pd.read_csv(RESULTS_TABLES / "evaluation_metrics.csv", index_col=0).to_markdown()); o.append("")
    o.append("## 4.8 Uji Diebold-Mariano\n")
    if (RESULTS_TABLES / "diebold_mariano.csv").exists():
        o.append(pd.read_csv(RESULTS_TABLES / "diebold_mariano.csv").to_markdown(index=False)); o.append("")
    o.append("## 4.9 Optimasi Inventori\n")
    if (RESULTS_TABLES / "inventory_parameters.csv").exists():
        o.append(pd.read_csv(RESULTS_TABLES / "inventory_parameters.csv", index_col=0).to_markdown()); o.append("")
    o.append("## 4.10 Analisis Dampak Biaya\n")
    if (RESULTS_TABLES / "cost_simulation.csv").exists():
        o.append(pd.read_csv(RESULTS_TABLES / "cost_simulation.csv", index_col=0).to_markdown()); o.append("")
    if (RESULTS_TABLES / "savings_vs_sarima.csv").exists():
        o.append(pd.read_csv(RESULTS_TABLES / "savings_vs_sarima.csv", index_col=0).to_markdown()); o.append("")
    o.append(f"![Cost]({RESULTS_FIGURES}/fig_4_13_cost_comparison.png)\n")

    path = PROJECT_ROOT / "BAB4_REPORT.md"
    open(path, "w").write("\n".join(o))
    print(f"Report generated: {path}")


if __name__ == "__main__":
    main()
```
Run: `python src/generate_report.py`

---

## Master Run Script

Create `run_all.sh`:
```bash
#!/bin/bash
set -e
source venv/bin/activate
python src/explore_raw_data.py
python src/data_collection.py        # Google Trends
python src/preprocessing.py
python src/data_split.py
python src/model_sarima.py &
SARIMA_PID=$!
python src/model_random_forest.py
python src/model_lstm.py
wait $SARIMA_PID
python src/evaluation.py
python src/inventory.py
python src/impact_simulation.py
python src/generate_report.py
echo "ALL STEPS COMPLETED -> BAB4_REPORT.md"
```
```bash
chmod +x run_all.sh && ./run_all.sh
```

---

## Final Project Tree
```
TA_Filan_RealData/
├── venv/
├── data/
│   ├── raw/pos_transactions_raw.csv      (63,674 rows, consolidated)
│   ├── trends/trends_combined.csv
│   └── processed/{weekly_features,train,val,test,*_scaled}.csv
├── src/
│   ├── config.py
│   ├── consolidate_raw.py        (only if rebuilding from the xlsx)
│   ├── explore_raw_data.py
│   ├── data_collection.py        (Google Trends only)
│   ├── preprocessing.py
│   ├── data_split.py
│   ├── model_sarima.py
│   ├── model_random_forest.py
│   ├── model_lstm.py
│   ├── evaluation.py
│   ├── inventory.py
│   ├── impact_simulation.py
│   └── generate_report.py
├── results/{figures,tables,models}/
├── BAB4_REPORT.md
├── requirements.txt
└── run_all.sh
```

---

## Quick Reference — Order of Execution

| # | Script | Time Est. | Critical to Bab IV? |
|---|---|---|---|
| 0 | Setup | 10 min | – |
| 1 | consolidate_raw / explore_raw_data | 2 min | Yes |
| 2 | data_collection (Trends) | 5–10 min | Yes |
| 3 | preprocessing | 1 min | Yes |
| 4 | data_split | <1 min | Yes |
| 5 | model_sarima | 5–60 min | Yes |
| 6 | model_random_forest | 5–15 min | Yes |
| 7 | model_lstm | 10–30 min | Yes |
| 8 | evaluation | <1 min | **Critical** |
| 9 | inventory | <1 min | Yes |
| 10 | impact_simulation | <1 min | **Critical** |
| 11 | generate_report | <1 min | Convenience |

---

## Optional — Per-SKU Sensitivity (supports the §4.12.1 discussion)

The data has 15 SKUs. For a stronger sensitivity analysis, re-run STEP 3 onward
grouping `aggregate_to_weekly` by `sku` (one series per model). Per-item series
are more volatile (higher CV) than the store aggregate, which is exactly the
regime where the thesis already hypothesizes Random Forest may overtake SARIMA.
Testing that on real per-item data — rather than a synthetic split — makes the
finding substantially more credible. Keep the store-aggregate run as the primary
Bab IV result for a clean 1:1 comparison with the original methodology.
```
```
