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
