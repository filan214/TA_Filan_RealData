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
