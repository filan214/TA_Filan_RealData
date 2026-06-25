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
