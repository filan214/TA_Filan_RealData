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
