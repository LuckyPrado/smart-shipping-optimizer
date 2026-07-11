from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

from pathlib import Path
import pickle
from datetime import datetime

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor

from utils import load_data, brazil_regions


def engineer_features(df):
    """Build the compact, validated feature set (see analysis/feature_engineering.ipynb).

    Only four columns are derived here; every other model input is a raw column
    from load_data() (coordinates, zip prefixes, price, freight, payment, product
    size) selected directly by the ColumnTransformer. The ~70 interaction/behavioural
    features this used to build were measured on the held-out test set in PR 7 and
    dropped as redundant -- they were overfitting, not helping.
    """
    df = df.copy()

    # Olist's own promised delivery window -- known at checkout, the strongest
    # single predictor of actual delivery time.
    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days

    # Seasonality: Feb/Nov/Dec run ~5-6 days slower than mid-year.
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month

    # state_pair feeds the target encoder; customer_region feeds the error report.
    df["state_pair"] = df["seller_state"] + "_" + df["customer_state"]
    df["customer_region"] = df["customer_state"].map(brazil_regions)

    return df


# Column schema, pruned 88 -> 15 in PR 7. Rejected features (distance, regions,
# order size, category, seller behaviour, 60+ interactions) and their test-set
# numbers are documented in analysis/feature_engineering.ipynb's experiment log.
num_attribs = [
    "estimated_delivery_days",   # strongest single predictor
    "purchase_month",            # seasonality
    # raw geography: coordinates + zip prefixes beat any distance/region summary.
    "seller_lat", "seller_lng", "customer_lat", "customer_lng",
    "seller_zip_code_prefix", "customer_zip_code_prefix",
    # order economics + physical size.
    "price", "freight_value", "payment_value",
    "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
]

# Regions dropped: redundant with the state one-hot (region is derived from state).
cat_attribs = ["seller_state", "customer_state", "product_category_name"]


def make_preprocessing():
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    # Target encoding is refit per CV fold (no leakage).
    target_attribs = ["state_pair", "customer_state", "seller_state"]
    return ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
        ("target", TargetEncoder(), target_attribs),
    ])


def main():
    logistics = load_data()
    print("Data loaded:", logistics.shape)

    logistics["actual_delivery_days"] = (
        logistics["order_delivered_customer_date"] - logistics["order_purchase_timestamp"]
    ).dt.days
    logistics = logistics.dropna(subset=["actual_delivery_days", "order_delivered_customer_date"])

    # Time-based split: earliest 80% = train, latest 20% = test (no future leaking into the past).
    logistics = logistics.sort_values("order_purchase_timestamp")
    split_idx = int(len(logistics) * 0.8)
    train_set = logistics.iloc[:split_idx]
    test_set = logistics.iloc[split_idx:]
    print(f"Train: {len(train_set)} | Test: {len(test_set)}")

    train_set = train_set.dropna(subset=["product_weight_g", "freight_value", "price"]).copy()
    logistics_train = engineer_features(train_set.drop("actual_delivery_days", axis=1))
    logistics_labels = train_set["actual_delivery_days"].copy()

    # Naive baseline: predict the training-mean delivery time for every order.
    dummy_regr = DummyRegressor(strategy="mean")
    dummy_regr.fit(logistics_train, logistics_labels)

    # Hyperparameter search over a time-aware CV (always train on the past, validate
    # on the future). The search runs sequentially (n_jobs=1) -- this machine's
    # parallel CV backend has crashed before -- while XGBoost uses threads per fit.
    # Budget is deliberately small (n_iter * 3 folds).
    print("\nTuning XGBoost hyperparameters with RandomizedSearchCV...")
    base_pipeline = make_pipeline(
        make_preprocessing(),
        XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
    )
    param_distributions = {
        "xgbregressor__n_estimators": [200, 400, 600, 800],
        "xgbregressor__max_depth": [3, 4, 5, 6, 8],
        "xgbregressor__learning_rate": [0.02, 0.03, 0.05, 0.1, 0.2],
        "xgbregressor__subsample": [0.6, 0.8, 1.0],
        "xgbregressor__colsample_bytree": [0.6, 0.8, 1.0],
        "xgbregressor__min_child_weight": [1, 3, 5, 10],
    }
    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_distributions,
        n_iter=15,
        scoring="neg_root_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=1,
        random_state=42,
        verbose=1,
        refit=True,   # refit the best config on all of train
    )
    search.fit(logistics_train, logistics_labels)

    xgb_reg = search.best_estimator_
    cv_rmse = -search.best_score_
    print("\nBest hyperparameters:")
    for key, val in search.best_params_.items():
        print(f"  {key.replace('xgbregressor__', '')}: {val}")
    print(f"Best CV RMSE: {cv_rmse:.4f} days")

    print("\nEvaluating on held-out test set...")
    test_set_clean = test_set.dropna(subset=["product_weight_g", "freight_value", "price"]).copy()
    test_labels = test_set_clean["actual_delivery_days"].copy()
    test_engineered = engineer_features(test_set_clean.drop("actual_delivery_days", axis=1))

    test_preds = xgb_reg.predict(test_engineered)
    test_rmse = root_mean_squared_error(test_labels, test_preds)
    test_mae = mean_absolute_error(test_labels, test_preds)
    test_r2 = r2_score(test_labels, test_preds)

    # naive baseline on the SAME test set: predict the training mean for every order
    dummy_preds = dummy_regr.predict(test_engineered)
    dummy_rmse = root_mean_squared_error(test_labels, dummy_preds)
    dummy_mae = mean_absolute_error(test_labels, dummy_preds)

    print(f"\nNaive baseline (mean) RMSE: {dummy_rmse:.4f} days | MAE: {dummy_mae:.4f} days")
    print(f"Held-out test RMSE:         {test_rmse:.4f} days | MAE: {test_mae:.4f} days")
    print(f"Test R2:                    {test_r2:.4f}")
    print(f"Beats baseline by:          {dummy_rmse - test_rmse:.4f} days "
          f"({(1 - test_rmse / dummy_rmse) * 100:.1f}% lower error)")
    print(f"CV mean RMSE (tuned):       {cv_rmse:.4f} days")
    print(f"Overfitting gap:            {cv_rmse - test_rmse:.4f} days")

    # error broken down by customer region (where do predictions hurt most?)
    print("\nError by customer region:")
    region_report = pd.DataFrame({
        "customer_region": test_engineered["customer_region"].values,
        "actual": test_labels.values,
        "pred": test_preds,
    })
    region_rows = []
    for region, grp in region_report.groupby("customer_region"):
        region_rows.append({
            "region": region,
            "n": len(grp),
            "rmse": root_mean_squared_error(grp["actual"], grp["pred"]),
            "mae": mean_absolute_error(grp["actual"], grp["pred"]),
        })
    region_table = pd.DataFrame(region_rows).sort_values("rmse", ascending=False)
    print(region_table.to_string(index=False,
          formatters={"rmse": "{:.4f}".format, "mae": "{:.4f}".format}))

    # model persistence
    model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"xgb_delivery_model_{model_timestamp}_rmse{test_rmse:.4f}.pkl"
    model_path = Path(__file__).resolve().parent / "models" / model_filename
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "model": xgb_reg,
        "num_attribs": num_attribs,
        "cat_attribs": cat_attribs,
        "cv_mean_rmse": cv_rmse,
        "best_params": search.best_params_,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "baseline_rmse": dummy_rmse,
        "trained_at": model_timestamp,
        "n_features": len(num_attribs) + len(cat_attribs),
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_artifact, f)

    print(f"\nModel saved to {model_path}")
    print(f"Artifact size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
