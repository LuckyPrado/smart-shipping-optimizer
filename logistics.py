from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

from pathlib import Path
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.base import clone
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
    "order_item_count",          # bigger orders plausibly ship slower
    "order_unique_sellers",      # multi-seller orders ship in parts
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


def metric_panel(y_true, preds):
    """Business-facing metric panel for one set of predictions.

    late_rate  = fraction of orders that arrive LATER than predicted (broken promise);
                 asymmetric on purpose -- under-promising loses customers, over-promising
                 is a pleasant surprise, and RMSE hides that difference.
    p90_ae     = 90th-percentile absolute error (tail behaviour RMSE only gestures at).
    median_ae  = robust central error, unaffected by the 60+ day outliers.
    """
    y_true = np.asarray(y_true, dtype=float)
    preds = np.asarray(preds, dtype=float)
    abs_err = np.abs(y_true - preds)
    return {
        "rmse": root_mean_squared_error(y_true, preds),
        "mae": mean_absolute_error(y_true, preds),
        "median_ae": float(np.median(abs_err)),
        "p90_ae": float(np.quantile(abs_err, 0.90)),
        "late_%": float(np.mean(y_true > preds)) * 100.0,
    }


def evaluate(model, dummy, eval_set, label):
    """Score the model against the naive-mean and Olist-estimate baselines."""
    y = eval_set["actual_delivery_days"].copy()
    X = engineer_features(eval_set.drop("actual_delivery_days", axis=1))

    preds = model.predict(X)
    r2 = r2_score(y, preds)

    dummy_preds = dummy.predict(X)
    # Olist's own promised delivery window as a zero-model baseline -- already a column.
    olist_preds = X["estimated_delivery_days"]

    model_panel = metric_panel(y, preds)
    panel = pd.DataFrame([
        {"predictor": "naive mean",     **metric_panel(y, dummy_preds)},
        {"predictor": "Olist estimate", **metric_panel(y, olist_preds)},
        {"predictor": "model (XGB)",    **model_panel},
    ])

    print(f"\n=== {label} ===   (R2 = {r2:.4f})")
    print(panel.to_string(index=False, formatters={
        "rmse":      "{:.2f}".format,
        "mae":       "{:.2f}".format,
        "median_ae": "{:.2f}".format,
        "p90_ae":    "{:.2f}".format,
        "late_%":    "{:.1f}".format,
    }))

    # error + late rate broken down by customer region (where do predictions hurt most?)
    print(f"\nError by customer region ({label}):")
    region_report = pd.DataFrame({
        "customer_region": X["customer_region"].values,
        "actual": y.values,
        "pred": preds,
    })
    region_rows = []
    for region, grp in region_report.groupby("customer_region"):
        region_rows.append({
            "region": region,
            "n": len(grp),
            "rmse": root_mean_squared_error(grp["actual"], grp["pred"]),
            "mae": mean_absolute_error(grp["actual"], grp["pred"]),
            "late_%": float((grp["actual"] > grp["pred"]).mean()) * 100.0,
        })
    region_table = pd.DataFrame(region_rows).sort_values("rmse", ascending=False)
    print(region_table.to_string(index=False, formatters={
        "rmse": "{:.4f}".format, "mae": "{:.4f}".format, "late_%": "{:.1f}".format}))

    return {"rmse": model_panel["rmse"], "mae": model_panel["mae"], "r2": r2,
            "median_ae": model_panel["median_ae"], "p90_ae": model_panel["p90_ae"],
            "late_%": model_panel["late_%"],
            "baseline_rmse": metric_panel(y, dummy_preds)["rmse"],
            "olist_rmse": metric_panel(y, olist_preds)["rmse"]}


def main():
    logistics = load_data()
    print("Data loaded:", logistics.shape)

    logistics["actual_delivery_days"] = (
        logistics["order_delivered_customer_date"] - logistics["order_purchase_timestamp"]
    ).dt.days
    logistics = logistics.dropna(subset=["actual_delivery_days", "order_delivered_customer_date"])

    # Time-based split: earliest 70% train, next 15% validation, latest 15% test.
    # Test stays frozen until PR 13 (see EVALUATE_ON_TEST below); all decisions use validation.
    logistics = logistics.sort_values("order_purchase_timestamp")
    n = len(logistics)
    train_set = logistics.iloc[: int(n* 0.70)]
    val_set = logistics.iloc[int(n*0.70):int(n*0.85)]
    test_set = logistics.iloc[int(n*0.85):] 
    print(f"Train: {len(train_set)} | Validation: {len(val_set)} | Test: {len(test_set)}")

    # PR 10.3: these rows used to be dropped from train AND test; now the pipeline's
    # median imputer handles them. Report how many the old dropna would have removed.
    impute_cols = ["product_weight_g", "freight_value", "price"]
    n_missing_train = train_set[impute_cols].isna().any(axis=1).sum()
    n_missing_val = val_set[impute_cols].isna().any(axis=1).sum()
    n_missing_test = test_set[impute_cols].isna().any(axis=1).sum()
    print(f"Rows with missing {impute_cols} (kept + imputed, not dropped): "
          f"train {n_missing_train} | validation {n_missing_val} | test {n_missing_test}")

    logistics_train = engineer_features(train_set.drop("actual_delivery_days", axis=1))
    logistics_labels = train_set["actual_delivery_days"].copy()

    # Naive baseline: predict the training-mean delivery time for every order.
    dummy_regr = DummyRegressor(strategy="mean")
    dummy_regr.fit(logistics_train, logistics_labels)

    # ---- PR 13.1: recency weighting ----
    # Most training data describes a slower (earlier) Olist than the evaluation window.
    # Down-weight older orders: the weight halves every HALFLIFE_DAYS going back in time.
    # sample_weight is a per-row fit argument, not an estimator hyperparameter, so it
    # cannot live in param_distributions; the half-life is chosen by an explicit sweep on
    # the VALIDATION window below (same discipline as the rest of PR 11-13).
    train_end = train_set["order_purchase_timestamp"].max()
    age_days = (train_end - train_set["order_purchase_timestamp"]).dt.days.values

    def recency_weights(halflife):
        if halflife is None:            # None == no weighting (uniform)
            return None
        return 0.5 ** (age_days / halflife)

    HALFLIFE_TUNE = 120   # weighting used while the OTHER hyperparameters are searched

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
    search.fit(logistics_train, logistics_labels,
               xgbregressor__sample_weight=recency_weights(HALFLIFE_TUNE))

    xgb_reg = search.best_estimator_
    cv_rmse = -search.best_score_
    print("\nBest hyperparameters:")
    for key, val in search.best_params_.items():
        print(f"  {key.replace('xgbregressor__', '')}: {val}")
    print(f"Best CV RMSE: {cv_rmse:.4f} days")

    # Choose the recency half-life on VALIDATION, holding the tuned hyperparameters fixed.
    # 100000 days is effectively "no decay" -- if it wins, weighting isn't helping.
    print("\nTuning recency half-life on validation (tuned hyperparameters fixed)...")
    val_labels = val_set["actual_delivery_days"]
    val_X = engineer_features(val_set.drop("actual_delivery_days", axis=1))
    best_hl, best_hl_rmse = None, float("inf")
    for hl in [60, 120, 240, None]:
        est = clone(search.best_estimator_)
        est.fit(logistics_train, logistics_labels,
                xgbregressor__sample_weight=recency_weights(hl))
        hl_rmse = root_mean_squared_error(val_labels, est.predict(val_X))
        tag = "no weighting" if hl is None else f"{hl}d"
        print(f"  half-life {tag:>12}: val RMSE {hl_rmse:.4f}")
        if hl_rmse < best_hl_rmse:
            best_hl, best_hl_rmse, xgb_reg = hl, hl_rmse, est
    print(f"Selected half-life: {'none' if best_hl is None else str(best_hl) + 'd'} "
          f"(val RMSE {best_hl_rmse:.4f})")

    EVALUATE_ON_TEST = False   # flip to True exactly once, in PR 13's final commit

    print("\nScoring on the VALIDATION window (test stays frozen)...")
    val_metrics = evaluate(xgb_reg, dummy_regr, val_set, "Validation")
    print(f"\nCV mean RMSE (tuned):       {cv_rmse:.4f} days")
    # The difference between the cv_rmse and val_rmse it is not only fit, but also the drift due to the non-stationarity property of our data.
    print(f"CV-vs-Holdout gap (drift + fit):            {cv_rmse - val_metrics['rmse']:.4f} days")

    if EVALUATE_ON_TEST:
        print("\n*** Touching the TEST set (one-shot, PR 13 final) ***")
        evaluate(xgb_reg, dummy_regr, test_set, "TEST (frozen)")
    else:
        print("\nTEST set frozen (EVALUATE_ON_TEST=False) -- not scored this run.")

    # model persistence
    model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"xgb_delivery_model_{model_timestamp}_valrmse{val_metrics['rmse']:.4f}.pkl"
    model_path = Path(__file__).resolve().parent / "models" / model_filename
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "model": xgb_reg,
        "num_attribs": num_attribs,
        "cat_attribs": cat_attribs,
        "cv_mean_rmse": cv_rmse,
        "best_params": search.best_params_,
        "val_rmse": val_metrics["rmse"],
        "val_mae": val_metrics["mae"],
        "val_r2": val_metrics["r2"],
        "baseline_rmse": val_metrics["baseline_rmse"],
        "trained_at": model_timestamp,
        "n_features": len(num_attribs) + len(cat_attribs),
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_artifact, f)

    print(f"\nModel saved to {model_path}")
    print(f"Artifact size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
