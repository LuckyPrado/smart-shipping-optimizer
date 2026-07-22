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
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
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


# PR 13.4 historical aggregates. Order grain (PR 10.2) dropped seller_id, so the
# seller's zip prefix (~2000 values) is the granular stand-in for its track record.
HIST_KEYS = {
    "sellerzip_hist": "seller_zip_code_prefix",
    "route_hist": "state_pair",
}


def leak_safe_expanding_mean(df, key, target="actual_delivery_days",
                             purchase="order_purchase_timestamp",
                             delivered="order_delivered_customer_date"):
    """Per-row historical mean of `target` over prior same-`key` orders whose delivery
    completed strictly BEFORE this order was purchased -- i.e. only outcomes already
    knowable at checkout. Returns a Series on df.index, NaN where a row has no usable
    history. This delivered-before-purchased shift is what stops the seller/route
    track-record features from leaking the label.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, grp in df.groupby(key, sort=False):
        comp = grp[[delivered, target]].dropna().sort_values(delivered)
        if comp.empty:
            continue
        dtimes = comp[delivered].values
        cumsum = comp[target].to_numpy().cumsum()
        # count of completions delivered strictly before each order's purchase
        idx = np.searchsorted(dtimes, grp[purchase].values, side="left")
        sums = np.where(idx > 0, cumsum[np.clip(idx - 1, 0, len(cumsum) - 1)], 0.0)
        out.loc[grp.index] = np.divide(
            sums, idx, out=np.full(len(idx), np.nan), where=idx > 0)
    return out


def fit_historical_aggregates(train_set):
    """Build the leak-safe training columns AND the serving-time lookup maps.

    Training rows get an expanding (past-completed-only) mean per key; validation/test
    rows later look up a single training-window mean per key -- all of which is
    chronologically in their past, so no leak. Returns (train_cols_df, maps) where maps
    holds the per-key mean Series plus the median used to fill unseen keys.
    """
    keyed = train_set.copy()
    keyed["state_pair"] = keyed["seller_state"] + "_" + keyed["customer_state"]
    fill = keyed["actual_delivery_days"].median()
    cols, maps = {}, {"fill": fill}
    for name, key in HIST_KEYS.items():
        cols[name] = leak_safe_expanding_mean(keyed, key).fillna(fill)
        maps[name] = keyed.groupby(key)["actual_delivery_days"].mean()
    return pd.DataFrame(cols, index=train_set.index), maps


def add_historical_aggregates(X, maps):
    """Attach serving-time aggregate columns to an engineered validation/test frame:
    map each key to its training-window mean, median-fill unseen keys. X must carry
    state_pair (from engineer_features) and the raw seller_zip_code_prefix column.
    """
    X = X.copy()
    for name, key in HIST_KEYS.items():
        X[name] = X[key].map(maps[name]).fillna(maps["fill"])
    return X


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
    # PR 13.4 leak-safe historical track records (built in fit_historical_aggregates,
    # not engineer_features -- they need the training labels and a fitted lookup map).
    "sellerzip_hist",            # avg past delivery time from this seller's zip prefix
    "route_hist",                # avg past delivery time on this seller->customer route
]

# PR 13.5: only product_category_name is one-hot encoded now. seller_state and
# customer_state used to be one-hot AND target-encoded -- the one-hot copy is dropped
# (target encoding below keeps them, and state_pair covers the interaction). Regions
# were dropped earlier as redundant with state.
cat_attribs = ["product_category_name"]


def make_preprocessing():
    # No StandardScaler: trees are scale-invariant, so it was a no-op. Median imputation
    # is still needed to fill the occasional missing numeric input.
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"))
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


def evaluate(model, dummy, eval_set, label, agg_maps):
    """Score the model against the naive-mean and Olist-estimate baselines."""
    y = eval_set["actual_delivery_days"].copy()
    X = engineer_features(eval_set.drop("actual_delivery_days", axis=1))
    X = add_historical_aggregates(X, agg_maps)   # PR 13.4 serving-time lookups

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


def promise_report(quantile_model, X, y, label):
    """Coverage + average promised days for the PR 13.6 P90 delivery-promise model.

    coverage = fraction of orders that arrive on or under the promised day (the number
    a customer-facing "arrives by" promise is actually judged on -- should sit near the
    90% the model was trained to target). X must already carry the engineered columns
    and the serving-time historical aggregates.
    """
    p = quantile_model.predict(X)
    y = np.asarray(y, dtype=float)
    coverage = float(np.mean(y <= p)) * 100.0
    print(f"\nP90 delivery promise ({label}):")
    print(f"  Empirical coverage: {coverage:.1f}%  (target 90% arrive on/under promise)")
    print(f"  Avg promised days: {np.mean(p):.1f} | "
          f"Olist estimate: {X['estimated_delivery_days'].mean():.1f} | "
          f"actual: {y.mean():.1f}")
    return coverage


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

    # ---- PR 13.4: leak-safe seller/route historical aggregates ----
    # Training rows get an expanding past-only mean; the fitted maps serve val/test.
    hist_cols, agg_maps = fit_historical_aggregates(train_set)
    for col in hist_cols.columns:
        logistics_train[col] = hist_cols[col]

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
    # PR 13.3: delivery times are right-skewed with a long tail (60+ day outliers),
    # so RMSE on the raw target is dominated by the tail. Train on log1p(days) and
    # invert predictions with expm1 via TransformedTargetRegressor -- a standard fix
    # that lets the model spend capacity on the common short deliveries. The wrapper
    # does the transform/inverse itself, so CV scoring and every downstream metric
    # stay in day-units. sample_weight still routes as xgbregressor__sample_weight
    # (forwarded through the wrapper into the pipeline step); only grid keys gain the
    # regressor__ prefix.
    base_pipeline = TransformedTargetRegressor(
        regressor=make_pipeline(
            make_preprocessing(),
            XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    param_distributions = {
        "regressor__xgbregressor__n_estimators": [200, 400, 600, 800],
        "regressor__xgbregressor__max_depth": [3, 4, 5, 6, 8],
        "regressor__xgbregressor__learning_rate": [0.02, 0.03, 0.05, 0.1, 0.2],
        "regressor__xgbregressor__subsample": [0.6, 0.8, 1.0],
        "regressor__xgbregressor__colsample_bytree": [0.6, 0.8, 1.0],
        "regressor__xgbregressor__min_child_weight": [1, 3, 5, 10],
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
        print(f"  {key.replace('regressor__xgbregressor__', '')}: {val}")
    print(f"Best CV RMSE: {cv_rmse:.4f} days")

    # Choose the recency half-life on VALIDATION, holding the tuned hyperparameters fixed.
    # 100000 days is effectively "no decay" -- if it wins, weighting isn't helping.
    print("\nTuning recency half-life on validation (tuned hyperparameters fixed)...")
    val_labels = val_set["actual_delivery_days"]
    val_X = engineer_features(val_set.drop("actual_delivery_days", axis=1))
    val_X = add_historical_aggregates(val_X, agg_maps)   # PR 13.4 serving-time lookups
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

    # ---- PR 13.6: P90 delivery-promise model ----
    # A point estimate of the mean is the wrong product for a shipping promise -- by
    # construction ~half of orders arrive after it. Train a SECOND model on the 0.9
    # pinball loss so its prediction is a day X that ~90% of orders beat: an honest
    # "arrives by" promise. Quantiles pass through the monotone log1p/expm1 transform
    # unchanged, so the 0.9-quantile in log space is the 0.9-quantile in days -- we can
    # reuse the same TTR wrapper, the tuned hyperparameters, and the selected recency
    # weights (no separate search). This is the number you'd actually show a customer.
    print("\nTraining P90 delivery-promise model (reg:quantileerror, alpha=0.9)...")
    tuned_params = {k.replace("regressor__xgbregressor__", ""): v
                    for k, v in search.best_params_.items()}
    p90_model = TransformedTargetRegressor(
        regressor=make_pipeline(
            make_preprocessing(),
            XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.9,
                         tree_method="hist", random_state=42, n_jobs=-1, **tuned_params),
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    p90_model.fit(logistics_train, logistics_labels,
                  xgbregressor__sample_weight=recency_weights(best_hl))
    p90_coverage = promise_report(p90_model, val_X, val_labels, "Validation")

    EVALUATE_ON_TEST = False   # flip to True exactly once, in PR 13's final commit

    print("\nScoring on the VALIDATION window (test stays frozen)...")
    val_metrics = evaluate(xgb_reg, dummy_regr, val_set, "Validation", agg_maps)
    print(f"\nCV mean RMSE (tuned):       {cv_rmse:.4f} days")
    # The difference between the cv_rmse and val_rmse it is not only fit, but also the drift due to the non-stationarity property of our data.
    print(f"CV-vs-Holdout gap (drift + fit):            {cv_rmse - val_metrics['rmse']:.4f} days")

    if EVALUATE_ON_TEST:
        print("\n*** Touching the TEST set (one-shot, PR 13 final) ***")
        evaluate(xgb_reg, dummy_regr, test_set, "TEST (frozen)", agg_maps)
    else:
        print("\nTEST set frozen (EVALUATE_ON_TEST=False) -- not scored this run.")

    # model persistence
    model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"xgb_delivery_model_{model_timestamp}_valrmse{val_metrics['rmse']:.4f}.pkl"
    model_path = Path(__file__).resolve().parent / "models" / model_filename
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "model": xgb_reg,
        "quantile_model": p90_model,   # PR 13.6 P90 "arrives by" promise model
        "quantile_alpha": 0.9,
        "val_p90_coverage": p90_coverage,
        "agg_maps": agg_maps,   # PR 13.4 serving-time lookups (needed to rebuild features)
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
