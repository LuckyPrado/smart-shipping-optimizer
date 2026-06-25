import sys
from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import pickle 
from datetime import datetime
from utils import haversine, load_data, brazil_regions, category_complexity_map, SP_LAT, SP_LNG


def engineer_features(df, seller_volume_map, route_freq_map,
                      state_pair_avg_map, customer_state_avg_map,
                      seller_state_avg_map, route_variability_map,
                      seller_avg_review_map, seller_review_volatility_map,
                      seller_high_installment_rate_map, seller_avg_order_value_map,
                      seller_age_map, city_density_map, daily_order_map,
                      rolling_7d_map, product_first_sale_map,
                      seller_state_reach_map, seller_price_range_map,
                      category_avg_photos_map, city_seller_concentration_map,
                      is_train=True):
    df = df.copy()

    df["real_distance_km"] = haversine(df["seller_lat"].to_numpy(), df["seller_lng"].to_numpy(),
                              df["customer_lat"].to_numpy(), df["customer_lng"].to_numpy())
    
    df["seller_customer_lat_diff"] = abs(df["customer_lat"] - df["seller_lat"])
    df["seller_customer_lng_diff"] = abs(df["customer_lng"] - df["seller_lng"])
    df["zip_distance"] = abs(df["customer_zip_code_prefix"] - df["seller_zip_code_prefix"])
    df["customer_zip_prefix_bin"] = df["customer_zip_code_prefix"] // 10000
    df["same_state"] = (df["seller_state"] == df["customer_state"]).astype(int)
    df["state_pair"] = df["seller_state"] + "_" + df["customer_state"]
    df["state_pair_cat"] = df["state_pair"].astype("category").cat.codes
    df["seller_region"] = df["seller_state"].map(brazil_regions)
    df["customer_region"] = df["customer_state"].map(brazil_regions)
    df["seller_state_cat"] = df["seller_state"].astype("category").cat.codes
    df["customer_state_cat"] = df["customer_state"].astype("category").cat.codes
    df["north_involved"] = (
        (df["seller_region"] == "North") | (df["customer_region"] == "North")
    ).astype(int)
    df["remote_state_flag"] = (df["customer_region"] == "North").astype(int)
    df["seller_remote_flag"] = df["seller_state"].isin(
        ["AM", "RO", "PA", "AC", "RR", "AP", "TO"]
    ).astype(int)
    df["southeast_seller"] = (df["seller_region"] == "Southeast").astype(int)
    df["log_distance"] = np.log1p(df["real_distance_km"])
    df["extreme_longhaul_flag"] = (df["real_distance_km"] > 1450).astype(int)
    df["long_heavy"] = (
        (df["real_distance_km"] > 500) & (df["product_weight_g"] > 2000)
    ).astype(int)
    df["short_heavy"] = (
        (df["real_distance_km"] < 200) & (df["product_weight_g"] > 5000)
    ).astype(int)

    df["product_volume"] = (
        df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    )
    df["freight_ratio"] = df["freight_value"] / df["price"].replace(0, np.nan)
    df["freight_per_km"] = df["freight_value"] / df["real_distance_km"].replace(0, np.nan)
    df["price_per_km"] = df["price"] / df["real_distance_km"].replace(0, np.nan)
    df["freight_per_weight"] = df["freight_value"] / df["product_weight_g"].replace(0, np.nan)
    df["max_dimension_cm"] = df[["product_length_cm", "product_height_cm", "product_width_cm"]].max(axis=1)
    df["log_freight"] = np.log1p(df["freight_value"])
    df["log_weight"] = np.log1p(df["product_weight_g"])
    df["log_volume"] = np.log1p(df["product_volume"])
    df["log_total_weight"] = np.log1p(df["product_weight_g"])
    df["heavy_item_flag"] = (df["product_weight_g"] > 5000).astype(int)
    df["local_heavy"] = (
        (df["same_state"] == 1) & (df["product_weight_g"] > 5000)
    ).astype(int)
    df["ultra_cheap_freight"] = (df["freight_value"] < 10).astype(int)
    df["freight_tier"] = pd.cut(
        df["freight_value"], bins=[0, 10, 20, 50, np.inf], labels=[0, 1, 2, 3]
    ).astype(float)

    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["quarter"] = df["order_purchase_timestamp"].dt.quarter
    df["purchase_date"] = df["order_purchase_timestamp"].dt.date
    df["holiday_pressure"] = df["purchase_month"].isin([1, 2, 6, 12]).astype(int)
    df["fast_season"] = df["purchase_month"].isin([7, 8, 9]).astype(int)
    df["daily_order_count"] = df["purchase_date"].map(daily_order_map).fillna(0)
    df["rolling_7d_orders"] = df["purchase_date"].map(rolling_7d_map).fillna(0)
    df["log_rolling_7d"] = np.log1p(df["rolling_7d_orders"])

    if is_train:
        df["seller_order_volume"] = df.groupby("seller_id")["order_id"].transform("count")
        df["route_frequency"] = df.groupby("state_pair")["order_id"].transform("count")
    else:
        df["seller_order_volume"] = df["seller_id"].map(seller_volume_map).fillna(0)
        df["route_frequency"] = df["state_pair"].map(route_freq_map).fillna(0)

    df["log_distance_route_ratio"] = np.log1p(
        df["real_distance_km"] / df["route_frequency"].replace(0, np.nan)
    )
    df["rare_route_flag"] = (
        df["route_frequency"] < df["route_frequency"].quantile(0.25)
    ).astype(int)
    df["route_variability"] = df["state_pair"].map(route_variability_map).fillna(
        route_variability_map.mean()
    )
    df["state_pair_avg_days"] = df["state_pair"].map(state_pair_avg_map).fillna(
        state_pair_avg_map.mean()
    )
    df["customer_state_avg_days"] = df["customer_state"].map(customer_state_avg_map).fillna(
        customer_state_avg_map.mean()
    )
    df["seller_state_avg_days"] = df["seller_state"].map(seller_state_avg_map).fillna(
        seller_state_avg_map.mean()
    )

    df["shipping_limit_days"] = (
        df["shipping_limit_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["seller_processing_window"] = (
        df["shipping_limit_date"] - df["order_approved_at"]
    ).dt.total_seconds() / 3600
    df["payment_approval_delay"] = (
        df["order_approved_at"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600
    df["log_approval_delay"] = np.log1p(
        df["payment_approval_delay"].replace([np.inf, -np.inf], np.nan)
    )

    df["category_complexity"] = df["product_category_name"].map(
        category_complexity_map
    ).fillna(2)

    order_agg = df.groupby("order_id").agg(
        items_per_order=("order_item_id", "count"),
        unique_sellers_per_order=("seller_id", "nunique"),
        total_weight=("product_weight_g", "sum"),
        total_volume=("product_volume", "sum"),
        order_total_price=("price", "sum")
    )
    df = df.join(order_agg, on="order_id", rsuffix="_agg")
    df["log_total_order_weight"] = np.log1p(df["total_weight"])
    df["log_installments"] = np.log1p(df["payment_installments"])

    df["seller_avg_review"] = df["seller_id"].map(seller_avg_review_map).fillna(3.0)
    df["seller_review_volatility"] = df["seller_id"].map(
        seller_review_volatility_map
    ).fillna(0)
    df["seller_high_installment_rate"] = df["seller_id"].map(
        seller_high_installment_rate_map
    ).fillna(0)
    df["seller_avg_order_value"] = df["seller_id"].map(
        seller_avg_order_value_map
    ).fillna(seller_avg_order_value_map.mean())
    df["seller_age_days"] = df["seller_id"].map(seller_age_map).fillna(0)
    df["log_seller_age"] = np.log1p(df["seller_age_days"])

    df["customer_city_order_density"] = df["customer_city"].map(
        city_density_map
    ).fillna(0)
    df["log_city_density"] = np.log1p(df["customer_city_order_density"])
    df["log_clv"] = np.log1p(df["price"])

    df["payment_value_vs_order_value"] = (
        df["payment_value"] / df["order_total_price"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    df["high_complexity_installment"] = (
        df["category_complexity"] * df["payment_installments"]
    )
    df["installment_approval_lag"] = (
        df["payment_installments"] *
        df["log_approval_delay"].replace([np.inf, -np.inf], np.nan)
    )

    df["product_first_sale"] = df["product_id"].map(product_first_sale_map)
    df["product_catalog_age_days"] = (
        df["order_purchase_timestamp"] - df["product_first_sale"]
    ).dt.days
    df["log_catalog_age"] = np.log1p(df["product_catalog_age_days"].fillna(0).clip(lower=0))
    df["log_review_comment"] = np.log1p(df["review_comment_length"].fillna(0))
    df["log_order_value_ratio"] = np.log1p(
        df["price"] / df["seller_avg_order_value"].replace(0, np.nan)
    )

    df["freight_burden"] = (
        df["freight_ratio"].replace([np.inf, -np.inf], np.nan) *
        df["log_total_order_weight"]
    )
    df["complex_heavy_order"] = df["category_complexity"] * df["log_total_order_weight"]
    df["clv_x_complexity"] = df["log_clv"] * df["category_complexity"]
    df["weight_x_sellers"] = df["log_total_order_weight"] * df["unique_sellers_per_order"]
    df["operational_stress"] = (
        df["log_approval_delay"].fillna(0) +
        df["unique_sellers_per_order"].fillna(1) +
        df["log_installments"].fillna(0)
    )
    df["pressure_x_weight_sellers"] = (
        df["rolling_7d_orders"] * df["weight_x_sellers"]
    ).replace([np.inf, -np.inf], np.nan)
    df["pressure_x_operational_stress"] = (
        df["daily_order_count"] * df["operational_stress"]
    ).replace([np.inf, -np.inf], np.nan)

    df["shipping_window_x_complexity"] = (
        df["shipping_limit_days"] * df["category_complexity"]
    )
    df["shipping_window_x_clv"] = (
        df["shipping_window_x_complexity"] * df["log_clv"]
    )
    df["shipping_window_x_seller_age"] = (
        df["shipping_limit_days"] * df["log_seller_age"]
    )
    df["shipping_window_x_operational_stress"] = (
        df["shipping_limit_days"] * df["operational_stress"]
    )
    df["shipping_window_x_city_density"] = (
        df["shipping_window_x_complexity"] * df["log_city_density"]
    )
    df["shipping_window_x_installment_lag"] = (
        df["shipping_limit_days"] *
        df["installment_approval_lag"].replace([np.inf, -np.inf], np.nan)
    )
    df["high_installment_x_shipping_window"] = (
        df["high_complexity_installment"] * df["shipping_limit_days"]
    )
    df["load_x_shipping_window"] = (
        df["log_rolling_7d"] * df["shipping_limit_days"]
    )

    df["processing_window_x_complexity"] = (
        df["seller_processing_window"] * df["category_complexity"]
    ).replace([np.inf, -np.inf], np.nan)
    df["processing_window_x_clv_complexity"] = (
        df["seller_processing_window"] * df["clv_x_complexity"]
    ).replace([np.inf, -np.inf], np.nan)
    df["processing_window_x_seller_age"] = (
        df["seller_processing_window"] * df["log_seller_age"]
    ).replace([np.inf, -np.inf], np.nan)
    df["processing_window_x_weight"] = (
        df["seller_processing_window"] * df["log_total_order_weight"]
    ).replace([np.inf, -np.inf], np.nan)
    df["processing_window_x_installments"] = (
        df["seller_processing_window"] * df["log_installments"]
    ).replace([np.inf, -np.inf], np.nan)
    df["payment_complexity_x_processing"] = (
        df["payment_value_vs_order_value"].replace([np.inf, -np.inf], np.nan) *
        df["seller_processing_window"]
    ).replace([np.inf, -np.inf], np.nan)

    df["city_density_x_seller_age"] = df["log_city_density"] * df["log_seller_age"]
    df["city_density_x_clv"] = df["log_city_density"] * df["log_clv"]
    df["city_density_x_weight_sellers"] = df["log_city_density"] * df["weight_x_sellers"]
    df["city_density_x_operational_stress"] = (
        df["log_city_density"] * df["operational_stress"]
    )

    df["seller_hub_distance"] = haversine(df["seller_lat"].to_numpy(), df["seller_lng"].to_numpy(), SP_LAT, SP_LNG)
        
    df["seller_customer_state_reach"] = df["seller_id"].map(
        seller_state_reach_map).fillna(0)
    
    df["seller_price_range"] = df["seller_id"].map(seller_price_range_map).fillna(0)

    df["photos_vs_category_avg"] = (
        df["product_photos_qty"] /
        df["product_category_name"].map(
            category_avg_photos_map).replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
    )
    df["order_item_diversity"] = df["order_id"].map(
        df.groupby("order_id")["product_category_name"].nunique()
    )
    df["order_unique_sellers"] = df["order_id"].map(
        df.groupby("order_id")["seller_id"].nunique()
    )
    df["max_item_price"] = df["order_id"].map(
        df.groupby("order_id")["price"].max()
    )
    df["order_diversity_x_sellers"] = (
        df["order_item_diversity"] * df["order_unique_sellers"]
    )
    df["avg_installment_value"] = (
        df["payment_value"] / df["payment_installments"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    df["weekend_purchase_x_installments"] = (
        (df["purchase_dayofweek"] >= 5).astype(int) * df["payment_installments"]
    )
    df["payment_sequential_x_complexity"] = (
        df["payment_sequential"] * df["category_complexity"]
    )
    df["customer_city_seller_concentration"] = df["customer_city"].map(
        city_seller_concentration_map).fillna(0)
    df["shipping_window_x_seller_review"] = (
        df["shipping_limit_days"] * df["seller_avg_review"].fillna(3)
    ).replace([np.inf, -np.inf], np.nan)
    df["seller_hub_distance_x_shipping_window"] = (
        df["seller_hub_distance"] * df["shipping_limit_days"]
    ).replace([np.inf, -np.inf], np.nan)
    df["seller_hub_distance_x_complexity"] = (
        df["seller_hub_distance"] * df["category_complexity"]
    ).replace([np.inf, -np.inf], np.nan)
    df["seller_hub_distance_x_review"] = (
        df["seller_hub_distance"] * df["seller_avg_review"].fillna(3)
    ).replace([np.inf, -np.inf], np.nan)
    df["seller_hub_distance_x_seller_age"] = (
        df["seller_hub_distance"] * df["log_seller_age"]
    ).replace([np.inf, -np.inf], np.nan)
    df["seller_reach_x_hub_distance"] = (
        df["seller_customer_state_reach"] * df["seller_hub_distance"]
    ).replace([np.inf, -np.inf], np.nan)

    return df


logistics = load_data()
print("Data loaded:", logistics.shape)

logistics["estimated_delivery_days"] = (
    logistics["order_estimated_delivery_date"] - logistics["order_purchase_timestamp"]
).dt.days
logistics = logistics.dropna(subset=["estimated_delivery_days", "order_delivered_customer_date"])

train_set, test_set = train_test_split(logistics, test_size=0.2, random_state=42)
print(f"Train: {len(train_set)} | Test: {len(test_set)}")

train_set = train_set.dropna(subset=["product_weight_g", "freight_value", "price"])
train_set = train_set.copy()
train_set["estimated_delivery_days"] = (
    train_set["order_estimated_delivery_date"] - train_set["order_purchase_timestamp"]
).dt.days
train_set["state_pair"] = train_set["seller_state"] + "_" + train_set["customer_state"]
train_set["purchase_date"] = train_set["order_purchase_timestamp"].dt.date

#training maps
seller_volume_map = train_set.groupby("seller_id")["order_id"].count()
route_freq_map = train_set.groupby("state_pair")["order_id"].count()
state_pair_avg_map = train_set.groupby("state_pair")["estimated_delivery_days"].mean()
customer_state_avg_map = train_set.groupby("customer_state")["estimated_delivery_days"].mean()
seller_state_avg_map = train_set.groupby("seller_state")["estimated_delivery_days"].mean()
route_variability_map = train_set.groupby("state_pair")["estimated_delivery_days"].std()
seller_avg_review_map = train_set.groupby("seller_id")["review_score"].mean()
seller_review_volatility_map = train_set.groupby("seller_id")["review_score"].std().fillna(0)
seller_high_installment_map = train_set.groupby("seller_id")["payment_installments"].apply(
    lambda x: (x > 3).mean()
)
seller_avg_order_value_map = train_set.groupby("seller_id")["price"].mean()
seller_first_sale_map = train_set.groupby("seller_id")["order_purchase_timestamp"].min()
seller_age_map = (train_set["order_purchase_timestamp"].max() - seller_first_sale_map).dt.days
city_density_map = train_set.groupby("customer_city")["order_id"].count()
product_first_sale_map = train_set.groupby("product_id")["order_purchase_timestamp"].min()
daily_orders = train_set.groupby("purchase_date").size().reset_index(name="daily_order_count")
daily_orders["purchase_date"] = pd.to_datetime(daily_orders["purchase_date"])
daily_orders = daily_orders.sort_values("purchase_date")
daily_orders["rolling_7d_orders"] = daily_orders["daily_order_count"].rolling(7).mean()
daily_order_map = daily_orders.set_index("purchase_date")["daily_order_count"].to_dict()
rolling_7d_map = daily_orders.set_index("purchase_date")["rolling_7d_orders"].to_dict()
seller_state_reach_map = train_set.groupby("seller_id")["customer_state"].nunique()
seller_price_range_map = train_set.groupby("seller_id")["price"].apply(
    lambda x: x.max() - x.min()
)
category_avg_photos_map = train_set.groupby("product_category_name")["product_photos_qty"].mean()
city_seller_concentration_map = train_set.groupby("customer_city")["seller_state"].nunique()

#feature engineering
logistics_train = train_set.drop("estimated_delivery_days", axis=1)
logistics_labels = train_set["estimated_delivery_days"].copy()

logistics_train = engineer_features(
    logistics_train, seller_volume_map, route_freq_map,
    state_pair_avg_map, customer_state_avg_map, seller_state_avg_map,
    route_variability_map, seller_avg_review_map, seller_review_volatility_map,
    seller_high_installment_map, seller_avg_order_value_map, seller_age_map,
    city_density_map, daily_order_map, rolling_7d_map, product_first_sale_map,
    seller_state_reach_map, seller_price_range_map,
    category_avg_photos_map, city_seller_concentration_map,
    is_train=True
)

#column schema
num_attribs = [
    "real_distance_km", "log_distance", "seller_customer_lng_diff",
    "seller_customer_lat_diff", "zip_distance", "customer_zip_code_prefix",
    "customer_zip_prefix_bin", "customer_lat", "seller_zip_code_prefix",
    "same_state", "north_involved", "remote_state_flag", "seller_remote_flag",
    "southeast_seller", "extreme_longhaul_flag", "long_heavy", "short_heavy",
    "local_heavy", "heavy_item_flag", "log_distance_route_ratio",
    "freight_value", "log_freight", "freight_tier", "ultra_cheap_freight",
    "freight_ratio", "freight_per_km", "freight_per_weight",
    "product_weight_g", "log_weight", "product_volume", "log_volume",
    "max_dimension_cm", "price_per_km",
    "purchase_month", "purchase_dayofweek", "quarter",
    "holiday_pressure", "fast_season",
    "daily_order_count", "rolling_7d_orders", "log_rolling_7d",
    "route_frequency", "rare_route_flag", "route_variability",
    "state_pair_cat", "state_pair_avg_days",
    "customer_state_avg_days", "seller_state_avg_days",
    "customer_state_cat", "seller_state_cat",
    "shipping_limit_days", "seller_processing_window",
    "payment_approval_delay", "log_approval_delay",
    "category_complexity", "log_catalog_age",
    "items_per_order", "unique_sellers_per_order",
    "total_weight", "log_total_order_weight",
    "payment_installments", "log_installments",
    "payment_value_vs_order_value",
    "high_complexity_installment", "installment_approval_lag",
    "seller_avg_review", "seller_review_volatility",
    "seller_high_installment_rate", "seller_age_days", "log_seller_age",
    "log_order_value_ratio", "log_review_comment",
    "customer_city_order_density", "log_city_density", "log_clv",
    "freight_burden", "complex_heavy_order", "clv_x_complexity",
    "weight_x_sellers", "operational_stress",
    "pressure_x_weight_sellers", "pressure_x_operational_stress",
    "shipping_window_x_complexity", "shipping_window_x_clv",
    "shipping_window_x_seller_age", "shipping_window_x_operational_stress",
    "shipping_window_x_city_density", "shipping_window_x_installment_lag",
    "high_installment_x_shipping_window", "load_x_shipping_window",
    "processing_window_x_complexity", "processing_window_x_clv_complexity",
    "processing_window_x_seller_age", "processing_window_x_weight",
    "processing_window_x_installments", "payment_complexity_x_processing",
    "city_density_x_seller_age", "city_density_x_clv",
    "city_density_x_weight_sellers", "city_density_x_operational_stress",
    "seller_hub_distance", "seller_price_range", "max_item_price",
    "avg_installment_value", "weekend_purchase_x_installments",
    "payment_sequential_x_complexity", "customer_city_seller_concentration",
    "shipping_window_x_seller_review", "seller_hub_distance_x_shipping_window",
    "seller_hub_distance_x_complexity", "seller_hub_distance_x_review",
    "seller_hub_distance_x_seller_age", "seller_reach_x_hub_distance",
    "order_diversity_x_sellers", "photos_vs_category_avg",
]

cat_attribs = ["seller_state", "customer_state", "product_category_name",
               "seller_region", "customer_region"]


#preprocessing pipeline
def make_preprocessing():
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )
    return ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])


#model training
print("\nTraining XGBoost conservative baseline...")
xgb_reg = make_pipeline(make_preprocessing(), XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=1.5,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    early_stopping_rounds=None
))

xgb_reg.fit(logistics_train, logistics_labels)

#cross validation
print("Running 10-fold cross validation...")
xgb_scores = -cross_val_score(
    xgb_reg, logistics_train, logistics_labels,
    scoring="neg_root_mean_squared_error",
    cv=10,
    n_jobs=-1,
    verbose=2
)

print("\nXGBoost Conservative Baseline Results:")
print(pd.Series(xgb_scores).describe())
print(f"\nMean RMSE:   {xgb_scores.mean():.4f} days")
print(f"Std RMSE:    {xgb_scores.std():.4f} days")
print(f"Best fold:   {xgb_scores.min():.4f} days")
print(f"Worst fold:  {xgb_scores.max():.4f} days")

#test evaluation
print("\nEvaluating on held-out test set...")
test_set_clean = test_set.dropna(subset=["product_weight_g", "freight_value", "price"]).copy()
test_set_clean["estimated_delivery_days"] = (
    test_set_clean["order_estimated_delivery_date"] - test_set_clean["order_purchase_timestamp"]
).dt.days
test_labels = test_set_clean["estimated_delivery_days"].copy()

test_engineered = engineer_features(
    test_set_clean.drop("estimated_delivery_days", axis=1),
    seller_volume_map, route_freq_map,
    state_pair_avg_map, customer_state_avg_map, seller_state_avg_map,
    route_variability_map, seller_avg_review_map, seller_review_volatility_map,
    seller_high_installment_map, seller_avg_order_value_map, seller_age_map,
    city_density_map, daily_order_map, rolling_7d_map, product_first_sale_map,
    seller_state_reach_map, seller_price_range_map,
    category_avg_photos_map, city_seller_concentration_map,
    is_train=False
)

test_preds = xgb_reg.predict(test_engineered)
test_rmse = root_mean_squared_error(test_labels, test_preds)
print(f"\nHeld-out test RMSE: {test_rmse:.4f} days")
print(f"CV mean RMSE:       {xgb_scores.mean():.4f} days")
print(f"Overfitting gap:    {xgb_scores.mean() - test_rmse:.4f} days")

#model persistence
model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"xgb_delivery_model_{model_timestamp}_rmse{test_rmse:.4f}.pkl"
model_path = Path(__file__).resolve().parent / "models" / model_filename

model_path.parent.mkdir(parents=True, exist_ok=True)

model_artifact = {
    "model": xgb_reg,
    "num_attribs": num_attribs,
    "cat_attribs": cat_attribs,
    "cv_mean_rmse": xgb_scores.mean(),
    "cv_std_rmse": xgb_scores.std(),
    "test_rmse": test_rmse,
    "trained_at": model_timestamp,
    "n_features": len(num_attribs) + len(cat_attribs),
    "maps": {
        "seller_volume_map": seller_volume_map,
        "route_freq_map": route_freq_map,
        "state_pair_avg_map": state_pair_avg_map,
        "customer_state_avg_map": customer_state_avg_map,
        "seller_state_avg_map": seller_state_avg_map,
        "route_variability_map": route_variability_map,
        "seller_avg_review_map": seller_avg_review_map,
        "seller_review_volatility_map": seller_review_volatility_map,
        "seller_high_installment_map": seller_high_installment_map,
        "seller_avg_order_value_map": seller_avg_order_value_map,
        "seller_age_map": seller_age_map,
        "city_density_map": city_density_map,
        "daily_order_map": daily_order_map,
        "rolling_7d_map": rolling_7d_map,
        "product_first_sale_map": product_first_sale_map,
        "seller_state_reach_map": seller_state_reach_map,
        "seller_price_range_map": seller_price_range_map,
        "category_avg_photos_map": category_avg_photos_map,
        "city_seller_concentration_map": city_seller_concentration_map,
    }
}

with open(model_path, "wb") as f:
    pickle.dump(model_artifact, f)

print(f"\nModel saved to {model_path}")
print(f"Artifact size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")