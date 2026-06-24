from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def load_data():
    path = Path(__file__).resolve().parent / "data"
    orders = pd.read_csv(path / "olist_orders_dataset.csv")
    order_items = pd.read_csv(path / "olist_order_items_dataset.csv")
    order_payments = pd.read_csv(path / "olist_order_payments_dataset.csv")
    order_reviews = pd.read_csv(path / "olist_order_reviews_dataset.csv")
    products = pd.read_csv(path / "olist_products_dataset.csv")
    sellers = pd.read_csv(path / "olist_sellers_dataset.csv")
    customers = pd.read_csv(path / "olist_customers_dataset.csv")
    geo = pd.read_csv(path / "olist_geolocation_dataset.csv")
    geo = geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()
    df = orders.merge(order_items, on="order_id")
    df = df.merge(products, on="product_id")
    df = df.merge(sellers, on="seller_id")
    df = df.merge(customers, on="customer_id")
    df = df.merge(geo, left_on="seller_zip_code_prefix", right_index=True)
    df = df.rename(columns={"geolocation_lat": "seller_lat", "geolocation_lng": "seller_lng"})
    df = df.merge(geo, left_on="customer_zip_code_prefix", right_index=True)
    df = df.rename(columns={"geolocation_lat": "customer_lat", "geolocation_lng": "customer_lng"})
    payments_agg = order_payments.groupby("order_id").agg(
        payment_installments=("payment_installments", "max"),
        payment_type=("payment_type", lambda x: x.mode()[0]),
        payment_value=("payment_value", "sum")
    ).reset_index()
    df = df.merge(payments_agg, on="order_id", how="left")
    reviews_agg = order_reviews.groupby("order_id").agg(
        review_score=("review_score", "mean")
    ).reset_index()
    df = df.merge(reviews_agg, on="order_id", how="left")
    return df

logistics = load_data()

logistics["shipping_limit_days"] = (logistics["shipping_limit_date"] - logistics["order_purchase_timestamp"]).dt.days
logistics["seller_processing_window"] = (logistics["shipping_limit_date"] - logistics["order_approved_at"]).dt.total_seconds() / 3600
logistics["shipping_window_x_complexity"] = logistics["shipping_limit_days"] * logistics["category_complexity"]
logistics["log_city_density"] = np.log1p(logistics["customer_city_order_density"])
logistics["log_seller_age"] = np.log1p(logistics["seller_age_days"])
logistics["seller_review_volatility"] = logistics["seller_id"].map(logistics.groupby("seller_id")["review_score"].std()).fillna(0)
logistics["log_review_comment"] = np.log1p(logistics["review_comment_length"].fillna(0))
logistics["clv_x_complexity"] = logistics["log_clv"] * logistics["category_complexity"]
logistics["freight_burden"] = logistics["freight_ratio"].replace([np.inf, -np.inf], np.nan) * logistics["log_total_weight"]
logistics["complex_heavy_order"] = logistics["category_complexity"] * logistics["log_total_weight"]
logistics["operational_stress"] = logistics["log_approval_delay"].fillna(0) + logistics["unique_sellers_per_order"].fillna(1) + logistics["log_installments"].fillna(0)
logistics["weight_x_sellers"] = logistics["log_total_weight"] * logistics["unique_sellers_per_order"]
logistics["pressure_x_weight_sellers"] = (logistics["rolling_7d_orders"] * logistics["weight_x_sellers"]).replace([np.inf, -np.inf], np.nan)
logistics["installment_approval_lag"] = logistics["payment_installments"] * logistics["log_approval_delay"].replace([np.inf, -np.inf], np.nan)
logistics["pressure_x_operational_stress"] = (logistics["daily_order_count"] * logistics["operational_stress"]).replace([np.inf, -np.inf], np.nan)
logistics["high_complexity_installment"] = logistics["category_complexity"] * logistics["payment_installments"]
logistics["seller_high_installment_rate"] = logistics["seller_id"].map(logistics.groupby("seller_id")["payment_installments"].apply(lambda x: (x > 3).mean())).fillna(0)
logistics["payment_value_vs_order_value"] = (logistics["payment_value"] / logistics["order_total_price"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
logistics["log_catalog_age"] = np.log1p(logistics["product_catalog_age_days"])
logistics["log_order_value_ratio"] = np.log1p(logistics["price"] / logistics["seller_avg_order_value"].replace(0, np.nan))

logistics["shipping_window_x_clv"] = logistics["shipping_window_x_complexity"] * logistics["log_clv"]
logistics["processing_window_x_clv_complexity"] = logistics["seller_processing_window"] * logistics["clv_x_complexity"]
logistics["processing_window_x_complexity"] = logistics["seller_processing_window"] * logistics["category_complexity"]
logistics["city_density_x_seller_age"] = logistics["log_city_density"] * logistics["log_seller_age"]
logistics["shipping_window_x_seller_age"] = logistics["shipping_limit_days"] * logistics["log_seller_age"]
logistics["payment_complexity_x_processing"] = logistics["payment_value_vs_order_value"].replace([np.inf, -np.inf], np.nan) * logistics["seller_processing_window"]
logistics["processing_window_x_seller_age"] = logistics["seller_processing_window"] * logistics["log_seller_age"]
logistics["load_x_shipping_window"] = logistics["log_rolling_7d"] * logistics["shipping_limit_days"]
logistics["processing_window_x_weight"] = logistics["seller_processing_window"] * logistics["log_total_weight"]
logistics["shipping_window_x_operational_stress"] = logistics["shipping_limit_days"] * logistics["operational_stress"]
logistics["processing_window_x_installments"] = logistics["seller_processing_window"] * logistics["log_installments"]
logistics["city_density_x_clv"] = logistics["log_city_density"] * logistics["log_clv"]
logistics["high_installment_x_shipping_window"] = logistics["high_complexity_installment"] * logistics["shipping_limit_days"]
logistics["city_density_x_weight_sellers"] = logistics["log_city_density"] * logistics["weight_x_sellers"]
logistics["shipping_window_x_city_density"] = logistics["shipping_window_x_complexity"] * logistics["log_city_density"]
logistics["city_density_x_operational_stress"] = logistics["log_city_density"] * logistics["operational_stress"]
logistics["shipping_window_x_installment_lag"] = logistics["shipping_limit_days"] * logistics["installment_approval_lag"].replace([np.inf, -np.inf], np.nan)
logistics["daily_order_pressure"] = logistics["daily_order_count"]
logistics["rolling_7d_orders"] = logistics["rolling_7d_orders"]
logistics["log_clv"] = np.log1p(logistics["customer_lifetime_value"])
logistics["pre_holiday_flag"] = (logistics["days_to_holiday"] <= 7).astype(int)
logistics["holiday_window"] = pd.cut(
    logistics["days_to_holiday"],
    bins=[0, 3, 7, 14, 30, 60, 365],
    labels=[0, 1, 2, 3, 4, 5]
).astype(float)

logistics["seller_hub_distance"] = logistics.apply(
    lambda row: haversine(row["seller_lat"], row["seller_lng"], -23.5505, -46.6333), axis=1
)
logistics["seller_customer_state_reach"] = logistics["seller_id"].map(
    logistics.groupby("seller_id")["customer_state"].nunique()
)
logistics["seller_price_range"] = logistics["seller_id"].map(
    logistics.groupby("seller_id")["price"].apply(lambda x: x.max() - x.min())
)
logistics["photos_vs_category_avg"] = (
    logistics["product_photos_qty"] /
    logistics["product_category_name"].map(
        logistics.groupby("product_category_name")["product_photos_qty"].mean()
    ).replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan)
logistics["order_item_diversity"] = logistics["order_id"].map(
    logistics.groupby("order_id")["product_category_name"].nunique()
)
logistics["order_unique_sellers"] = logistics["order_id"].map(
    logistics.groupby("order_id")["seller_id"].nunique()
)
logistics["max_item_price"] = logistics["order_id"].map(
    logistics.groupby("order_id")["price"].max()
)
logistics["order_diversity_x_sellers"] = logistics["order_item_diversity"] * logistics["order_unique_sellers"]
logistics["avg_installment_value"] = (
    logistics["payment_value"] / logistics["payment_installments"].replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan)
logistics["weekend_purchase_x_installments"] = (
    (logistics["purchase_dayofweek"] >= 5).astype(int) * logistics["payment_installments"]
)
logistics["payment_sequential_x_complexity"] = logistics["payment_sequential"] * logistics["category_complexity"]
logistics["customer_city_seller_concentration"] = logistics["customer_city"].map(
    logistics.groupby("customer_city")["seller_state"].nunique()
).fillna(0)
logistics["shipping_window_x_seller_review"] = (
    logistics["shipping_limit_days"] * logistics["seller_avg_review"].fillna(3)
).replace([np.inf, -np.inf], np.nan)
logistics["seller_hub_distance_x_shipping_window"] = (
    logistics["seller_hub_distance"] * logistics["shipping_limit_days"]
).replace([np.inf, -np.inf], np.nan)
logistics["seller_hub_distance_x_complexity"] = (
    logistics["seller_hub_distance"] * logistics["category_complexity"]
).replace([np.inf, -np.inf], np.nan)
logistics["seller_hub_distance_x_review"] = (
    logistics["seller_hub_distance"] * logistics["seller_avg_review"].fillna(3)
).replace([np.inf, -np.inf], np.nan)
logistics["seller_hub_distance_x_seller_age"] = (
    logistics["seller_hub_distance"] * logistics["log_seller_age"]
).replace([np.inf, -np.inf], np.nan)
logistics["seller_reach_x_hub_distance"] = (
    logistics["seller_customer_state_reach"] * logistics["seller_hub_distance"]
).replace([np.inf, -np.inf], np.nan)

