from pathlib import Path
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2


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
        payment_value=("payment_value", "sum"),
        payment_sequential=("payment_sequential", "max")
    ).reset_index()
    df = df.merge(payments_agg, on="order_id", how="left")
    reviews_agg = order_reviews.groupby("order_id").agg(
        review_score=("review_score", "mean")
    ).reset_index()
    df = df.merge(reviews_agg, on="order_id", how="left")
    return df


logistics = load_data()

logistics["order_purchase_timestamp"] = pd.to_datetime(logistics["order_purchase_timestamp"])
logistics["order_estimated_delivery_date"] = pd.to_datetime(logistics["order_estimated_delivery_date"])
logistics["order_approved_at"] = pd.to_datetime(logistics["order_approved_at"])
logistics["shipping_limit_date"] = pd.to_datetime(logistics["shipping_limit_date"])

logistics["estimated_delivery_days"] = (
    logistics["order_estimated_delivery_date"] - logistics["order_purchase_timestamp"]
).dt.days
logistics = logistics.dropna(subset=["estimated_delivery_days"])

logistics["product_volume"] = (
    logistics["product_length_cm"] *
    logistics["product_height_cm"] *
    logistics["product_width_cm"]
)
logistics["purchase_hour"] = logistics["order_purchase_timestamp"].dt.hour
logistics["purchase_dayofweek"] = logistics["order_purchase_timestamp"].dt.dayofweek
logistics["purchase_month"] = logistics["order_purchase_timestamp"].dt.month
logistics["purchase_week_of_year"] = logistics["order_purchase_timestamp"].dt.isocalendar().week.astype(int)
logistics["purchase_quarter"] = logistics["order_purchase_timestamp"].dt.quarter
logistics["purchase_date"] = logistics["order_purchase_timestamp"].dt.date

logistics["payment_approval_delay"] = (
    logistics["order_approved_at"] - logistics["order_purchase_timestamp"]
).dt.total_seconds() / 3600
logistics["shipping_limit_days"] = (
    logistics["shipping_limit_date"] - logistics["order_purchase_timestamp"]
).dt.days
logistics["seller_processing_window"] = (
    logistics["shipping_limit_date"] - logistics["order_approved_at"]
).dt.total_seconds() / 3600

category_complexity = {
    "moveis_escritorio": 5, "moveis_decoracao": 5, "moveis_quarto": 5,
    "moveis_sala": 5, "moveis_colchao_e_estofado": 5,
    "eletrodomesticos": 4, "eletrodomesticos_2": 4, "eletroportateis": 4,
    "informatica_acessorios": 3, "eletronicos": 3, "telefonia": 3,
    "esporte_lazer": 3, "brinquedos": 3, "ferramentas_jardim": 3,
    "beleza_saude": 2, "utilidades_domesticas": 2, "cama_mesa_banho": 2,
    "livros_tecnicos": 1, "livros_interesse_geral": 1, "musica": 1,
}
logistics["category_complexity"] = logistics["product_category_name"].map(category_complexity).fillna(2)

SP_LAT, SP_LNG = -23.5505, -46.6333
logistics["real_distance_km"] = logistics.apply(
    lambda row: haversine(row["seller_lat"], row["seller_lng"],
                          row["customer_lat"], row["customer_lng"]), axis=1
)
logistics["seller_hub_distance"] = logistics.apply(
    lambda row: haversine(row["seller_lat"], row["seller_lng"], SP_LAT, SP_LNG), axis=1
)

seller_avg_review = logistics.groupby("seller_id")["review_score"].mean()
logistics["seller_avg_review"] = logistics["seller_id"].map(seller_avg_review).fillna(3.0)

seller_tenure = logistics.groupby("seller_id")["order_purchase_timestamp"].min()
logistics["seller_age_days"] = (
    logistics["order_purchase_timestamp"] - logistics["seller_id"].map(seller_tenure)
).dt.days
logistics["log_seller_age"] = np.log1p(logistics["seller_age_days"])

PLATFORM_LAUNCH = pd.Timestamp("2016-09-01")
brazil_capitals = [
    "sao paulo", "rio de janeiro", "brasilia", "salvador", "fortaleza",
    "belo horizonte", "manaus", "curitiba", "recife", "porto alegre",
    "belem", "goiania", "sao luis", "maceio", "natal", "teresina",
    "campo grande", "joao pessoa", "aracaju", "porto velho", "macapa",
    "florianopolis", "vitoria", "cuiaba", "palmas", "rio branco", "boa vista"
]

neighboring_states = {
    "SP": ["RJ", "MG", "PR", "MS", "GO"],
    "RJ": ["SP", "MG", "ES"],
    "MG": ["SP", "RJ", "ES", "BA", "GO", "MS", "MT", "DF"],
    "RS": ["SC", "PR"],
    "SC": ["RS", "PR"],
    "PR": ["SC", "RS", "SP", "MS"],
    "BA": ["SE", "AL", "PE", "PI", "TO", "GO", "MG", "ES"],
    "GO": ["MT", "MS", "MG", "BA", "TO", "DF"],
    "AM": ["RR", "PA", "MT", "RO", "AC"],
    "PA": ["AM", "RR", "AP", "MA", "TO", "MT"],
}

print("Engineering features...")

logistics["seller_zip_prefix_bin"] = logistics["seller_zip_code_prefix"] // 10000

logistics["seller_zip_to_customer_zip_ratio"] = (
    logistics["seller_zip_code_prefix"] /
    logistics["customer_zip_code_prefix"].replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan)

logistics["seller_in_capital_city"] = logistics["seller_city"].str.lower().isin(brazil_capitals).astype(int)
logistics["customer_in_capital_city"] = logistics["customer_city"].str.lower().isin(brazil_capitals).astype(int)

logistics["days_until_end_of_month"] = (
    logistics["order_purchase_timestamp"].dt.days_in_month -
    logistics["order_purchase_timestamp"].dt.day
)
logistics["purchase_week_of_year"] = logistics["purchase_week_of_year"]
logistics["purchase_quarter_x_dayofweek"] = logistics["purchase_quarter"] * logistics["purchase_dayofweek"]
logistics["days_since_platform_launch"] = (
    logistics["order_purchase_timestamp"] - PLATFORM_LAUNCH
).dt.days
logistics["purchase_hour_bucket"] = pd.cut(
    logistics["purchase_hour"],
    bins=[-1, 8, 12, 18, 24],
    labels=[0, 1, 2, 3]
).astype(float)

logistics["product_length_to_height_ratio"] = (
    logistics["product_length_cm"] /
    logistics["product_height_cm"].replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan)
logistics["product_weight_to_volume_ratio"] = (
    logistics["product_weight_g"] /
    logistics["product_volume"].replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan)
logistics["product_max_to_min_dimension_ratio"] = (
    logistics[["product_length_cm", "product_height_cm", "product_width_cm"]].max(axis=1) /
    logistics[["product_length_cm", "product_height_cm", "product_width_cm"]].min(axis=1).replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan)
logistics["product_surface_area"] = 2 * (
    logistics["product_length_cm"] * logistics["product_height_cm"] +
    logistics["product_length_cm"] * logistics["product_width_cm"] +
    logistics["product_height_cm"] * logistics["product_width_cm"]
)
logistics["is_oversized"] = (
    logistics[["product_length_cm", "product_height_cm", "product_width_cm"]].max(axis=1) > 100
).astype(int)

logistics["payment_value_per_item"] = (
    logistics["payment_value"] /
    logistics["order_id"].map(logistics.groupby("order_id")["order_item_id"].count()).replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan)
logistics["is_single_payment"] = (logistics["payment_installments"] == 1).astype(int)
logistics["payment_value_x_installments"] = logistics["payment_value"] * logistics["payment_installments"]
logistics["installment_coverage_ratio"] = (
    logistics["payment_installments"] /
    pd.cut(logistics["price"], bins=5, labels=[1, 2, 3, 4, 5]).astype(float)
).replace([np.inf, -np.inf], np.nan)

seller_weekend_ratio = logistics.groupby("seller_id").apply(
    lambda x: (x["purchase_dayofweek"] >= 5).mean()
)
logistics["seller_weekend_order_ratio"] = logistics["seller_id"].map(seller_weekend_ratio).fillna(0)

seller_night_ratio = logistics.groupby("seller_id").apply(
    lambda x: ((x["purchase_hour"] >= 0) & (x["purchase_hour"] < 8)).mean()
)
logistics["seller_night_order_ratio"] = logistics["seller_id"].map(seller_night_ratio).fillna(0)

seller_approval_hour = logistics.groupby("seller_id")["order_approved_at"].apply(
    lambda x: pd.to_datetime(x).dt.hour.mean()
)
logistics["seller_avg_approval_hour"] = logistics["seller_id"].map(seller_approval_hour).fillna(12)

logistics["customer_zip_prefix_x_seller_state_cat"] = (
    logistics["customer_zip_code_prefix"] // 10000 *
    logistics["seller_state"].astype("category").cat.codes
)

logistics["category_x_purchase_month"] = (
    logistics["category_complexity"] * logistics["purchase_month"]
)
logistics["category_x_seller_region"] = (
    logistics["category_complexity"] *
    logistics["seller_state"].astype("category").cat.codes
)
logistics["category_x_customer_region"] = (
    logistics["category_complexity"] *
    logistics["customer_state"].astype("category").cat.codes
)

order_agg = logistics.groupby("order_id").agg(
    order_total_volume=("product_volume", "sum"),
    order_max_dimension=("product_length_cm", "max"),
    order_item_count=("order_item_id", "count"),
    order_unique_sellers=("seller_id", "nunique"),
    order_total_price=("price", "sum")
).reset_index()
logistics = logistics.merge(order_agg, on="order_id", how="left")
logistics["items_per_seller"] = (
    logistics["order_item_count"] /
    logistics["order_unique_sellers"].replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan)
logistics["order_price_x_unique_sellers"] = logistics["order_total_price"] * logistics["order_unique_sellers"]
logistics["order_total_price_x_installments"] = logistics["order_total_price"] * logistics["payment_installments"]

state_pair_freq = logistics.groupby(["seller_state", "customer_state"]).size()
logistics["seller_customer_state_pair_frequency"] = logistics.set_index(
    ["seller_state", "customer_state"]
).index.map(state_pair_freq).values

logistics["is_same_city"] = (
    logistics["seller_city"].str.lower() == logistics["customer_city"].str.lower()
).astype(int)

logistics["is_neighboring_state"] = logistics.apply(
    lambda row: int(row["customer_state"] in neighboring_states.get(row["seller_state"], [])), axis=1
)

logistics["seller_to_customer_direction"] = np.sign(
    logistics["customer_lat"] - logistics["seller_lat"]
) * 2 + np.sign(logistics["customer_lng"] - logistics["seller_lng"])

logistics["log_surface_area"] = np.log1p(logistics["product_surface_area"])
logistics["log_days_since_launch"] = np.log1p(logistics["days_since_platform_launch"])
logistics["log_payment_value_x_installments"] = np.log1p(logistics["payment_value_x_installments"])
logistics["log_order_total_volume"] = np.log1p(logistics["order_total_volume"])

features = {
    "seller_zip_prefix_bin": logistics["seller_zip_prefix_bin"],
    "seller_zip_to_customer_zip_ratio": logistics["seller_zip_to_customer_zip_ratio"],
    "seller_in_capital_city": logistics["seller_in_capital_city"],
    "customer_in_capital_city": logistics["customer_in_capital_city"],
    "days_until_end_of_month": logistics["days_until_end_of_month"],
    "purchase_week_of_year": logistics["purchase_week_of_year"],
    "purchase_quarter_x_dayofweek": logistics["purchase_quarter_x_dayofweek"],
    "days_since_platform_launch": logistics["days_since_platform_launch"],
    "log_days_since_launch": logistics["log_days_since_launch"],
    "purchase_hour_bucket": logistics["purchase_hour_bucket"],
    "product_length_to_height_ratio": logistics["product_length_to_height_ratio"],
    "product_weight_to_volume_ratio": logistics["product_weight_to_volume_ratio"],
    "product_max_to_min_dimension_ratio": logistics["product_max_to_min_dimension_ratio"],
    "product_surface_area": logistics["product_surface_area"],
    "log_surface_area": logistics["log_surface_area"],
    "is_oversized": logistics["is_oversized"],
    "payment_value_per_item": logistics["payment_value_per_item"],
    "is_single_payment": logistics["is_single_payment"],
    "payment_value_x_installments": logistics["payment_value_x_installments"],
    "log_payment_value_x_installments": logistics["log_payment_value_x_installments"],
    "installment_coverage_ratio": logistics["installment_coverage_ratio"],
    "seller_weekend_order_ratio": logistics["seller_weekend_order_ratio"],
    "seller_night_order_ratio": logistics["seller_night_order_ratio"],
    "seller_avg_approval_hour": logistics["seller_avg_approval_hour"],
    "customer_zip_prefix_x_seller_state_cat": logistics["customer_zip_prefix_x_seller_state_cat"],
    "category_x_purchase_month": logistics["category_x_purchase_month"],
    "category_x_seller_region": logistics["category_x_seller_region"],
    "category_x_customer_region": logistics["category_x_customer_region"],
    "order_total_volume": logistics["order_total_volume"],
    "log_order_total_volume": logistics["log_order_total_volume"],
    "order_max_dimension": logistics["order_max_dimension"],
    "items_per_seller": logistics["items_per_seller"],
    "order_price_x_unique_sellers": logistics["order_price_x_unique_sellers"],
    "order_total_price_x_installments": logistics["order_total_price_x_installments"],
    "log_order_total_price_x_installments": logistics["log_payment_value_x_installments"],
    "seller_customer_state_pair_frequency": logistics["seller_customer_state_pair_frequency"],
    "is_same_city": logistics["is_same_city"],
    "is_neighboring_state": logistics["is_neighboring_state"],
    "seller_to_customer_direction": logistics["seller_to_customer_direction"],
}

print("\n" + "=" * 60)
print("CORRELATION SUMMARY")
print("=" * 60)
print(f"\n{'Feature':<45} {'Pearson r':>10} {'Keep?':>8}")
print("-" * 65)

results = {}
for feat, series in features.items():
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    corr = clean.corr(logistics.loc[clean.index, "estimated_delivery_days"])
    results[feat] = corr
    keep = "YES" if abs(corr) > 0.03 else "WEAK"
    print(f"{feat:<45} {corr:>10.4f} {keep:>8}")

print("\n" + "=" * 60)
print("QUINTILE BREAKDOWN — FEATURES ABOVE 0.03")
print("=" * 60)

for feat, corr in sorted(results.items(), key=lambda x: abs(x[1]), reverse=True):
    if abs(corr) < 0.03:
        continue
    clean = logistics[[feat, "estimated_delivery_days"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 100:
        continue
    try:
        bins = pd.qcut(clean[feat], 5, duplicates="drop")
        quintile_stats = clean.groupby(bins, observed=True)["estimated_delivery_days"].agg(["mean", "std", "count"])
        spread = quintile_stats["mean"].max() - quintile_stats["mean"].min()
        print(f"\n=== {feat} (r={corr:.4f}, spread={spread:.2f} days) ===")
        print(quintile_stats.to_string())
    except Exception:
        flag_stats = logistics.groupby(feat)["estimated_delivery_days"].agg(["mean", "std", "count"])
        spread = flag_stats["mean"].max() - flag_stats["mean"].min()
        print(f"\n=== {feat} (r={corr:.4f}, spread={spread:.2f} days) ===")
        print(flag_stats.to_string())

print("\n" + "=" * 60)
print("FINAL RECOMMENDATION")
print("=" * 60)
print(f"\n{'Feature':<45} {'Pearson r':>10} {'Spread':>8} {'Add?':>6}")
print("-" * 71)

for feat, corr in sorted(results.items(), key=lambda x: abs(x[1]), reverse=True):
    clean = logistics[[feat, "estimated_delivery_days"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 100:
        spread = 0.0
    else:
        try:
            bins = pd.qcut(clean[feat], 5, duplicates="drop")
            quintile_means = clean.groupby(bins, observed=True)["estimated_delivery_days"].mean()
            spread = quintile_means.max() - quintile_means.min()
        except Exception:
            flag_means = logistics.groupby(feat)["estimated_delivery_days"].mean()
            spread = flag_means.max() - flag_means.min()
    add = "YES" if abs(corr) > 0.03 and spread > 1.0 else "NO"
    print(f"{feat:<45} {corr:>10.4f} {spread:>8.2f} {add:>6}")