from pathlib import Path
import numpy as np
import pandas as pd


brazil_regions = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North",
    "RO": "North", "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast",
    "PB": "Northeast", "PE": "Northeast", "PI": "Northeast", "RN": "Northeast", "SE": "Northeast",
    "DF": "Central-West", "GO": "Central-West", "MT": "Central-West", "MS": "Central-West",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South"
}

category_complexity_map = {
    "moveis_escritorio": 5, "moveis_decoracao": 5, "moveis_quarto": 5,
    "moveis_sala": 5, "moveis_colchao_e_estofado": 5,
    "eletrodomesticos": 4, "eletrodomesticos_2": 4, "eletroportateis": 4,
    "informatica_acessorios": 3, "eletronicos": 3, "telefonia": 3,
    "esporte_lazer": 3, "brinquedos": 3, "ferramentas_jardim": 3,
    "beleza_saude": 2, "utilidades_domesticas": 2, "cama_mesa_banho": 2,
    "livros_tecnicos": 1, "livros_interesse_geral": 1, "musica": 1,
}

SP_LAT, SP_LNG = -23.5505, -46.6333

def haversine(lat1, lon1, lat2, lon2):
    """ Calculate distance between two point on earth"""
    R = 6378.137 # Earth radius in Km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    distance = R * 2 * np.arcsin(np.sqrt(a))
    return distance 

def load_data():
    path = Path(__file__).resolve().parent / "data"
    cache_file = path / "_cache.parquet"

    # if cache exist, use it
    if cache_file.exists():
        print("Using cached files.", flush=True)
        return pd.read_parquet(cache_file)
    
    # else build dataframe from CSVs
    print("Building DataFrame", flush=True)

    print("Loading orders...", flush=True)
    orders = pd.read_csv(path / "olist_orders_dataset.csv",engine="pyarrow")
    print("Done", flush=True)

    print("Loading order_items...", flush=True)
    order_items = pd.read_csv(path / "olist_order_items_dataset.csv",engine="pyarrow")
    print("Done", flush=True)

    print("Loading order_payments...", flush=True)
    order_payments = pd.read_csv(path / "olist_order_payments_dataset.csv",engine="pyarrow")
    print("Done", flush=True)

    print("Loading order_reviews...", flush=True) 
    order_reviews = pd.read_csv(path / "olist_order_reviews_dataset.csv",engine="pyarrow")
    print("Done", flush=True)
    
    print("Loading products...", flush=True)
    products = pd.read_csv(path / "olist_products_dataset.csv",engine="pyarrow")
    print("Done", flush=True)
    
    print("Loading sellers...", flush=True)
    sellers = pd.read_csv(path / "olist_sellers_dataset.csv",engine="pyarrow")
    print("Done", flush=True)
    
    print("Loading customers...", flush=True)
    customers = pd.read_csv(path / "olist_customers_dataset.csv",engine="pyarrow")
    print("Done", flush=True)
    
    print("Loading geo...", flush=True)
    geo = pd.read_csv(path / "olist_geolocation_dataset.csv", engine="pyarrow", usecols=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"])
    geo = geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()
    print("Done", flush=True)

    n_orders = len(orders)
    df = orders.merge(order_items, on="order_id")
    df = df.merge(products, on="product_id")
    df = df.merge(sellers, on="seller_id")
    df = df.merge(customers, on="customer_id")
    df = df.merge(geo, left_on="seller_zip_code_prefix", right_index=True)
    df = df.rename(columns={"geolocation_lat": "seller_lat", "geolocation_lng": "seller_lng"})
    df = df.merge(geo, left_on="customer_zip_code_prefix", right_index=True)
    df = df.rename(columns={"geolocation_lat": "customer_lat", "geolocation_lng": "customer_lng"})
    print(f"Orders surviving inner merges: {n_orders} -> {df['order_id'].nunique()}")
    n0 = len(df)
    df = df[df["order_status"] == "delivered"]
    print(f"Filtered delivered orders: {n0} -> {len(df)}")

    payments_agg = order_payments.groupby("order_id").agg(
        payment_installments=("payment_installments", "max"),
        payment_type=("payment_type", lambda x: x.mode()[0]),
        payment_value=("payment_value", "sum"),
        payment_sequential=("payment_sequential", "max")
    ).reset_index()
    df = df.merge(payments_agg, on="order_id", how="left")
    reviews_agg = order_reviews.groupby("order_id").agg(
        review_score=("review_score", "mean"),
        review_comment_length=("review_comment_message", lambda x: x.dropna().str.len().mean())
    ).reset_index()
    df = df.merge(reviews_agg, on="order_id", how="left")
    print(f"DataFrame after merging: {len(df)}")

    # ---- Aggregate order items to one row per order (PR 10.2) ----
    # Sort heaviest item first so "first" picks the heaviest item's category
    # (and, incidentally, that seller's geography for multi-seller orders).
    df = df.sort_values("product_weight_g", ascending=False)

    agg_spec = {
        # sum: order economics + total shipment weight
        "price": "sum",
        "freight_value": "sum",
        "product_weight_g": "sum",
        # max: the biggest box drives handling
        "product_length_cm": "max",
        "product_height_cm": "max",
        "product_width_cm": "max",
        # counts -> two new features
        "product_id": "count",     # renamed to order_item_count below
        "seller_id": "nunique",    # renamed to order_unique_sellers below
        # heaviest item's category (df is sorted by weight desc)
        "product_category_name": "first",
        # everything constant within an order
        "order_status": "first",
        "order_purchase_timestamp": "first",
        "order_approved_at": "first",
        "order_delivered_carrier_date": "first",
        "order_delivered_customer_date": "first",
        "order_estimated_delivery_date": "first",
        "shipping_limit_date": "first",
        "customer_id": "first",
        "customer_state": "first",
        "customer_zip_code_prefix": "first",
        "customer_lat": "first",
        "customer_lng": "first",
        "seller_state": "first",
        "seller_zip_code_prefix": "first",
        "seller_lat": "first",
        "seller_lng": "first",
        # payment + review columns are already order-level, so "first" just carries them
        "payment_installments": "first",
        "payment_type": "first",
        "payment_value": "first",
        "payment_sequential": "first",
        "review_score": "first",
        "review_comment_length": "first",
    }

    df = df.groupby("order_id", as_index=False).agg(agg_spec)
    df = df.rename(columns={
        "product_id": "order_item_count",
        "seller_id": "order_unique_sellers",
    })
    print(f"Aggregated to one row per order: {len(df)}")


    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_approved_at"] = pd.to_datetime(df["order_approved_at"])
    df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"])
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
    df["order_delivered_carrier_date"] = pd.to_datetime(df["order_delivered_carrier_date"])

    # save cache
    df.to_parquet(cache_file, index=False)
    
    return df

if __name__ == "__main__":
    load_data()

