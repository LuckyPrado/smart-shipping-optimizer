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
        review_score=("review_score", "mean"),
        review_comment_length=("review_comment_message", lambda x: x.dropna().str.len().mean())
    ).reset_index()
    df = df.merge(reviews_agg, on="order_id", how="left")
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

