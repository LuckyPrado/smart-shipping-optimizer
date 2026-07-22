"""Inference entry point for the delivery-time model.

Loads the newest portable artifact from ``models/`` (see PR 14.2: XGBoost-JSON boosters +
joblib preprocessing + JSON metadata), rebuilds the model's features, and prints the
predicted delivery time in days together with the P90 "arrives by" promise.

Usage:
    python predict.py --demo            # score 5 rows from the frozen test window
    python predict.py --csv orders.csv  # score an order-grain CSV (columns as load_data())
    python predict.py --demo --model models/xgb_delivery_model_..._valrmse6.2268

The input columns are the order-grain columns produced by ``utils.load_data`` (raw Olist
fields plus coordinates/zip prefixes); ``engineer_features`` derives the rest.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from logistics import (add_historical_aggregates, engineer_features, latest_artifact,
                       load_artifact, predict_days, predict_promise)
from utils import load_data

MODELS_DIR = Path(__file__).resolve().parent / "models"


def demo_rows(n):
    """Return n rows from the frozen test window (latest 15%), where the true delivery time
    is known, so predictions can be shown next to actuals."""
    df = load_data()
    df["actual_delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    df = (df.dropna(subset=["actual_delivery_days", "order_delivered_customer_date"])
            .sort_values("order_purchase_timestamp"))
    test = df.iloc[int(len(df) * 0.85):]
    return test.tail(n)


def read_orders_csv(path):
    """Read an order-grain CSV, coercing any date/timestamp columns to datetime so
    engineer_features can do its date arithmetic."""
    orders = pd.read_csv(path)
    for col in orders.columns:
        if "timestamp" in col or "date" in col:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")
    return orders


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="CSV of orders to score.")
    source.add_argument("--demo", action="store_true",
                        help="Score rows from the frozen test window.")
    ap.add_argument("--model", type=str, default=None,
                    help="Artifact stem or any of its files (default: newest in models/).")
    ap.add_argument("--n", type=int, default=5, help="Rows to score in --demo mode.")
    args = ap.parse_args()

    stem = args.model or latest_artifact(MODELS_DIR)
    bundle = load_artifact(stem)
    meta = bundle["meta"]
    val_rmse = meta.get("val_rmse")
    print(f"Model: {Path(stem).name}"
          + (f"  (validation RMSE {val_rmse:.4f} days)" if val_rmse is not None else ""))

    if args.demo:
        rows = demo_rows(args.n)
        actual = rows["actual_delivery_days"].to_numpy()
        orders = rows.drop("actual_delivery_days", axis=1)
    else:
        orders = read_orders_csv(args.csv)
        actual = None

    X = add_historical_aggregates(engineer_features(orders), bundle["agg_maps"])
    days = predict_days(bundle, X)
    promise = predict_promise(bundle, X)

    out = pd.DataFrame({
        "order_id": orders["order_id"].to_numpy() if "order_id" in orders
        else np.arange(len(orders)),
        "predicted_days": np.round(days, 1),
        "p90_promise_days": np.round(promise, 1),
    })
    if actual is not None:
        out["actual_days"] = actual
        out["error_days"] = np.round(days - actual, 1)
        out["kept_promise"] = actual <= promise

    print(f"\nScored {len(out)} order(s):")
    print(out.to_string(index=False))
    print(f"\nMean predicted: {days.mean():.1f} days | "
          f"mean P90 promise: {promise.mean():.1f} days")
    if actual is not None:
        print(f"Promise kept for {int((actual <= promise).sum())}/{len(actual)} orders.")


if __name__ == "__main__":
    main()
