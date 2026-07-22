"""Regenerate the two figures embedded in the README.

  docs/delivery_drift.png      -- monthly mean delivery time across 2016->2018, with the
                                  train/validation/test windows shaded. This downward drift
                                  is what recency weighting (PR 13.1) exists to counter.
  docs/predicted_vs_actual.png -- the shipped model's predictions vs. the truth on the
                                  frozen test window, using the newest models/ artifact.

Run:  python analysis/readme_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")   # headless: write PNGs, never open a window

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error

from logistics import (add_historical_aggregates, engineer_features, latest_artifact,
                       load_artifact, predict_days)
from utils import load_data

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)
TARGET = "actual_delivery_days"

plt.style.use("ggplot")

df = load_data()
df[TARGET] = (df["order_delivered_customer_date"]
              - df["order_purchase_timestamp"]).dt.days
df = (df.dropna(subset=[TARGET, "order_delivered_customer_date"])
        .sort_values("order_purchase_timestamp"))
n = len(df)
train_end = df["order_purchase_timestamp"].iloc[int(n * 0.70)]
val_end = df["order_purchase_timestamp"].iloc[int(n * 0.85)]


def _midpoint(start, end):
    return start + (end - start) / 2


# ---- Figure 1: the drift that motivates recency weighting ----
monthly = df.set_index("order_purchase_timestamp")[TARGET].resample("ME").mean()
start, end = monthly.index.min(), monthly.index.max()

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(monthly.index, monthly.values, marker="o", color="#c0392b", linewidth=2)
ax.axvspan(start, train_end, alpha=0.06, color="green")
ax.axvspan(train_end, val_end, alpha=0.10, color="orange")
ax.axvspan(val_end, end, alpha=0.12, color="red")
ax.axvline(train_end, color="gray", linestyle="--", linewidth=1)
ax.axvline(val_end, color="gray", linestyle="--", linewidth=1)

y_lab = np.nanmax(monthly.values)
for label, a, b in [("train", start, train_end),
                    ("validation", train_end, val_end),
                    ("test", val_end, end)]:
    ax.text(_midpoint(a, b), y_lab, label, ha="center", va="top",
            fontsize=9, color="#444")

ax.set_title("Delivery times drift downward over time — why recent orders are weighted more")
ax.set_xlabel("Order purchase month")
ax.set_ylabel("Mean delivery days")
fig.tight_layout()
fig.savefig(DOCS / "delivery_drift.png", dpi=120)
plt.close(fig)

# ---- Figure 2: predicted vs. actual on the frozen test window ----
bundle = load_artifact(latest_artifact(ROOT / "models"))
test = df.iloc[int(n * 0.85):]
X = add_historical_aggregates(
    engineer_features(test.drop(TARGET, axis=1)), bundle["agg_maps"])
pred = predict_days(bundle, X)
actual = test[TARGET].to_numpy()
rmse = root_mean_squared_error(actual, pred)

lim = 45
fig, ax = plt.subplots(figsize=(7, 6))
hb = ax.hexbin(actual, pred, gridsize=45, cmap="viridis", bins="log",
               extent=(0, lim, 0, lim))
ax.plot([0, lim], [0, lim], "r--", linewidth=2, label="perfect prediction")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("Actual delivery days")
ax.set_ylabel("Predicted delivery days")
ax.set_title("Predicted vs. actual — frozen test window", fontsize=12)
ax.text(0.96, 0.06, f"RMSE {rmse:.2f} days", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85})
ax.legend(loc="upper left")
fig.colorbar(hb, ax=ax, label="orders (log scale)")
fig.tight_layout()
fig.savefig(DOCS / "predicted_vs_actual.png", dpi=120)
plt.close(fig)

print(f"Saved {DOCS / 'delivery_drift.png'}")
print(f"Saved {DOCS / 'predicted_vs_actual.png'}  (test RMSE {rmse:.2f})")
