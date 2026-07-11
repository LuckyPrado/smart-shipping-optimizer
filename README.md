# Smart Shipping Optimizer

Predicts order **delivery time** (in days) for a Brazilian e-commerce marketplace, using
the public [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
It joins orders, items, products, sellers, customers, payments, reviews, and geolocation
data, then trains an XGBoost regression model — evaluated honestly on a **time-based**
held-out test set, with no data leakage.

## Results

| Model | Held-out test RMSE | R² |
|-------|--------------------|-----|
| Naive baseline (predict the training mean) | 7.47 days | — |
| **XGBoost (tuned, 15 features)** | **5.47 days** | **+0.11** |

The model beats the naive baseline by ~27%. R² is modest because delivery times drift over
time (Olist got faster from 2016→2018) — a genuine non-stationarity in the data, not a bug.
Every feature and hyperparameter decision was validated by measuring on the time-based test
set; the full record — including ~20 candidate features that were tested and **rejected** as
redundant — is in `analysis/feature_engineering.ipynb`.

### Key modeling decisions
- **Target:** actual delivery time (`order_delivered_customer_date − order_purchase_timestamp`),
  *not* the platform's own estimate (which is what an earlier version leaked).
- **Split:** time-based (earliest 80% train, latest 20% test) so the model is judged on
  forecasting the future, never on peeking at it. CV uses `TimeSeriesSplit`.
- **Features:** a compact, proven set of 15 — Olist's promised delivery window, purchase-month
  seasonality, raw coordinates + zip prefixes, and order economics/size. Distance, region,
  order-size, category, and seller-behaviour features were all measured and dropped as redundant.
- **Encoding:** state-level target encoding is refit *inside* the CV pipeline (per fold) so it
  can't leak.
- **Tuning:** `RandomizedSearchCV` over a `TimeSeriesSplit`, not hardcoded guesses.

## Project layout

| Path | Purpose |
|------|---------|
| `logistics.py` | **Main pipeline** — loads data, engineers features, tunes + trains the XGBoost model, evaluates it (time-aware CV + held-out test), and saves the model to `models/`. |
| `utils.py` | Shared helpers — cached `load_data()`, vectorized `haversine()`, region/category lookups. |
| `analysis/feature_engineering.ipynb` | Feature-engineering lab notebook, incl. the experiment log of what was tried and rejected. |
| `analysis/correlation.py` | Exploratory analysis — correlations, plots, shipping-corridor network graph, PCA, residuals. |
| `analysis/featurevisual.py` | Feature screening — ranks candidate features by correlation with delivery time. |
| `experiments/` | Scratch / experimental work (not part of the pipeline). |
| `data/` | Olist CSVs (not tracked in git). |
| `models/` | Saved model artifacts (not tracked in git). |

## Requirements

- Python 3.12
- Dependencies listed in `requirements.txt`

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Data

The Olist CSV files are expected in the `data/` folder:

```
data/
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_products_dataset.csv
├── olist_sellers_dataset.csv
├── olist_customers_dataset.csv
├── olist_geolocation_dataset.csv
└── product_category_name_translation.csv
```

If they are missing, download the dataset from
[Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place the CSVs
in `data/`. The first run builds a cached `data/_cache.parquet` to speed up later runs
(delete it to force a rebuild).

## Usage

Train, tune, and evaluate the model (the main entry point):

```bash
python logistics.py
```

This runs the hyperparameter search, prints the chosen params, cross-validation RMSE, and
held-out test RMSE/MAE/R² next to the naive baseline, and saves a timestamped model artifact
to `models/`.

Run the exploratory analysis (opens matplotlib plots):

```bash
python analysis/correlation.py
python analysis/featurevisual.py
```
