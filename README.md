# Smart Shipping Optimizer

Predicts order **delivery time** (in days) for a Brazilian e-commerce marketplace, using
the public [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
The project joins orders, items, products, sellers, customers, payments, reviews, and
geolocation data, engineers a rich set of logistics features (distance, route frequency,
seller behavior, seasonality, etc.), and trains an XGBoost regression model.

## Project layout

| File | Purpose |
|------|---------|
| `logistics.py` | **Main pipeline** — loads data, engineers features, trains the XGBoost model, evaluates it (cross-validation + held-out test RMSE), and saves the model to `models/`. |
| `correlation.py` | Exploratory data analysis — correlations, plots, a shipping-corridor network graph, PCA, and residual analysis. |
| `featurevisual.py` | Feature screening — ranks candidate features by their correlation with delivery time. |
| `features.py` | Experimental / scratch feature-engineering work. |

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

The Olist CSV files are expected in the `data/` folder next to the scripts:

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
in `data/`.

## Usage

Train and evaluate the model (this is the main entry point):

```bash
python logistics.py
```

This prints cross-validation and held-out test RMSE (in days) and saves a timestamped
model artifact to `models/`.

Run the exploratory analysis (opens matplotlib plots):

```bash
python correlation.py
python featurevisual.py
```
