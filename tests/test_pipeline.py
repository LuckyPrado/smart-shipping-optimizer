"""Fast, data-light regression tests for the delivery-time pipeline.

Covers the pure, deterministic pieces: feature derivation, the haversine helper, the
PR-10 order-grain aggregation contract, and -- most importantly -- a leak guard that
turns the project's hardest-won lesson (no post-outcome columns as model inputs) into a
test that fails loudly if anyone reintroduces one.
"""
from pathlib import Path

import pandas as pd
import pytest

from logistics import (cat_attribs, engineer_features, leak_safe_expanding_mean,
                       num_attribs)
from utils import aggregate_order_items, haversine

DATA_CACHE = Path(__file__).resolve().parent.parent / "data" / "_cache.parquet"


def test_engineer_features_derivations():
    df = pd.DataFrame({
        "order_purchase_timestamp": pd.to_datetime(
            ["2018-01-01", "2018-02-15", "2018-03-10", "2018-06-20", "2018-11-05"]),
        "order_estimated_delivery_date": pd.to_datetime(
            ["2018-01-11", "2018-02-25", "2018-03-20", "2018-06-30", "2018-11-20"]),
        "seller_state": ["SP", "MG", "RJ", "SP", "PR"],
        "customer_state": ["SP", "BA", "AM", "RS", "SP"],
    })
    out = engineer_features(df)

    assert list(out["estimated_delivery_days"]) == [10, 10, 10, 10, 15]
    assert list(out["purchase_month"]) == [1, 2, 3, 6, 11]
    assert list(out["state_pair"]) == ["SP_SP", "MG_BA", "RJ_AM", "SP_RS", "PR_SP"]
    assert list(out["customer_region"]) == [
        "Southeast", "Northeast", "North", "South", "Southeast"]
    # engineer_features must not mutate its input frame
    assert "estimated_delivery_days" not in df.columns


def test_haversine_known_city_pair():
    sao_paulo = (-23.5505, -46.6333)
    rio = (-22.9068, -43.1729)

    dist = haversine(*sao_paulo, *rio)
    assert 355 < dist < 370                      # SP->Rio great-circle is ~360 km
    assert haversine(*sao_paulo, *sao_paulo) == pytest.approx(0.0, abs=1e-9)
    assert dist == pytest.approx(haversine(*rio, *sao_paulo))   # symmetric


def test_aggregate_order_items():
    """Exercise the REAL PR-10.2 aggregation (utils.aggregate_order_items, the same code
    load_data runs) on a hand-built item-grain order: sum economics + weight, max the box
    dimensions, count items, count distinct sellers, keep the heaviest item's category."""
    items = pd.DataFrame({
        "order_id": ["A", "A", "B"],
        "seller_id": ["s1", "s2", "s3"],
        "product_id": ["p1", "p2", "p3"],
        "price": [50.0, 30.0, 20.0],
        "freight_value": [10.0, 5.0, 7.0],
        "product_weight_g": [1000.0, 3000.0, 800.0],
        "product_length_cm": [20.0, 40.0, 15.0],
        "product_height_cm": [10.0, 5.0, 8.0],
        "product_width_cm": [12.0, 30.0, 9.0],
        "product_category_name": ["books", "furniture", "toys"],
    })

    out = aggregate_order_items(items).set_index("order_id")

    assert out.loc["A", "price"] == 80.0 and out.loc["A", "freight_value"] == 15.0
    assert out.loc["A", "product_weight_g"] == 4000.0            # summed weight (1000+3000)
    assert out.loc["A", "product_length_cm"] == 40.0            # max dimension
    assert out.loc["A", "order_item_count"] == 2
    assert out.loc["A", "order_unique_sellers"] == 2
    assert out.loc["A", "product_category_name"] == "furniture"  # heaviest item's category
    assert out.loc["B", "order_item_count"] == 1
    assert out.loc["B", "order_unique_sellers"] == 1


def test_leak_safe_expanding_mean_only_sees_completed_history():
    """The leak-safe shift: an order may use a same-key order's outcome only if that order
    was DELIVERED strictly before this one was PURCHASED. History completing the day before
    a purchase counts; history completing the day after (or with no history) does not."""
    df = pd.DataFrame({
        "key": ["K", "K", "K"],
        "order_purchase_timestamp": pd.to_datetime(
            ["2017-12-30", "2018-01-11", "2018-01-09"]),
        "order_delivered_customer_date": pd.to_datetime(
            ["2018-01-10", "2018-01-20", "2018-01-15"]),
        "actual_delivery_days": [11.0, 9.0, 6.0],
    })

    out = leak_safe_expanding_mean(df, "key")

    # row 0: the history order itself -- nothing completed before it -> NaN
    assert pd.isna(out.iloc[0])
    # row 1: purchased 01-11, history delivered 01-10 (day BEFORE) -> sees it -> 11.0
    assert out.iloc[1] == 11.0
    # row 2: purchased 01-09, history delivered 01-10 (day AFTER) -> must not see it -> NaN
    assert pd.isna(out.iloc[2])


# --- the leak guard: the whole reason this project exists -------------------------------

# Columns knowable only after the order ships/arrives (or that were the leaky target).
# None of these may ever appear as a model input.
POST_OUTCOME_COLUMNS = {
    "order_delivered_customer_date", "order_delivered_carrier_date",
    "shipping_limit_date", "order_approved_at", "order_status",
    "review_score", "review_comment_length",
}


def test_no_post_outcome_columns_in_feature_lists():
    features = set(num_attribs) | set(cat_attribs)

    leaked = features & POST_OUTCOME_COLUMNS
    assert not leaked, f"post-outcome column(s) used as model input: {sorted(leaked)}"

    # Also guard against renamed derivatives that still reference post-delivery info.
    for col in features:
        assert "delivered" not in col, f"feature '{col}' references delivery outcome"
        assert "shipping_limit" not in col, f"feature '{col}' references shipping_limit"


@pytest.mark.skipif(not DATA_CACHE.exists(), reason="requires data/_cache.parquet")
def test_load_data_is_order_grain():
    """When the data is present, load_data() must return exactly one row per order with
    the two PR-10 count features populated."""
    from utils import load_data

    df = load_data()
    assert df["order_id"].is_unique
    assert {"order_item_count", "order_unique_sellers"} <= set(df.columns)
    assert (df["order_item_count"] >= 1).all()
    assert (df["order_unique_sellers"] >= 1).all()
