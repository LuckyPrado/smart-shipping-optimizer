# Repo root on sys.path so `from utils import ...` works from analysis/.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import haversine, load_data, brazil_regions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy import stats


if __name__ == "__main__":
    logistics = load_data()

    logistics["order_purchase_timestamp"] = pd.to_datetime(logistics["order_purchase_timestamp"])
    logistics["order_estimated_delivery_date"] = pd.to_datetime(logistics["order_estimated_delivery_date"])
    logistics["order_delivered_customer_date"] = pd.to_datetime(logistics["order_delivered_customer_date"])

    # Screen features against the ACTUAL delivery time (PR 5), not the platform's
    # estimate. The estimate was the leaky target; conclusions drawn against it
    # (which features matter, the PCA "regimes") may not hold for real delivery time.
    logistics["actual_delivery_days"] = (logistics["order_delivered_customer_date"] - logistics["order_purchase_timestamp"]).dt.days
    logistics = logistics.dropna(subset=["actual_delivery_days"])

    logistics["real_distance_km"] = logistics.apply(
        lambda row: haversine(row["seller_lat"], row["seller_lng"], row["customer_lat"], row["customer_lng"]), axis=1
    )
    logistics["same_state"] = (logistics["seller_state"] == logistics["customer_state"]).astype(int)
    logistics["state_pair"] = logistics["seller_state"] + "_" + logistics["customer_state"]
    logistics["product_volume"] = logistics["product_length_cm"] * logistics["product_height_cm"] * logistics["product_width_cm"]
    logistics["log_distance"] = np.log1p(logistics["real_distance_km"])
    logistics["distance_weight"] = logistics["real_distance_km"] * logistics["product_weight_g"]
    logistics["freight_per_km"] = logistics["freight_value"] / logistics["real_distance_km"].replace(0, np.nan)
    logistics["volume_distance_interaction"] = logistics["product_volume"] * logistics["real_distance_km"]
    logistics["route_frequency"] = logistics.groupby("state_pair")["order_id"].transform("count")
    logistics["seller_order_volume"] = logistics.groupby("seller_id")["order_id"].transform("count")
    logistics["freight_ratio"] = logistics["freight_value"] / logistics["price"].replace(0, np.nan)
    logistics["product_density"] = logistics["product_weight_g"] / logistics["product_volume"].replace(0, np.nan)
    logistics["max_dimension_cm"] = logistics[["product_length_cm", "product_height_cm", "product_width_cm"]].max(axis=1)
    logistics["price_per_km"] = logistics["price"] / logistics["real_distance_km"].replace(0, np.nan)

    sample = logistics.dropna(subset=["real_distance_km", "actual_delivery_days", "product_weight_g"]).sample(15000, random_state=42)

    print("=" * 60)
    print("1. DISTANCE vs DELIVERY DAYS")
    print("=" * 60)
    # Drop NaNs from BOTH columns together so the two series stay row-aligned. Dropping
    # each independently only happens to work while the geo merge guarantees no NaNs --
    # the moment either column gains one, the rows silently mispair.
    dist_pair = logistics[["real_distance_km", "actual_delivery_days"]].dropna()
    pearson_r, pearson_p = stats.pearsonr(
        dist_pair["real_distance_km"], dist_pair["actual_delivery_days"])
    spearman_r, spearman_p = stats.spearmanr(
        dist_pair["real_distance_km"], dist_pair["actual_delivery_days"])
    print(f"Pearson r:  {pearson_r:.4f}  (p={pearson_p:.2e})")
    print(f"Spearman r: {spearman_r:.4f}  (p={spearman_p:.2e})")
    print(f"log_distance Pearson: {logistics['log_distance'].corr(logistics['actual_delivery_days']):.4f}")

    distance_bins = pd.qcut(logistics["real_distance_km"], 10, duplicates="drop")
    print("\nMean delivery days by distance quantile:")
    print(logistics.groupby(distance_bins, observed=True)["actual_delivery_days"].agg(["mean", "std", "count"]).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].hexbin(sample["real_distance_km"], sample["actual_delivery_days"], gridsize=70, cmap="viridis", bins="log")
    axes[0].set_xlabel("Distance KM")
    axes[0].set_ylabel("Actual Delivery Days")
    axes[0].set_title("Hexbin: Distance vs Delivery Days")
    smoothed = lowess(sample["actual_delivery_days"], sample["real_distance_km"], frac=0.12)
    axes[1].scatter(sample["real_distance_km"], sample["actual_delivery_days"], alpha=0.05, s=5)
    axes[1].plot(smoothed[:, 0], smoothed[:, 1], color="red", linewidth=3, label="LOWESS")
    axes[1].set_xlabel("Distance KM")
    axes[1].set_ylabel("Actual Delivery Days")
    axes[1].set_title("LOWESS: Nonlinear Distance Relationship")
    axes[1].legend()
    sns.boxplot(data=logistics, x=distance_bins, y="actual_delivery_days", ax=axes[2])
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha="right")
    axes[2].set_title("Distance Regime Distribution")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("2. PRODUCT WEIGHT vs DELIVERY DAYS")
    print("=" * 60)
    weight_bins = pd.cut(logistics["product_weight_g"], bins=[0, 500, 1000, 2000, 5000, 10000, np.inf], labels=["<500g", "500g-1kg", "1-2kg", "2-5kg", "5-10kg", "10kg+"])
    weight_stats = logistics.groupby(weight_bins, observed=True)["actual_delivery_days"].agg(["mean", "std", "count"])
    print(weight_stats.to_string())
    print(f"\nWeight-target Pearson: {logistics['product_weight_g'].corr(logistics['actual_delivery_days']):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.violinplot(data=logistics, x=weight_bins, y="actual_delivery_days", ax=axes[0])
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
    axes[0].set_title("Weight Bins vs Delivery Days (Violin)")
    logistics["heavy_item_flag"] = (logistics["product_weight_g"] > 5000).astype(int)
    sns.boxplot(data=logistics, x="heavy_item_flag", y="actual_delivery_days", ax=axes[1])
    axes[1].set_xticklabels(["Normal", "Heavy (>5kg)"])
    axes[1].set_title("Heavy Item Flag vs Delivery Days")
    print(f"\nHeavy item mean days: {logistics[logistics['heavy_item_flag']==1]['actual_delivery_days'].mean():.2f}")
    print(f"Normal item mean days: {logistics[logistics['heavy_item_flag']==0]['actual_delivery_days'].mean():.2f}")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("3. FREIGHT VALUE vs DELIVERY DAYS")
    print("=" * 60)
    freight_bins = pd.qcut(logistics["freight_value"], 10, duplicates="drop")
    freight_stats = logistics.groupby(freight_bins, observed=True)["actual_delivery_days"].agg(["mean", "std"])
    print(freight_stats.to_string())
    print(f"\nFreight-target Pearson: {logistics['freight_value'].corr(logistics['actual_delivery_days']):.4f}")
    print(f"log_freight-target Pearson: {np.log1p(logistics['freight_value']).corr(logistics['actual_delivery_days']):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].hexbin(sample["freight_value"], sample["actual_delivery_days"], gridsize=50, cmap="plasma", bins="log")
    axes[0].set_xlabel("Freight Value")
    axes[0].set_ylabel("Actual Delivery Days")
    axes[0].set_title("Hexbin: Freight vs Delivery Days")
    freight_smoothed = lowess(sample["actual_delivery_days"], sample["freight_value"], frac=0.15)
    axes[1].scatter(sample["freight_value"], sample["actual_delivery_days"], alpha=0.05, s=5)
    axes[1].plot(freight_smoothed[:, 0], freight_smoothed[:, 1], color="red", linewidth=3)
    axes[1].set_title("LOWESS: Freight Nonlinear Relationship")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("4. ROUTE FREQUENCY vs DELIVERY DAYS")
    print("=" * 60)
    freq_bins = pd.qcut(logistics["route_frequency"], 5, duplicates="drop")
    freq_stats = logistics.groupby(freq_bins, observed=True)["actual_delivery_days"].agg(["mean", "std", "count"])
    print(freq_stats.to_string())
    logistics["rare_route_flag"] = (logistics["route_frequency"] < logistics["route_frequency"].quantile(0.25)).astype(int)
    print(f"\nRare route mean days: {logistics[logistics['rare_route_flag']==1]['actual_delivery_days'].mean():.2f}")
    print(f"Frequent route mean days: {logistics[logistics['rare_route_flag']==0]['actual_delivery_days'].mean():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(data=logistics, x="route_frequency", bins=50, ax=axes[0])
    axes[0].set_title("Route Frequency Distribution")
    sns.boxplot(data=logistics, x=freq_bins, y="actual_delivery_days", ax=axes[1])
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].set_title("Route Frequency vs Delivery Days")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("5. SELLER ORDER VOLUME vs DELIVERY DAYS")
    print("=" * 60)
    vol_bins = pd.qcut(logistics["seller_order_volume"], 5, duplicates="drop")
    vol_stats = logistics.groupby(vol_bins, observed=True)["actual_delivery_days"].agg(["mean", "std"])
    print(vol_stats.to_string())
    print(f"\nSeller volume-target Pearson: {logistics['seller_order_volume'].corr(logistics['actual_delivery_days']):.4f}")
    logistics["high_volume_seller_flag"] = (logistics["seller_order_volume"] > logistics["seller_order_volume"].quantile(0.75)).astype(int)
    print(f"High volume seller mean days: {logistics[logistics['high_volume_seller_flag']==1]['actual_delivery_days'].mean():.2f}")
    print(f"Low volume seller mean days: {logistics[logistics['high_volume_seller_flag']==0]['actual_delivery_days'].mean():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    vol_smoothed = lowess(sample["actual_delivery_days"], sample["seller_order_volume"], frac=0.15)
    axes[0].scatter(sample["seller_order_volume"], sample["actual_delivery_days"], alpha=0.05, s=5)
    axes[0].plot(vol_smoothed[:, 0], vol_smoothed[:, 1], color="red", linewidth=3)
    axes[0].set_title("LOWESS: Seller Volume vs Delivery Days")
    sns.boxplot(data=logistics, x="high_volume_seller_flag", y="actual_delivery_days", ax=axes[1])
    axes[1].set_xticklabels(["Low Volume", "High Volume"])
    axes[1].set_title("Seller Volume Flag vs Delivery Days")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("6. PURCHASE MONTH SEASONALITY")
    print("=" * 60)
    logistics["purchase_month"] = logistics["order_purchase_timestamp"].dt.month
    monthly_stats = logistics.groupby("purchase_month")["actual_delivery_days"].agg(["mean", "std"])
    print(monthly_stats.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(monthly_stats.index, monthly_stats["mean"], marker="o", linewidth=2)
    axes[0].fill_between(monthly_stats.index,
                         monthly_stats["mean"] - monthly_stats["std"],
                         monthly_stats["mean"] + monthly_stats["std"], alpha=0.2)
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Mean Delivery Days")
    axes[0].set_title("Monthly Seasonality")
    logistics["purchase_dayofweek"] = logistics["order_purchase_timestamp"].dt.dayofweek
    dow_stats = logistics.groupby("purchase_dayofweek")["actual_delivery_days"].mean()
    axes[1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow_stats.values)
    axes[1].set_title("Day of Week vs Delivery Days")
    print("\nDay of week mean days:")
    print(dow_stats.to_string())
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("7. FREIGHT PER KM — SHIPPING EFFICIENCY")
    print("=" * 60)
    fpk_clean = logistics["freight_per_km"].replace([np.inf, -np.inf], np.nan).dropna()
    fpk_percentiles = fpk_clean.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print("Freight per km percentiles:")
    print(fpk_percentiles.to_string())
    logistics["expensive_route_flag"] = (logistics["freight_per_km"] > logistics["freight_per_km"].quantile(0.75)).astype(int)
    print(f"\nExpensive route mean days: {logistics[logistics['expensive_route_flag']==1]['actual_delivery_days'].mean():.2f}")
    print(f"Efficient route mean days: {logistics[logistics['expensive_route_flag']==0]['actual_delivery_days'].mean():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fpk_sample = logistics["freight_per_km"].replace([np.inf, -np.inf], np.nan).dropna()
    sns.kdeplot(fpk_sample.clip(upper=fpk_sample.quantile(0.99)), ax=axes[0])
    axes[0].set_title("KDE: Freight Per KM Distribution")
    sns.boxplot(data=logistics, x="expensive_route_flag", y="actual_delivery_days", ax=axes[1])
    axes[1].set_xticklabels(["Efficient Route", "Expensive Route"])
    axes[1].set_title("Route Efficiency vs Delivery Days")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("8. GEOGRAPHIC TOPOLOGY — NORTH/SOUTH PENALTY")
    print("=" * 60)
    logistics["customer_region"] = logistics["customer_state"].map(brazil_regions)
    logistics["seller_region"] = logistics["seller_state"].map(brazil_regions)
    region_stats = logistics.groupby("customer_region")["actual_delivery_days"].agg(["mean", "std", "count"])
    print("Customer region delivery stats:")
    print(region_stats.to_string())
    logistics["remote_state_flag"] = logistics["customer_region"].isin(["North"]).astype(int)
    print(f"\nRemote (North) mean days: {logistics[logistics['remote_state_flag']==1]['actual_delivery_days'].mean():.2f}")
    print(f"Non-remote mean days: {logistics[logistics['remote_state_flag']==0]['actual_delivery_days'].mean():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    region_order = region_stats["mean"].sort_values(ascending=False).index
    sns.boxplot(data=logistics, x="customer_region", y="actual_delivery_days", order=region_order, ax=axes[0])
    axes[0].set_title("Customer Region vs Delivery Days")
    axes[0].tick_params(axis="x", rotation=30)
    pivot = logistics.pivot_table(values="actual_delivery_days", index="seller_state", columns="customer_state", aggfunc="mean")
    sns.heatmap(pivot, cmap="magma", linewidths=0.3, ax=axes[1])
    axes[1].set_title("State Pair Delivery Heatmap")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("9. PRODUCT DENSITY AND VOLUME")
    print("=" * 60)
    density_clean = logistics["product_density"].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Product density-target Pearson: {density_clean.corr(logistics.loc[density_clean.index, 'actual_delivery_days']):.4f}")
    print(f"Product volume-target Pearson: {logistics['product_volume'].corr(logistics['actual_delivery_days']):.4f}")
    vol_bins2 = pd.qcut(logistics["product_volume"].replace([np.inf, -np.inf], np.nan).dropna(), 5, duplicates="drop")
    vol_target = logistics.loc[vol_bins2.index].copy()
    vol_target["vol_bin"] = vol_bins2.values
    print("\nVolume bins mean delivery days:")
    print(vol_target.groupby("vol_bin", observed=True)["actual_delivery_days"].agg(["mean", "std"]).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    vol_clean = logistics[["product_volume", "actual_delivery_days"]].replace([np.inf, -np.inf], np.nan).dropna()
    axes[0].hexbin(np.log1p(vol_clean["product_volume"]), vol_clean["actual_delivery_days"], gridsize=50, cmap="viridis", bins="log")
    axes[0].set_xlabel("log(Product Volume)")
    axes[0].set_ylabel("Actual Delivery Days")
    axes[0].set_title("Hexbin: log Volume vs Delivery Days")
    density_sample = logistics[["product_density", "actual_delivery_days"]].replace([np.inf, -np.inf], np.nan).dropna().sample(10000, random_state=42)
    density_clipped = density_sample["product_density"].clip(upper=density_sample["product_density"].quantile(0.99))
    sns.kdeplot(x=density_clipped, y=density_sample["actual_delivery_days"], fill=True, ax=axes[1])
    axes[1].set_title("KDE: Product Density vs Delivery Days")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("10. ROUTE VARIABILITY")
    print("=" * 60)
    route_stats = logistics.groupby("state_pair")["actual_delivery_days"].agg(["mean", "std", "count"])
    route_stats = route_stats[route_stats["count"] >= 10]
    print("Top 10 most variable routes:")
    print(route_stats.sort_values("std", ascending=False).head(10).to_string())
    print("\nTop 10 most stable routes:")
    print(route_stats.sort_values("std").head(10).to_string())
    logistics["route_variability"] = logistics["state_pair"].map(route_stats["std"]).fillna(route_stats["std"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].scatter(route_stats["mean"], route_stats["std"], alpha=0.5, s=route_stats["count"]/50)
    axes[0].set_xlabel("Mean Delivery Days")
    axes[0].set_ylabel("Std Delivery Days")
    axes[0].set_title("Route Mean vs Variability")
    sns.histplot(route_stats["std"].dropna(), bins=40, ax=axes[1])
    axes[1].set_title("Route Variability Distribution")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("11. NETWORK GRAPH — SHIPPING CORRIDORS")
    print("=" * 60)
    route_graph = logistics.groupby(["seller_state", "customer_state"]).size().reset_index(name="route_count")
    G = nx.DiGraph()
    for _, row in route_graph.iterrows():
        G.add_edge(row["seller_state"], row["customer_state"], weight=row["route_count"])
    degree_centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight="weight")
    pagerank = nx.pagerank(G, weight="weight")
    print("Top 10 states by degree centrality:")
    print(sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10])
    print("\nTop 10 states by betweenness centrality:")
    print(sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10])
    print("\nTop 10 states by PageRank:")
    print(sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10])

    top_routes = route_graph.sort_values("route_count", ascending=False).head(35)
    G_small = nx.DiGraph()
    for _, row in top_routes.iterrows():
        G_small.add_edge(row["seller_state"], row["customer_state"], weight=row["route_count"])
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    pos = nx.circular_layout(G_small)
    weights = [G_small[u][v]["weight"] / 4000 for u, v in G_small.edges()]
    node_sizes = [degree_centrality.get(n, 0.1) * 8000 for n in G_small.nodes()]
    nx.draw_networkx_nodes(G_small, pos, node_size=node_sizes, ax=axes[0])
    nx.draw_networkx_labels(G_small, pos, font_size=9, ax=axes[0])
    nx.draw_networkx_edges(G_small, pos, width=weights, arrows=True, alpha=0.7, ax=axes[0])
    axes[0].set_title("Top Shipping Corridors (node size = centrality)")
    axes[0].axis("off")
    pagerank_vals = [pagerank.get(n, 0) * 10000 for n in G_small.nodes()]
    nx.draw_networkx_nodes(G_small, pos, node_size=pagerank_vals, node_color="orange", ax=axes[1])
    nx.draw_networkx_labels(G_small, pos, font_size=9, ax=axes[1])
    nx.draw_networkx_edges(G_small, pos, width=weights, arrows=True, alpha=0.7, ax=axes[1])
    axes[1].set_title("PageRank Centrality (node size = PageRank)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("12. PCA CLUSTER ANALYSIS — HIDDEN SHIPPING REGIMES")
    print("=" * 60)
    pca_features = ["real_distance_km", "product_weight_g", "freight_value",
                    "product_volume", "route_frequency", "seller_order_volume",
                    "freight_per_km", "distance_weight"]
    pca_data = logistics[pca_features].replace([np.inf, -np.inf], np.nan).dropna().sample(20000, random_state=42)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    pca_scaled = scaler.fit_transform(pca_data)
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(pca_scaled)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")
    target_vals = logistics.loc[pca_data.index, "actual_delivery_days"].values

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    scatter = axes[0].scatter(pca_result[:, 0], pca_result[:, 1], c=target_vals, cmap="viridis", alpha=0.3, s=5)
    plt.colorbar(scatter, ax=axes[0], label="Delivery Days")
    axes[0].set_title("PCA: Feature Space Colored by Delivery Days")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    components_df = pd.DataFrame(pca.components_, columns=pca_features, index=["PC1", "PC2"])
    sns.heatmap(components_df, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1])
    axes[1].set_title("PCA Component Loadings")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("13. RESIDUAL ANALYSIS — WHERE MODEL FAILS")
    print("=" * 60)
    rf_features = pca_features.copy()
    rf_data = logistics[rf_features + ["actual_delivery_days"]].replace([np.inf, -np.inf], np.nan).dropna()
    X_rf = rf_data[rf_features]
    y_rf = rf_data["actual_delivery_days"]
    X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    residuals = y_test.values - y_pred
    print(f"Mean residual: {residuals.mean():.4f}")
    print(f"Std residual:  {residuals.std():.4f}")
    dist_test = X_test["real_distance_km"].values
    weight_test = X_test["product_weight_g"].values
    route_freq_test = X_test["route_frequency"].values
    dist_bins = pd.qcut(dist_test, 10, duplicates="drop")
    print("\nMean residual by distance quantile:")
    resid_by_dist = pd.Series(residuals).groupby(dist_bins).agg(["mean", "std"])
    print(resid_by_dist.to_string())

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes[0, 0].scatter(dist_test, residuals, alpha=0.1, s=5)
    axes[0, 0].axhline(0, color="red", linewidth=2)
    axes[0, 0].set_xlabel("Distance KM")
    axes[0, 0].set_ylabel("Residual")
    axes[0, 0].set_title("Residuals vs Distance")
    axes[0, 1].scatter(weight_test, residuals, alpha=0.1, s=5)
    axes[0, 1].axhline(0, color="red", linewidth=2)
    axes[0, 1].set_xlabel("Product Weight g")
    axes[0, 1].set_title("Residuals vs Weight")
    axes[0, 2].scatter(route_freq_test, residuals, alpha=0.1, s=5)
    axes[0, 2].axhline(0, color="red", linewidth=2)
    axes[0, 2].set_xlabel("Route Frequency")
    axes[0, 2].set_title("Residuals vs Route Frequency")
    axes[1, 0].scatter(y_pred, residuals, alpha=0.1, s=5)
    axes[1, 0].axhline(0, color="red", linewidth=2)
    axes[1, 0].set_xlabel("Predicted Days")
    axes[1, 0].set_ylabel("Residual")
    axes[1, 0].set_title("Residuals vs Predicted")
    sns.histplot(residuals, bins=60, ax=axes[1, 1], kde=True)
    axes[1, 1].set_title("Residual Distribution")
    axes[1, 2].scatter(y_test, y_pred, alpha=0.1, s=5)
    axes[1, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2)
    axes[1, 2].set_xlabel("Actual Days")
    axes[1, 2].set_ylabel("Predicted Days")
    axes[1, 2].set_title("Actual vs Predicted")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("14. DISTANCE × WEIGHT INTERACTION SURFACE")
    print("=" * 60)
    interaction_corr = logistics["distance_weight"].corr(logistics["actual_delivery_days"])
    print(f"distance_weight Pearson: {interaction_corr:.4f}")
    vol_dist_corr = logistics["volume_distance_interaction"].replace([np.inf, -np.inf], np.nan).corr(logistics["actual_delivery_days"])
    print(f"volume_distance_interaction Pearson: {vol_dist_corr:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    dw_clean = logistics[["distance_weight", "actual_delivery_days"]].replace([np.inf, -np.inf], np.nan).dropna()
    dw_clipped = dw_clean["distance_weight"].clip(upper=dw_clean["distance_weight"].quantile(0.99))
    axes[0].hexbin(dw_clipped, dw_clean["actual_delivery_days"], gridsize=50, cmap="inferno", bins="log")
    axes[0].set_xlabel("Distance × Weight")
    axes[0].set_ylabel("Delivery Days")
    axes[0].set_title("2D Contour: Distance×Weight vs Delivery Days")
    vd_clean = logistics[["volume_distance_interaction", "actual_delivery_days"]].replace([np.inf, -np.inf], np.nan).dropna()
    vd_clipped = vd_clean["volume_distance_interaction"].clip(upper=vd_clean["volume_distance_interaction"].quantile(0.99))
    axes[1].hexbin(np.log1p(vd_clipped), vd_clean["actual_delivery_days"], gridsize=50, cmap="plasma", bins="log")
    axes[1].set_xlabel("log(Volume × Distance)")
    axes[1].set_ylabel("Delivery Days")
    axes[1].set_title("Volume×Distance Interaction")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("16. DISTANCE ROUTE RATIO — PC2 AXIS")
    print("=" * 60)
    logistics["distance_route_ratio"] = logistics["real_distance_km"] / logistics["route_frequency"].replace(0, np.nan)
    drr_corr = logistics["distance_route_ratio"].replace([np.inf, -np.inf], np.nan).corr(logistics["actual_delivery_days"])
    print(f"distance_route_ratio Pearson: {drr_corr:.4f}")

    drr_clean = logistics[["distance_route_ratio", "actual_delivery_days"]].replace([np.inf, -np.inf], np.nan).dropna()
    drr_clipped = drr_clean["distance_route_ratio"].clip(upper=drr_clean["distance_route_ratio"].quantile(0.99))
    drr_bins = pd.qcut(drr_clipped, 10, duplicates="drop")
    print("\nMean delivery days by distance_route_ratio decile:")
    print(drr_clean.loc[drr_clipped.index].groupby(drr_bins, observed=True)["actual_delivery_days"].agg(["mean", "std"]).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].hexbin(np.log1p(drr_clipped), drr_clean.loc[drr_clipped.index, "actual_delivery_days"], gridsize=50, cmap="viridis", bins="log")
    axes[0].set_xlabel("log(Distance / Route Frequency)")
    axes[0].set_ylabel("Actual Delivery Days")
    axes[0].set_title("Hexbin: Distance-Route Ratio vs Delivery Days")
    drr_sample = drr_clean.sample(10000, random_state=42)
    drr_smoothed = lowess(drr_sample["actual_delivery_days"], np.log1p(drr_sample["distance_route_ratio"].clip(upper=drr_sample["distance_route_ratio"].quantile(0.99))), frac=0.15)
    axes[1].scatter(np.log1p(drr_sample["distance_route_ratio"].clip(upper=drr_sample["distance_route_ratio"].quantile(0.99))), drr_sample["actual_delivery_days"], alpha=0.05, s=5)
    axes[1].plot(drr_smoothed[:, 0], drr_smoothed[:, 1], color="red", linewidth=3)
    axes[1].set_title("LOWESS: Distance-Route Ratio Relationship")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("17. FREIGHT TIER ANALYSIS")
    print("=" * 60)
    logistics["freight_tier"] = pd.cut(
        logistics["freight_value"],
        bins=[0, 10, 20, 50, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(float)
    logistics["ultra_cheap_freight"] = (logistics["freight_value"] < 10).astype(int)
    tier_stats = logistics.groupby("freight_tier", observed=True)["actual_delivery_days"].agg(["mean", "std", "count"])
    print(tier_stats.to_string())
    print(f"\nUltra cheap freight mean days: {logistics[logistics['ultra_cheap_freight']==1]['actual_delivery_days'].mean():.2f}")
    print(f"Normal freight mean days: {logistics[logistics['ultra_cheap_freight']==0]['actual_delivery_days'].mean():.2f}")
    print(f"Difference: {logistics[logistics['ultra_cheap_freight']==1]['actual_delivery_days'].mean() - logistics[logistics['ultra_cheap_freight']==0]['actual_delivery_days'].mean():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(data=logistics, x="freight_tier", y="actual_delivery_days", ax=axes[0])
    axes[0].set_xticklabels(["<10", "10-20", "20-50", "50+"])
    axes[0].set_title("Freight Tier vs Delivery Days")
    sns.violinplot(data=logistics, x="ultra_cheap_freight", y="actual_delivery_days", ax=axes[1])
    axes[1].set_xticklabels(["Normal Freight", "Ultra Cheap (<10)"])
    axes[1].set_title("Ultra Cheap Freight vs Delivery Days")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("18. LONG HEAVY AND SHORT HEAVY FLAGS")
    print("=" * 60)
    logistics["long_heavy"] = (
        (logistics["real_distance_km"] > 500) &
        (logistics["product_weight_g"] > 2000)
    ).astype(int)
    logistics["short_heavy"] = (
        (logistics["real_distance_km"] < 200) &
        (logistics["product_weight_g"] > 5000)
    ).astype(int)
    logistics["extreme_longhaul_flag"] = (logistics["real_distance_km"] > 1450).astype(int)

    for flag, desc in [("long_heavy", "Long distance + heavy"), ("short_heavy", "Short distance + heavy"), ("extreme_longhaul_flag", "Extreme longhaul >1450km")]:
        mean_0 = logistics[logistics[flag]==0]["actual_delivery_days"].mean()
        mean_1 = logistics[logistics[flag]==1]["actual_delivery_days"].mean()
        count_1 = logistics[flag].sum()
        print(f"\n{desc}:")
        print(f"  Count: {count_1}")
        print(f"  Flag=0 mean: {mean_0:.2f} days")
        print(f"  Flag=1 mean: {mean_1:.2f} days")
        print(f"  Difference:  {mean_1 - mean_0:.2f} days")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, (flag, desc) in enumerate([("long_heavy", "Long+Heavy"), ("short_heavy", "Short+Heavy"), ("extreme_longhaul_flag", "Extreme Longhaul")]):
        sns.boxplot(data=logistics, x=flag, y="actual_delivery_days", ax=axes[i])
        axes[i].set_title(f"{desc} vs Delivery Days")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("19. HOLIDAY AND SEASONAL PRESSURE")
    print("=" * 60)
    logistics["holiday_pressure"] = logistics["purchase_month"].isin([1, 2, 6, 12]).astype(int)
    logistics["fast_season"] = logistics["purchase_month"].isin([7, 8, 9]).astype(int)
    logistics["end_of_year"] = logistics["purchase_month"].isin([11, 12]).astype(int)

    for flag, desc in [("holiday_pressure", "Holiday months (Jan,Feb,Jun,Dec)"), ("fast_season", "Fast season (Jul,Aug,Sep)"), ("end_of_year", "End of year (Nov,Dec)")]:
        mean_0 = logistics[logistics[flag]==0]["actual_delivery_days"].mean()
        mean_1 = logistics[logistics[flag]==1]["actual_delivery_days"].mean()
        print(f"\n{desc}:")
        print(f"  Flag=0 mean: {mean_0:.2f} days")
        print(f"  Flag=1 mean: {mean_1:.2f} days")
        print(f"  Difference:  {mean_1 - mean_0:.2f} days")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, (flag, desc) in enumerate([("holiday_pressure", "Holiday Pressure"), ("fast_season", "Fast Season"), ("end_of_year", "End of Year")]):
        sns.violinplot(data=logistics, x=flag, y="actual_delivery_days", ax=axes[i])
        axes[i].set_title(f"{desc} vs Delivery Days")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("20. PRICE ANALYSIS")
    print("=" * 60)
    price_bins = pd.qcut(logistics["price"], 10, duplicates="drop")
    price_stats = logistics.groupby(price_bins, observed=True)["actual_delivery_days"].agg(["mean", "std"])
    print(price_stats.to_string())
    print(f"\nPrice-target Pearson: {logistics['price'].corr(logistics['actual_delivery_days']):.4f}")
    print(f"log_price-target Pearson: {np.log1p(logistics['price']).corr(logistics['actual_delivery_days']):.4f}")
    logistics["premium_product"] = (logistics["price"] > logistics["price"].quantile(0.9)).astype(int)
    print(f"\nPremium product mean days: {logistics[logistics['premium_product']==1]['actual_delivery_days'].mean():.2f}")
    print(f"Standard product mean days: {logistics[logistics['premium_product']==0]['actual_delivery_days'].mean():.2f}")
    print(f"Difference: {logistics[logistics['premium_product']==1]['actual_delivery_days'].mean() - logistics[logistics['premium_product']==0]['actual_delivery_days'].mean():.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].hexbin(sample["price"], sample["actual_delivery_days"], gridsize=50, cmap="plasma", bins="log")
    axes[0].set_xlabel("Price")
    axes[0].set_ylabel("Actual Delivery Days")
    axes[0].set_title("Hexbin: Price vs Delivery Days")
    price_smoothed = lowess(sample["actual_delivery_days"], sample["price"], frac=0.15)
    axes[1].scatter(sample["price"], sample["actual_delivery_days"], alpha=0.05, s=5)
    axes[1].plot(price_smoothed[:, 0], price_smoothed[:, 1], color="red", linewidth=3)
    axes[1].set_title("LOWESS: Price Relationship")
    sns.boxplot(data=logistics, x="premium_product", y="actual_delivery_days", ax=axes[2])
    axes[2].set_xticklabels(["Standard", "Premium (top 10%)"])
    axes[2].set_title("Premium Product vs Delivery Days")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("21. SELLER STATE ANALYSIS")
    print("=" * 60)
    seller_state_stats = logistics.groupby("seller_state")["actual_delivery_days"].agg(["mean", "std", "count"])
    print(seller_state_stats.sort_values("mean", ascending=False).to_string())
    logistics["seller_remote_flag"] = logistics["seller_state"].isin(["AM", "RO", "PA", "AC", "RR", "AP", "TO"]).astype(int)
    print(f"\nRemote seller mean days: {logistics[logistics['seller_remote_flag']==1]['actual_delivery_days'].mean():.2f}")
    print(f"Non-remote seller mean days: {logistics[logistics['seller_remote_flag']==0]['actual_delivery_days'].mean():.2f}")
    print(f"Difference: {logistics[logistics['seller_remote_flag']==1]['actual_delivery_days'].mean() - logistics[logistics['seller_remote_flag']==0]['actual_delivery_days'].mean():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    seller_means = seller_state_stats["mean"].sort_values(ascending=False)
    axes[0].barh(seller_means.index, seller_means.values)
    axes[0].set_title("Mean Delivery Days by Seller State")
    axes[0].set_xlabel("Mean Delivery Days")
    sns.boxplot(data=logistics, x="seller_remote_flag", y="actual_delivery_days", ax=axes[1])
    axes[1].set_xticklabels(["Non-remote Seller", "Remote Seller"])
    axes[1].set_title("Remote Seller Flag vs Delivery Days")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("22. CROSS REGIONAL SHIPPING")
    print("=" * 60)
    logistics["same_region"] = (logistics["seller_region"] == logistics["customer_region"]).astype(int)
    logistics["north_involved"] = (
        (logistics["seller_region"] == "North") |
        (logistics["customer_region"] == "North")
    ).astype(int)
    logistics["southeast_seller"] = (logistics["seller_region"] == "Southeast").astype(int)

    for flag, desc in [("same_region", "Same region shipment"), ("north_involved", "North region involved"), ("southeast_seller", "Southeast seller")]:
        mean_0 = logistics[logistics[flag]==0]["actual_delivery_days"].mean()
        mean_1 = logistics[logistics[flag]==1]["actual_delivery_days"].mean()
        print(f"\n{desc}:")
        print(f"  Flag=0 mean: {mean_0:.2f} days")
        print(f"  Flag=1 mean: {mean_1:.2f} days")
        print(f"  Difference:  {mean_1 - mean_0:.2f} days")

    region_pair_stats = logistics.groupby(["seller_region", "customer_region"])["actual_delivery_days"].agg(["mean", "count"])
    print("\nRegion pair delivery averages:")
    print(region_pair_stats.sort_values("mean", ascending=False).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    region_pivot = logistics.pivot_table(values="actual_delivery_days", index="seller_region", columns="customer_region", aggfunc="mean")
    sns.heatmap(region_pivot, annot=True, fmt=".1f", cmap="magma", ax=axes[0])
    axes[0].set_title("Region Pair Average Delivery Days")
    sns.violinplot(data=logistics, x="north_involved", y="actual_delivery_days", ax=axes[1])
    axes[1].set_xticklabels(["No North", "North Involved"])
    axes[1].set_title("North Region Involvement vs Delivery Days")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("23. ZIP CODE PREFIX ANALYSIS")
    print("=" * 60)
    logistics["customer_zip_prefix_bin"] = logistics["customer_zip_code_prefix"] // 10000
    zip_stats = logistics.groupby("customer_zip_prefix_bin")["actual_delivery_days"].agg(["mean", "std", "count"])
    print("Delivery days by customer zip prefix bin:")
    print(zip_stats.sort_values("mean", ascending=False).head(20).to_string())
    print(f"\nCustomer zip prefix Pearson: {logistics['customer_zip_code_prefix'].corr(logistics['actual_delivery_days']):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    zip_sample = logistics[["customer_zip_code_prefix", "actual_delivery_days"]].sample(15000, random_state=42)
    zip_smoothed = lowess(zip_sample["actual_delivery_days"], zip_sample["customer_zip_code_prefix"], frac=0.15)
    axes[0].scatter(zip_sample["customer_zip_code_prefix"], zip_sample["actual_delivery_days"], alpha=0.05, s=5)
    axes[0].plot(zip_smoothed[:, 0], zip_smoothed[:, 1], color="red", linewidth=3)
    axes[0].set_title("LOWESS: Customer Zip vs Delivery Days")
    sns.boxplot(data=logistics, x="customer_zip_prefix_bin", y="actual_delivery_days", ax=axes[1])
    axes[1].set_title("Zip Prefix Bin vs Delivery Days")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("24. ITEM QUANTITY ANALYSIS")
    print("=" * 60)
    quantity_stats = logistics.groupby("order_item_id")["actual_delivery_days"].agg(["mean", "std", "count"])
    print("Delivery days by order item count:")
    print(quantity_stats.to_string())
    print(f"\nOrder item id Pearson: {logistics['order_item_id'].corr(logistics['actual_delivery_days']):.4f}")
    logistics["multi_item_order"] = (logistics["order_item_id"] > 1).astype(int)
    print(f"\nMulti-item mean days: {logistics[logistics['multi_item_order']==1]['actual_delivery_days'].mean():.2f}")
    print(f"Single-item mean days: {logistics[logistics['multi_item_order']==0]['actual_delivery_days'].mean():.2f}")
    print(f"Difference: {logistics[logistics['multi_item_order']==1]['actual_delivery_days'].mean() - logistics[logistics['multi_item_order']==0]['actual_delivery_days'].mean():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(data=logistics[logistics["order_item_id"] <= 10], x="order_item_id", y="actual_delivery_days", ax=axes[0])
    axes[0].set_title("Order Item Count vs Delivery Days")
    sns.violinplot(data=logistics, x="multi_item_order", y="actual_delivery_days", ax=axes[1])
    axes[1].set_xticklabels(["Single Item", "Multi Item"])
    axes[1].set_title("Multi Item Order vs Delivery Days")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("25. COMPACTNESS RATIO — WEIGHT VS VOLUME")
    print("=" * 60)
    logistics["compactness_ratio"] = logistics["product_weight_g"] / np.cbrt(logistics["product_volume"].replace(0, np.nan))
    compact_corr = logistics["compactness_ratio"].replace([np.inf, -np.inf], np.nan).corr(logistics["actual_delivery_days"])
    print(f"compactness_ratio Pearson: {compact_corr:.4f}")
    compact_clean = logistics[["compactness_ratio", "actual_delivery_days"]].replace([np.inf, -np.inf], np.nan).dropna()
    compact_bins = pd.qcut(compact_clean["compactness_ratio"].clip(upper=compact_clean["compactness_ratio"].quantile(0.99)), 5, duplicates="drop")
    print("\nDelivery days by compactness quintile:")
    print(compact_clean.groupby(compact_bins, observed=True)["actual_delivery_days"].agg(["mean", "std"]).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    compact_clipped = compact_clean["compactness_ratio"].clip(upper=compact_clean["compactness_ratio"].quantile(0.99))
    axes[0].hexbin(np.log1p(compact_clipped), compact_clean["actual_delivery_days"], gridsize=50, cmap="viridis", bins="log")
    axes[0].set_xlabel("log(Compactness Ratio)")
    axes[0].set_ylabel("Actual Delivery Days")
    axes[0].set_title("Hexbin: Compactness vs Delivery Days")
    compact_sample = compact_clean.sample(10000, random_state=42)
    compact_smoothed = lowess(compact_sample["actual_delivery_days"], np.log1p(compact_sample["compactness_ratio"].clip(upper=compact_sample["compactness_ratio"].quantile(0.99))), frac=0.15)
    axes[1].scatter(np.log1p(compact_sample["compactness_ratio"].clip(upper=compact_sample["compactness_ratio"].quantile(0.99))), compact_sample["actual_delivery_days"], alpha=0.05, s=5)
    axes[1].plot(compact_smoothed[:, 0], compact_smoothed[:, 1], color="red", linewidth=3)
    axes[1].set_title("LOWESS: Compactness Relationship")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("15. ENGINEERED FLAG SUMMARY")
    print("=" * 60)
    flags = {
        "heavy_item_flag": "Product > 5kg",
        "rare_route_flag": "Route frequency < 25th pct",
        "high_volume_seller_flag": "Seller volume > 75th pct",
        "remote_state_flag": "Customer in North region",
        "expensive_route_flag": "Freight/km > 75th pct"
    }
    for flag, description in flags.items():
        if flag in logistics.columns:
            mean_0 = logistics[logistics[flag]==0]["actual_delivery_days"].mean()
            mean_1 = logistics[logistics[flag]==1]["actual_delivery_days"].mean()
            print(f"\n{description}:")
            print(f"  Flag=0 mean: {mean_0:.2f} days")
            print(f"  Flag=1 mean: {mean_1:.2f} days")
            print(f"  Difference:  {mean_1 - mean_0:.2f} days")