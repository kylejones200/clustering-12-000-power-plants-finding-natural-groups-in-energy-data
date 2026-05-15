"""
Python code extracted from 03_clustering_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
    force=True,
)
logger = logging.getLogger(__name__)
plants = pd.read_parquet("egrid_all_plants_1996-2023.parquet")
plants_2023 = plants[plants["data_year"] == 2023].copy()
plants_2023["log_generation"] = np.log1p(
    plants_2023["Plant annual net generation (MWh)"]
)
plants_2023["log_co2"] = np.log1p(plants_2023["Plant annual CO2 emissions (tons)"])
plants_2023["carbon_intensity"] = (
    plants_2023["Plant annual CO2 emissions (tons)"]
    / plants_2023["Plant annual net generation (MWh)"]
)
plants_2023["capacity_factor"] = plants_2023["Plant annual net generation (MWh)"] / (
    plants_2023["Plant nameplate capacity (MW)"] * 8760
)
features = [
    "log_generation",
    "log_co2",
    "carbon_intensity",
    "capacity_factor",
    "nox_intensity",
    "so2_intensity",
]
X = plants_2023[features].dropna()
logger.info(f"Clustering {len(X):,} plants on {len(features)} features")
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
results = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    results.append(
        {
            "k": k,
            "inertia": kmeans.inertia_,
            "silhouette": silhouette_score(X_scaled, labels),
            "calinski": calinski_harabasz_score(X_scaled, labels),
        }
    )
    logger.info(f"K={k}: Silhouette={results[-1]['silhouette']:.3f}")
optimal_k = 5
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)
plants_2023.loc[X.index, "cluster"] = cluster_labels
logger.info("\nCluster sizes:")
logger.info(pd.Series(cluster_labels).value_counts().sort_index())
cluster_profiles = (
    plants_2023.groupby("cluster")
    .agg(
        {
            "log_generation": "median",
            "carbon_intensity": "median",
            "capacity_factor": "median",
            "Plant nameplate capacity (MW)": "median",
        }
    )
    .round(3)
)
logger.info("\nCluster Profiles:")
logger.info(cluster_profiles)
from sklearn.mixture import GaussianMixture

bic_scores = []
for n in range(2, 11):
    gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    logger.info(f"n={n}: BIC={gmm.bic(X_scaled):.1f}")
optimal_n = range(2, 11)[np.argmin(bic_scores)]
logger.info(f"\nOptimal components: {optimal_n}")
gmm_final = GaussianMixture(n_components=5, covariance_type="full", random_state=42)
gmm_labels = gmm_final.fit_predict(X_scaled)
gmm_probas = gmm_final.predict_proba(X_scaled)
plants_2023.loc[X.index, "gmm_cluster"] = gmm_labels
plants_2023.loc[X.index, "gmm_probability"] = gmm_probas.max(axis=1)
uncertain = plants_2023[plants_2023["gmm_probability"] < 0.7]
logger.info(f"\nPlants with uncertain cluster membership: {len(uncertain)}")
logger.info("These plants have characteristics of multiple clusters")
import hdbscan

hdb = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, metric="euclidean")
hdb_labels = hdb.fit_predict(X_scaled)
n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
n_noise = list(hdb_labels).count(-1)
logger.info(f"HDBSCAN found {n_clusters} natural clusters")
logger.info(
    f"Outliers/noise: {n_noise} plants ({n_noise / len(hdb_labels) * 100:.1f}%)"
)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plants_2023.loc[X.index, "pca1"] = X_pca[:, 0]
plants_2023.loc[X.index, "pca2"] = X_pca[:, 1]
logger.info(f"PC1 explains {pca.explained_variance_ratio_[0] * 100:.1f}% of variance")
logger.info(f"PC2 explains {pca.explained_variance_ratio_[1] * 100:.1f}% of variance")
logger.info(f"Total: {pca.explained_variance_ratio_.sum() * 100:.1f}%")
from sklearn.manifold import TSNE

sample_idx = np.random.choice(len(X_scaled), 2000, replace=False)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled[sample_idx])
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="tab10", alpha=0.6, s=20)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")
plt.colorbar(label="Cluster")
plt.subplot(1, 2, 2)
plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=cluster_labels[sample_idx],
    cmap="tab10",
    alpha=0.6,
    s=20,
)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE Projection (sample)")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.savefig("cluster_visualization.png", dpi=150)


def benchmark_plant(plant_row, cluster_data):
    cluster_median = cluster_data["carbon_intensity"].median()
    plant_value = plant_row["carbon_intensity"]
    percentile = (cluster_data["carbon_intensity"] < plant_value).mean() * 100
    return {
        "cluster_median": cluster_median,
        "plant_value": plant_value,
        "percentile": percentile,
        "vs_median": (plant_value - cluster_median) / cluster_median * 100,
    }


coal_cluster_id = 0
coal_plants = plants_2023[plants_2023["cluster"] == coal_cluster_id]
example_plant = coal_plants.sample(1).iloc[0]
benchmark = benchmark_plant(example_plant, coal_plants)
logger.info(f"Plant: {example_plant.get('Plant name', 'Unknown')}")
logger.info(f"Carbon intensity: {benchmark['plant_value']:.3f} tons/MWh")
logger.info(f"Cluster median: {benchmark['cluster_median']:.3f} tons/MWh")
logger.info(f"Performance: {benchmark['vs_median']:+.1f}% vs peers")
logger.info(f"Percentile: {benchmark['percentile']:.0f}th")
state_profiles = plants_2023.groupby("Plant state abbreviation").apply(
    lambda df: pd.Series(
        {
            "pct_cluster_0": (df["cluster"] == 0).sum() / len(df) * 100,
            "pct_cluster_1": (df["cluster"] == 1).sum() / len(df) * 100,
            "pct_cluster_2": (df["cluster"] == 2).sum() / len(df) * 100,
            "pct_cluster_3": (df["cluster"] == 3).sum() / len(df) * 100,
            "pct_cluster_4": (df["cluster"] == 4).sum() / len(df) * 100,
        }
    )
)
state_scaler = StandardScaler()
state_profiles_scaled = state_scaler.fit_transform(state_profiles)
state_kmeans = KMeans(n_clusters=4, random_state=42)
state_clusters = state_kmeans.fit_predict(state_profiles_scaled)
state_profiles["state_cluster"] = state_clusters
logger.info("\nState Cluster Profiles:")
for i in range(4):
    states_in_cluster = state_profiles[
        state_profiles["state_cluster"] == i
    ].index.tolist()
    logger.info(f"\nCluster {i}: {', '.join(states_in_cluster)}")
    logger.info(state_profiles[state_profiles["state_cluster"] == i].mean())
from sklearn.neighbors import NearestNeighbors


def find_similar_plants(target_plant_idx, X_scaled, n_neighbors=10):
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors([X_scaled[target_plant_idx]])
    return (indices[0][1:], distances[0][1:])


target_idx = 1000
similar_idx, similar_dist = find_similar_plants(target_idx, X_scaled)
logger.info("Most similar plants (competitors):")
for i, (idx, dist) in enumerate(zip(similar_idx, similar_dist), 1):
    plant = plants_2023.iloc[X.index[idx]]
    logger.info(
        f"{i}. {plant.get('Plant name', 'Unknown')} ({plant.get('Plant state abbreviation', '??')}) - Distance: {dist:.3f}"
    )
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
results.append(
    {
        "k": k,
        "inertia": kmeans.inertia_,
        "silhouette": silhouette_score(X_scaled, labels),
        "calinski": calinski_harabasz_score(X_scaled, labels),
    }
)
logger.info(f"K={k}: Silhouette={results[-1]['silhouette']:.3f}")
gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
gmm.fit(X_scaled)
bic_scores.append(gmm.bic(X_scaled))
logger.info(f"n={n}: BIC={gmm.bic(X_scaled):.1f}")
n_components = (5,)
covariance_type = ("full",)
random_state = 42
min_cluster_size = (50,)
min_samples = (10,)
metric = "euclidean"
c = (cluster_labels[sample_idx],)
cluster_median = cluster_data["carbon_intensity"].median()
plant_value = plant_row["carbon_intensity"]
percentile = (cluster_data["carbon_intensity"] < plant_value).mean() * 100
return {
    "cluster_median": cluster_median,
    "plant_value": plant_value,
    "percentile": percentile,
    "vs_median": (plant_value - cluster_median) / cluster_median * 100,
}
lambda df: pd.Series(
    {
        "pct_cluster_0": (df["cluster"] == 0).sum() / len(df) * 100,
        "pct_cluster_1": (df["cluster"] == 1).sum() / len(df) * 100,
        "pct_cluster_2": (df["cluster"] == 2).sum() / len(df) * 100,
        "pct_cluster_3": (df["cluster"] == 3).sum() / len(df) * 100,
        "pct_cluster_4": (df["cluster"] == 4).sum() / len(df) * 100,
    }
)
states_in_cluster = state_profiles[state_profiles["state_cluster"] == i].index.tolist()
logger.info(f"\nCluster {i}: {', '.join(states_in_cluster)}")
logger.info(state_profiles[state_profiles["state_cluster"] == i].mean())
nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
nn.fit(X_scaled)
distances, indices = nn.kneighbors([X_scaled[target_plant_idx]])
return (indices[0][1:], distances[0][1:])
plant = plants_2023.iloc[X.index[idx]]
logger.info(
    f"{i}. {plant.get('Plant name', 'Unknown')} ({plant.get('Plant state abbreviation', '??')}) - Distance: {dist:.3f}"
)
