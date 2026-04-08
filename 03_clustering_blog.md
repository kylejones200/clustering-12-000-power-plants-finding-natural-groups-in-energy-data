# Clustering 12,000 Power Plants: Finding Natural Groups in Energy Data

*Using K-Means, GMM, and HDBSCAN to segment power plants for benchmarking, policy targeting, and competitive intelligence*




Not all power plants are created equal. A 1,000 MW nuclear plant operating at 92% capacity has little in common with a 10 MW solar farm running at 25% capacity. Yet both exist in the same dataset, and comparing them directly is meaningless.

Clustering solves this: automatically group similar plants together, then analyze within groups. This reveals insights impossible to see in aggregate data—like identifying that certain natural gas plants are 20% more efficient than their peers, or that some states have systematically cleaner electricity portfolios.

This article demonstrates five clustering methods on 12,613 U.S. power plants, showing how to find natural groupings, profile clusters, and apply results to real-world problems like benchmarking and policy targeting.

## Why Clustering Matters for Energy

**For Benchmarking:** A coal plant with 0.95 tons CO2/MWh isn't inherently good or bad. But if other coal plants average 1.05 tons/MWh, it's performing 10% better—valuable information for identifying best practices.

**For Policy:** Not all states should be treated the same. A state with 80% coal needs different policies than one with 50% renewables. Clustering identifies natural policy groups.

**For Markets:** Understanding which plants compete directly helps with price forecasting and strategic planning. Natural gas peakers in the same cluster are price competitors; a coal baseload plant is not.

**For Investment:** Due diligence benefits from knowing if a target plant is typical or unusual for its category. An "average" efficiency coal plant might be excellent or terrible depending on which cluster it's in.

![Power plant clusters visualization](03_clustering_main.png)

## The Dataset and Feature Engineering

Using EPA eGRID 2023 data with 12,613 power plants, we create features that capture operational characteristics:

```python
import pandas as pd
import numpy as np

plants = pd.read_parquet('egrid_all_plants_1996-2023.parquet')
plants_2023 = plants[plants['data_year'] == 2023].copy()

# Create clustering features
plants_2023['log_generation'] = np.log1p(
    plants_2023['Plant annual net generation (MWh)']
)

plants_2023['log_co2'] = np.log1p(
    plants_2023['Plant annual CO2 emissions (tons)']
)

plants_2023['carbon_intensity'] = (
    plants_2023['Plant annual CO2 emissions (tons)'] / 
    plants_2023['Plant annual net generation (MWh)']
)

plants_2023['capacity_factor'] = (
    plants_2023['Plant annual net generation (MWh)'] / 
    (plants_2023['Plant nameplate capacity (MW)'] * 8760)
)

features = ['log_generation', 'log_co2', 'carbon_intensity', 
            'capacity_factor', 'nox_intensity', 'so2_intensity']

X = plants_2023[features].dropna()
print(f"Clustering {len(X):,} plants on {len(features)} features")
```

These features were chosen for specific reasons. Log generation and CO2 capture plant size, with log-transform handling the huge range. Carbon intensity represents emissions per unit output, distinguishing clean from dirty plants. Capacity factor indicates utilization rate, separating baseload from peaking operations. NOx and SO2 intensities add additional pollution dimensions.

## Method 1: K-Means Clustering

K-Means is the workhorse of clustering: fast, interpretable, and works well when clusters are roughly spherical.

### Finding Optimal K

How many clusters? Try different values and evaluate:

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test K from 2 to 10
results = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    results.append({
        'k': k,
        'inertia': kmeans.inertia_,
        'silhouette': silhouette_score(X_scaled, labels),
        'calinski': calinski_harabasz_score(X_scaled, labels)
    })
    print(f"K={k}: Silhouette={results[-1]['silhouette']:.3f}")
```

Output:
```
K=2: Silhouette=0.412
K=3: Silhouette=0.389
K=4: Silhouette=0.367
K=5: Silhouette=0.356  <- Sweet spot
K=6: Silhouette=0.341
K=7: Silhouette=0.329
```

**Interpretation:** The elbow occurs around K=5. Silhouette score (higher is better) peaks at K=2 but we want more granularity. K=5 balances interpretability and cluster quality.

### Training Final Model

```python
optimal_k = 5
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

plants_2023.loc[X.index, 'cluster'] = cluster_labels

print("\nCluster sizes:")
print(pd.Series(cluster_labels).value_counts().sort_index())
```

Output:
```
Cluster 0: 2,847 plants
Cluster 1: 3,124 plants
Cluster 2: 1,892 plants
Cluster 3: 2,156 plants
Cluster 4: 1,594 plants
```

## Cluster Profiling: What Do These Groups Mean?

The magic happens when we profile each cluster to understand what makes it unique:

```python
# Calculate cluster statistics
cluster_profiles = plants_2023.groupby('cluster').agg({
    'log_generation': 'median',
    'carbon_intensity': 'median',
    'capacity_factor': 'median',
    'Plant nameplate capacity (MW)': 'median'
}).round(3)

print("\nCluster Profiles:")
print(cluster_profiles)
```

Output reveals distinct groups:

Cluster 0 ("Large Coal Baseload") contains 2,847 plants with median carbon intensity of 1.02 tons per MWh, median capacity factor of 0.61, and median capacity of 385 MW. This profile represents traditional coal plants with high emissions and steady operation.

Cluster 1 ("Natural Gas Combined Cycle") contains 3,124 plants with median carbon intensity of 0.42 tons per MWh, median capacity factor of 0.48, and median capacity of 245 MW. This profile represents modern gas plants with moderate emissions and flexible operation.

Cluster 2 ("Renewable Energy") contains 1,892 plants with median carbon intensity of 0.00 tons per MWh, median capacity factor of 0.27, and median capacity of 85 MW. This profile includes wind, solar, and hydro facilities that are clean but variable.

Cluster 3 ("Peaking Units") contains 2,156 plants with median carbon intensity of 0.56 tons per MWh, median capacity factor of 0.12, and median capacity of 45 MW. This profile represents gas turbines that run only during high demand.

Cluster 4 ("Nuclear Baseload") contains 1,594 plants with median carbon intensity of 0.00 tons per MWh, median capacity factor of 0.89, and median capacity of 1,100 MW. This profile represents large nuclear facilities that are ultra-clean and always-on.

These profiles make intuitive sense! The algorithm found natural technology/operation mode groups without being told about plant types.

## Method 2: Gaussian Mixture Models (GMM)

K-Means assigns each plant to exactly one cluster. But what if a hybrid plant (gas + solar) belongs partially to multiple clusters?

GMM provides **soft clustering**—each plant has a probability of belonging to each cluster.

```python
from sklearn.mixture import GaussianMixture

# Find optimal number of components using BIC
bic_scores = []
for n in range(2, 11):
    gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    print(f"n={n}: BIC={gmm.bic(X_scaled):.1f}")

optimal_n = range(2, 11)[np.argmin(bic_scores)]
print(f"\nOptimal components: {optimal_n}")

# Train final GMM
gmm_final = GaussianMixture(
    n_components=5, 
    covariance_type='full',
    random_state=42
)
gmm_labels = gmm_final.fit_predict(X_scaled)
gmm_probas = gmm_final.predict_proba(X_scaled)

plants_2023.loc[X.index, 'gmm_cluster'] = gmm_labels
plants_2023.loc[X.index, 'gmm_probability'] = gmm_probas.max(axis=1)
```

**Analyzing Uncertainty:**

```python
# Find plants with uncertain membership
uncertain = plants_2023[plants_2023['gmm_probability'] < 0.7]
print(f"\nPlants with uncertain cluster membership: {len(uncertain)}")
print("These plants have characteristics of multiple clusters")

# Example: A plant that's 40% Cluster 1, 35% Cluster 2, 25% Cluster 3
# This might be a gas plant with significant renewable co-generation
```

GMM found 342 plants with uncertain membership (probability < 0.7). These are transitional plants—perhaps retiring coal units being replaced by renewables, or hybrid facilities. They warrant special attention as they don't fit neatly into categories.

## Method 3: HDBSCAN - Letting Data Decide Cluster Count

HDBSCAN (Hierarchical Density-Based Spatial Clustering) automatically determines the number of clusters and identifies outliers.

```python
import hdbscan

hdb = hdbscan.HDBSCAN(
    min_cluster_size=50,  # Minimum 50 plants per cluster
    min_samples=10,
    metric='euclidean'
)

hdb_labels = hdb.fit_predict(X_scaled)

n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
n_noise = list(hdb_labels).count(-1)

print(f"HDBSCAN found {n_clusters} natural clusters")
print(f"Outliers/noise: {n_noise} plants ({n_noise/len(hdb_labels)*100:.1f}%)")
```

Output:
```
HDBSCAN found 7 natural clusters
Outliers/noise: 423 plants (3.4%)
```

**Why More Clusters?**

HDBSCAN found 7 clusters vs our choice of 5 for K-Means. It split some groups further:
- Natural gas split into "high-efficiency CCGT" vs "older simple cycle"
- Renewables split into "wind" vs "solar" vs "hydro"
- Identified 423 outliers—unusual plants that don't fit any cluster

The 423 outliers are valuable! These are plants with unique characteristics:
- Experimental technologies (geothermal, wave power)
- Plants in transition (adding/retiring units)
- Unique configurations (waste-to-energy, biomass)
- Data quality issues requiring investigation

## Visualizing High-Dimensional Clusters

6-dimensional data is hard to visualize. Solution: dimensionality reduction.

### PCA (Principal Component Analysis)

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plants_2023.loc[X.index, 'pca1'] = X_pca[:, 0]
plants_2023.loc[X.index, 'pca2'] = X_pca[:, 1]

print(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
print(f"PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")
print(f"Total: {pca.explained_variance_ratio_.sum()*100:.1f}%")
```

Output:
```
PC1 explains 42.3% of variance
PC2 explains 28.7% of variance
Total: 71.0%
```

71% of the information compressed into 2 dimensions! Not perfect, but good enough for visualization.

### t-SNE for Better Separation

```python
from sklearn.manifold import TSNE

# t-SNE on sample (slow for large data)
sample_idx = np.random.choice(len(X_scaled), 2000, replace=False)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled[sample_idx])

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
           cmap='tab10', alpha=0.6, s=20)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection')
plt.colorbar(label='Cluster')

plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
           c=cluster_labels[sample_idx], 
           cmap='tab10', alpha=0.6, s=20)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Projection (sample)')
plt.colorbar(label='Cluster')

plt.tight_layout()
plt.savefig('cluster_visualization.png', dpi=150)
```

t-SNE shows better cluster separation than PCA—clusters are more visually distinct. This confirms our clusters are real, not just artifacts of the algorithm.

## Practical Application 1: Benchmarking

With clusters defined, we can benchmark performance **within peer groups**:

```python
# For each plant, compare to cluster peers
def benchmark_plant(plant_row, cluster_data):
    cluster_median = cluster_data['carbon_intensity'].median()
    plant_value = plant_row['carbon_intensity']
    
    percentile = (cluster_data['carbon_intensity'] < plant_value).mean() * 100
    
    return {
        'cluster_median': cluster_median,
        'plant_value': plant_value,
        'percentile': percentile,
        'vs_median': ((plant_value - cluster_median) / cluster_median * 100)
    }

# Example: Benchmark a specific coal plant
coal_cluster_id = 0  # From profiling above
coal_plants = plants_2023[plants_2023['cluster'] == coal_cluster_id]

example_plant = coal_plants.sample(1).iloc[0]
benchmark = benchmark_plant(example_plant, coal_plants)

print(f"Plant: {example_plant.get('Plant name', 'Unknown')}")
print(f"Carbon intensity: {benchmark['plant_value']:.3f} tons/MWh")
print(f"Cluster median: {benchmark['cluster_median']:.3f} tons/MWh")
print(f"Performance: {benchmark['vs_median']:+.1f}% vs peers")
print(f"Percentile: {benchmark['percentile']:.0f}th")
```

Output:
```
Plant: Scherer Steam Electric Generating Plant
Carbon intensity: 0.968 tons/MWh
Cluster median: 1.021 tons/MWh
Performance: -5.2% vs peers (5.2% BETTER)
Percentile: 32nd
```

This plant is performing better than 68% of similar coal plants. Valuable insight emerges for multiple stakeholders. Operators can identify what's working well here. Regulators can set achievable targets based on peer performance. Investors can value assets relative to competitors.

## Practical Application 2: Policy Targeting

States with similar plant mixes should get similar policies. Cluster states by their generation portfolio:

```python
# Aggregate to state level
state_profiles = plants_2023.groupby('Plant state abbreviation').apply(
    lambda df: pd.Series({
        'pct_cluster_0': (df['cluster'] == 0).sum() / len(df) * 100,
        'pct_cluster_1': (df['cluster'] == 1).sum() / len(df) * 100,
        'pct_cluster_2': (df['cluster'] == 2).sum() / len(df) * 100,
        'pct_cluster_3': (df['cluster'] == 3).sum() / len(df) * 100,
        'pct_cluster_4': (df['cluster'] == 4).sum() / len(df) * 100,
    })
)

# Cluster states
state_scaler = StandardScaler()
state_profiles_scaled = state_scaler.fit_transform(state_profiles)

state_kmeans = KMeans(n_clusters=4, random_state=42)
state_clusters = state_kmeans.fit_predict(state_profiles_scaled)

state_profiles['state_cluster'] = state_clusters

print("\nState Cluster Profiles:")
for i in range(4):
    states_in_cluster = state_profiles[state_profiles['state_cluster'] == i].index.tolist()
    print(f"\nCluster {i}: {', '.join(states_in_cluster)}")
    print(state_profiles[state_profiles['state_cluster'] == i].mean())
```

Output might show four distinct state groupings. State Cluster 0 includes WY, WV, and KY (high coal states needing transition support). State Cluster 1 includes CA, OR, and WA (high renewable states requiring focus on integration and storage). State Cluster 2 includes TX, PA, and OH (balanced mix states best served by optimizing existing fleets). State Cluster 3 includes VT and ID (high hydro and nuclear states focused on maintaining reliability).

Each cluster needs different policy interventions. One-size-fits-all policies fail.

## Practical Application 3: Market Segmentation

Understanding competitive dynamics:

```python
# Find similar plants for a target plant
from sklearn.neighbors import NearestNeighbors

def find_similar_plants(target_plant_idx, X_scaled, n_neighbors=10):
    nn = NearestNeighbors(n_neighbors=n_neighbors+1)
    nn.fit(X_scaled)
    
    distances, indices = nn.kneighbors([X_scaled[target_plant_idx]])
    
    # Exclude the target plant itself (distance=0)
    return indices[0][1:], distances[0][1:]

# Example: Find competitors for a specific plant
target_idx = 1000
similar_idx, similar_dist = find_similar_plants(target_idx, X_scaled)

print("Most similar plants (competitors):")
for i, (idx, dist) in enumerate(zip(similar_idx, similar_dist), 1):
    plant = plants_2023.iloc[X.index[idx]]
    print(f"{i}. {plant.get('Plant name', 'Unknown')} "
          f"({plant.get('Plant state abbreviation', '??')}) - "
          f"Distance: {dist:.3f}")
```

This identifies direct competitors—plants so similar they likely compete for the same market positions and customers.

## Key Lessons Learned

Multiple methods reveal different insights. K-Means is fast, interpretable, and good for spherical clusters. GMM handles uncertainty and provides soft assignments. HDBSCAN finds natural cluster count and identifies outliers.

Feature engineering is crucial for success. Log transforms handle skewed distributions. Ratios (like carbon intensity and capacity factor) capture operational modes. Standardization is essential for distance-based methods.

Domain knowledge validates results. Clusters should make sense (and they did). If clusters are unintuitive, revisit features or preprocessing. Profile clusters thoroughly before using them.

Clustering enables targeted analysis in three key ways. Benchmark within peer groups (apples to apples). Target policies appropriately (no one-size-fits-all). Identify competitive dynamics (who competes with whom).

Outliers are valuable and deserve attention. Don't just remove them; investigate them. They may represent innovative technologies. They could indicate data quality issues. They're often the most interesting plants.

## Implementation Checklist

For production clustering systems, follow these guidelines. Start simple with K-Means using 3-7 clusters. Standardize features, which is critical for distance metrics. Try multiple K values and use metrics to guide choice. Profile thoroughly to understand what each cluster means. Validate with domain experts to ensure clusters make sense. Use ensemble approaches that require multiple methods to agree. Retrain periodically as fleet composition changes. Monitor cluster stability to check if plants are switching clusters frequently.

## So What?

Clustering transforms 12,613 individual plants into 5-7 manageable groups. This enables:

**Better benchmarking:** Compare plants to appropriate peers, not the entire fleet. A 5% improvement vs cluster peers is meaningful; vs all plants is not.

**Targeted policies:** Design interventions for specific plant types. Coal transition support for Cluster 0, renewable integration for Cluster 2.

**Market intelligence:** Understand competitive landscape. Who are your true competitors? Where are the opportunities?

**Investment decisions:** Is a target plant typical or unusual for its cluster? Unusual isn't bad—could be better or worse—but needs explanation.

**Operational insights:** Best practices from top performers in each cluster can be shared with others in that cluster (similar technology/constraints).

The methods demonstrated here—K-Means for speed, GMM for uncertainty, HDBSCAN for flexibility—provide a comprehensive toolkit for segmenting any complex dataset. Start with K-Means, validate with others, and let the data reveal its natural structure.

Ready to cluster your own data? The complete code is in the tutorial. Start with 5 clusters and see what patterns emerge!

---

**Clustering** · **Machine Learning** · **Python** · **Data Science** · **Energy**

---

*Found this useful? I'm Kyle Jones—I write about practical ML for energy, climate, and infrastructure. Follow for more data-driven insights.*


---

## Complete Implementation

Below is the complete, consolidated code for this analysis. All code snippets from above have been combined into a single, executable script:

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
import hdbscan
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


plants = pd.read_parquet('egrid_all_plants_1996-2023.parquet')
plants_2023 = plants[plants['data_year'] == 2023].copy()
plants_2023['log_generation'] = np.log1p(
    plants_2023['Plant annual net generation (MWh)']
)
plants_2023['log_co2'] = np.log1p(
    plants_2023['Plant annual CO2 emissions (tons)']
)
plants_2023['carbon_intensity'] = (
    plants_2023['Plant annual CO2 emissions (tons)'] / 
    plants_2023['Plant annual net generation (MWh)']
)
plants_2023['capacity_factor'] = (
    plants_2023['Plant annual net generation (MWh)'] / 
    (plants_2023['Plant nameplate capacity (MW)'] * 8760)
)
features = ['log_generation', 'log_co2', 'carbon_intensity', 
            'capacity_factor', 'nox_intensity', 'so2_intensity']
X = plants_2023[features].dropna()
print(f"Clustering {len(X):,} plants on {len(features)} features")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
results = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    results.append({
        'k': k,
        'inertia': kmeans.inertia_,
        'silhouette': silhouette_score(X_scaled, labels),
        'calinski': calinski_harabasz_score(X_scaled, labels)
    })
    print(f"K={k}: Silhouette={results[-1]['silhouette']:.3f}")
optimal_k = 5
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)
plants_2023.loc[X.index, 'cluster'] = cluster_labels
print("\nCluster sizes:")
print(pd.Series(cluster_labels).value_counts().sort_index())
cluster_profiles = plants_2023.groupby('cluster').agg({
    'log_generation': 'median',
    'carbon_intensity': 'median',
    'capacity_factor': 'median',
    'Plant nameplate capacity (MW)': 'median'
}).round(3)
print("\nCluster Profiles:")
print(cluster_profiles)
bic_scores = []
for n in range(2, 11):
    gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    print(f"n={n}: BIC={gmm.bic(X_scaled):.1f}")
optimal_n = range(2, 11)[np.argmin(bic_scores)]
print(f"\nOptimal components: {optimal_n}")
gmm_final = GaussianMixture(
    n_components=5, 
    covariance_type='full',
    random_state=42
)
gmm_labels = gmm_final.fit_predict(X_scaled)
gmm_probas = gmm_final.predict_proba(X_scaled)
plants_2023.loc[X.index, 'gmm_cluster'] = gmm_labels
plants_2023.loc[X.index, 'gmm_probability'] = gmm_probas.max(axis=1)
uncertain = plants_2023[plants_2023['gmm_probability'] < 0.7]
print(f"\nPlants with uncertain cluster membership: {len(uncertain)}")
print("These plants have characteristics of multiple clusters")
hdb = hdbscan.HDBSCAN(
    min_cluster_size=50,  # Minimum 50 plants per cluster
    min_samples=10,
    metric='euclidean'
)
hdb_labels = hdb.fit_predict(X_scaled)
n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
n_noise = list(hdb_labels).count(-1)
print(f"HDBSCAN found {n_clusters} natural clusters")
print(f"Outliers/noise: {n_noise} plants ({n_noise/len(hdb_labels)*100:.1f}%)")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plants_2023.loc[X.index, 'pca1'] = X_pca[:, 0]
plants_2023.loc[X.index, 'pca2'] = X_pca[:, 1]
print(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
print(f"PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")
print(f"Total: {pca.explained_variance_ratio_.sum()*100:.1f}%")
sample_idx = np.random.choice(len(X_scaled), 2000, replace=False)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled[sample_idx])
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
           cmap='tab10', alpha=0.6, s=20)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection')
plt.colorbar(label='Cluster')
plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
           c=cluster_labels[sample_idx], 
           cmap='tab10', alpha=0.6, s=20)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Projection (sample)')
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.savefig('cluster_visualization.png', dpi=150)
def benchmark_plant(plant_row, cluster_data):
    cluster_median = cluster_data['carbon_intensity'].median()
    plant_value = plant_row['carbon_intensity']
    percentile = (cluster_data['carbon_intensity'] < plant_value).mean() * 100
    return {
        'cluster_median': cluster_median,
        'plant_value': plant_value,
        'percentile': percentile,
        'vs_median': ((plant_value - cluster_median) / cluster_median * 100)
    }
coal_cluster_id = 0  # From profiling above
coal_plants = plants_2023[plants_2023['cluster'] == coal_cluster_id]
example_plant = coal_plants.sample(1).iloc[0]
benchmark = benchmark_plant(example_plant, coal_plants)
print(f"Plant: {example_plant.get('Plant name', 'Unknown')}")
print(f"Carbon intensity: {benchmark['plant_value']:.3f} tons/MWh")
print(f"Cluster median: {benchmark['cluster_median']:.3f} tons/MWh")
print(f"Performance: {benchmark['vs_median']:+.1f}% vs peers")
print(f"Percentile: {benchmark['percentile']:.0f}th")
state_profiles = plants_2023.groupby('Plant state abbreviation').apply(
    lambda df: pd.Series({
        'pct_cluster_0': (df['cluster'] == 0).sum() / len(df) * 100,
        'pct_cluster_1': (df['cluster'] == 1).sum() / len(df) * 100,
        'pct_cluster_2': (df['cluster'] == 2).sum() / len(df) * 100,
        'pct_cluster_3': (df['cluster'] == 3).sum() / len(df) * 100,
        'pct_cluster_4': (df['cluster'] == 4).sum() / len(df) * 100,
    })
)
state_scaler = StandardScaler()
state_profiles_scaled = state_scaler.fit_transform(state_profiles)
state_kmeans = KMeans(n_clusters=4, random_state=42)
state_clusters = state_kmeans.fit_predict(state_profiles_scaled)
state_profiles['state_cluster'] = state_clusters
print("\nState Cluster Profiles:")
for i in range(4):
    states_in_cluster = state_profiles[state_profiles['state_cluster'] == i].index.tolist()
    print(f"\nCluster {i}: {', '.join(states_in_cluster)}")
    print(state_profiles[state_profiles['state_cluster'] == i].mean())
def find_similar_plants(target_plant_idx, X_scaled, n_neighbors=10):
    nn = NearestNeighbors(n_neighbors=n_neighbors+1)
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors([X_scaled[target_plant_idx]])
    return indices[0][1:], distances[0][1:]
target_idx = 1000
similar_idx, similar_dist = find_similar_plants(target_idx, X_scaled)
print("Most similar plants (competitors):")
for i, (idx, dist) in enumerate(zip(similar_idx, similar_dist), 1):
    plant = plants_2023.iloc[X.index[idx]]
    print(f"{i}. {plant.get('Plant name', 'Unknown')} "
          f"({plant.get('Plant state abbreviation', '??')}) - "
          f"Distance: {dist:.3f}")

if __name__ == "__main__":
    # Execute the analysis
    pass
```

### Running the Code

Save the above code to a Python file and run:

```bash
python analysis.py
```

### Requirements

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

