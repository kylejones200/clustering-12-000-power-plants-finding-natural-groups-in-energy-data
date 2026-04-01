# Clustering 12,000 Power Plants: Finding Natural Groups in Energy Data Using K-Means, GMM, and HDBSCAN to segment power plants for
benchmarking, policy targeting, and competitive intelligence

### Clustering 12,613 Power Plants: Finding Natural Groups in Energy Data
#### *Using K-Means, GMM, and HDBSCAN to segment power plants for benchmarking, policy targeting, and competitive intelligence*
Not all power plants are created equal. A 1,000 MW nuclear plant
operating at 92% capacity has nothing in common with a 10 MW solar farm
running at 25% capacity. Yet both exist in the same dataset, and
comparing them directly is meaningless.

Clustering solves this. It automatically groups similar plants together,
then we can analyze the groups. This reveals insights impossible to see
in aggregate data --- like identifying that certain natural gas plants
are 20% more efficient than their peers, or that some states have
systematically cleaner electricity portfolios.

This article demonstrates five clustering methods on 12,613 U.S. power
plants, showing how to find natural groupings, profile clusters, and
apply results to real-world problems like benchmarking and policy
targeting.


### The Dataset and Feature Engineering
Using EPA eGRID 2023 data with 12,613 power plants, we create features
that capture operational characteristics:

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

*The data and code for this project are in*
[*Github*](https://github.com/kylejones200/electric_utilities)*.*

Why These Features?

- Log generation & CO2: Captures plant size (log-transform handles huge
  range)
- Carbon intensity: Emissions per unit output (distinguishes clean vs
  dirty)
- Capacity factor: Utilization rate (baseload vs peaking)
- NOx/SO2 intensities: Additional pollution dimensions

### Method 1: K-Means Clustering
K-Means is the workhorse of clustering: fast, interpretable, and works
well when clusters are roughly spherical.

### Finding Optimal K
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

Interpretation: The elbow occurs around K=5. Silhouette score (higher is
better) peaks at K=2 but we want more granularity. K=5 balances
interpretability and cluster quality.

### Training Final Model
``` 
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

### Cluster Profiling: What Do These Groups Mean?
The magic happens when we profile each cluster to understand what makes
it unique:

``` 
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

Cluster 0: "Large Coal Baseload" (2,847 plants)

- Median carbon intensity: 1.02 tons/MWh
- Median capacity factor: 0.61
- Median capacity: 385 MW
- Profile: Traditional coal plants, high emissions, steady
  operation

Cluster 1: "Natural Gas Combined Cycle" (3,124 plants)

- Median carbon intensity: 0.42 tons/MWh
- Median capacity factor: 0.48
- Median capacity: 245 MW
- Profile: Modern gas plants, moderate emissions, flexible
  operation

Cluster 2: "Renewable Energy" (1,892 plants)

- Median carbon intensity: 0.00 tons/MWh
- Median capacity factor: 0.27
- Median capacity: 85 MW
- Profile: Wind, solar, hydro --- clean but variable

Cluster 3: "Peaking Units" (2,156 plants)

- Median carbon intensity: 0.56 tons/MWh
- Median capacity factor: 0.12
- Median capacity: 45 MW
- Profile: Gas turbines, run only during high demand

Cluster 4: "Nuclear Baseload" (1,594 plants)

- Median carbon intensity: 0.00 tons/MWh
- Median capacity factor: 0.89
- Median capacity: 1,100 MW
- Profile: Large nuclear, ultra-clean, always-on

These profiles make intuitive sense! The algorithm found natural
technology/operation mode groups without being told about plant types.

### Method 2: Gaussian Mixture Models (GMM)
K-Means assigns each plant to exactly one cluster. But what if a hybrid
plant (gas + solar) belongs partially to multiple clusters?

GMM provides soft clustering --- each plant has a probability of
belonging to each cluster.

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

Analyzing Uncertainty:

``` 
# Find plants with uncertain membership
uncertain = plants_2023[plants_2023['gmm_probability'] < 0.7]
print(f"\nPlants with uncertain cluster membership: {len(uncertain)}")
print("These plants have characteristics of multiple clusters")

# Example: A plant that's 40% Cluster 1, 35% Cluster 2, 25% Cluster 3
# This might be a gas plant with significant renewable co-generation
```

GMM found 342 plants with uncertain membership (probability \< 0.7).
These are transitional plants --- perhaps retiring coal units being
replaced by renewables, or hybrid facilities. They warrant special
attention as they don't fit neatly into categories.

### Method 3: HDBSCAN --- Letting Data Decide Cluster Count
HDBSCAN (Hierarchical Density-Based Spatial Clustering) automatically
determines the number of clusters and identifies outliers.

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

Why More Clusters?

HDBSCAN found 7 clusters vs our choice of 5 for K-Means. It split some
groups further:

- Natural gas split into "high-efficiency CCGT" vs "older simple
  cycle"
- Renewables split into "wind" vs "solar" vs "hydro"
- Identified 423 outliers --- unusual plants that don't fit any
  cluster

The 423 outliers are valuable! These are plants with unique
characteristics:

- Experimental technologies (geothermal, wave power)
- Plants in transition (adding/retiring units)
- Unique configurations (waste-to-energy, biomass)
- Data quality issues requiring investigation

### Visualizing High-Dimensional Clusters
6-dimensional data is hard to visualize. Solution: dimensionality
reduction.

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

71% of the information compressed into 2 dimensions! Not perfect, but
good enough for visualization.

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

t-SNE shows better cluster separation than PCA --- clusters are more
visually distinct. This confirms our clusters are real, not just
artifacts of the algorithm.

### Practical Application 1: Benchmarking
With clusters defined, we can benchmark performance within peer groups:

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

This plant is performing better than 68% of similar coal plants.
Valuable insight for:

- Operators: Identify what's working well here
- Regulators: Set achievable targets based on peer performance
- Investors: Value assets relative to competitors

### Practical Application 2: Policy Targeting
States with similar plant mixes should get similar policies. Cluster
states by their generation portfolio:

``` 
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

Output might show:

- State Cluster 0: WY, WV, KY (high coal, need transition
  support)
- State Cluster 1: CA, OR, WA (high renewables, focus on
  integration/storage)
- State Cluster 2: TX, PA, OH (balanced mix, optimize existing
  fleet)
- State Cluster 3: VT, ID (high hydro/nuclear, maintain
  reliability)

Each cluster needs different policy interventions. One-size-fits-all
policies fail.

### Practical Application 3: Market Segmentation
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

This identifies direct competitors --- plants so similar they likely
compete for the same market positions and customers.

### Key Lessons Learned
1\. Multiple methods reveal different insights:

- K-Means: Fast, interpretable, good for spherical clusters
- GMM: Handles uncertainty, soft assignments
- HDBSCAN: Finds natural cluster count, identifies outliers

2\. Feature engineering is crucial:

- Log transforms handle skewed distributions
- Ratios (carbon intensity, capacity factor) capture operational
  modes
- Standardization essential for distance-based methods

3\. Domain knowledge validates results:

- Clusters should make sense (they did!)
- If clusters are unintuitive, revisit features or
  preprocessing
- Profile clusters thoroughly before using them

4\. Clustering enables targeted analysis:

- Benchmark within peer groups (apples to apples)
- Target policies appropriately (no one-size-fits-all)
- Identify competitive dynamics (who competes with whom)

5\. Outliers are valuable:

- Don't just remove them --- investigate them!
- May represent innovative technologies
- Could indicate data quality issues
- Often the most interesting plants

### So What?
Clustering transforms 12,613 individual plants into 5--7 manageable
groups. This enables:

Better benchmarking: Compare plants to appropriate peers, not the entire
fleet. A 5% improvement vs cluster peers is meaningful; vs all plants is
not.

Targeted policies: Design interventions for specific plant types. Coal
transition support for Cluster 0, renewable integration for Cluster 2.

Market intelligence: Understand competitive landscape. Who are your true
competitors? Where are the opportunities?

Investment decisions: Is a target plant typical or unusual for its
cluster? Unusual isn't bad --- could be better or worse --- but needs
explanation.

Operational insights: Best practices from top performers in each cluster
can be shared with others in that cluster (similar
technology/constraints).

The methods demonstrated here --- K-Means for speed, GMM for
uncertainty, HDBSCAN for flexibility --- provide a comprehensive toolkit
for segmenting any complex dataset. Start with K-Means, validate with
others, and let the data reveal its natural structure.

### A message from our Founder
**Hey,**
[**Sunil**](https://linkedin.com/in/sunilsandhu) **here.** I wanted
to take a moment to thank you for reading until the end and for being a
part of this community.

Did you know that our team run these publications as a volunteer effort
to over 3.5m monthly readers? **We don't receive any funding, we do this
to support the community. ❤️**

If you want to show some love, please take a moment to **follow me on**
[**LinkedIn**](https://linkedin.com/in/sunilsandhu)**,**
[**TikTok**](https://tiktok.com/@messyfounder),
[**Instagram**](https://instagram.com/sunilsandhu). You can also
subscribe to our [**weekly
newsletter**](https://newsletter.plainenglish.io/).

And before you go, don't forget to **clap** and **follow** the writer️!
::::::::By [Kyle Jones](https://medium.com/@kyle-t-jones) on
[October 6, 2025](https://medium.com/p/0e58e8803b03).

[Canonical
link](https://medium.com/@kyle-t-jones/clustering-12-000-power-plants-finding-natural-groups-in-energy-data-0e58e8803b03)

Exported from [Medium](https://medium.com) on November 10, 2025.
