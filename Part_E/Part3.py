# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score
from sklearn.utils import shuffle

# Load your dataset
data = pd.read_csv('Part_E\\data\\solar_weather.csv')

# Exclude the first column since it's a string
data = data.iloc[:, 1:]

# Reduce Dataset Size by Sampling
sampled_data = shuffle(data, random_state=42).sample(frac=0.1, random_state=42)  

# Preprocessing: Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(sampled_data)

# Dimensionality Reduction using PCA
pca = PCA(n_components=0.95)  
reduced_data = pca.fit_transform(scaled_data)

# Agglomerative Clustering
def agglomerative_clustering_experiments(data):
    linkage_methods = ['ward', 'complete', 'average']
    for linkage in linkage_methods:
        print(f"Agglomerative Clustering with linkage method: {linkage}")
        model = AgglomerativeClustering(n_clusters=10, linkage=linkage)
        clusters = model.fit_predict(data)
        print(f"Cluster labels: {clusters}")

        children = model.children_
        n_samples = len(data)
        linkage_matrix = np.column_stack([children, np.arange(n_samples, 2*n_samples - 1, dtype=float), np.ones(n_samples - 1)])
        
         # Plot dendrogram
        plt.figure(figsize=(12, 6))
        plt.title(f"Dendrogram for Agglomerative Clustering with {linkage} linkage")
        dendrogram(linkage_matrix, labels=clusters, truncate_mode='level', p=3)
        plt.xlabel("Sample Index")
        plt.ylabel("Cluster Distance")
        plt.show()

# K-Means Clustering
def kmeans_experiments(data):
    k_values = [2, 5, 10, 15, 20]
    for k in k_values:
        print(f"K-Means Clustering with {k} clusters")
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        clusters = model.fit_predict(data)
        print(f"Cluster labels: {clusters}")
        silhouette_avg = silhouette_score(data, clusters)
        print(f"Silhouette Score: {silhouette_avg}")
        # Plotting
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.5)
        plt.title(f"K-Means Clustering with {k} clusters")
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='Cluster')
        plt.show()

def main():
    print("Agglomerative Clustering Experiments:")
    print(f"Data shape after PCA: {reduced_data.shape}")
    agglomerative_clustering_experiments(reduced_data)

    print("\nK-Means Clustering Experiments:")
    kmeans_experiments(reduced_data)

if __name__ == "__main__":
    main()
