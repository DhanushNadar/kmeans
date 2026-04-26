import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    print("--- Online Learning Behavior ---")
    
    files = []
    for root, _, filenames in os.walk('dataset'):
        for file in filenames:
            if file.endswith('.csv') or file.endswith('.data'):
                files.append(os.path.join(root, file))
                
    if not files:
        print("Dataset not found!")
        return
        
    try:
        df = pd.read_csv(files[0])
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("\n--- Part A: Preprocessing & EDA ---")
    
    df_numeric = df.select_dtypes(include=[np.number])
    
    if df_numeric.empty:
        print("No numeric data available for clustering.")
        return
        
    print("Missing values before cleaning:")
    print(df_numeric.isnull().sum())
    
    for col in df_numeric.columns:
        df_numeric[col] = df_numeric[col].fillna(df_numeric[col].median())
        
    print("\nFeature scaling...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    print("Visualizing feature distributions...")
    plot_cols = df_numeric.columns[:4]
    df_numeric[plot_cols].hist(figsize=(8, 6), bins=15)
    plt.tight_layout()
    plt.show()

    print("\n--- Part B: Model Building ---")
    print("Applying Elbow Method to determine optimal K...")
    
    wcss = []
    max_k = min(10, len(df_numeric) - 1)
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, wcss, marker='o', linestyle='--')
    plt.title('Elbow Method For Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS (Inertia)')
    plt.tight_layout()
    plt.show()
    
    optimal_k = min(3, max_k)
    print(f"Training K-Means model using optimal K={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    df_numeric['Cluster'] = clusters
    
    print("\n--- Part C: Evaluation & Interpretation ---")
    inertia = kmeans.inertia_
    sil_score = silhouette_score(scaled_data, clusters) if optimal_k > 1 else 0
    
    print(f"Inertia (WCSS):     {inertia:.2f}")
    print(f"Silhouette Score:   {sil_score:.4f}")
    
    print("\nVisualizing clusters...")
    if scaled_data.shape[1] > 2:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)
        reduced_centroids = pca.transform(kmeans.cluster_centers_)
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette='viridis', legend='full', alpha=0.6)
        plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
        plt.title('K-Means Clusters (2D PCA Projection)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(8, 6))
        col1, col2 = df_numeric.columns[0], df_numeric.columns[1]
        sns.scatterplot(x=df_numeric[col1], y=df_numeric[col2], hue=clusters, palette='viridis', legend='full', alpha=0.6)
        plt.title('K-Means Clusters')
        plt.tight_layout()
        plt.show()
        
    print("\n--- Interpretation ---")
    print("Clusters have been formed grouping similar data points. The Centroids (red X) represent the center of each group.")

if __name__ == "__main__":
    main()
