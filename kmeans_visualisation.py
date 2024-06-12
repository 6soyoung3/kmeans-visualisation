import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, load_iris, load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set the seaborn style
sns.set(style="whitegrid", palette="muted")

def plot_voronoi(X, kmeans, ax, n_clusters):
    """
    Plots the Voronoi diagram for the current state of the KMeans algorithm.
    """
    # Determine the minimum and maximum values for the mesh grid, extending the range by 1 unit
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a mesh grid over the data range with a step size of 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict the cluster for each point in the mesh grid
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)  # Reshape the prediction result to match the mesh grid shape

    # Plot the Voronoi diagram using the contourf function
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='Pastel1')

    # Scatter plot of the data points coloured by their cluster assignment
    scatter = ax.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=50, cmap='viridis', marker='o', edgecolor='k')

    # Scatter plot of the cluster centroids marked with a red 'X'
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')

    # Set the title and labels for the axes
    ax.set_title(f'K-Means Clustering with {n_clusters} Centroids')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

def visualise_kmeans(X, title):
    """
    Visualises the K-Means clustering process with different numbers of centroids.
    """
    n_clusters_list = [1, 2, 3, 4]
    fig, axes = plt.subplots(1, len(n_clusters_list), figsize=(20, 5), sharey=True)
    
    # Set the window title for the figure
    fig.canvas.manager.set_window_title(title)

    for i, n_clusters in enumerate(n_clusters_list):
        kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
        kmeans.fit(X)  # Fit the KMeans algorithm to the data
        plot_voronoi(X, kmeans, axes[i], n_clusters)  # Plot the Voronoi diagram for the current number of centroids

    plt.suptitle(title, y=1.05)
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85)
    plt.show()  # Display the plots

if __name__ == "__main__":
    # Sample 1: Synthetic dataset using make_blobs
    X_blobs, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
    
    # Sample 2: Iris dataset
    iris = load_iris()
    X_iris = iris.data[:, :2]  # Use only the first two features for 2D visualisation
    X_iris = StandardScaler().fit_transform(X_iris)  # Standardise the features
    
    # Sample 3: Wine dataset
    wine = load_wine()
    X_wine = wine.data[:, :2]  # Use only the first two features for 2D visualisation
    X_wine = StandardScaler().fit_transform(X_wine)  # Standardise the features

    # Visualise the K-Means clustering process for each dataset
    visualise_kmeans(X_blobs, 'Synthetic Blobs')
    visualise_kmeans(X_iris, 'Iris Dataset')
    visualise_kmeans(X_wine, 'Wine Dataset')
