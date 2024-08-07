
# Weighted-K-Means Clustering from Scratch

## Introduction

This project implements the K-Means clustering algorithm from scratch in Python, inspired by the paper "Automated Variable Weighting in k-means Type Clustering" by Huang et al. The implementation includes functions for centroid initialization using the k-means++ method, distance calculation, cluster assignment, centroid updates, and visualization of the clustering process.

## Files

- `wk_means.py`: Contains the implementation of the K-Means algorithm and supporting functions.
- `README.md`: Documentation of the project and explanation of the implementation.

## Installation

To run this code, you'll need Python 3 and the following libraries:

- NumPy
- Pandas
- Matplotlib

You can install the necessary libraries using pip:

```bash
pip install numpy pandas matplotlib
```

## Usage

The main function to use is `K_Means`, which performs the Weighted-K-Means clustering on a given dataset.

### Example

```python
import numpy as np
from k_means import K_Means

# Create a sample dataset
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

n, m = X.shape
k = 2
max_iter = 100
beta = 0.5

# Run K-M

eans
centroids, W, U = K_Means(n, k, m, X, beta, max_iter)

print("Centroids:\n", centroids)
print("Weights:\n", W)
print("Cluster Assignments:\n", U)
```

## Functions

### `init_centroids_kmeanspp(X, k)`

Initializes centroids using the k-means++ method to ensure better clustering results.

### `init_weights(m)`

Initializes the feature weights `W`.

### `init_matrixes(m,n,k)`

Initializes the matrices `D`, `U`, and `W_D` used in the algorithm.

### `dist(X, Z, i, j, l)`

Calculates the squared distance between a data point and a centroid.

### `compute_W_D_i_l(m, beta, X, Z, W, i, l)`

Computes the weighted distance for a single data point and a single centroid.

### `modify_W_D(n, k, m, beta, X, Z, W, W_D)`

Updates the weighted distance matrix `W_D` for all data points and centroids.

### `find_min_W_D(W_D, i, k)`

Finds the index of the centroid with the minimum weighted distance to a data point.

### `modify_U(n, k, U, W_D)`

Updates the cluster assignment matrix `U` based on the current weighted distances.

### `modify_Z(n, k, m, X, Z, U)`

Updates the centroids `Z` based on the current cluster assignments.

### `modify_D(n, k, m, X, U, Z, D)`

Updates the distance matrix `D` with the distances between all data points and centroids.

### `find_min_D(m, D)`

Finds the index of the feature with the minimum distance.

### `modify_W(m, beta, W, D)`

Updates the feature weights `W` based on the current distances.

### `show_clusters(X, cluster, cg)`

Visualizes the current state of the clusters and centroids.

### `K_Means(n, k, m, X, beta, max_iter)`

Main function to perform Weighted-K-Means clustering. It initializes the centroids, iteratively updates the assignments and centroids, and checks for convergence.

## Reference

The following paper inspires the implementation of the Weighted-K-Means algorithm in this project:

Huang JZ, Ng MK, Rong H, Li Z. Automated variable weighting in k-means type clustering. IEEE Trans Pattern Anal Mach Intell. 2005 May;27(5):657-68. doi: 10.1109/TPAMI.2005.95. PMID: 15875789.

## License

This project is licensed under the MIT License.
