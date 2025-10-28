import numpy as np
import matplotlib.pyplot as plt
from wk_means import init_centroids_kmeanspp, init_matrices, K_Means

np.random.seed(0)
X1 = np.random.normal(loc=[0, 0], scale=0.5, size=(100, 2))
X2 = np.random.normal(loc=[4, 4], scale=0.5, size=(100, 2))
X3 = np.random.normal(loc=[0, 5], scale=0.5, size=(100, 2))
X = np.vstack((X1, X2, X3))
n, m = X.shape
k = 3

beta = 8.0   
max_iter = 100

Z, W, U, iters, obj = K_Means(n, k, m, X, beta, max_iter=max_iter)
labels = np.argmax(U, axis=1)

print("\nFinal Results:")
print("Iterations:", iters)
print("Variable Weights (W):", np.round(W, 4))
print("Objective Value:", round(obj, 4))

plt.figure(figsize=(7, 7))
for i in range(k):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f"Cluster {i+1}")
plt.scatter(Z[:, 0], Z[:, 1], c='red', marker='*', s=200, label='Centroids')
plt.xlabel("X₁")
plt.ylabel("X₂")
plt.title("Weighted K-Means Clustering Result")
plt.legend()
plt.show()
