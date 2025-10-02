import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, TSNE
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Wholesale_customers_data.csv")
numeric_data = data.select_dtypes(include=np.number)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_data)

# Dimensionality Reduction Models
n_neighbors = 10  # for LLE and Isomap
methods = {
    "Isomap": Isomap(n_neighbors=n_neighbors, n_components=2),
    "Standard LLE": LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method="standard"),
    "MDS": MDS(n_components=2, n_init=4, random_state=42),
    "t-SNE": TSNE(n_components=2, random_state=42)
}

# 2D Visualization
plt.figure(figsize=(12, 10))
for i, (name, model) in enumerate(methods.items(), 1):
    X_2d = model.fit_transform(X_scaled)
    plt.subplot(2, 2, i)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=30, alpha=0.7)
    plt.title(f"{name} (2D)")
plt.tight_layout()
plt.show()

#  3D Visualization
isomap_3d = Isomap(n_neighbors=n_neighbors, n_components=3)
X_iso_3d = isomap_3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_iso_3d[:, 0], X_iso_3d[:, 1], X_iso_3d[:, 2], c='skyblue', s=40, alpha=0.7)
ax.set_title("Isomap 3D Embedding")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")
plt.show()
