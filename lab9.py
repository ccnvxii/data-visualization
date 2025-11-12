import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load the dataset
data = pd.read_csv('Wholesale_customers_data.csv')
X = data.iloc[:, 2:].values  # Select numerical features: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
y = data['Channel'].values  # Use Channel for coloring in plots
m, n = X.shape  # m samples, n features

# Using PCA to visualize data in 2D and 3D spaces
print("PCA visualization in 2D and 3D")

# 2D PCA
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis')
plt.title('PCA 2D Visualization')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.show()

# 3D PCA
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y, cmap='viridis')
ax.set_title('PCA 3D Visualization')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.colorbar(scatter)
plt.show()

# Center the data for SVD
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# Compute SVD
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Eigenvalues from singular values (lambda_i = sigma_i^2 / (m-1))
eigenvalues = (S ** 2) / (m - 1)

# Plot the dependence of the eigenvalues on their number (descending order)
print("Plot eigenvalues in descending order")
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.title('Eigenvalues in Descending Order')
plt.xlabel('Eigenvalue Number')
plt.ylabel('Eigenvalue')
plt.grid(True)
plt.show()

# Determine the smallest i such that sum of top i eigenvalues / total >= 0.8
print("Find smallest d for 80% significance")
total_variance = np.sum(eigenvalues)
cumulative_variance = np.cumsum(eigenvalues)
explained_ratio = cumulative_variance / total_variance
d = np.argmax(explained_ratio >= 0.8) + 1  # 1-based index
print(f"Smallest d: {d} (explained variance: {explained_ratio[d - 1]:.4f})")

# Set lambda_i to zero for i >= d (0-based i >= d-1), reconstruct, compare
print("Reconstruct with d={} and compare".format(d))
S_trunc = S.copy()
S_trunc[d:] = 0  # Set singular values from d onwards to 0 (0-based)
Sigma_trunc = np.diag(S_trunc)
X_approx = U @ Sigma_trunc @ Vt + X_mean  # Reconstruct and add mean back
mse = mean_squared_error(X, X_approx)
print(f"Mean Squared Error between original and approximated: {mse:.4f}")

# Optionally plot original vs approx for one feature, e.g., first feature (Fresh)
plt.figure(figsize=(8, 6))
plt.plot(X[:, 0], label='Original')
plt.plot(X_approx[:, 0], label='Approximated')
plt.title('Comparison of First Feature: Original vs Approximated')
plt.legend()
plt.show()

# Set d=2 and d=3, reconstruct reduced data, plot first d "columns" of reconstructed data
# For d=2
print(" For d=2")
d = 2
# Reconstruct with first 2 components
S_trunc_2 = S.copy()
S_trunc_2[2:] = 0
Sigma_trunc_2 = np.diag(S_trunc_2)
X_approx_2d = (U @ Sigma_trunc_2 @ Vt) + X_mean
plt.figure(figsize=(8, 6))
plt.scatter(X_approx_2d[:, 0], X_approx_2d[:, 1], c=y, cmap='viridis')
plt.title('Reconstructed 2D Visualization (First 2 Columns)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()
print("Compare to PCA 2D: Differences may occur due to reconstruction approximation.")

# For d=3
print("For d=3")
d = 3
# Reconstruct with first 3 components
S_trunc_3 = S.copy()
S_trunc_3[3:] = 0
Sigma_trunc_3 = np.diag(S_trunc_3)
X_approx_3d = (U @ Sigma_trunc_3 @ Vt) + X_mean
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_approx_3d[:, 0], X_approx_3d[:, 1], X_approx_3d[:, 2], c=y, cmap='viridis')
ax.set_title('Reconstructed 3D Visualization (First 3 Columns)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.colorbar(scatter)
plt.show()
print("Compare to PCA 3D: Differences may occur due to reconstruction approximation.")
