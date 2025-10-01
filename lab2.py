import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------
# 1) Scalar field u(x,y)
# ----------------------
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x, y)

# Define scalar field
U = 7 * np.log(X ** 2 + 1 / 13) - 4 * np.sin(X * Y)

# Gradient of scalar field (∇u)
dx = x[1] - x[0]
dy = y[1] - y[0]
Uy, Ux = np.gradient(U, dy, dx)

# Plot scalar field as a colored surface
plt.figure(figsize=(8, 6))
pcm = plt.pcolormesh(X, Y, U, shading='auto', cmap='viridis')
plt.colorbar(pcm, label='u(x,y)')

# Add contour lines on top
contours = plt.contour(X, Y, U, colors='k', linewidths=0.6, alpha=0.6)
plt.clabel(contours, inline=True, fontsize=8)

# Overlay gradient vectors (downsampled for clarity)
step = 10
plt.quiver(X[::step, ::step], Y[::step, ::step],
           Ux[::step, ::step], Uy[::step, ::step],
           color='white', scale=50, label='∇u (gradient)')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Scalar field u(x,y) and its gradient')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------
# 2) 2D vector field F = (x^2 y, -y)
# ----------------------------------
x = np.linspace(-10, 10, 20)
y = np.linspace(-10, 10, 20)
X, Y = np.meshgrid(x, y)

Fx = X ** 2 * Y
Fy = -Y

plt.figure(figsize=(12, 5))

# Vector field using quiver (arrows)
plt.subplot(1, 2, 1)
plt.quiver(X, Y, Fx, Fy)
plt.title("Vector field F(x,y) with quiver")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

# Vector field using streamlines
plt.subplot(1, 2, 2)
plt.streamplot(X, Y, Fx, Fy, color="blue")
plt.title("Vector field F(x,y) with streamlines")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# 3) 3D vector field F = ((X+Z)/(X^2+eps), 1/(Y+eps), 1/(Z+eps))
# --------------------------------------------------------------
x = np.linspace(-3, 3, 8)
y = np.linspace(-3, 3, 8)
z = np.linspace(-3, 3, 8)
X, Y, Z = np.meshgrid(x, y, z)
eps = 1e-3

# Components of vector field
U = (X + Z) / (X ** 2 + eps)
V = 1.0 / (Y + eps)
W = 1.0 / (Z + eps)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Normalize vectors for visualization
length = np.sqrt(U ** 2 + V ** 2 + W ** 2)
length[length == 0] = 1.0
Un = U / length
Vn = V / length
Wn = W / length

# 3D quiver plot
ax.quiver(X, Y, Z, Un, Vn, Wn, length=0.4, linewidth=0.6)
ax.set_title('3D vector field F=((x+z)/(x^2+ε), 1/(y+ε), 1/(z+ε))')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(True)
ax.legend([Patch(color='grey')], ['3D quiver (normalized)'], loc='upper left')
plt.tight_layout()
plt.show()

# ----------------------------------------------
# 4) Tensor field visualization using glyphs
#    (ellipsoids, cuboids, cylinders, superquadrics)
# ----------------------------------------------

# Tensor matrix definition
def tensor_matrix(x, y, z):
    return np.array([
        [np.log(x) / np.sin(x), np.sqrt(x) / y, np.sqrt(y) / z],
        [0.0, np.log(y) / np.sin(y), np.sqrt(z) / x],
        [0.0, 0.0, np.log(z) / np.sin(z)]
    ])

# Draw an ellipsoid
def plot_ellipsoid(ax, center, radii, rotation=np.eye(3), color='C0', alpha=0.5):
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Apply radii and rotation
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            vec = np.array([x[i, j], y[i, j], z[i, j]])
            p = rotation @ (radii * vec)
            x[i, j], y[i, j], z[i, j] = p + center

    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True)

# Draw a cuboid
def plot_cuboid(ax, center, size, color='C1', alpha=0.5):
    l, w, h = size
    x = np.array([-l / 2, l / 2]) + center[0]
    y = np.array([-w / 2, w / 2]) + center[1]
    z = np.array([-h / 2, h / 2]) + center[2]
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, np.full_like(X, z[0]), color=color, alpha=alpha)
    ax.plot_surface(X, Y, np.full_like(X, z[1]), color=color, alpha=alpha)
    Y, Z = np.meshgrid(y, z)
    ax.plot_surface(np.full_like(Y, x[0]), Y, Z, color=color, alpha=alpha)
    ax.plot_surface(np.full_like(Y, x[1]), Y, Z, color=color, alpha=alpha)
    X, Z = np.meshgrid(x, z)
    ax.plot_surface(X, np.full_like(X, y[0]), Z, color=color, alpha=alpha)
    ax.plot_surface(X, np.full_like(X, y[1]), Z, color=color, alpha=alpha)

# Draw a cylinder
def plot_cylinder(ax, center, radius, height, color='C2', alpha=0.5):
    z = np.linspace(center[2] - height / 2, center[2] + height / 2, 20)
    theta = np.linspace(0, 2 * np.pi, 20)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center[0]
    y_grid = radius * np.sin(theta_grid) + center[1]
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha)

# Draw a superquadric (generalized ellipsoid)
def plot_superquadric(ax, center, a, b, c, n1=1, n2=1, color='C3', alpha=0.5):
    u = np.linspace(-np.pi / 2, np.pi / 2, 20)
    v = np.linspace(-np.pi, np.pi, 20)
    u, v = np.meshgrid(u, v)
    cosu = np.sign(np.cos(u)) * np.abs(np.cos(u)) ** n1
    sinu = np.sign(np.sin(u)) * np.abs(np.sin(u)) ** n1
    cosv = np.sign(np.cos(v)) * np.abs(np.cos(v)) ** n2
    sinv = np.sign(np.sin(v)) * np.abs(np.sin(v)) ** n2
    x = a * cosu * cosv + center[0]
    y = b * cosu * sinv + center[1]
    z = c * sinu + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

# Main visualization loop
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Grid of points where tensors will be visualized
xs = np.linspace(1.1, 3.0, 3)
ys = np.linspace(1.1, 3.0, 3)
zs = np.linspace(1.1, 3.0, 3)

colors = ['C0', 'C1', 'C2', 'C3']

for i, xi in enumerate(xs):
    for j, yi in enumerate(ys):
        for k, zi in enumerate(zs):
            try:
                T = tensor_matrix(xi, yi, zi)
            except Exception:
                continue
            Ts = 0.5 * (T + T.T)  # symmetric part of tensor
            w, v = np.linalg.eigh(Ts)  # eigenvalues/eigenvectors
            radii = np.sqrt(np.abs(w))
            radii = np.clip(radii, 0.05, 1.0)
            center = [xi, yi, zi]
            color = colors[(i + j + k) % len(colors)]

            # Visualize with different glyph types
            plot_ellipsoid(ax, center, radii, rotation=v, color=color, alpha=0.5)
            plot_cuboid(ax, center, size=radii, color=color, alpha=0.3)
            plot_cylinder(ax, center, radius=radii[0] / 2, height=radii[1] * 2, color=color, alpha=0.3)
            plot_superquadric(ax, center, a=radii[0], b=radii[1], c=radii[2],
                              n1=1.0, n2=1.0, color=color, alpha=0.3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Tensor field visualization: ellipsoids, cuboids, cylinders, superquadrics')
ax.grid(True)
plt.tight_layout()
plt.show()