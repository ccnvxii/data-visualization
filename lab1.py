import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------
# 1. 2D plot of a function y(x) and z(x)
# --------------------------------------
# Create data points
x = np.linspace(-2, 2, 1000)
y = np.cos(5 * np.pi * x) * (np.sin(3 * np.pi * x) ** 2) + 3 * np.sin(np.pi * x) * (np.cos(3 * np.pi * x) ** 3)

# Define z function
def z_func(x):
    return np.where(x <= 0, np.sqrt(1 + x ** 2), (1 + x ** 3) / (1 + np.sqrt(1 + np.exp(-0.5 * x))))

z = z_func(x)

# Plot y function
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, y, label='y')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = cos(5πx)sin²(3πx) + 3sin(πx)cos³(3πx)')
plt.grid(True)
plt.legend()
plt.text(-1.8, -1.5, 'y = cos(5πx)sin²(3πx) + 3sin(πx)cos³(3πx)', fontsize=10)

# Plot z function
plt.subplot(1, 2, 2)
plt.plot(x, z, label='z', color='orange')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Plot of z = √(1+x²) if x≤0, else (1+x³)/(1+√(1+e⁻⁰·⁵ˣ)) if x > 0')
plt.grid(True)
plt.legend()
plt.text(-1.8, 2.5, "z(x) = √(1+x²), if x <= 0\n"
                    "z(x) = (1+x³)/(1+√(1+e⁻⁰·⁵ˣ)), if x > 0", fontsize=10)

plt.tight_layout()
plt.show()

# ----------------------------------------
# 2. Surface plot of z = 10x²cos⁵(x) - 2y³
# ----------------------------------------
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)  # Create a 2D grid of x and y values

# Compute Z values for surface
Z = 10 * X ** 2 * np.cos(X) ** 5 - 2 * Y ** 3

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
plt.colorbar(surf, label='z value')  # Add color bar

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Surface Plot of z = 10x²cos⁵(x) - 2y³')
ax.grid(True)

# Add equation as text on plot
ax.text2D(0.05, 0.95, 'z = 10x²cos⁵(x) - 2y³', transform=ax.transAxes)
plt.show()

# -----------------------------------------------
# 3. Surface plot from polar-like coordinates (ρ)
# -----------------------------------------------
# Define parameters
a = 1  # scaling factor
b = 0.5  # secondary scaling factor
phi = np.linspace(0, 2*np.pi, 200)  # azimuthal angles
theta = np.linspace(0, np.pi, 100)  # polar angles

# 2D Polar Plot (Equatorial cut)
# Compute rho at theta = π/2 (equatorial plane)
cos_2phi = np.cos(2 * phi)  # cos(2*phi) term
term1 = cos_2phi + np.sqrt(np.maximum(cos_2phi**2 * ((b**2 / a**2) - 1), 0))  # avoid negative sqrt
rho = a * np.sqrt(np.maximum(term1, 0))  # ensure non-negative values

# Create polar plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='polar')
ax.plot(phi, rho, color='blue', lw=2)

# Configure polar plot
ax.set_rmax(1.2)  # maximum radius
ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])  # radial ticks
ax.set_rlabel_position(-22.5)  # radial label position
ax.grid(True)
ax.set_title("Polar Plot of ρ(φ) at θ = π/2", va='bottom')
plt.show()

# 3D Surface Plot
# Create 2D grids of angles
PHI, THETA = np.meshgrid(phi, theta)

# Compute rho for all (phi, theta) avoiding imaginary numbers
cos_2phi = np.cos(2 * PHI)
term1 = cos_2phi + np.sqrt(np.maximum(cos_2phi**2 * ((b**2 / a**2) - 1), 0))
rho = a * np.sqrt(np.maximum(term1, 0))

# Convert polar-like coordinates to Cartesian coordinates
X = rho * np.sin(THETA) * np.cos(PHI)
Y = rho * np.sin(THETA) * np.sin(PHI)
Z = rho * np.cos(THETA)

# Plot the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.9)

# Label axes and add title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(r'Surface plot: $\rho = a \sqrt{\cos(2\phi) + \sqrt{\cos^2(2\phi)(b^2/a^2 - 1)}}$')

# Add color bar
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

# ---------------------------------------------
# 4. Surface of 2nd order (elliptic paraboloid)
# ---------------------------------------------
a = 2.0  # semi-major axis
b = 1.5  # semi-minor axis

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# z = (x²/a² + y²/b²)/2
Z = (X ** 2 / a ** 2 + Y ** 2 / b ** 2) / 2

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')

plt.colorbar(surf, label='z value')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Surface Plot of x²/a² + y²/b² = 2z')
ax.grid(True)
ax.text2D(0.05, 0.95, r'$\frac{x^2}{a^2} + \frac{y^2}{b^2} = 2z$', transform=ax.transAxes)
plt.show()

# -------------------------------------
# 5. 2D and 3D Bar Charts of Population
# -------------------------------------
countries = ['USA', 'Germany', 'France', 'Japan', 'USSR']
years = [1900, 1913, 1929, 1938, 1950, 1960, 1970, 1980, 1990, 2000]
population = {
    'USA': [76.4, 97.6, 122.2, 130.5, 153, 176, 200.5, 227, 247, 277],
    'Germany': [45.7, 54.7, 58.7, 62.3, 67, 72, 77, 78.5, 79, 82],
    'France': [40.8, 41.8, 42, 42, 42, 46, 50.5, 54, 56.5, 59],
    'Japan': [44, 51.6, 63.2, 71.8, 83, 93, 104, 116.8, 123.5, 127],
    'USSR': [123, 158, 171.5, 186.5, 205.5, 226.5, 247, 258.5, 290, 290]
}

# --- 2D Bar Chart ---
x = np.arange(len(years))  # X-axis positions
width = 0.15  # Width of bars

fig, ax = plt.subplots(figsize=(12, 6))
for i, country in enumerate(countries):
    ax.bar(x + i * width, population[country], width, label=country)

ax.set_xlabel('Year')
ax.set_ylabel('Population (million)')
ax.set_title('Population by Country (2D Bar Chart)')
ax.set_xticks(x + width * 2)  # Center tick labels
ax.set_xticklabels(years)
ax.legend()
plt.show()

# --- 3D Bar Chart ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

_x = np.arange(len(years))
_y = np.arange(len(countries))
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

# Heights of bars
top = []
for country in countries:
    top.extend(population[country])
top = np.array(top)

z = np.zeros_like(x)
dx = dy = 0.8  # Bar width

ax.bar3d(x, y, z, dx, dy, top, shade=True)
ax.set_xticks(np.arange(len(years)))
ax.set_xticklabels(years)
ax.set_yticks(np.arange(len(countries)))
ax.set_yticklabels(countries)
ax.set_xlabel('Year')
ax.set_ylabel('Country')
ax.set_zlabel('Population (million)')
ax.set_title('Population by Country (3D Bar Chart)')
plt.show()
