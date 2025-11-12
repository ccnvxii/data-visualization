# Task
# Develop a program that builds animation according to the option, save it in gif format.
# The options:
# A rectangle with rounded edges moving along a sine wave;

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyBboxPatch

# 1. Create figure and axes
fig = plt.figure()
fig.set_dpi(100)  # Set resolution
fig.set_size_inches(7, 6.5)  # Set figure size
ax = plt.axes(xlim=(0, 10), ylim=(0, 10))  # Set axes limits

# 2. Create a rectangle with rounded edges
width, height = 1.5, 1
patch = FancyBboxPatch((0, 5), width, height, boxstyle="round,pad=0.1", fc='skyblue')

# 3. Initialize the animation
def init():
    patch.set_bounds(0, 5, width, height)  # Set initial position
    ax.add_patch(patch)  # Add rectangle to axes
    return patch,

# 4. Animation function (called for each frame)
def animate(i):
    x = 0.05 * i  # Move along x-axis
    y = 5 + 2 * np.sin(0.5 * x * np.pi)  # Move along sine wave on y-axis
    patch.set_bounds(x, y, width, height)  # Update rectangle position
    return patch,

# 5. Create the animation
anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=200,  # Number of frames
                               interval=50,  # Delay between frames (ms)
                               blit=True)

# 6. Save the animation as a GIF
anim.save("rounded_rectangle_sine.gif", writer='pillow')

# Display the plot
plt.show()
