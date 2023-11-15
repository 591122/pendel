import numpy as np
import math
import matplotlib as mpl
mpl.use('pgf')
import matplotlib.pyplot as plt
plt.style.use('IEEE_report')

# Generate data|
theta_radians = np.linspace(0, np.pi/3, 1000)
theta_degrees = np.degrees(theta_radians)
sin_theta = np.sin(theta_radians)
theta_approx = theta_radians  # Small angle approximation

# Create a figure and axis with customized style
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(theta_radians, sin_theta, label='sin($\\theta$)')
ax.plot(theta_radians, theta_approx, label='$\\theta$')

# Set axis labels and title
ax.set_xlabel('(Radians | Degrees)')
ax.set_ylabel('Function Value')
ax.set_title('Small Angle Approximation')

# Customize tick labels for both radians and degrees
radian_ticks = np.linspace(0, np.pi/3, 5)
degree_ticks = np.arange(0, 61, 15)
all_ticks = np.concatenate([radian_ticks, np.radians(degree_ticks)])
all_tick_labels = [f'{int(math.degrees(x))}° | {x:.2f}' if x in np.radians(degree_ticks) else f'{x:.2f} / {int(math.degrees(x))}°' for x in all_ticks]

ax.set_xticks(all_ticks)
ax.set_xticklabels(all_tick_labels) # Rotate the labels by 45 degrees
ax.tick_params(axis='y')

# Add a legend
ax.legend(loc='upper left')

# Show the grid
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('norm.pgf', format='pgf')
plt.show()