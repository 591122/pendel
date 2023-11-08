import numpy as np
import matplotlib.pyplot as plt


import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Import Matplotlib


# Open the CSV file and read the data




# Define constants
m = 1
L = 0.68509933
g = 9.8
c = 0.1


current_dir = os.getcwd()
csv_file_path = os.path.join(current_dir, 'pendel/recorded_data/45_lang_2.csv')
df = pd.read_csv(csv_file_path)




# Define the range of variables for plotting
v = df['Vinkel']
u = df['Vinkel_hastighet']


# Generate a grid of points
# Calculate the rates of change
du_dt = v
dv_dt = (-c*L*u-m*g*np.sin(u))/(m*L**2)
# Plot the vector field
plt.quiver(u, v, du_dt, dv_dt)
# Set axis labels and title
plt.xlabel('theta')
plt.ylabel('thetadt')
plt.title('Phase Plane')
# Show the plot
plt.show()