import pandas as pd
import matplotlib.pyplot as plt
import os  # Import the 'os' module

# Get the current working directory
current_directory = os.getcwd()

# Specify the full file path
file_path = os.path.join(current_directory, 'recorded_data.csv')

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Extract the 'Position_X' and 'Time (s)' columns
position_x = df['Vinkel']
time = df['Time (s)']

# Create a scatter plot
plt.scatter(time, position_x)

# Create a scatter of vinkel_hastighet
plt.scatter(time, df['Vinkel_hastighet'])

# Set labels and title
plt.xlabel('Time (s)')
plt.ylabel('Position_X')
plt.title('Position_X vs. Time')

# Show the plot
plt.show()