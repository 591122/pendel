import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the current working directory
current_directory = os.getcwd()

# Specify the directory containing the CSV files
data_folder = os.path.join(current_directory, 'recorded_data')

# List all CSV files in the directory
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# Loop through each CSV file
for csv_file in csv_files:
    # Construct the full file path
    file_path = os.path.join(data_folder, csv_file)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Extract the 'Position_X' and 'Time (s)' columns
    position_x = df['Vinkel']
    time = df['Time (s)']

    # Create a scatter plot
    plt.scatter(time, position_x)

    # Create a scatter of vinkel_hastighet
    #plt.scatter(time, df['Vinkel_hastighet'])

    # Set labels and title
    plt.xlabel('Tid (s)')
    plt.ylabel('Vinkel')
    plt.title(f'Vinkel over tid for: {csv_file}')

    # Show the plot
    plt.show()
