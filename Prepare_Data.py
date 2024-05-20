# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:47:11 2024

@author: najdi.boubker
"""

import os
import pandas as pd

# Define the root directory containing the bearing folders
root_dir = "C:\\Users\\najdi.boubker\\Desktop\\phm-ieee-2012-data-challenge-dataset-master\\phm-ieee-2012-data-challenge-dataset-master\\Learning_set"  # Change this to your directory path

# List of bearing folders
bearing_folders = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2"]

# Initialize a list to hold the data
data_list = []

# Process each bearing folder
for folder in bearing_folders:
    folder_path = os.path.join(root_dir, folder)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            # Read the CSV file
            df = pd.read_csv(file_path, header=None)
            # Extract the 5th column (index 4) with 2560 observations
            x_vibrational_signal = df.iloc[:, 4].values
            # Append to the data list
            data_list.append(x_vibrational_signal)
            print(file_name)

# Convert the list to a DataFrame
final_df = pd.DataFrame(data_list)

# Save the concatenated DataFrame to a new CSV file
final_df.to_csv("Train_Data_H.csv", index=False)
