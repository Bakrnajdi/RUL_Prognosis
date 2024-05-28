# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:47:11 2024

@author: najdi.boubker
"""

import os
import pandas as pd

# Define the root directory containing the bearing folders
# root_dir = "C:\\Users\\najdi.boubker\\Desktop\\phm-ieee-2012-data-challenge-dataset-master\\phm-ieee-2012-data-challenge-dataset-master\\Learning_set"  # Change this to your directory path
root_dir = "/Users/mac/Desktop/PhD/Prognosis_RUL/phm-ieee-2012-data-challenge-dataset-master/Learning_set/"  # Change this to your directory path

# List of bearing folders
bearing_folders = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2"]

# Initialize a list to hold the data
data_list = []

# Dictionary to hold the number of files processed in each folder
files_count = {}

# Process each test bearing folder
for folder in bearing_folders:
    folder_path = os.path.join(root_dir, folder)
    count = 0  # Initialize the count of files for this folder
    for file_name in os.listdir(folder_path):
        if file_name.startswith('acc') and file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            # Read the CSV file
            df = pd.read_csv(file_path, header=None)
            # Extract the 5th column (index 4) with 2560 observations
            x_vibrational_signal = df.iloc[:, 4].values
            # Append to the data list
            data_list.append(x_vibrational_signal)
            count += 1  # Increment the count
    # Save the count for this folder
    files_count[folder] = count

# Convert the list to a DataFrame
final_df = pd.DataFrame(data_list)

# Save the concatenated DataFrame to a new CSV file
final_df.to_csv("Train_Data_H.csv", index=False)

# Save the file counts to a new CSV file
files_count_df = pd.DataFrame(list(files_count.items()), columns=['Folder', 'File_Count'])
files_count_df.to_csv("Train_files_count.csv", index=False)

# Display the files count
print(files_count_df)




import os
import pandas as pd

# Define the root directory containing the bearing folders
root_dir = "/Users/mac/Desktop/PhD/Prognosis_RUL/phm-ieee-2012-data-challenge-dataset-master/Test_set"  # Change this to your directory path

# List of test bearing folders
test_bearing_folders = [
    "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7","Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7","Bearing3_3"
]

# Initialize a list to hold the data
data_list = []

# Dictionary to hold the number of files processed in each folder
files_count = {}

# Process each test bearing folder
for folder in test_bearing_folders:
    folder_path = os.path.join(root_dir, folder)
    count = 0  # Initialize the count of files for this folder
    
    # List and sort the files in the folder
    file_names = sorted([f for f in os.listdir(folder_path) if f.startswith('acc') and f.endswith('.csv')])
    
    
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        print(file_path)
        # Read the CSV filess
        df = pd.read_csv(file_path, header=None)
        # Extract the 5th column (index 4) with 2560 observations
        x_vibrational_signal = df.iloc[:, 4].values
        # Append to the data list
        data_list.append(x_vibrational_signal)
        count += 1  # Increment the count
    # Save the count for this folder
    files_count[folder] = count

# Convert the list to a DataFrame
final_df = pd.DataFrame(data_list)

# Save the concatenated DataFrame to a new CSV file
final_df.to_csv("Test_Data.csv", index=False)

# Save the file counts to a new CSV file
files_count_df = pd.DataFrame(list(files_count.items()), columns=['Folder', 'File_Count'])
files_count_df.to_csv("files_count.csv", index=False)

# Display the files count
print(files_count_df)

