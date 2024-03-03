import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    # Initialize an empty list to store dataframes
    dfs = []
    filenames_records = []
    dfs_rows_number_records = []
    # Loop through files in the input folder
    for file_name in os.listdir(input_folder_path):
        # Check if file is a CSV file
        if file_name.endswith('.csv'):
            # Append filename to list
            filenames_records.append(file_name)
            # Read CSV file into a dataframe
            file_path = os.path.join(input_folder_path, file_name)
            df = pd.read_csv(file_path)
            # Append dataframe size rows to list
            dfs_rows_number_records.append(str(len(df.index)))
            # Append dataframe to list
            dfs.append(df)
    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat(dfs, ignore_index=True)
    # Drop duplicate rows
    combined_df = combined_df.drop_duplicates()
    # Save the combined dataset to a CSV file in the output folder
    output_file_path = os.path.join(output_folder_path, 'finaldata.csv')
    combined_df.to_csv(output_file_path, index=False)
    
    with open(f'{output_folder_path}/ingestedfiles.txt', "w") as f:
        f.write(f"Ingestion folder path: {input_folder_path}\n")
        f.write(f"Ingestion files names: {','.join(filenames_records)}\n")
        f.write(f"Ingestion rows number: {','.join(dfs_rows_number_records)}\n")
        f.write(f"Ingestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"{'-'*2}\n")

if __name__ == '__main__':
    merge_multiple_dataframe()
