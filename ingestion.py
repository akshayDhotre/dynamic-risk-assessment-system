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
    df_list = []
    ingested_file_names = []
    for dir_path, _, file_names in os.walk(input_folder_path):
        for file_name in file_names:
            if file_name.endswith('.csv'):
                full_file_path = os.path.join(dir_path, file_name)
                candidate_df = pd.read_csv(full_file_path, encoding='utf-8')
                df_list.append(candidate_df)
                ingested_file_names.append(file_name)
    main_df = pd.concat(df_list)
    main_df.drop_duplicates(inplace=True)
    main_df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False, encoding='utf-8')

    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as md_file:
        md_file.write(str(ingested_file_names))

if __name__ == '__main__':
    merge_multiple_dataframe()
