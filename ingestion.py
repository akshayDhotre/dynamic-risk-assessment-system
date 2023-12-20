"""
This script used to ingested data

Author: Akshay Dhotre
Date: December 2023
"""
import os
import sys
import json
import logging
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and get input and output paths
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def merge_multiple_dataframe():
    """
    Function for data ingestion. Check for datasets, combine them together,
    drops duplicates and write metadata ingestedfiles.txt and ingested data
    to finaldata.csv
    """
    logging.info(f"Reading datasets from {input_folder_path}")

    df_list = []
    ingested_file_names = []
    for dir_path, _, file_names in os.walk(input_folder_path):
        for file_name in file_names:
            if file_name.endswith('.csv'):
                full_file_path = os.path.join(dir_path, file_name)
                candidate_df = pd.read_csv(full_file_path, encoding='utf-8')
                df_list.append(candidate_df)
                ingested_file_names.append(file_name)

    logging.info("Combining datasets and removing duplicates")
    main_df = pd.concat(df_list)
    main_df.drop_duplicates(inplace=True)

    logging.info("Saving ingested dataset")
    main_df.to_csv(
        os.path.join(
            output_folder_path,
            'finaldata.csv'),
        index=False,
        encoding='utf-8')

    logging.info("Saving ingested metadata")
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'),
              'w', encoding='utf-8') as md_file:
        md_file.write(str(ingested_file_names))


if __name__ == '__main__':
    logging.info("Running data ingestion process")
    merge_multiple_dataframe()
