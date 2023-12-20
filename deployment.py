"""
This script is to deply the trained model

Author: Akshay Dhotre
Date: December 2023
"""
import os
import sys
import logging
import json
import shutil

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and correct path variable
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])


def store_model_into_pickle():
    """
    Copy the latest model pickle file, the latestscore.txt value,
    and the ingestedfiles.txt file into the deployment directory
    """
    for file in ['latestscore.txt', 'trainedmodel.pkl']:
        logging.info(f'deploying file - {file}')
        shutil.copy(
            os.path.join(
                model_path, file), os.path.join(
                prod_deployment_path, file))

    logging.info('Deploying ingestedfiles metadata')
    shutil.copy(os.path.join(dataset_csv_path, 'ingestedfiles.txt'),
                os.path.join(prod_deployment_path, 'ingestedfiles.txt'))


if __name__ == '__main__':
    logging.info('Running deployment process')
    store_model_into_pickle()
