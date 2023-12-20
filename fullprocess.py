"""
This script used to run comeplete process

Author: Akshay Dhotre
Date: December 2023
"""

import os
import sys
import ast
import re
import json
import logging
import pandas as pd
from sklearn import metrics
import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

logging.info('Check and read new data')
# first, read ingestedfiles.txt
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']
model_path = config['output_model_path']

MOVE_TO_NEXT_STEP = False
logging.info("Starting automated monitoring")
with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt'), 'r', encoding='utf-8') as f:
    ingestedfiles = ast.literal_eval(f.read())
# second, determine whether the source data folder has files that aren't
# listed in ingestedfiles.txt
files = os.listdir(input_folder_path)
files = [file for file in files if file not in ingestedfiles]


# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if files != []:
    logging.info("ingesting new files")
    ingestion.merge_multiple_dataframe()
    MOVE_TO_NEXT_STEP = True
else:
    logging.info("No new files - ending process")

# Checking for model drift
# check whether the score from the deployed model is different from the
# score from the model that uses the newest ingested data
if MOVE_TO_NEXT_STEP:
    with open(os.path.join(prod_deployment_path, 'latestscore.txt'), encoding='utf-8') as f:
        latest_score = re.findall(r'\d*\.?\d+', f.read())[0]
        latest_score = float(latest_score)

    data_df = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))

    new_y = data_df['exited']
    new_y_pred = diagnostics.model_predictions(data_df)

    new_score = metrics.f1_score(new_y, new_y_pred)

    logging.info(f'latest score: {latest_score}, new score: {new_score}')

    if new_score >= latest_score:
        MOVE_TO_NEXT_STEP = False
        logging.info('No model drift found')


# if you found model drift, you should proceed. otherwise, do end the
# process here
if MOVE_TO_NEXT_STEP:
    logging.info('Training and scoring new model')
    training.train_model()
    scoring.score_model()


# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
if MOVE_TO_NEXT_STEP:
    logging.info('Deployment of new model')
    deployment.store_model_into_pickle()


# Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
if MOVE_TO_NEXT_STEP:
    logging.info('Produce report and call APIs for statistics')
    reporting.score_model()
    os.system('python apicalls.py')
