

import training
import scoring
import deployment
import diagnostics
import reporting
import json
import os
import ast
import ingestion, deployment, diagnostics, reporting
import pandas as pd
from sklearn import metrics

##################Check and read new data
#first, read ingestedfiles.txt
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']
model_path = config['output_model_path']

move_to_next_step = False

with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt'), 'r') as f:
    ingestedfiles = ast.literal_eval(f.read())
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
files = os.listdir(input_folder_path)
files = [file for file in files if file not in ingestedfiles]


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if files !=[]:
    ingestion.merge_multiple_dataframe()
    move_to_next_step = True
else:
    pass

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
if move_to_next_step:
    with open(os.path.join(prod_deployment_path, 'latestscore.txt')) as f:
        latest_score = ast.literal_eval(f.read())
    
    data_df = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))

    new_y = data_df['exited']
    new_y_pred = diagnostics.model_predictions(data_df)
    
    new_score = metrics.f1_score(new_y, new_y_pred)

    if new_score >= latest_score:
        # no model drift
        move_to_next_step = False


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if move_to_next_step:
    training.train_model()
    scoring.score_model()



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
if move_to_next_step:
    deployment.store_model_into_pickle()


##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
if move_to_next_step:
    reporting.score_model()
    os.system('python apicalls.py')








