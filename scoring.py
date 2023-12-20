"""
This script is for scoring model after training

Author: Akshay Dhotre
Date: December 2023
"""
import os
import sys
import pickle
import json
import logging
import pandas as pd
from sklearn import metrics

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and get path variables
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')


def score_model():
    """
    Load a trained model and the test data, calculate an F1 score
    for the model on the test data then saves the result to
    the latestscore.txt file
    """
    logging.info('Loading and preparing testdata')
    test_df = pd.read_csv(test_data_path, encoding='utf-8')

    y = test_df.pop('exited')
    X = test_df.drop(['corporation'], axis=1)

    logging.info('Load model')
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    logging.info("Predicting on test data")
    y_pred = model.predict(X)

    f1 = metrics.f1_score(y, y_pred)

    logging.info('Saving score to text file')
    with open(os.path.join(config['output_model_path'], 'latestscore.txt'),
              'w', encoding='utf-8') as md_file:
        md_file.write(f"f1 score = {f1}")
    return f1


if __name__ == '__main__':
    logging.info('Running scoring process')
    score_model()
