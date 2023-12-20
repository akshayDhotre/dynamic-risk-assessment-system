"""
This script is to train model for ingested data

Author: Akshay Dhotre
Date: December 2023
"""
import os
import sys
import json
import pickle
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')


def split_target_from_dataset(dataset):
    """
    Eliminate features not used
    segregate the dataset into X and y
    input: dataset to segregate
    output: X and y
    """
    logging.info('Splitting target variable from features')

    # Target variable
    y = dataset.pop('exited')

    # Features
    X = dataset.drop(['corporation'], axis=1)

    return X, y

# Function for training the model


def train_model():
    """
    Train a logistic regression model for churn classification
    Input: None
    Output: trained model saved to disk
    """
    logging.info('Read data to get features and target')
    data_df = pd.read_csv(dataset_csv_path)
    X, y = split_target_from_dataset(data_df)

    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    logging.info('Train model')
    model.fit(X, y)

    logging.info('Save trained model')
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)


if __name__ == '__main__':
    logging.info('Running model training process')
    train_model()
