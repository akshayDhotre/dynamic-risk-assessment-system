"""
This script is for creating functions for model predictions
and diagnostics

Author: Akshay Dhotre
Date: December 2023
"""
import timeit
import os
import sys
import json
import pickle
import logging
import subprocess
from io import StringIO
import pandas as pd
import numpy as np


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')


def model_predictions(data_df):
    """
    Loads deployed model to predict on data

    Input:
        X_df (pandas.DataFrame): Dataframe with features

    Returns:
        y_pred: Model predictions
    """
    logging.info('Load and prepare input data')
    if data_df is None:
        data_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    y = data_df.pop('exited')
    X = data_df.drop(['corporation'], axis=1)

    logging.info('Load model and generate poredictions')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X)

    return y_pred


def dataframe_summary():
    """
    Loads finaldata.csv and calculates mean, median and std
    on numerical data

    Returns: Dataset summary in list format
    """
    logging.info('Load dataset and select numeric columns')
    data_df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    numeric_col_index = np.where(data_df.dtypes != object)[0]
    numeric_col = data_df.columns[numeric_col_index].tolist()

    logging.info('Compute statistics per numeric column')
    means = data_df[numeric_col].mean(axis=0).tolist()
    medians = data_df[numeric_col].median(axis=0).tolist()
    stddevs = data_df[numeric_col].std(axis=0).tolist()

    statistics = means
    statistics.extend(medians)
    statistics.extend(stddevs)

    return statistics


def missing_data():
    """
    Find missing data in the dasatet
    Returns: Dictionary of percentage of missing data per column
    """
    logging.info('Loading dataset and checking for missing data')
    data_df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    missing_list = {
        col: {
            'percentage': perc} for col,
        perc in zip(
            data_df.columns,
            data_df.isna().sum() /
            data_df.shape[0] *
            100)}

    return missing_list


def _ingestion_time():
    """
    Runs ingestion.py script and measures execution time

    Returns:
        float: running time
    """
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'ingestion.py'], capture_output=True)
    timing = timeit.default_timer() - starttime
    return timing


def _training_time():
    """
    Runs training.py script and measures execution time

    Returns:
        float: running time
    """
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'training.py'], capture_output=True)
    timing = timeit.default_timer() - starttime
    return timing


def execution_time():
    """
    Gets average execution time for data ingestion and model training
    by running each process 10 times

    Returns:
        list[dict]: mean of execution times for each script
    """
    logging.info('Calculating ingestion process timing')
    ingestion_time = []
    for _ in range(10):
        time = _ingestion_time()
        ingestion_time.append(time)

    logging.info('Calculating training process timing')
    training_time = []
    for _ in range(10):
        time = _training_time()
        training_time.append(time)

    timing_measures = [
        {'ingest_time_mean': np.mean(ingestion_time)},
        {'train_time_mean': np.mean(training_time)}
    ]

    return timing_measures


def execute_cmd(cmd):
    """
    Execute a pip list type cmd
    Input: pip list type of cmd
    Returns: Output of command in dataframe format
    """
    logging.info('Running subprocess for given command')
    subprocess_instance = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    subprocess_output = StringIO(
        subprocess_instance.communicate()[0].decode('utf-8'))

    df = pd.read_csv(subprocess_output, sep="\\s+")
    df.drop(index=[0], axis=0, inplace=True)
    df = df.set_index('Package')

    return df


def outdated_packages_list():
    """
    Check dependencies status from requirements.txt file using pip-outdated
    which checks each package status if it is outdated or not

    Returns:
        str: stdout of the pip-outdated command
    """
    logging.info('Collect outdated dependencies (for current virtual env)')
    cmd = ['pip', 'list', '--outdated']
    df = execute_cmd(cmd)
    df.drop(['Version', 'Type'], axis=1, inplace=True)

    logging.info('Collect all dependencies (for current virtual env)')
    cmd = ['pip', 'list']
    df1 = execute_cmd(cmd)
    df1 = df1.rename(columns={'Version': 'Latest'})

    logging.info('Collect dependencies as per requirements.txt file')
    requirements = pd.read_csv(
        'requirements.txt',
        sep='==',
        header=None,
        names=[
            'Package',
            'Version'],
        engine='python')
    requirements = requirements.set_index('Package')

    logging.info(
        'Assemble target and latest versions for requirements.txt dependencies')
    target_dependencies = requirements.join(df1)
    for p in df.index:
        if p in target_dependencies.index:
            target_dependencies.at[p, 'Latest'] = df.at[p, 'Latest']

    logging.info('Keep only outdated dependencies (ie latest version exists)')
    target_dependencies.dropna(inplace=True)

    return target_dependencies


if __name__ == '__main__':

    logging.info('Running diagnostics process')

    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    print("Model predictions on testdata.csv:",
          model_predictions(test_df), end='\n\n')

    print("Summary statistics")
    print(json.dumps(dataframe_summary(), indent=4), end='\n\n')

    print("Missing percentage")
    print(json.dumps(missing_data(), indent=4), end='\n\n')

    print("Execution time")
    print(json.dumps(execution_time(), indent=4), end='\n\n')

    print("Outdated Packages")
    dependencies = outdated_packages_list()
    print(dependencies)
