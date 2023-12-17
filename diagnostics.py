
from io import StringIO
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl') 


##################Function to get model predictions
def model_predictions(data_df):
    #read the deployed model and a test dataset, calculate predictions
    if data_df is None:
        data_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    y = data_df.pop('exited')
    X = data_df.drop(['corporation'], axis=1)

    y_pred = model.predict(X)

    return y_pred

##################Function to get summary statistics
def dataframe_summary():
    data_df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    # select numeric values
    # data_df = data_df.drop(['exited'], axis=1)
    # data_df = data_df.select_dtypes('number')

    # stats_dict = {}

    # for col in data_df.columns:
    #     col_mean = data_df[col].mean()
    #     col_median = data_df[col].median()
    #     col_std = data_df[col].std()

    #     stats_dict[col] = {'mean': col_mean, 'median': col_median, 'std': col_std}

        # Select numeric columns
    numeric_col_index = np.where(data_df.dtypes != object)[0]
    numeric_col = data_df.columns[numeric_col_index].tolist()

    # compute statistics per numeric column
    means = data_df[numeric_col].mean(axis=0).tolist()
    medians = data_df[numeric_col].median(axis=0).tolist()
    stddevs = data_df[numeric_col].std(axis=0).tolist()

    statistics = means
    statistics.extend(medians)
    statistics.extend(stddevs)

    return statistics

def missing_data():
    data_df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    missing_list = {col: {'percentage': perc} for col, perc in zip(data_df.columns, data_df.isna().sum() / data_df.shape[0] * 100)}

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
    ingestion_time = []
    
    for _ in range(10):
        time = _ingestion_time()
        ingestion_time.append(time)

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
    """execute a pip list type cmd
    input: pip list type of cmd
    return: output of cmd in dataframe format
    """
    a = subprocess.Popen(cmd, stdout=subprocess.PIPE)  # text=True then no need to decode bytes to str
    b = StringIO(a.communicate()[0].decode('utf-8'))
    df = pd.read_csv(b, sep="\s+")
    df.drop(index=[0], axis=0, inplace=True)
    df = df.set_index('Package')
    return df

##################Function to check dependencies
def outdated_packages_list():
    # collect outdated dependencies (for current virtual env)
    cmd = ['pip', 'list', '--outdated']
    df = execute_cmd(cmd)
    df.drop(['Version','Type'], axis=1, inplace=True)

    # collect all dependencies (for current virtual env)
    cmd = ['pip', 'list']
    df1 = execute_cmd(cmd)
    df1 = df1.rename(columns = {'Version':'Latest'})

    # collect dependencies as per requirements.txt file
    requirements = pd.read_csv('requirements.txt', sep='==', header=None, names=['Package','Version'], engine='python')
    requirements = requirements.set_index('Package')

    # assemble target and latest versions for requirements.txt dependencies
    dependencies = requirements.join(df1)
    for p in df.index:
        if p in dependencies.index:
            dependencies.at[p, 'Latest'] = df.at[p,'Latest']
    
    # keep only outdated dependencies (ie latest version exists)
    dependencies.dropna(inplace=True)

    return dependencies


if __name__ == '__main__':
    # model_predictions()
    # dataframe_summary()
    # missing_data()
    # execution_time()
    # outdated_packages_list()

    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    # y = test_df.pop('exited')
    # X = test_df.drop(['corporation'], axis=1)

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





    
