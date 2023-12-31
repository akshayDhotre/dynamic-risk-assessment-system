"""
This script used to generate a confusion matrix and generate PDF report

Author: Akshay Dhotre
Date: December 2023
"""
import json
import os
import ast
import sys
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import metrics
from sklearn.model_selection import train_test_split
from diagnostics import model_predictions, dataframe_summary, missing_data, execution_time, outdated_packages_list

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])


def score_model():
    """
    Calculate a confusion matrix using the test data and the deployed model
    and prepare report for model scores
    """
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    y = test_df['exited']

    y_pred = model_predictions(test_df)

    cm = metrics.confusion_matrix(y, y_pred)

    logging.info('Create confusion matrix plot')
    f, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        cmap='viridis',
        fmt='d',
        linewidths=.5,
        annot_kws={
            "fontsize": 15})
    plt.xlabel('Predicted Class', fontsize=15)
    ax.xaxis.set_ticklabels(['Not Churned', 'Churned'])
    plt.ylabel('True Class', fontsize=15)
    ax.yaxis.set_ticklabels(['Not Churned', 'Churned'])
    plt.title('Confusion matrix', fontsize=20)

    logging.info('write the confusion matrix to the workspace')
    savepath = os.path.join(model_path, 'confusionmatrix.png')
    f.savefig(savepath)

    logging.info('Collect statistics')
    statistics = dataframe_summary()
    missingdata = missing_data()
    timings = execution_time()
    dependencies = outdated_packages_list()
    # collect ingested files
    filepath = os.path.join(dataset_csv_path, 'ingestedfiles.txt')
    with open(filepath, 'r') as f:
        ingestedfiles = ast.literal_eval(f.read())

    logging.info('Produce pdf report')

    # 1- list of ingested files
    ingestedfiles = pd.DataFrame(ingestedfiles, columns=['Ingested files'])
    col_names = ingestedfiles.columns.tolist()
    data = ingestedfiles.values
    rowLabels = ingestedfiles.index.tolist()

    # Plot table
    fig, ax = plt.subplots(1, figsize=(3, 3))
    plt.title('Ingested files', fontsize=20)
    ax.axis('off')
    table = plt.table(
        cellText=data,
        colLabels=col_names,
        loc='center',
        colLoc='right',
        rowLabels=rowLabels)
    plt.tight_layout()

    # 2- summary statistics
    col_names = [
        'lastmonth_activity',
        'lastyear_activity',
        'number_of_employees',
        'exited']
    data = np.array(statistics).reshape(3, 4)

    # Plot table
    fig, ax = plt.subplots(1, figsize=(10, 2))
    plt.title('Summary statistics', fontsize=20)
    ax.axis('off')
    table = plt.table(
        cellText=data,
        colLabels=col_names,
        loc='center',
        colLoc='right',
        rowLabels=[
            'mean',
            'median',
            'std'])

    # Additional Statistics
    logging.info('Compute classification report')
    cr = metrics.classification_report(y, y_pred, output_dict=True)

    # 4- classification report
    df = pd.DataFrame(cr).transpose()
    col_names = df.columns.tolist()
    data = df.values
    rowLabels = df.index.tolist()
    # Plot table
    fig, ax = plt.subplots(1, figsize=(10, 5))
    ax.axis('off')
    table = plt.table(
        cellText=data,
        colLabels=col_names,
        loc='center',
        colLoc='right',
        rowLabels=rowLabels)
    plt.title('Classification report', fontsize=20)
    plt.tight_layout()

    # 5- Missing data
    df = pd.DataFrame(
        data=missingdata,
        index=test_df.columns.tolist(),
        columns=['missing data'])
    col_names = df.columns.tolist()
    data = df.values
    rowLabels = df.index.tolist()
    # Plot table
    fig, ax = plt.subplots(1, figsize=(4, 6))
    ax.axis('off')
    table = plt.table(
        cellText=data,
        colLabels=col_names,
        loc='center',
        colLoc='right',
        rowLabels=rowLabels)
    plt.title('Missing data', fontsize=20)
    plt.tight_layout()

    # 6- Timing of execution
    timing = pd.DataFrame(timings, columns=['Duration (sec)'])
    col_names = timing.columns.tolist()
    data = timing.values
    rowLabels = ["Ingestion step", 'Training step']
    # Plot table
    fig, ax = plt.subplots(1, figsize=(4, 4))
    plt.title('Execution time', fontsize=20)
    ax.axis('off')
    table = plt.table(
        cellText=data,
        colLabels=col_names,
        loc='center',
        colLoc='right',
        rowLabels=rowLabels)
    plt.tight_layout()

    # 7- dependencies status
    col_names = dependencies.columns.tolist()
    data = dependencies.values
    rowLabels = dependencies.index.tolist()
    # Plot table
    fig, ax = plt.subplots(1, figsize=(5, 5))
    plt.title('dependencies status', fontsize=20)
    ax.axis('off')
    table = plt.table(
        cellText=data,
        colLabels=col_names,
        loc='center',
        colLoc='right',
        rowLabels=rowLabels)
    plt.tight_layout()

    def save_multi_image(filename):
        pp = PdfPages(filename)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()

    filename = "report.pdf"
    save_multi_image(filename)


if __name__ == '__main__':
    score_model()
