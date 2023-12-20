# Dynamic risk assessment system
Dynamic risk assessment system using MLOps best prectices as part of Udacity Nanodegree

# Project details
This project is part of Course: Machine Learning Model Scoring and Monitoring. The problem is to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's clients. Also setting up processes to re-train, re-deploy, monitor and report on the ML model.

# Workflow steps
The workflow consiststhe following components which are all separated:
- data ingestion
- model training (logisticRegression sklearn model for binary classification)
- model scoring
- deployment of pipeline artifacts into production
- model monitoring, reporting and statistics - API set-up for ML diagnostics and results
- process automation with data ingestion and model drift detection using CRON job
- retraining / redeployment in case of model drift

# Project Steps Overview

- **Data ingestion**
    
    Automatically check a database for new data that can be used for model training. Compile all training data to a training dataset and save it to persistent storage. Write metrics related to the completed data ingestion tasks to persistent storage.

- **Training, scoring, and deploying** 
    
    Write scripts that train an ML model that predicts attrition risk, and score the model. Write the model and the scoring metrics to persistent storage.

- **Diagnostics**
    
    Determine and save summary statistics related to a dataset. Time the performance of model training and scoring scripts. Check for dependency changes and package updates.

- **Reporting**
    
    Automatically generate plots and documents that report on model metrics. Provide an API endpoint that can return model predictions and metrics.

- **Process Automation** 
    
    Create a script and cron job that automatically run all previous steps at regular intervals.

## Prerequisites
- Python 3 required
- Linux or Linux environment within windows through WSL 2

## Dependencies
This project dependencies is available in the ```requirements.txt``` file.


## How to use?

### Change config.json file to use practice data and models

```bash
{
"input_folder_path": "practicedata",
"output_folder_path": "ingesteddata", 
"test_data_path": "testdata", 
"output_model_path": "practicemodels", 
"prod_deployment_path": "production_deployment"
}
```

### Run data ingestion
```python
python ingestion.py
```
Artifacts output:
```
data/ingesteddata/finaldata.csv
data/ingesteddata/ingestedfiles.txt
```

### Model training
```python
python training.py
```
Artifacts output:
```
models/practicemodels/trainedmodel.pkl
```

### Model scoring 
```python
python scoring.py
```
Artifacts output: 
```
models/practicemodels/latestscore.txt
``` 

### Model deployment
```python
python deployment.py
```
Artifacts output:
```
models/prod_deployment_path/ingestedfiles.txt
models/prod_deployment_path/trainedmodel.pkl
models/prod_deployment_path/latestscore.txt
``` 

### Run diagnostics
```python
python diagnostics.py
```

### Run reporting
```python
python reporting.py
```
Artifacts output:
```
models/practicemodels/confusionmatrix.png
models/practicemodels/summary_report.pdf
```

### Run Flask App
```python
python app.py
```

### Run API endpoints
```python
python apicalls.py
```
Artifacts output:
```
models/practicemodels/apireturns.txt
```

### Change config.json file to use production data and models

```bash
{
"input_folder_path": "sourcedata",
"output_folder_path": "ingesteddata", 
"test_data_path": "testdata", 
"output_model_path": "models", 
"prod_deployment_path": "production_deployment"
}
```

### Full process automation
```python
python fullprocess.py
```
### Cron job

Start cron service
```bash
sudo service cron start
```

Edit crontab file
```bash
sudo crontab -e
```
   - Select **option 3** to edit file using vim text editor
   - Press **i** to insert a cron job
   - Write the cron job in ```cronjob.txt``` which runs ```fullprocces.py``` every 10 mins
   - Save after editing, press **esc key**, then type **:wq** and press enter
  
View crontab file
```bash
sudo crontab -l
```
