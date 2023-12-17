from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
import diagnostics, scoring, ingestion 
# import predict_exited_from_saved_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


@app.route('/')
def greetings():
    return "Welcome to Dynamic Risk Assessment System!"

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    file_path = request.get_json()['filepath']

    df = pd.read_csv(file_path)

    preds = diagnostics.model_predictions(df)
    return preds

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def get_score():        
    #check the score of the deployed model
    return {'F1 Score': scoring.score_model()}

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def get_stats():        
    #check means, medians, and modes for each column
    return jsonify(diagnostics.dataframe_summary())

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def get_diagnostics():        
    #check timing and percent NA values
    missing_data = diagnostics.missing_data()
    execution_timings = diagnostics.execution_time()
    dependency_check = diagnostics.outdated_packages_list()

    # diagnostics_data = {
    #     'missing_percentage':missing_data, 
    #     'execution_time': execution_timings,
    #     'outdated_packages': dependency_check
    # }

    diagnostics_data = {'execution time': {step:duration 
                for step, duration in zip(['ingestion step','training step'],
                                            execution_timings)}, 
            'missing data': {col:pct 
                for col, pct in zip(['lastmonth_activity',
                                    'lastyear_activity',
                                    'number_of_employees',
                                    'exited'], missing_data)},
            'dependency check':[{'Module':row[0], 
                                'Version':row[1][0], 
                                'Vlatest':row[1][1]} 
                                for row in dependency_check.iterrows()]
            }
    
    return jsonify(diagnostics_data)

if __name__ == "__main__":    
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)
