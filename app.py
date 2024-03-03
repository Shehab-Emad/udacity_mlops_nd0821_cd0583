from flask import Flask, session, jsonify, request
import pandas as pd
import json
import os
import diagnostics
import subprocess
import scoring

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    """
    predict endpoint that calls the prediction function in diagnostics.py

    Returns:
        json: predictions
    """
    file_path = request.get_json()['file_path']
    print()
    df = pd.read_csv(file_path)
    df.drop(columns=['corporation', 'exited'], inplace=True)

    preds = diagnostics.model_predictions(df)
    return jsonify(preds.tolist()) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():
    """
    Scoring endpoint that runs the script scoring.py

    Returns:
        str:f1 score
    """     
    #check the score of the deployed model
    return jsonify(scoring.score_model()) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():
    """
    stats endpoint that calls dataframe summary function from diagnostics.py

    Returns:
        json: summary statistics
    """ 
    #check means, medians, and modes for each column
    return jsonify(diagnostics.dataframe_summary()) #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diag():
    """
    diagnostics endpoint thats calls execution_time, missing_data,
    and outdated_packages_list from diagnostics.py

    Returns:
        json:execution_time, missing_data and outdated_packages_list
    """
    #check timing, missing data, and dependency
    execution_time = diagnostics.execution_time()
    missing_data = diagnostics.missing_data()
    outdated_packages_list = diagnostics.outdated_packages_list()

    ret = {
        'execution_time': execution_time,
        'missing_data': missing_data,
        'outdated_packages_list': outdated_packages_list
    }

    return jsonify(ret) #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
