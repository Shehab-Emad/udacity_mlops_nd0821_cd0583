
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

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
prod_deployment_model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')

##################Function to get model predictions
def model_predictions(df):
    #read the deployed model and a test dataset, calculate predictions
    # Load trained model
    with open(prod_deployment_model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Predict on test data
    preds = model.predict(df)
    return preds #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    df = pd.read_csv(dataset_csv_path)
    df.drop(columns=['corporation', 'exited'], inplace=True)
    summary_list = []
    for col in df.columns:
        summary = {}
        mean = df[col].mean()
        median = df[col].median()
        std = df[col].std()

        summary[col] = {'mean': mean, 'median': median, 'std': std}
        summary_list.append(summary)
    return summary_list #return value should be a list containing all summary statistics


def missing_data():
    # check for missing data
    df = pd.read_csv(dataset_csv_path)
    df.drop(columns=['corporation', 'exited'], inplace=True)

    missing = {col: {'percentage': percentage} for col, percentage in zip(df.columns, df.isna().sum() / df.shape[0] * 100)}

    return missing

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    exec_time_list = []
    for file in ["ingestion.py", "training.py"]:
        starttime = timeit.default_timer()
        subprocess.run(['python', file], capture_output=True)
        timing = timeit.default_timer() - starttime
        exec_time_list.append(timing)

    return exec_time_list #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    # Read currently installed versions from requirements.txt
    with open('requirements.txt', 'r') as requirements_file:
        installed_modules = requirements_file.readlines()

    # Clean module names and create a dictionary to store version information
    module_versions = {}
    for module in installed_modules:
        module_name = module.strip().split('==')[0]
        module_versions[module_name] = {'current_version': None, 'latest_version': None}

    # Get currently installed versions
    for module_name in module_versions.keys():
        try:
            result = subprocess.run(['pip', 'show', module_name], capture_output=True, text=True)
            output = result.stdout
            info = [line.split(': ')[1].strip() for line in output.split('\n') if 'Version' in line]
            if info:
                module_versions[module_name]['current_version'] = info[0]
        except:
            pass

    # Get latest versions from PyPI
    for module_name in module_versions.keys():
        try:
            result = subprocess.run(['pip', 'search', module_name, '--json'], capture_output=True, text=True)
            output = result.stdout
            info = json.loads(output)
            if info:
                module_versions[module_name]['latest_version'] = info[0]['version']
        except:
            pass

    # Print table of module versions
    print('Module', 'Current Version', 'Latest Version')
    print("-" * 70)
    for module_name, versions in module_versions.items():
        print(f"{module_name}, {versions['current_version']}, {versions['latest_version']}")



if __name__ == '__main__':

    df = pd.read_csv(test_data_path)
    df.drop(columns=['corporation', 'exited'], inplace=True)
    print(model_predictions(df))
    print(dataframe_summary())
    print(execution_time())
    print(missing_data())
    print(outdated_packages_list())





    
