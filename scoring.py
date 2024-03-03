from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
output_score_path = os.path.join(config['output_model_path'], 'latestscore.txt')

#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # Read test data
    test_data = pd.read_csv(test_data_path)

    # Split features and target variable
    X_test = test_data.drop(columns=['corporation', 'exited'])
    y_test = test_data['exited']

    # Load trained model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate F1 score
    f1 = metrics.f1_score(y_test, y_pred)

    # Write F1 score to file
    with open(output_score_path, 'w') as score_file:
        score_file.write(str(f1))
    return f1

if __name__ == '__main__':
    score_model()
