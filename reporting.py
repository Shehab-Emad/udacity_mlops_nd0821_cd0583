import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import diagnostics



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
output_model_path = os.path.join(config['output_model_path'])




##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    
    # Read test data
    test_data = pd.read_csv(test_data_path)

    # Split features and target variable
    X_test = test_data.drop(columns=['corporation', 'exited'])
    y_test = test_data['exited']

    preds = diagnostics.model_predictions(X_test)

    # Calculate confusion_matrix
    confusion_matrix = metrics.confusion_matrix(y_test, preds)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.savefig(os.path.join(output_model_path, 'confusionmatrix.png'))


    # fig.savefig(os.path.join(output_model_path, 'confusionmatrix.png'))
    return preds #return value should be a list containing all predictions



if __name__ == '__main__':
    score_model()
