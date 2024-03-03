

import training
import scoring
import deployment
import diagnostics
import reporting
import json, os
import ingestion
import pandas as pd
from sklearn.metrics import f1_score


with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


dataset_csv_path = os.path.join(config['output_folder_path'],'ingestedfiles.csv') 
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
output_score_path = os.path.join(config['output_model_path'], 'latestscore.txt')

prod_deployment_path = os.path.join(config['prod_deployment_path']) 


def run():
##################Check and read new data
#first, read ingestedfiles.txt
    ingested_files = []
    with open(os.path.join(prod_deployment_path, "ingestedfiles.txt")) as file:
        lines = [line.strip('\n') for line in file.readlines()[-4:]]
        for line in lines:
            if "Ingestion files names" in line:
                ingested_files = line.split(": ")[-1].split(",")
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    source_files = set(os.listdir(input_folder_path))


# ##################Deciding whether to proceed, part 1
# #if you found new data, you should proceed. otherwise, do end the process here
    if len(source_files.difference(set(ingested_files))) == 0:
        # No new data found 
        print("not found")
        return None
    else:
        print("found")

    # Ingesting new data
    ingestion.merge_multiple_dataframe()


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    deployed_score = 0
    with open(os.path.join(prod_deployment_path, "latestscore.txt")) as file:
        deployed_score = float(file.read())
        # deployed_score = float(deployed_score)

    data_df = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    
    X_df = data_df.drop(columns=['corporation', 'exited'])
    y_df = data_df['exited']

    y_pred = diagnostics.model_predictions(X_df)
    new_score = f1_score(y_df.values, y_pred)


# ##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
    print(f"Deployed score = {deployed_score}")
    print(f"New score = {new_score}")


# ##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
    if(new_score <= deployed_score):
        print("No model drift occurred")
        return None
    # Re-training
    print("retraining model")
    training.train_model()
    print("rescoring model")
    scoring.score_model()

    # Re-deployment
    print("redeploying model")
    deployment.store_model_into_pickle()
# ##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
    diagnostics.dataframe_summary
    reporting.score_model()


if __name__ == '__main__':
    run()
