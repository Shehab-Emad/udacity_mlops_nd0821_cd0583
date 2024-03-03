import requests
import json
import os
#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 
output_model_path = config['output_model_path']
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 

#Call each API endpoint and store the responses
response1 = requests.post(f'{URL}/prediction', json={'file_path': test_data_path}).text
response2 = requests.get(f'{URL}/scoring').text
response3 = requests.get(f'{URL}/summarystats').text
response4 = requests.get(f'{URL}/diagnostics').text

#combine all API responses
responses = [response1, response2, response3, response4] #combine reponses here

#write the responses to your workspace
with open(os.path.join(output_model_path, 'apireturns.txt'), 'w') as f:
    for res in responses:
        f.write(res)
        f.write("\n")


