import os

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["INATURALIST_API_KEY"] = os.getenv("INATURALIST_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_GEN_AI_API_KEY")


import setup
from S1_model import ZeroShot
from S2_model import rag_model
from s3_model import agent_model

import pandas as pd
import numpy as np
import json
import pickle
import utils


def gather_responses(system_prompt, queries, params):

    model_systems = {}
    model_systems['s1'] = ZeroShot(system_template=system_prompt,llm_choice = params['llm_choice'], model = params['model'],temperature=params['temperature'])
    model_systems['s2'] =  rag_model(dossier_path='data/retrieval_dossier/wikipedia-en-dwca-species-descriptions.csv', system_prompt= system_prompt,
                llm_choice = params['llm_choice'], model = params['model'], temperature=params['temperature'], persist_directory=params['persist_directory'])
    model_systems['s3'] = agent_model(system_prompt = system_prompt,llm_choice = params['llm_choice'], model = params['model'], temperature=params['temperature'])

    # s1_responses = []
    # s2_responses = []
    responses = {}

    models = ['s1','s2','s3']


    if not os.path.isfile(params['pkl_out']):
        print("Running model responses")
        #read first test question.
        for i, question in enumerate(queries):
            responses[str(i)] = {}
            print(question)
            for model in models:
                #if not os.path.isfile(f'output/Q{i}_{model}.json'):    
                    responses[str(i)][model] = [model_systems[model].invoke_response(question) for i in range(params['Nreplicates'])]

                #with open(f'output/Q{i}_{model}.json', 'w') as f:
                    # for r in responses[model]:
                    #     f.write(f"{r}\n")
                #    json.dump(responses[model],f)

        out_file = open(params['pkl_out'], 'wb')
        pickle.dump(responses, out_file)
        out_file.close()

    return responses

### Evaluation questions. These are split into groups,
# 1. species specific questions
# 2. threats and interventions


#Species presence absences
file_path = 'eval/species_point_presence_absence.csv'
sp_pa_df = pd.read_csv(file_path)

#System prompt for presence/absences
system_prompt = "You are a Foundational Nature AI capable of informing questions about biodiversity and conservation relevant for real-world decisions. Ensure that you respond with a score between 0 and 1 (where 1 indicates that you think the species is very likely to be present there and 0 indicates the species is highly unlikely to be found there) Please provide your response with the following fields, separated by commas: the score, the justification,  then a measure of your confidence - again scored from 0 to 1 (0 is very low confidence, 1 is very confident in your answer)"
# List to hold all questions filled with data
question_template = "Can you tell me if {binomial} can be found at the point {y} degress latitude and {x} degrees longitude"
filled_questions = []

# Fill the question template with data from each row
for index, row in sp_pa_df.iterrows():
    filled_question = question_template.format(**row.to_dict())
    filled_questions.append(filled_question)

####  Create model approaches ####
#Get the parameters
params = setup.get_parameters()
params['pkl_out'] = f"output/species_point_pres_abs_{params['llm_choice']}_{params['model']}_All_Model_Q_responses.pkl"
#TBD: Create log files and pass into the models as they are initialised



sp_p_a_responses = gather_responses(system_prompt=system_prompt,queries=filled_questions,params=params)









#Species locations
file_path = 'eval/species_geospatial_queries_responses - species_presences.csv'  # Replace this with the path to your file
try:
    system_prompt, queries, data = utils.read_query_file_and_construct_questions(file_path)
    print(system_prompt)

except Exception as e:
    print("Error reading queries:", e)

####  Create model approaches ####
#Get the parameters
params = setup.get_parameters()
params['pkl_out'] = f"output/species_toponym_presence_likelihood_{params['llm_choice']}_{params['model']}_All_Model_Q_responses.pkl"
#TBD: Create log files and pass into the models as they are initialised


sp_p_a_responses = gather_responses()



# Construct container to hold 
diffs = {}
vals = {}
mean_diff = {}
filtered_responses = {}

if not responses:
    print(f'Loading responses from: {params['pkl_out']}')
    in_file = open(params['pkl_out'], 'rb')
    responses = pickle.load(in_file)
    in_file.close()

for i, question in enumerate(queries):
    
    diffs[str(i)] = {}
    mean_diff[str(i)] = {}
    for model in models:
        filtered_responses = [utils.filter_response(r) for r in responses[str(i)][model]]
 
        vals[str(i)],diffs[str(i)][model], mean_diff[str(i)][model] = quantitative_species_presence_metric(filtered_responses,i)
        
pd.DataFrame([vals,diffs]).to_csv(f'output/{params['llm_choice']}_diffs.csv')
print(mean_diff)

