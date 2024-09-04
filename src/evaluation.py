import os

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["INATURALIST_API_KEY"] = os.getenv("INATURALIST_API_KEY")


from S1_model import ZeroShot
from S2_model import rag_model
from s3_model import agent_model

import pandas as pd
import numpy as np
import json
import pickle

def read_query_file_and_construct_questions(file_path):
    """
    Reads a custom-formatted file where the first line contains a question with placeholders,
    and the subsequent lines contain data in CSV format. This function replaces the placeholders
    in the question with data from each row.

    Parameters:
    file_path (str): The path to the file to be read.
    
    Returns:
    list: A list of questions with data inserted from each row.
    """
    with open(file_path, 'r') as file:
        # Extract the first line and get the question template
        first_line = file.readline().strip()
        if first_line.startswith('#System_prompt: '):
            system_prompt = first_line[len('"#System_prompt:'):].strip()
        else:
            raise ValueError("The file does not start with a proper system prompt header.")
        
        second_line = file.readline().strip()
        if second_line.startswith('#Prompt: '):
            question_template = second_line[len('#Prompt: '):].strip()
        else:
            raise ValueError("The file does not start with a proper question header.")

        # Read the CSV data that follows
        data = pd.read_csv(file, skiprows=0)  # We already read the first and second lines, so the cursor is at the second line

    # print(repr(question_template))
    # q_data_columns = re.findall(r'\{([^}]+)\}',question_template)
    # print(q_data_columns)


    # List to hold all questions filled with data
    filled_questions = []

    # Fill the question template with data from each row
    for index, row in data.iterrows():
        filled_question = question_template.format(**row.to_dict())
        filled_questions.append(filled_question)
    
    return system_prompt,filled_questions, data

def filter_response(response):
    if {'answer'} <= response.keys():
        filtered_response = response['answer']
    elif {'output'} <= response.keys():
        filtered_response = response['output']

    return filtered_response

def read_and_process_species_responses(responses):
    """
    Reads a text file and splits each line into a numeric value and a text description.
    Assumes each line starts with a numeric value followed by a comma, then the text.

    Parameters:
    responses (dict): dictionary of .
    
    Returns:
    DataFrame: A pandas DataFrame with two columns: 'Numeric' and 'Text'.
    """
    data = []  # List to store the split data

    # Open the file and process each line
    #with open(file_path, 'r', encoding='utf-8') as file:
    for r in responses:
        # Strip whitespace and split the line at the first comma
        parts = r.strip().split(',', 1)
        #Check that there are 2 parts and that it doesn't start with "Note:"
        if len(parts) == 2 and not parts[0].startswith("Note:"):
            numeric_value = float(parts[0].strip())  # Convert the numeric part to float
            text_description = parts[1].strip()
            data.append([numeric_value, text_description])

    # Convert the list of data into a DataFrame
    df = pd.DataFrame(data, columns=['Value', 'Justification'])
    return df
### Begin by reading in the evaluation questions. These are split into groups,
# 1. species specific questions
# 2. threats and interventions


params = {}

params['Nreplicates'] = 10
params['llm_choice'] = "ChatGPT"
params['model'] = None #llama3.1" #model choice if using a local Ollama model
params['pkl_out'] = f"output/{params['llm_choice']}_All_Model_Q_responses.pkl"

#For each category of question:
# - Read the system prompt
# - Set up a set of models with the system prompt
# - Construct the set of questions for the model
# - Invoke questions
# - Record responses
# - Repeat invocation and recording N times
# - Calculate performances


#Species locations
file_path = 'eval/species_geospatial_queries_responses - species_presences.csv'  # Replace this with the path to your file
try:
    system_prompt, queries, data = read_query_file_and_construct_questions(file_path)
    print(system_prompt)

except Exception as e:
    print("Error reading queries:", e)

#TBD: Create log files and pass into the models as they are initialised

#Create model approaches
model_systems = {}
model_systems['s1'] = ZeroShot(system_template=system_prompt, llm_choice=params['llm_choice'], model = params['model'])
model_systems['s2'] = rag_model(dossier_path='data/retrieval_dossier/wikipedia-en-dwca-species-descriptions.csv', system_prompt= system_prompt,
               llm_choice = params['llm_choice'], model = params['model'], persist_directory='training/wikipedia')
model_systems['s3'] = agent_model(system_prompt = system_prompt,llm_choice = params['llm_choice'], model = params['model'])


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



# #Calculate performance measures for each response

def quantitative_species_presence_metric(responses,i):
    df = read_and_process_species_responses(responses)
    diffs = df['Value'] - data['Value'][i]
    mean_diff = np.mean(diffs)
    return df['Value'], diffs, mean_diff


def qualitative_evaluation(fname, i):
    # implement code for comparing the justification responses to the supplied responses
    
    return None

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
        filtered_responses = [filter_response(r) for r in responses[str(i)][model]]
 
        vals[str(i)],diffs[str(i)][model], mean_diff[str(i)][model] = quantitative_species_presence_metric(filtered_responses,i)
        
pd.DataFrame([vals,diffs]).to_csv(f'output/{params['llm_choice']}_diffs.csv')
print(mean_diff)

