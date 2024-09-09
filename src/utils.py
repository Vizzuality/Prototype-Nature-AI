
from pydantic import BaseModel, Field
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from langchain_core.messages.ai import AIMessage

class LLMResponse(BaseModel):
    answer: str = Field(..., description="The textual answer provided by the LLM")
    #confidence: float = Field(..., gt=0, lt=1, description="Confidence score of the answer, between 0 and 1")

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
    if isinstance(response,AIMessage):
        response = response.dict()
    if {'answer'} <= response.keys():
        filtered_response = response['answer']
    elif {'output'} <= response.keys():
        filtered_response = response['output']
    elif {'content'} <= response.keys():
        filtered_response = response['content']

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





# #Calculate performance measures for each response

def quantitative_species_presence_metric(responses,data, i):
    df = read_and_process_species_responses(responses)
    diffs = df['Value'] - data['Value'][i]
    mean_diff = np.mean(diffs)
    return df['Value'], diffs, mean_diff


def qualitative_evaluation(fname, i):
    # implement code for comparing the justification responses to the supplied responses
    
    return None