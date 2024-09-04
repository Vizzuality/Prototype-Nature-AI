
import streamlit as st
import pandas as pd
import csv
from datetime import datetime
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim

import os

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["INATURALIST_API_KEY"] = os.getenv("INATURALIST_API_KEY")


from S1_model import ZeroShot
from S2_model import rag_model
from s3_model import agent_model

params = {}

params['llm_choice'] = "ChatGPT"
params['model'] = None #llama3.1" #model choice if using a local Ollama model




# Initialize the models

system_prompt = """You are a Foundational Nature AI capable of informing questions about biodiversity and conservation 
    relevant for real-world decisions. Please respond as accurately and precisely as possible"""

model_instances = {
    "Zero-shot": ZeroShot(system_template=system_prompt, llm_choice=params['llm_choice'], model = params['model']),
    "Default dossier": rag_model(dossier_path='data/retrieval_dossier/abbrev_wikipedia-en-dwca-species-descriptions.csv', system_prompt= system_prompt,
        llm_choice = params['llm_choice'], model = params['model'], persist_directory='app_testing/abbrev_wikipedia'),
    "Dynamic call": agent_model(system_prompt = system_prompt,llm_choice = params['llm_choice'], model = params['model'])
}

# Function to save feedback to CSV
def save_feedback(data):
    fieldnames = ['timestamp', 'question', 'context', 'model_name', 'response', 'confidence', 'rating']
    with open('model_feedback.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(data)


def check_for_location(answer):
    # Dummy function to determine if the answer contains a location
    # In practice, you might use NLP techniques or regex to find location names
    geolocator = Nominatim(user_agent="geoapiExercises")
    try:
        location = geolocator.geocode(answer, exactly_one=True)
        if location:
            return location.latitude, location.longitude
        else:
            return None
    except:
        return None

def create_map(location):
    if location:
        m = folium.Map(location=location, zoom_start=10)
        folium.Marker(location).add_to(m)
        return m
    return None 


def filter_response(response):
    if {'answer'} <= response.keys():
        filtered_response = response['answer']
    elif {'output'} <= response.keys():
        filtered_response = response['output']

    return filtered_response

def setup_state():
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    if 'ratings' not in st.session_state:
        st.session_state.ratings = {}
    if 'all_rated' not in st.session_state:
        st.session_state.all_rated = False

def reset_state():
    st.session_state.responses = {}
    st.session_state.ratings = {}
    st.session_state.all_rated = False
    st.session_state.key = None                


# Streamlit app
def main():
    st.empty()
    st.title("Open Nature AI - Species related queries with feedback")
    question = st.text_input("Enter your question:")
    context = st.text_area("Enter the context for the question:")

    setup_state()  # Ensure the session state is initialized

    placeholder = {}

    if st.button("Get Answers"):
        

        st.write(f"**Question**: {question}")
        st.write(f"**Context**: {context}")

        with st.spinner(text=f"Retrieving answers from {len(model_instances)} models"):
            # Store responses in session state
            st.session_state.responses = {model_name: model_instance.invoke_response(question + '. ' +context) for model_name, model_instance in model_instances.items()}

        
        # Reset ratings each time new answers are fetched
        st.session_state.ratings = {model_name: 0 for model_name in model_instances.keys()}
        st.session_state.all_rated = False  # Reset rating completion flag

        
    if st.session_state.responses:
        for model_name, model_instance in model_instances.items():
            response = st.session_state.responses.get(model_name, "")
            if response:
                placeholder[model_name] = st.empty()
                with placeholder[model_name].container():
                    st.write(f"**Answer from {model_name}**: {filter_response(response)}")
                    # Check if the answer includes location data
                    location = check_for_location(st.session_state.responses[model_name])
                    if location:
                        folium_map = create_map(location)
                        folium_static(folium_map)

                    # Collect ratings
                    st.session_state.ratings[model_name] = st.slider(f"Rate the answer from {model_name} (1 = Poor, 5 = Excellent)", 1, 5, key=model_name)
                
        if st.button("Submit Ratings"):
            # Check if all ratings are submitted (i.e., not zero if zero is not a default valid value)
            if all(rating > 0 for rating in st.session_state.ratings.values()):
                st.session_state.all_rated = True
                st.success("Thank you for your feedback!")
                for model, rate in st.session_state.ratings.items():
                    st.write(f"You rated the answer from {model}: {rate}/5")
                    data = {
                        'timestamp': datetime.now(),
                        'question': question,
                        'context': context,
                        'model_name': model,
                        'response': st.session_state.responses[model],
                        'rating': rate
                    }
                    save_feedback(data)#
                
                if st.button("Do enter another question"):
                    st.empty()
                    question = ""
                    context = ""
                    reset_state()
                    for model_name in model_instances.keys():placeholder[model_name].empty()
                
            else:
                st.error("Please rate all responses before submitting.")
                

if __name__ == "__main__":
    main()









