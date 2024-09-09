import streamlit as st
import pandas as pd
import csv
from datetime import datetime
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
import os
from dotenv import load_dotenv

from langchain_core.rate_limiters import InMemoryRateLimiter

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["INATURALIST_API_KEY"] = os.getenv("INATURALIST_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_GEN_AI_API_KEY")

from S1_model import ZeroShot
from S2_model import rag_model
from s3_model import agent_model
import setup
import utils

params = setup.get_app_parameters()


print()

# Initialize the models
system_prompt = """You are a Foundational Nature AI capable of informing questions about biodiversity and conservation 
    relevant for real-world decisions. Please respond as accurately and precisely as possible"""

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)

@st.cache_resource
def create_model_instances():
    model_instances = {
        "Zero-shot": ZeroShot(system_template=system_prompt,llm_choice = params['llm_choice'], model = params['model'],temperature=params['temperature'],rate_limiter=rate_limiter),
        "Default dossier": rag_model(dossier_path='data/retrieval_dossier/wikipedia-en-dwca-species-descriptions.csv', system_prompt= system_prompt,
                    llm_choice = params['llm_choice'], model = params['model'], temperature=params['temperature'], persist_directory=params['persist_directory'], rate_limiter=rate_limiter),
        "Dynamic call": agent_model(system_prompt = system_prompt,llm_choice = params['llm_choice'], model = params['model'], temperature=params['temperature'], rate_limiter=rate_limiter)
    }
    return model_instances


model_instances = create_model_instances()

# Function to save feedback to CSV
def save_feedback(data):
    fieldnames = ['timestamp', 'question', 'context', 'model_name', 'response', 'confidence', 'rating']
    with open('model_feedback.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(data)

def check_for_location(answer):
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


def setup_state():
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    if 'ratings' not in st.session_state:
        st.session_state.ratings = {}
    if 'all_rated' not in st.session_state:
        st.session_state.all_rated = False
    if 'question' not in st.session_state:
        st.session_state.question = ""
    if 'clear_triggered' not in st.session_state:
        st.session_state.clear_triggered = False

# Reset session state and force re-run
def reset_app():
    st.session_state.clear()  # Clear all session state
    st.rerun()  # Force re-run to clear the app

# Streamlit app
def main():
    st.title("Open Nature AI - Species related queries with feedback")
    
    setup_state()  # Ensure the session state is initialized
    
    # Use session state to control the input value of the question field
    question = st.text_input("Enter your question:", value=st.session_state.question)

    # Use @st.cache_data to cache the model responses
    @st.cache_data(show_spinner=False)
    def get_model_responses(question):
        return {model_name: model_instance.invoke_response(question) for model_name, model_instance in model_instances.items()}

    if st.button("Get Answers") and question:
        #st.session_state.question = question  # Save question to session state
        
        with st.spinner(text=f"Retrieving answers from {len(model_instances)} models"):
            # Store responses in session state
            st.session_state.responses = get_model_responses(question)
        
        st.session_state.ratings = {model_name: 0 for model_name in model_instances.keys()}
        st.session_state.all_rated = False
        st.session_state.question = ""  # Clear the question input field

    if st.session_state.responses:
        for model_name, model_instance in model_instances.items():
            response = st.session_state.responses.get(model_name, "")
            if response:
                st.write(f"**Answer from {model_name}**: {utils.filter_response(response)}")
                
                location = check_for_location(utils.filter_response(response))
                if location:
                    folium_map = create_map(location)
                    folium_static(folium_map)

                # Collect ratings
                st.session_state.ratings[model_name] = st.slider(
                    f"Rate the answer from {model_name} (1 = Poor, 5 = Excellent)", 1, 5, key=model_name
                )

        if st.button("Submit Ratings"):
            if all(rating > 0 for rating in st.session_state.ratings.values()):
                st.session_state.all_rated = True
                st.success("Thank you for your feedback!")
                
                for model, rate in st.session_state.ratings.items():
                    st.write(f"You rated the answer from {model}: {rate}/5")
                    data = {
                        'timestamp': datetime.now(),
                        'question': st.session_state.question,
                        'context': "",  # Context removed as per your request
                        'model_name': model,
                        'response': st.session_state.responses[model],
                        'rating': rate
                    }
                    save_feedback(data)

                # Add the "Ask another question" button to reset the state
                if st.button("Ask another question"):
                    reset_app()  # Clear session state and force re-run
        
        # Clear the responses and ratings after submitting ratings or asking another question
        if st.session_state.all_rated or st.session_state.clear_triggered:
            st.session_state.responses = {}
            st.session_state.ratings = {}
            st.session_state.all_rated = False
            st.session_state.question = ""  # Clear the question input field
            st.session_state.clear_triggered = True
            #st.rerun()

if __name__ == "__main__":
    main()