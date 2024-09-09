import os
from typing import Type, Union

import requests
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.chat_models.ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

from utils import LLMResponse

class ZeroShot:

    def __init__(self, system_template, llm_choice,model, temperature, rate_limiter):
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        self.chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        self.llm = None

        # Initialise the chat model
        match llm_choice:
            case "ChatGPT":
                self.llm = ChatOpenAI(model=model, temperature=temperature, rate_limiter=rate_limiter)
            case "Ollama":
                self.llm = ChatOllama(model=model,temperature=temperature, rate_limiter=rate_limiter)
            case "GoogleGenerativeAI":
                convert_system_to_human = False
                if model == 'gemini-1.0-pro':
                    convert_system_to_human = True
                    
                self.llm = ChatGoogleGenerativeAI(
                            model=model,
                            temperature=temperature, convert_system_message_to_human = convert_system_to_human, rate_limiter=rate_limiter)
        
        self.structured_llm = self.llm.with_structured_output(LLMResponse)
        

    def invoke_response(self, prompt):
        #prompt = 
        response = self.llm.invoke(self.chat_prompt_template.format_prompt(
                        text=prompt
                    ).to_messages())
        return response