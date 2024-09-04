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

from utils import LLMResponse

class ZeroShot:

    def __init__(self, system_template, llm_choice,model):
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        self.chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        # Initialise the chat model
        match llm_choice:
            case "ChatGPT":
                self.llm = ChatOpenAI(model="gpt-4o", max_tokens=1024)
            case "Ollama":
                self.llm = ChatOllama(model="llama3.1")
        
        self.structured_llm = self.llm.with_structured_output(LLMResponse)
        

    def invoke_response(self, prompt):

        prompt = self.chat_prompt_template.format_prompt(
            text=prompt
        ).to_messages()
        response = self.structured_llm.invoke(prompt)
        return response