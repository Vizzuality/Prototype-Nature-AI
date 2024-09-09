import os
from typing import Type, Union

import json

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
from langchain.pydantic_v1 import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_ollama import ChatOllama

from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from iNaturalistTool import INaturalistTool
from utils import LLMResponse


class agent_model:

    def __init__(self, system_prompt,llm_choice, model,temperature, rate_limiter):

        self.llm_choice = llm_choice

        inat_tool = INaturalistTool()

        search_tool = TavilySearchResults(max_results=1)

        api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
        wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt + 
                    "For the justification of your score, please give full, detailed and accurate answers. If you do not get any result from the inaturalist tool use the wikipedia tool"
                    "and if you do not get any result from wikipedia use the tavily_search_results_json tool.",
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        self.tools = [wiki_tool, search_tool, inat_tool]


        match llm_choice:
            case "ChatGPT":
                self.llm = ChatOpenAI(model=model, temperature = temperature, rate_limiter=rate_limiter)
                                
            case "Ollama":
                #self.llm = Ollama(model=model)
                # self.llm = ChatOpenAI(
                #     api_key="ollama",
                #     model=model,
                #     base_url="http://localhost:11434/v1",
                # )
                self.llm = ChatOllama(model=model, temperature = temperature, rate_limiter=rate_limiter)
            case "GoogleGenerativeAI":
                convert_system_to_human = False
                if model == 'gemini-1.0-pro':
                    convert_system_to_human = True

                self.llm = ChatGoogleGenerativeAI(
                            model=model,
                            temperature=temperature, convert_system_message_to_human = convert_system_to_human, rate_limiter=rate_limiter)                

        #self.structured_llm = self.llm.with_structured_output(LLMResponse)

        # Construct the Tools agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        



    def invoke_response(self, prompt_input):
        # Create an agent executor by passing in the agent and tools
        agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
        
        results = agent_executor.invoke(
            {"input":prompt_input}
        )


        # parser = PydanticOutputParser(pydantic_object=LLMResponse)
        
        # new_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm)
        # parsed_results = new_parser.parse(results)

        return results
    

    
