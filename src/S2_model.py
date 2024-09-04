import os
from typing import Type, Union

import requests
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.chat_models.ollama import ChatOllama

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from tqdm import tqdm
from itertools import batched
from utils import LLMResponse

class rag_model:

    def __init__(self,dossier_path, system_prompt, llm_choice, model,persist_directory):
        # Need to add code to be able to input multiple documents
        
        #loader = PyPDFLoader(file_path=dossier_path)
        loader = TextLoader(file_path = dossier_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        
        print(len(splits))

        match llm_choice:
            case "ChatGPT":
                self.llm = ChatOpenAI(model="gpt-4o", max_tokens=1024)
                self.embeddings = OpenAIEmbeddings()
            case "Ollama":
                self.llm = ChatOllama(model=model)
                self.embeddings = OllamaEmbeddings(model=model)

        self.structured_llm = self.llm.with_structured_output(None, method = 'json_mode')        


        vectorstore = None
        if os.path.exists(persist_directory+'_'+llm_choice):
            #If local store exists then read
            vectorstore=FAISS.load_local(persist_directory+'_'+llm_choice, embeddings=self.embeddings, allow_dangerous_deserialization = True)
        else:
            print("FAISS from documents")
            batch_size = 100
            with tqdm(total=len(splits), desc="Ingesting documents") as pbar:
                for i in range(0, len(splits), batch_size):
                    
                    if vectorstore:
                        vectorstore.add_documents(splits[i:i + batch_size])
                    else:
                        vectorstore = FAISS.from_documents(splits[i:i + batch_size], self.embeddings)
                    pbar.update(batch_size)  

            #vectorstore = Chroma.from_documents(documents=splits, embedding=self.embeddings,persist_directory=chroma_persist_directory)
            
            #vectorstore = FAISS.from_documents(documents=splits, embedding=self.embeddings)
            vectorstore.save_local(persist_directory+'_'+llm_choice)
            print("Finished constructing FAISS from documents")

        self.retriever = vectorstore.as_retriever()

        self.prompt = ChatPromptTemplate.from_messages(
            [   
                ("system", system_prompt + "\n\n {context}"),
                ("human", "{input}"),
            ]
        )

    def invoke_response(self, prompt_input):

        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)

        results = rag_chain.invoke(
            {
                "input": prompt_input
            }
        )

        return results
