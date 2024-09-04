

from typing import List, Dict

import wikipedia

import os
from typing import Type, Union
import requests

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool

class RequestInputWikipedia(BaseModel):
    """
    Request input class for the Wikipedia tool.
    """

    taxon: Union[str, int] = Field(description="Taxon name or scientific name of a species to search for.")


class INaturalistTool(BaseTool):
    """
    Tool for querying the Wikipedia API for species information
    """

    name = "wikipedia"
    description = "This tool is useful for when you need to get information about places where a"
    "species can be found."
    args_schema: Type[BaseModel] = RequestInputWikipedia

    def _run(self, taxon):
        

    def get_wiki_data(query: str) -> List[Dict]:
        wiki_data = []
        for search_result in wikipedia.search(query):
            try:
                page = wikipedia.page(search_result)
            except:
                continue
            wiki_data.append({'title': search_result, 'content': page.content, 'summary': page.summary,
                            'url': page.url})
        return wiki_data