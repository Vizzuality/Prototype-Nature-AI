
import os
from typing import Type, Union
import requests

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool

class RequestInputINaturalist(BaseModel):
    """
    Request input class for the iNaturalist tool.
    """
    taxon: Union[str, int] = Field(description="Taxon name or ID of a species to search for.")


class INaturalistTool(BaseTool):
    """
    Tool for querying the iNaturalist API to get information about places
    where a species can be found.
    """

    name = "inaturalist"
    description = "This tool is useful for when you need to get information about places where a"
    "species can be found."
    args_schema: Type[BaseModel] = RequestInputINaturalist
    api_token = os.getenv("INATURALIST_API_KEY")

    def _run(self, taxon):
        headers = {"Authorization": f"Bearer {str(self.api_token)}"}
        params = {
            "taxon": taxon,
        }
        response = requests.get(
            "https://www.inaturalist.org/places.json", headers=headers, params=params
        )
        if response.status_code == 200:
            data = response.json()
            filtered_data = [
                {
                    "name": item["name"],
                    "latitude": item["latitude"],
                    "longitude": item["longitude"],
                    "place_type_name": item["place_type_name"],
                }
                for item in data
            ]
            return filtered_data
        else:
            return {"error": response.status_code, "message": response.text}


