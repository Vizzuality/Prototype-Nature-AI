
from pydantic import BaseModel, Field
from typing import Optional, Tuple

class LLMResponse(BaseModel):
    answer: str = Field(..., description="The textual answer provided by the LLM")
    confidence: float = Field(..., gt=0, lt=1, description="Confidence score of the answer, between 0 and 1")