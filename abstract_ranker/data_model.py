from typing import List
from pydantic import BaseModel, Field


class AbstractLLMResponse(BaseModel):
    "Result back from LLM grading of an abstract"

    summary: str = Field(..., title="The one-line summary of the abstract")
    experiment: str = Field(
        ...,
        title="The most likely experiment associated with this work (ATLAS, CMS, LHCb, "
        "MATHUSLA, etc.). Blank if unknown.",
    )
    keywords: List[str] = Field(
        ..., title="List of keywords associated with the abstract"
    )
    interest: str = Field(
        ..., title="Interest level in the abstract (high, medium, or low)"
    )
    explanation: str = Field(
        ...,
        title="Explanation of the interest level in the abstract",
    )
