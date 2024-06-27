from typing import List
from pydantic import BaseModel, Field


class AbstractLLMResponse(BaseModel):
    "Result back from LLM grading of an abstract"

    # Summary of the abstract
    summary: str = Field(..., title="The one-line summary of the abstract")

    # The most likely experiment this is associated with
    experiment: str = Field(
        ...,
        title="The Experiment associated with this work if known (ATLAS, CMS, LHCb, "
        "MATHUSLA, etc.). Blank if unknown. No explanation.",
    )

    # List of keywords
    keywords: List[str] = Field(
        ..., title="List of keywords associated with the abstract"
    )

    # What is the interest level here?
    interest: str = Field(
        ..., title="Interest level in the abstract (high, medium, or low)"
    )

    # A short explanation of why the interest level is what it is
    explanation: str = Field(
        ...,
        title="Explanation of the interest level in the abstract. Very short.",
    )
