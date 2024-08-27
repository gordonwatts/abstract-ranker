from typing import List
from pydantic import BaseModel, Field


class AbstractLLMResponse(BaseModel):
    "Result back from LLM grading of an abstract"

    # Summary of the abstract
    summary: str = Field(
        ...,
        title="A short summary of the abstract that does not repeat the title, no more than 200 "
        "characters. If there is no abstract provided, just repeat the title.",
    )

    # The most likely experiment this is associated with
    experiment: str = Field(
        ...,
        title="The Experiment associated with this work if known (ATLAS, CMS, LHCb, "
        "MATHUSLA, etc.). Blank if unknown. No explanation.",
    )

    # List of keywords
    keywords: List[str] = Field(
        ..., title="Short JSON list of string-keywords associated with the abstract"
    )

    # What is the interest level here?
    interest: str = Field(
        ...,
        title="The string 'high', 'medium', or 'low' indicating how interesting I'll find "
        "the abstract.",
    )

    # A short explanation of why the interest level is what it is
    explanation: str = Field(
        ...,
        title="Very short explanation of the interest level in the abstract. No more than a "
        "single sentence, 100 words maximum.",
    )

    # How confident is the AI of its interest assignment?
    confidence: float = Field(
        ...,
        title="A float from 0 to 1 representing the confidence in the interest level.",
    )

    # Any terms in the abstract that the LLM does not know, but would probably make
    # the confidence level higher.
    unknown_terms: List[str] = Field(
        ...,
        title="Short JSON list of terms (strings) in the abstract whose definition would "
        "improve your confidence.",
    )
