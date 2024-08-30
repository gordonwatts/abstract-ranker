from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Contribution(BaseModel):
    "A contribution we are going to evaluate"
    # Title of the talk
    title: str

    # Abstract of the talk
    abstract: str

    # Poster, plenary, etc.
    type: Optional[str]

    # Time of the abstract/talk/paper
    startDate: Optional[datetime]

    # End date of the talk (or same as startDate)
    endDate: Optional[datetime]

    # The room metadata
    roomFullname: Optional[str]


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
