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
        title="Explanation of the interest level in the abstract. Very short!",
    )


# Please format your with a summary  (One line, terse, summary of the abstract that
# does not repeat the title. It should add extra information beyond the title, and should mention
# any key outcomes that are present in the abstract), an experiment name (If you can guess the
# experiment this abstract is associated with (e.g. ATLAS, CMS, LHCb, etc), place it here.
# Otherwise
# leave it blank), a list of keywords (json-list of 4 or less keywords or phrases describing topics
# in the below abstract and title, comma separated, pulled from my list of interests), and my
# expected interest(put: "high" (hits several of the interests listed above), "medium" (hits one
# interest), or "low" (hits a not interest). Be harsh, my time is valuable).
