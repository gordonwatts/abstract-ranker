from typing import Any, Callable, Dict, List

from pydantic import BaseModel, Field

from abstract_ranker.local_llms import query_hugging_face
from abstract_ranker.openai_utils import query_gpt


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


_llm_dispatch: Dict[str, Callable[[str, Dict[Any, Any]], AbstractLLMResponse]] = {
    "GPT4Turbo": lambda prompt, context: query_gpt(prompt, context, "gpt-4-turbo"),
    "GPT4o": lambda prompt, context: query_gpt(prompt, context, "gpt-4o"),
    "GPT35Turbo": lambda prompt, context: query_gpt(prompt, context, "gpt-3.5-turbo"),
    "phi3-mini": lambda prompt, context: query_hugging_face(
        prompt, context, "microsoft/Phi-3-mini-4k-instruct"
    ),
}


def get_llm_models() -> List[str]:
    """Get the available LLM models.

    Returns:
        Dict[str, str]: The available models.
    """
    return list(_llm_dispatch.keys())


def query_llm(prompt: str, context: Dict[str, str], model: str) -> AbstractLLMResponse:
    """Query the given LLM for a summary.

    Args:
        prompt (str): Prompt to use
        context (Dict[str, str]): The context and instructions
        model (str): The name of the model to use, short hand.

    Returns:
        dict: The results, parsed as json.
    """
    return _llm_dispatch[model](prompt, context)
