from typing import Dict, List

from abstract_ranker.local_llms import query_hugging_face
from abstract_ranker.openai_utils import query_gpt


_llm_dispatch = {
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


def query_llm(prompt: str, context: Dict[str, str], model: str) -> dict:
    """Query the given LLM for a summary.

    Args:
        prompt (str): Prompt to use
        context (Dict[str, str]): The context and instructions
        model (str): The name of the model to use, short hand.

    Returns:
        dict: The results, parsed as json.
    """
    return _llm_dispatch[model](prompt, context)
