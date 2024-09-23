from typing import Any, Callable, Dict, List

from joblib import Memory

from abstract_ranker.config import CACHE_DIR
from abstract_ranker.data_model import AbstractLLMResponse
from abstract_ranker.local_llms import query_hugging_face
from abstract_ranker.openai_utils import query_gpt, summarize_gpt

memory_llm_query = Memory(CACHE_DIR / "llm_queries", verbose=0)


_llm_dispatch: Dict[str, Callable[[str, Dict[Any, Any]], AbstractLLMResponse]] = {
    "GPT4Turbo": lambda prompt, context: query_gpt(prompt, context, "gpt-4-turbo"),
    "GPT4o": lambda prompt, context: query_gpt(prompt, context, "gpt-4o"),
    "GPT35Turbo": lambda prompt, context: query_gpt(prompt, context, "gpt-3.5-turbo"),
    "phi3-mini": lambda prompt, context: query_hugging_face(
        prompt, context, "microsoft/Phi-3-mini-4k-instruct"
    ),
}

_llm_summary_dispatch: Dict[str, Callable[[str, Dict[Any, Any]], str]] = {
    "GPT4o": lambda prompt, context: summarize_gpt(prompt, context, "gpt-4-turbo"),
}


def get_llm_models() -> List[str]:
    """Get the available LLM models.

    Returns:
        Dict[str, str]: The available models.
    """
    return list(_llm_dispatch.keys())


@memory_llm_query.cache
def query_llm(
    prompt: str, context: Dict[str, str | List[str]], model: str
) -> AbstractLLMResponse:
    """Query the given LLM for a summary.

    Args:
        prompt (str): Prompt to use
        context (Dict[str, str]): The context and instructions
        model (str): The name of the model to use, short hand.

    Returns:
        dict: The results, parsed as json.
    """
    return _llm_dispatch[model](prompt, context)


def summarize_llm(prompt: str, context: Dict[str, str], model: str) -> None:
    """Summarize the given context with the given model.

    Args:
        prompt (str): The prompt to use.
        context (Dict[str, str]): The context to use.
        model (str): The model to use.
    """
    response = query_llm(prompt, context, model)
