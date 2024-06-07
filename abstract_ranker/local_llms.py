import logging
from typing import Any, Dict

_hf_models: Dict[str, Any] = {}


def create_pipeline(model_name: str):
    """Create the pipeline for text generation using the specified model.

    Args:
        model_name (str): Name of the model to use.

    Returns:
        Pipeline: The text generation pipeline.
    """

    from transformers import pipeline

    if model_name not in _hf_models:
        _hf_models[model_name] = pipeline(
            "text-generation", model=model_name, trust_remote_code=True
        )

    return _hf_models[model_name]


def query_hugging_face(query: str, context: Dict[str, str], model_name: str) -> str:
    """Use the `transformers` library to run a query from huggingface.co.

    Args:
        query (str): The query text
        context (Dict[str, str]): Context for the query
        model_name (str): Which model we should be using

    Returns:
        str: The reply to the question.
    """
    # Build the content out of the context
    content = f"""{query}
Title: {context["title"]}
Abstract: {context["abstract"]}"""

    messages = [
        {
            "role": "All of your answers will be parsed by a yaml parser. "
            "Format answer as yaml (all output must be yaml). If the user gives you a template"
            " for the yaml, please use that (it will be in between ```yaml <template> ```.",
            "content": content,
        },
    ]
    logger = logging.getLogger(__name__)
    logger.debug(f"Loading in model {model_name}")
    pipe = create_pipeline(model_name)

    logger.debug("Running the pipeline")
    result = pipe(messages)
    logging.info(f"Result from hf inference for {context['title']}: {result}")
    assert isinstance(result, str)
    return result
