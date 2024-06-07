import logging
from typing import Dict


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

    from transformers import pipeline

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

    # Create the pipeline
    pipe = pipeline(
        "text-generation",
        model=model_name,
        trust_remote_code=True,
    )

    logger.debug("Running the pipeline")
    result = pipe(messages)
    logging.info(f"Result from hf inference for {context['title']}: {result}")
    assert isinstance(result, str)
    return result
