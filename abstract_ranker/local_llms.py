import logging
from typing import Any, Dict, List

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from abstract_ranker.data_model import AbstractLLMResponse

_hf_models: Dict[str, Any] = {}


def reset():
    """Use for testing - will trigger a clear of everything"""
    _hf_models.clear()


def create_pipeline(model_name: str):
    """Create the pipeline for text generation using the specified model.

    Args:
        model_name (str): Name of the model to use.

    Returns:
        Pipeline: The text generation pipeline.
    """

    if model_name not in _hf_models:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        _hf_models[model_name] = pipeline(
            "text-generation", model=model, tokenizer=tokenizer
        )

    return _hf_models[model_name]


@retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(ValidationError))
def query_hugging_face(
    query: str, context: Dict[str, str | List[str]], model_name: str
) -> AbstractLLMResponse:
    """Use the `transformers` library to run a query from huggingface.co.

    Args:
        query (str): The query text
        context (Dict[str, str]): Context for the query
        model_name (str): Which model we should be using

    Returns:
        str: The reply to the question.
    """

    # Build the content out of the context
    messages = [
        {
            "role": "system",
            "content": "You are my expert AI assistant will help me pick talks and posters I'm"
            " interested in.",
        },
        {
            "role": "user",
            "content": query,
        },
        {
            "role": "user",
            "content": "Topics I'm very interested in:\n - "
            + "\n - ".join(context["interested_topics"]),
        },
        {
            "role": "user",
            "content": "Topics I'm not at all interested in:\n - "
            + "\n - ".join(context["not_interested_topics"]),
        },
        {
            "role": "user",
            "content": f'Conference Talk Title: "{context["title"]}"',
        },
        {
            "role": "user",
            "content": f'Conference Talk Abstract: "{context["abstract"]}"',
        },
        {
            "role": "user",
            "content": "Your answer should be correct JSON using in the following schema. And "
            "everything should be short and succinct with no emoji. Here is the answer schema as "
            "a template:\n"
            f"{AbstractLLMResponse.model_json_schema()['properties']}",
        },
    ]

    logger = logging.getLogger(__name__)
    logger.debug(f"Loading in model {model_name}")
    pipe = create_pipeline(model_name)
    parser = JsonSchemaParser(AbstractLLMResponse.model_json_schema())

    prefix_function = build_transformers_prefix_allowed_tokens_fn(
        pipe.tokenizer, parser
    )

    logger.debug(f"Running the pipeline with args: {query}")
    generation_args = {
        "max_new_tokens": 1024,
        "return_full_text": False,
        "temperature": 1.1,
        "do_sample": True,
        "prefix_allowed_tokens_fn": prefix_function,
    }
    full_result = pipe(messages, **generation_args)
    logger.debug(f"Result from hf inference for {context['title']}: {full_result}")
    result = full_result[0]["generated_text"]
    assert isinstance(result, str)

    logger.debug(f"Text from hf LLM for {context['title']}: \n--**--\n{result}\n--**--")

    # Parse result into a dict

    try:
        return AbstractLLMResponse.model_validate_json(result)
    except Exception:
        logging.error(f"Bad JSON format for '{context['title']}': {result}")

    return AbstractLLMResponse.model_validate_json(result + '" }')
