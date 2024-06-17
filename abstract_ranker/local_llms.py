import logging
from typing import Any, Dict, List

from joblib import Memory
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from pydantic import BaseModel

from abstract_ranker.config import CACHE_DIR

memory_hf = Memory(CACHE_DIR / "huggingface_llms", verbose=0)


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


@memory_hf.cache
def query_hugging_face(query: str, context: Dict[str, str], model_name: str) -> str:
    """Use the `transformers` library to run a query from huggingface.co.

    Args:
        query (str): The query text
        context (Dict[str, str]): Context for the query
        model_name (str): Which model we should be using

    Returns:
        str: The reply to the question.
    """

    class AnswerFormat(BaseModel):
        one_line_summary: str
        experiment: str
        keywords: List[str]
        interest: str
        explanation_of_interest: str

    # Build the content out of the context
    content = f"""{query}
Title: {context["title"]}
Abstract: {context["abstract"]}

Please answer in the json schema: {AnswerFormat.schema_json()}
"""

    messages = [
        {
            "role": "system",
            "content": "You are my expert AI assistant will help me pick talks and posters I'm"
            " interested in.",
        },
        {
            "role": "user",
            "content": content,
        },
    ]

    logger = logging.getLogger(__name__)
    logger.debug(f"Loading in model {model_name}")
    pipe = create_pipeline(model_name)
    parser = JsonSchemaParser(AnswerFormat.schema())

    prefix_function = build_transformers_prefix_allowed_tokens_fn(
        pipe.tokenizer, parser
    )

    logger.debug(f"Running the pipeline with args: {content}")
    generation_args = {
        "max_new_tokens": 250,
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
    answer = AnswerFormat.model_validate_json(result)

    return ""
