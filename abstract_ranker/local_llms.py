import logging
from typing import Any, Dict

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
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

        _hf_models[model_name] = pipeline(
            "text-generation", model=model, tokenizer=tokenizer
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
            "role": "system",
            "content": "All of your answers will be parsed by a yaml parser. "
            "Format answer as yaml (all output must be yaml). If the user gives you a template"
            " for the yaml, please use that (it will be in between ```yaml <template> ```.",
        },
        {
            "role": "user",
            "content": content,
        },
    ]
    logger = logging.getLogger(__name__)
    logger.debug(f"Loading in model {model_name}")
    pipe = create_pipeline(model_name)

    logger.debug("Running the pipeline")
    generation_args = {
        "max_new_tokens": 250,
        "return_full_text": False,
        "temperature": 1.0,
        "do_sample": True,
    }
    logger.debug(f"Running the pipeline with args: {content}")
    full_result = pipe(messages, **generation_args)
    logger.debug(f"Result from hf inference for {context['title']}: {full_result}")
    result = full_result[0]["generated_text"]
    assert isinstance(result, str)
    logger.info(f"Text from hf LLM for {context['title']}: \n--**--\n{result}\n--**--")
    return result
