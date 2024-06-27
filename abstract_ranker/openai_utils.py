# Config items
from pathlib import Path
from typing import Dict

import openai

from abstract_ranker.data_model import AbstractLLMResponse


def get_key():
    return Path(".openai_key").read_text().strip()


def query_gpt(prompt: str, context: Dict[str, str], model: str) -> AbstractLLMResponse:
    """Queries LLM `model` with a prompt and context.

    Args:
        prompt (str): The prompt for the query.
        context (str): The context for the query.

    Returns:
        AbstractLLMResponse: The parsed json response from open AI.
    """
    # Build context:
    c_text = f"""Title: {context['title']}
Abstract: {context['abstract']}"""

    # Generate the completion using OpenAI GPT-3.5 Turbo
    openai_client = openai.OpenAI(api_key=get_key())
    response = openai_client.chat.completions.create(
        # model="gpt-3.5-turbo",
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant and expert in the field of experimental "
                "particle physics. Please return responses in JSON.",
            },
            {"role": "user", "content": prompt},
            {"role": "user", "content": c_text},
        ],
        max_tokens=1000,
        temperature=0.7,
        n=1,
        stop=None,
    )

    # Parse the YAML response
    r = response.choices[0].message.content
    if r is not None:
        parsed_response = AbstractLLMResponse.model_validate_json(r)
    else:
        parsed_response = AbstractLLMResponse(
            summary="No response from GPT-3.5 Turbo.",
            experiment="",
            keywords=[],
            interest="",
            explanation="",
        )

    return parsed_response
