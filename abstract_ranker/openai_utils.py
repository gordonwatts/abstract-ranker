# Config items
from pathlib import Path
from typing import Dict

import openai
import yaml
from joblib import Memory

from abstract_ranker.config import CACHE_DIR

memory_openapi = Memory(CACHE_DIR / "openapi", verbose=0)

openai_client = openai.OpenAI(api_key=Path(".openai_key").read_text().strip())


@memory_openapi.cache
def query_gpt(prompt: str, context: Dict[str, str], model: str) -> dict:
    """Queries LLM `model` with a prompt and context.

    Args:
        prompt (str): The prompt for the query.
        context (str): The context for the query.

    Returns:
        dict: The parsed YAML response from OpenAI.
    """
    # Build context:
    c_text = f"""Title: {context['title']}
Abstract: {context['abstract']}"""

    # Generate the completion using OpenAI GPT-3.5 Turbo
    response = openai_client.chat.completions.create(
        # model="gpt-3.5-turbo",
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": prompt,
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
        parsed_response = yaml.safe_load(r)
    else:
        parsed_response = {
            "summary:": "No response from GPT-3.5 Turbo.",
            "experiment": "",
        }

    return parsed_response
