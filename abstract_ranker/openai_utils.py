# Config items
import logging
from pathlib import Path
from typing import Dict, List

import openai

from abstract_ranker.data_model import AbstractLLMResponse


def get_key():
    return Path(".openai_key").read_text().strip()


def query_gpt(
    prompt: str, context: Dict[str, str | List[str]], model: str
) -> AbstractLLMResponse:
    """Queries LLM `model` with a prompt and context.

    Args:
        prompt (str): The prompt for the query.
        context (str): The context for the query.

    Returns:
        AbstractLLMResponse: The parsed json response from open AI.
    """
    # Fix up Schema to make it easier for the LLM to interpret.
    schema = AbstractLLMResponse.model_json_schema()["properties"]
    schema = {k: v["title"] for k, v in schema.items()}

    # Generate the completion using OpenAI
    openai_client = openai.OpenAI(api_key=get_key())
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant and expert in the field of experimental "
                "particle physics. All responses must be in the JSON format specified.",
            },
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": "Topics I'm very interested in\n - "
                + "\n - ".join(context["interested_topics"]),
            },
            {
                "role": "user",
                "content": "Topics I'm not at all interested in\n - "
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
                "content": "Your answer should be correct JSON using in the following JSON schema."
                " Everything should be short and succinct with no emoji, and properly escape "
                "latex directives. This is a JSON schema, so "
                "replace the title and type dict with the actual data: \n"
                f"{schema}",
            },
        ],
        max_tokens=1000,
        temperature=0.7,
        n=1,
        stop=None,
    )

    # Parse the YAML response
    r = response.choices[0].message.content
    if r is not None:
        # Remove leading text or trailing text
        start_bracket = r.find("{")
        if start_bracket != -1:
            logging.debug(f"Removing header from response: {r[:start_bracket]}")
            r = r[start_bracket:]

        end_bracket = r.rfind("}")
        if end_bracket != -1:
            logging.debug(f"Removing trailer from response: {r[end_bracket:]}")
            r = r[: end_bracket + 1]

        # Escape any latex in there
        r = r.replace("\\", "\\\\")

        # Parse the response
        try:
            parsed_response = AbstractLLMResponse.model_validate_json(r)
        except Exception as e:
            logging.error(f"Bad JSON format for '{context['title']}': {r} ({e})")
            raise

    else:
        parsed_response = AbstractLLMResponse(
            summary="No response from {model}.",
            experiment="",
            keywords=[],
            interest="",
            explanation="",
            confidence=0.0,
            unknown_terms=[],
        )

    return parsed_response
