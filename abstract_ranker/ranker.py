import csv
import re
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import openai
import requests
import yaml
from joblib import Memory
from pydantic import BaseModel

# Config items
CACHE_DIR = Path("./.abstract_cache")
memory_indico = Memory(CACHE_DIR / "indico", verbose=0)
memory_openapi = Memory(CACHE_DIR / "openapi", verbose=0)


# Some classes to help us out.
class Contribution(BaseModel):
    "And indico contribution"
    # Title of the talk
    title: str

    # Abstract of the talk
    description: str

    # Poster, plenary, etc.
    type: Optional[str]


@memory_indico.cache
def _load_indico_json(node: str, meeting_id: str) -> Dict[str, Any]:
    """Loads and returns the JSON info for an indico meeting.

    Args:
        node (str): The url stem for the indico instance we will access
        meeting_id (str): The meeting ID for the meeting we want to access

    Returns:
        _type_: The JSON data for the meeting
    """

    url = f"{node}/export/event/{meeting_id}.json?detail=contributions"

    # Make the request to the URL
    response = requests.get(url)
    return response.json()["results"][0]


def parse_indico_url(event_url: str) -> Tuple[str, str]:
    """Parses the indico event URL and extracts the node and meeting ID.

    Args:
        event_url (str): The URL of the indico event.

    Returns:
        Tuple[str, str]: The node and meeting ID extracted from the URL.
    """
    pattern = r"(https?://[^/]+)/event/(\d+)/"
    match = re.search(pattern, event_url)
    if match:
        node = match.group(1)
        meeting_id = match.group(2)
        return node, meeting_id
    else:
        raise ValueError("Invalid indico event URL")


def load_indico_json(event_url: str) -> Dict[str, Any]:
    """Returns the json for a url from any indico instance

    Args:
        event_url (str): The URL of anything in the meeting

    Returns:
        Dict[str, Any]: The info for the meeting
    """

    # Example usage
    node, meeting_id = parse_indico_url(event_url)
    return _load_indico_json(node, meeting_id)  # type: ignore


def contributions(event_data: Dict[str, Any]) -> Generator[Contribution, None, None]:
    """Yields the contributions from the event data.

    Args:
        event_data (Dict[str, Any]): The event data.

    Yields:
        Dict[str, Any]: The contribution data.
    """
    for contrib in event_data["contributions"]:
        yield Contribution(**contrib)


openai_client = openai.OpenAI(api_key=Path(".openai_key").read_text().strip())


@memory_openapi.cache
def query_gpt(prompt: str, context: Dict[str, str]) -> dict:
    """Queries OpenAI GPT-3.5 Turbo with a prompt and context.

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


if __name__ == "__main__":
    # Example usage
    event_url = "https://indico.cern.ch/event/1330797/contributions/"
    data = load_indico_json(event_url)
    prompt = """I am an expert in experimental particle physics as well as computing for
 particle physics. You are my expert AI assistant who is well versed in particle physics
 and particle physics computing. My interests are in the following areas:
    1. Hidden Sector Physics
    2. Long Lived Particles (Exotics or RPV SUSY)
    3. Analysis techniques and methods and frameworks, particularly those based around python or ROOT's DataFrame (RDF)
    4. Machine Learning and AI for particle physics
    5. Distributed computing for analysis (e.g. Dask, Spark, etc)
    6. Data Preservation and FAIR principles
    7. Differentiable Programming

I'm not very interested in:
    1. Quantum Computing
    2. Lattice QCD
    3. Neutrino Physics

Please summarize this conference abstract so I can quickly judge the abstract and if I want to
see the talk it represents. 

Your reply should have the following format:

summary: <One line, terse, summary of the abstract that does not repeat the title. It should add extra information beyond the title, and should mention any key outcomes that are present in the abstract>
experiment: <If you can guess the experiment this abstract is associated with (e.g. ATLAS, CMS, LHCb, etc), place it here. Otherwise blank.>
keywords: <comma separated list of keywords that match my interest list above. If you can't find any, leave blank.>
interest: <If you can guess how interested I am from above, put "low", "medium", or "high" here. Otherwise blank.>

Here is the talk title and Abstract:"""

    def safe_get(d, key):
        if key in d:
            return d[key] if d[key] else ""
        else:
            return ""

    # Define the CSV file path
    csv_file = "abstract_summary.csv"

    # Open the CSV file in write mode
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(
            ["Title", "Summary", "Experiment", "Keywords", "Interest", "Type"]
        )

        # Iterate over the contributions and write each row
        for contrib in contributions(data):
            if not (contrib.description is None or len(contrib.description) < 10):
                summary = query_gpt(
                    prompt,
                    {"title": contrib.title, "abstract": contrib.description},
                )

                # Write the row to the CSV file
                writer.writerow(
                    [
                        contrib.title,
                        safe_get(summary, "summary"),
                        safe_get(summary, "experiment"),
                        safe_get(summary, "keywords"),
                        safe_get(summary, "interest"),
                        contrib.type,
                    ]
                )

    # Print a message indicating the CSV file has been created
    print(f"CSV file '{csv_file}' has been created.")
