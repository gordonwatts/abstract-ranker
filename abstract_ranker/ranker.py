import csv
import logging
from typing import Any, Dict, Generator, Optional

from pydantic import BaseModel

from abstract_ranker.indico import IndicoDate, load_indico_json
from abstract_ranker.llm_utils import get_llm_models
from abstract_ranker.openai_utils import query_gpt
import argparse

from abstract_ranker.utils import generate_ranking_csv_filename


class Contribution(BaseModel):
    "And indico contribution"
    # Title of the talk
    title: str

    # Abstract of the talk
    description: str

    # Poster, plenary, etc.
    type: Optional[str]

    # Start date of the talk
    startDate: Optional[IndicoDate]

    # End date of the talk
    endDate: Optional[IndicoDate]

    # The room
    roomFullname: Optional[str]


def contributions(event_data: Dict[str, Any]) -> Generator[Contribution, None, None]:
    """Yields the contributions from the event data.

    Args:
        event_data (Dict[str, Any]): The event data.

    Yields:
        Dict[str, Any]: The contribution data.
    """
    for contrib in event_data["contributions"]:
        yield Contribution(**contrib)


def process_contributions(event_url: str, prompt: str, model: str) -> None:
    """
    Process contributions from the event URL and write them to a CSV file.

    Args:
        event_url (str): The URL of the event.
        prompt (str): The prompt for summarizing the abstracts.
        model(str): The LLM to use for summarization.
    """
    data = load_indico_json(event_url)

    def safe_get(d, key):
        if key in d:
            return d[key] if d[key] else ""
        else:
            return ""

    def as_a_number(interest):
        if interest == "high":
            return 3
        elif interest == "medium":
            return 2
        elif interest == "low":
            return 1
        else:
            return 0

    # Open the CSV file in write mode
    csv_file = generate_ranking_csv_filename(data)
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(
            [
                "Date",
                "Time",
                "Room",
                "Title",
                "Summary",
                "Experiment",
                "Keywords",
                "Interest",
                "Type",
            ]
        )

        # Iterate over the contributions and write each row
        for contrib in contributions(data):
            if not (contrib.description is None or len(contrib.description) < 10):
                summary = query_gpt(
                    prompt,
                    {"title": contrib.title, "abstract": contrib.description},
                    model,
                )

                # Write the row to the CSV file
                writer.writerow(
                    [
                        (
                            contrib.startDate.get_local_datetime()[0]
                            if contrib.startDate
                            else ""
                        ),
                        (
                            contrib.startDate.get_local_datetime()[1]
                            if contrib.startDate
                            else ""
                        ),
                        contrib.roomFullname if contrib.roomFullname else "",
                        contrib.title,
                        safe_get(summary, "summary"),
                        safe_get(summary, "experiment"),
                        safe_get(summary, "keywords"),
                        as_a_number(safe_get(summary, "interest")),
                        contrib.type,
                    ]
                )

    # Print a message indicating the CSV file has been created
    logging.info(f"CSV file '{csv_file}' has been created.")


def cmd_rank(args):
    # Example usage
    event_url = args.indico_url  # "https://indico.cern.ch/event/1330797/contributions/"
    prompt = """I am an expert in experimental particle physics as well as computing for
    particle physics. You are my expert AI assistant who is well versed in particle physics
    and particle physics computing. My interests are in the following areas:
    1. Hidden Sector Physics
    2. Long Lived Particles (Exotics or RPV SUSY)
    3. Analysis techniques and methods and frameworks, particularly those based around python or
       ROOT's DataFrame (RDF)
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

    summary: <One line, terse, summary of the abstract that does not repeat the title. It should
              add extra information beyond the title, and should mention any key outcomes that are
              present in the abstract>
    experiment: <If you can guess the experiment this abstract is associated with (e.g. ATLAS, CMS,
                 LHCb, etc), place it here. Otherwise blank.>
    keywords: <comma separated list of keywords that match my interest list above. If you can't
               find any, leave blank.>
    interest: <If you can guess how interested I am from above, put "low", "medium", or "high"
                here. Otherwise blank.>

    Here is the talk title and Abstract:"""

    process_contributions(event_url, prompt, args.model)


if __name__ == "__main__":

    # Define a command-line parser.
    parser = argparse.ArgumentParser(description="Abstract Ranker")
    parser.set_defaults(func=lambda _: parser.print_usage())

    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    rank_parser = subparsers.add_parser("rank", help="Rank contributions")
    rank_parser.add_argument("indico_url", type=str, help="URL of the indico event")
    rank_parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="GPT model to use",
        choices=get_llm_models(),
        default="GPT4o",
    )
    rank_parser.add_argument(
        "-v",
        action="count",
        default=0,
        help="Increase output verbosity",
    )
    rank_parser.set_defaults(func=cmd_rank)

    args = parser.parse_args()

    # Turn on logging. If the verbosity is 1, set the logging level to INFO. If the verbosity is 2,
    # set the logging level to DEBUG.
    if args.v == 1:
        logging.basicConfig(level=logging.INFO)
    elif args.v >= 2:
        logging.basicConfig(level=logging.DEBUG)

    # Next, call the appropriate command function.
    func = args.func
    func(args)
