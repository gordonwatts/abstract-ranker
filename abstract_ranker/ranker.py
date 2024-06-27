import csv
import logging
from typing import Any, Dict, Generator, Optional

from pydantic import BaseModel

from abstract_ranker.indico import IndicoDate, load_indico_json
from abstract_ranker.llm_utils import get_llm_models, query_llm
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
                summary = query_llm(
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
                        summary.summary,
                        summary.experiment,
                        summary.keywords,
                        as_a_number(summary.interest),
                        contrib.type,
                    ]
                )

    # Print a message indicating the CSV file has been created
    logging.info(f"CSV file '{csv_file}' has been created.")


def cmd_rank(args):
    # Example usage
    event_url = args.indico_url  # "https://indico.cern.ch/event/1330797/contributions/"
    prompt = """Help me judge the following conference presentation as interesting or not.
My interests are in the following areas:

    1. Hidden Sector Physics
    2. Long Lived Particles (Exotics or RPV SUSY)
    3. Analysis techniques and methods and frameworks, particularly those based around python or
       ROOT's DataFrame (RDF)
    4. Machine Learning and AI for particle physics
    5. The ServiceX tool
    6. Distributed computing for analysis (e.g. Dask, Spark, etc)
    7. Data Preservation and FAIR principles
    8. Differentiable Programming

I am *not interested* in:

    1. Quantum Computing
    2. Lattice Gauge Theory
    3. Neutrino Physics

"""
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
    elif args.v == 2:
        logging.basicConfig(level=logging.DEBUG)
        for p in ["httpx", "httpcore", "urllib3", "filelock"]:
            logging.getLogger(p).setLevel(logging.WARNING)
    elif args.v >= 3:
        logging.basicConfig(level=logging.DEBUG)

    # Next, call the appropriate command function.
    func = args.func
    func(args)
