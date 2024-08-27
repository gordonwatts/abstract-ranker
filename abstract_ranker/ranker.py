import argparse
import csv
import logging
from typing import Any, Dict, Generator, Optional

from pydantic import BaseModel
from rich.progress import Progress

from abstract_ranker.config import (
    abstract_ranking_prompt,
    interested_topics,
    not_interested_topics,
)
from abstract_ranker.indico import IndicoDate, load_indico_json
from abstract_ranker.llm_utils import get_llm_models, query_llm
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


def process_contributions(
    event_url: str, prompt: str, model: str, use_cache: bool, progress_bar: bool
) -> None:
    """
    Process contributions from the event URL and write them to a CSV file.

    Args:
        event_url (str): The URL of the event.
        prompt (str): The prompt for summarizing the abstracts.
        model(str): The LLM to use for summarization.
        progress_bar (bool): Whether to show a progress bar.
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
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
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
                "Confidence",
                "Unknown Terms",
            ]
        )

        # Iterate over the contributions and write each row
        with Progress() as progress:
            task = (
                progress.add_task(
                    "Ranking contributions", total=len(data["contributions"])
                )
                if progress_bar
                else None
            )
            for contrib in contributions(data):
                abstract_text = (
                    contrib.description
                    if not (
                        contrib.description is None or len(contrib.description) < 10
                    )
                    else "Not given"
                )
                summary = query_llm(
                    prompt,
                    {
                        "title": contrib.title,
                        "abstract": abstract_text,
                        "interested_topics": interested_topics,
                        "not_interested_topics": not_interested_topics,
                    },
                    model,
                    use_cache,
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
                        summary.confidence,
                        summary.unknown_terms,
                    ]
                )
                if task is not None:
                    progress.update(task, advance=1)

    # Print a message indicating the CSV file has been created
    logging.info(f"CSV file '{csv_file}' has been created.")


def cmd_rank(args):
    # Example usage
    event_url = args.indico_url  # "https://indico.cern.ch/event/1330797/contributions/"

    process_contributions(
        event_url,
        abstract_ranking_prompt,
        args.model,
        not args.ignore_cache,
        args.v == 0,
    )


def main():
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
    rank_parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Ignore the cache and re-run the queries",
        default=False,
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


if __name__ == "__main__":
    main()
