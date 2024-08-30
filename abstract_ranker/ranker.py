import argparse
import csv
import logging
from pathlib import Path
from typing import Generator, Tuple

from abstract_ranker.config import (
    abstract_ranking_prompt,
    interested_topics,
    not_interested_topics,
)
from abstract_ranker.data_model import AbstractLLMResponse, Contribution
from abstract_ranker.indico import (
    indico_contributions,
    load_indico_json,
)
from abstract_ranker.llm_utils import get_llm_models, query_llm
from abstract_ranker.utils import (
    as_a_number,
    generate_ranking_csv_filename,
    progress_bar,
)


def dump_to_csv_file(
    output_filename: Path,
    data: Generator[Tuple[Contribution, AbstractLLMResponse], None, None],
    progress_bar: bool,
):
    # Open the CSV file in write mode
    with output_filename.open(mode="w", newline="", encoding="utf-8") as file:
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

        for contrib, summary in data:
            # Write the row to the CSV file
            writer.writerow(
                [
                    (
                        contrib.startDate.strftime("%Y-%m-%d %H:%M:%S")
                        if contrib.startDate
                        else ""
                    ),
                    (
                        contrib.startDate.strftime("%Y-%m-%d %H:%M:%S")
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
    # Print a message indicating the CSV file has been created
    logging.info(f"CSV file '{output_filename}' has been created.")


def process_contributions(
    contributions: Generator[Contribution, None, None],
    prompt: str,
    model: str,
    use_cache: bool,
) -> Generator[Tuple[Contribution, AbstractLLMResponse], None, None]:

    for contrib in contributions:
        abstract_text = (
            contrib.description
            if not (contrib.description is None or len(contrib.description) < 10)
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

        yield contrib, summary


def cmd_rank_indico(args):
    event_url = args.indico_url  # "https://indico.cern.ch/event/1330797/contributions/"

    # Build the pipe-line.
    indico_data = load_indico_json(event_url)
    number_contributions = len(indico_data["contributions"])
    contributions = indico_contributions(indico_data)

    if args.v == 0:
        contributions = progress_bar(number_contributions, contributions)

    rankings = process_contributions(
        event_url,
        abstract_ranking_prompt,
        args.model,
        not args.ignore_cache,
    )

    csv_file = generate_ranking_csv_filename(indico_data)
    dump_to_csv_file(csv_file, rankings, args.v == 0)


def cmd_rank_arxiv(args):
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
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="GPT model to use",
        choices=get_llm_models(),
        default="GPT4o",
    )
    parser.add_argument(
        "-v",
        action="count",
        default=0,
        help="Increase output verbosity",
    )
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Ignore the cache and re-run the queries",
        default=False,
    )

    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    rank_indico_parser = subparsers.add_parser(
        "rank_indico",
        help="Rank contributions to an indico event",
        description="""
    Rank the contributions of an Indico event and write them to a CSV file by interest from
    low (1) to high (3). Includes a summary of the contribution's abstract if there was an
    abstract provided.""",
    )
    rank_indico_parser.add_argument(
        "indico_url", type=str, help="URL of the indico event"
    )
    rank_indico_parser.set_defaults(func=cmd_rank_indico)

    rank_arxiv_parser = subparsers.add_parser(
        "rank_arxiv",
        help="Rank contributions for arxiv categories",
        description="""
    Rank the contributions of the daily submissions on arxiv and write them to a CSV
    file by interest from low (1) to high (3). Includes a summary of the
    contribution's abstract.""",
    )
    rank_arxiv_parser.add_argument(
        "arxiv_categories", type=str, nargs="+", help="List of arxiv categories"
    )
    rank_arxiv_parser.set_defaults(func=cmd_rank_arxiv)

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
