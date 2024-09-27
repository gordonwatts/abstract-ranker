import argparse
import logging
from pathlib import Path
from typing import Generator
from datetime import datetime, timedelta

from abstract_ranker.config import (
    abstract_ranking_prompt,
)
from abstract_ranker.data_model import Contribution
from abstract_ranker.llm_utils import get_llm_models


def _generate_ranking_results(
    args,
    number_contributions: int,
    contributions: Generator[Contribution, None, None],
    csv_file: Path,
):
    """Generate the ranking results.

    Args:
        args (_type_): Command line arguments for common steering parameters.
        number_contributions (int): The total number of contributions.
        contributions (Generator[Contribution, None, None]): The list of contributions.
        csv_file (Path): Where we will write the csv file.
    """
    from abstract_ranker.driver import process_contributions
    from abstract_ranker.output import dump_to_csv_file
    from abstract_ranker.utils import progress_bar

    if args.v == 0:
        contributions = progress_bar(number_contributions, contributions)

    rankings = process_contributions(
        contributions,
        abstract_ranking_prompt,
        args.model,
        not args.ignore_cache,
    )

    dump_to_csv_file(csv_file, rankings, args.v == 0)


def cmd_rank_indico(args):
    from abstract_ranker.indico import (
        generate_ranking_csv_filename,
        indico_contributions,
        load_indico_json,
    )

    # Build the pipe-line.
    indico_data = load_indico_json(args.indico_url)
    number_contributions = len(indico_data["contributions"])
    contributions = indico_contributions(indico_data)

    csv_file = generate_ranking_csv_filename(indico_data)

    _generate_ranking_results(args, number_contributions, contributions, csv_file)


def cmd_rank_arxiv(args):
    """Driver to rank the arxiv abstracts

    Args:
        args (): Command line arguments
    """
    # Get yesterday's date - that was when things were submitted.
    the_date = datetime.now() - timedelta(days=1)
    the_date = the_date.replace(hour=0, minute=0, second=1, microsecond=0)

    from abstract_ranker.arxiv import (
        arxiv_contributions,
        arxiv_ranked_filename,
        load_arxiv_abstract,
    )

    # Now load in the submissions.
    arxiv_data = load_arxiv_abstract(args.arxiv_categories, the_date)
    contributions = arxiv_contributions(arxiv_data)
    csv_file = arxiv_ranked_filename(the_date, args.arxiv_categories)

    _generate_ranking_results(args, len(arxiv_data), contributions, csv_file)


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
