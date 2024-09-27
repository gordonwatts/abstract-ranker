import argparse
import logging
import fsspec

from abstract_ranker.llm_utils import get_llm_models, summarize_llm


def cmd_summarize(args):
    """Summarize a file.

    Args:
        args (_type_): The command line arguments.
    """
    with fsspec.open(args.file, "r") as f:
        text = f.read()  # type: ignore

    # Type of summary
    summary_type_prompt = """
The below text are meeting minutes. Please extract from the meeting minutes any action items,
decisions made, or what look like big successes. The output should be Markdown that looks like
this:

# <Meeting Title>

## Action Items
- Item 1
- Item 2

## Decisions Made
- Decision 1
- Decision 2

## Big Successes
- Success 1
- Success 2

"""

    r = summarize_llm(
        summary_type_prompt, {"text": text}, args.model, not args.ignore_cache
    )

    # Dump output to stdout
    print(r)


def main():
    """Main command line to summarize things.

    We take files (or eventually) other things.
    """
    parser = argparse.ArgumentParser(description="LLM Summarizer")
    parser.set_defaults(func=lambda _: parser.print_usage())

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

    rank_parser = subparsers.add_parser("file", help="Summarize a file")
    rank_parser.add_argument("file", type=str, help="File location")
    rank_parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="GPT model to use",
        choices=get_llm_models(),
        default="GPT4o",
    )
    rank_parser.set_defaults(func=cmd_summarize)

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
