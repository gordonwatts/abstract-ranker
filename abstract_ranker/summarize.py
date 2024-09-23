import argparse
import logging


def cmd_summarize(args):
    """Summarize a file.

    Args:
        args (_type_): The command line arguments.
    """
    print("hi")


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

    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    rank_parser = subparsers.add_parser("file", help="Summarize a file")
    rank_parser.add_argument("file", type=str, help="File location")
    # rank_parser.add_argument(
    #     "--model",
    #     "-m",
    #     type=str,
    #     help="GPT model to use",
    #     choices=get_llm_models(),
    #     default="GPT4o",
    # )
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
