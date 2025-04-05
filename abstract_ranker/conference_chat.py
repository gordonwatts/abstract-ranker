import typer
import logging

app = typer.Typer()


def setup_logging(verbose: int):
    level = logging.WARNING  # Default level
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level)


@app.command()
def main(
    indico_url: str = typer.Argument(..., help="URL to an Indico conference endpoint"),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True, min=0, max=2),
):
    """Process a URL to an Indico conference endpoint."""
    verbose = int(verbose)  # Ensure verbose is an integer
    setup_logging(verbose)
    logging.debug(f"Processing URL: {indico_url}")

    # Load what we need - late loading so command processing is fast
    from abstract_ranker.indico import (
        indico_contributions,
        load_indico_json,
    )

    # Next, download the contributions
    indico_data = load_indico_json(indico_url)
    number_contributions = len(indico_data["contributions"])
    logging.info(
        f"Number of contributions from '{indico_data['title']}': {number_contributions}"
    )
    # contributions = indico_contributions(indico_data, "")


if __name__ == "__main__":
    app()


def main_app():
    app()
