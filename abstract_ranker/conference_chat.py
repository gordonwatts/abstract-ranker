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
    url: str = typer.Argument(..., help="URL to an Indico conference endpoint"),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True, min=0, max=2),
):
    """Process a URL to an Indico conference endpoint."""
    verbose = int(verbose)  # Ensure verbose is an integer
    setup_logging(verbose)
    logging.info(f"Processing URL: {url}")
    # Add logic to process the URL here


if __name__ == "__main__":
    app()


def main_app():
    app()
