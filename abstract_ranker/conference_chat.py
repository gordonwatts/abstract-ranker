import logging
import tempfile
from pathlib import Path
import asyncio

import typer

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
    from abstract_ranker.indico import indico_contributions, load_indico_json
    from .utils import convert_contribution_to_data
    from .minirag_ingester import process_attachments

    # Next, download the list of contributions
    indico_data = load_indico_json(indico_url)
    number_contributions = len(indico_data["contributions"])
    logging.info(
        f"Number of contributions from '{indico_data['title']}': {number_contributions}"
    )
    contributions = indico_contributions(indico_data, "")
    contribution_data = [
        convert_contribution_to_data(contrib) for contrib in contributions
    ]

    # Create a download directory in our temp directory based on the conference name.
    conference_name = indico_data["title"].replace(" ", "_")
    temp_dir = tempfile.gettempdir()
    download_dir = Path(temp_dir) / "conference_chat" / conference_name
    download_dir.mkdir(parents=True, exist_ok=True)

    # Use asyncio to call the async function
    asyncio.run(
        process_attachments(contribution_data, download_dir, "https://hi-there")
    )


if __name__ == "__main__":
    app()


def main_app():
    app()
