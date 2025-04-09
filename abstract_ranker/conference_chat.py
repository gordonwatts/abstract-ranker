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
def ingest(
    indico_url: str = typer.Argument(..., help="URL to an Indico conference endpoint"),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True, min=0, max=2),
    skip_minirag_injection: bool = typer.Option(
        False, "--skip-minirag-injection", help="Skip injection into minirag"
    ),
    max_files: int = typer.Option(
        None, "--max-files", help="Maximum number of files to send to the RAG system"
    ),
):
    """Ingest all contributions from an indico conference into a RAG server."""
    verbose = int(verbose)  # Ensure verbose is an integer
    setup_logging(verbose)
    logging.debug(f"Processing URL: {indico_url}")

    # Load what we need - late loading so command processing is fast
    from abstract_ranker.indico import indico_contributions, load_indico_json
    from abstract_ranker.utils import convert_contribution_to_data
    from abstract_ranker.minirag_ingester import process_attachments

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

    # Apply max_files limit if specified
    if max_files is not None:
        contribution_data = contribution_data[:max_files]

    # Create a download directory in our temp directory based on the conference name.
    conference_name = indico_data["title"].replace(" ", "_")
    temp_dir = tempfile.gettempdir()
    download_dir = Path(temp_dir) / "conference_chat" / conference_name
    download_dir.mkdir(parents=True, exist_ok=True)

    # Use asyncio to call the async function
    asyncio.run(
        process_attachments(
            contribution_data,
            download_dir,
            "http://localhost:9621",  # light rag
            # "http://localhost:9721", # mini rag
            skip_injection=skip_minirag_injection,
        )
    )


@app.command()
def webui():
    """Launch the Web UI to chat with the conference database."""
    import subprocess
    from pathlib import Path

    # Get the path to the current file (conference_chat.py)
    current_file_path = Path(__file__).resolve()

    # Determine the path to conference_chat_streamlit.py relative to this file
    streamlit_file_path = current_file_path.parent / "conference_chat_streamlit.py"

    # Run the Streamlit command
    subprocess.run(["streamlit", "run", str(streamlit_file_path)])


if __name__ == "__main__":
    app()


def main_app():
    app()
