from pathlib import Path
from typing import Any, Dict, Generator, TypeVar

from rich.progress import Progress


def generate_ranking_csv_filename(event: Dict[str, Any]) -> Path:
    """Build a CSV filename for the summary of the abstracts and ranking.

        Format: "<year>-<monday>-<day> <title>.csv"
    Args:
        event (Dict[str, Any]): The indico data from the event

    Returns:
        str: valid filename for the CSV file
    """
    return Path(f"{event['startDate']['date']} - {event['title']}.csv")


def as_a_number(interest: str) -> int:
    """Convert the interest level to a number."""

    if interest == "high":
        return 3
    elif interest == "medium":
        return 2
    elif interest == "low":
        return 1
    else:
        return 0


T = TypeVar("T")


def progress_bar(
    length: int, data: Generator[T, None, None]
) -> Generator[T, None, None]:
    """A progress bar for the indicating how close we are to being done."""
    with Progress() as progress:
        task = progress.add_task("Ranking contributions", total=length)
        for contrib in data:
            yield contrib
            progress.update(task, advance=1)
