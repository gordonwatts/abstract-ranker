from typing import Any, Dict


def generate_ranking_csv_filename(event: Dict[str, Any]) -> str:
    """Build a CSV filename for the summary of the abstracts and ranking.

        Format: "<year>-<monday>-<day> <title>.csv"
    Args:
        event (Dict[str, Any]): The indico data from the event

    Returns:
        str: valid filename for the CSV file
    """
    return f"{event['startDate']['date']} - {event['title']}.csv"
