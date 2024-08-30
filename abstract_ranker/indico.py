from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, Generator, Optional, Tuple

import pytz
from joblib import Memory
from pydantic import BaseModel
import requests
from tzlocal import get_localzone

from abstract_ranker.config import CACHE_DIR
from abstract_ranker.data_model import Contribution

memory_indico = Memory(CACHE_DIR / "indico", verbose=0)


# Some classes to help us out.
class IndicoDate(BaseModel):
    "A date in the indico system"
    # The date
    date: str

    # The time
    time: str

    # The timezone
    tz: str

    def get_local_datetime(self) -> datetime:
        """Returns the date and time in the local timezone, taking into account
        `self.tz` and `self.date` and `self.time`.

        Returns:
            Tuple[str, str]: The date and time in the local timezone.
        """
        # Get the talk timezone offset and the proper time for the talk.
        talk_timezone = pytz.timezone(self.tz)
        now = datetime.now(talk_timezone)
        timezone_offset = now.strftime("%z")

        talk_time = datetime.strptime(
            f"{self.date} {self.time} {timezone_offset}", "%Y-%m-%d %H:%M:%S %z"
        )

        # Now, create that in local time.
        local_talk_time = talk_time.astimezone(get_localzone())
        return local_talk_time

        # local_date = local_talk_time.strftime("%Y-%m-%d")
        # local_time = local_talk_time.strftime("%I:%M:%S %p")
        # return local_date, local_time


@memory_indico.cache
def _load_indico_json(node: str, meeting_id: str) -> Dict[str, Any]:
    """Loads and returns the JSON info for an indico meeting.

    Args:
        node (str): The url stem for the indico instance we will access
        meeting_id (str): The meeting ID for the meeting we want to access

    Returns:
        _type_: The JSON data for the meeting
    """

    url = f"{node}/export/event/{meeting_id}.json?detail=contributions"

    # Make the request to the URL
    response = requests.get(url)
    return response.json()["results"][0]


def parse_indico_url(event_url: str) -> Tuple[str, str]:
    """Parses the indico event URL and extracts the node and meeting ID.

    Args:
        event_url (str): The URL of the indico event.

    Returns:
        Tuple[str, str]: The node and meeting ID extracted from the URL.
    """
    pattern = r"(https?://[^/]+)/event/(\d+)/?"
    match = re.search(pattern, event_url)
    if match:
        node = match.group(1)
        meeting_id = match.group(2)
        return node, meeting_id
    else:
        raise ValueError("Invalid indico event URL")


def load_indico_json(event_url: str) -> Dict[str, Any]:
    """Returns the json for a url from any indico instance

    Args:
        event_url (str): The URL of anything in the meeting

    Returns:
        Dict[str, Any]: The info for the meeting
    """

    # Example usage
    node, meeting_id = parse_indico_url(event_url)
    return _load_indico_json(node, meeting_id)  # type: ignore


class IndicoContribution(BaseModel):
    "And indico contribution"
    # Title of the talk
    title: str

    # Abstract of the talk
    description: str

    # Poster, plenary, etc.
    type: Optional[str]

    # Start date of the talk
    startDate: Optional[IndicoDate]

    # End date of the talk
    endDate: Optional[IndicoDate]

    # The room
    roomFullname: Optional[str]


def indico_contributions(
    event_data: Dict[str, Any]
) -> Generator[Contribution, None, None]:
    """Yields the contributions from the event data.

    Args:
        event_data (Dict[str, Any]): The event data.

    Yields:
        Dict[str, Any]: The contribution data.
    """
    for contrib in event_data["contributions"]:
        item = IndicoContribution(**contrib)
        start_date = (
            item.startDate.get_local_datetime() if item.startDate is not None else None
        )
        end_date = (
            item.endDate.get_local_datetime() if item.endDate is not None else None
        )

        contribution = Contribution(
            title=item.title,
            abstract=item.description,
            type=item.type,
            startDate=start_date,
            endDate=end_date,
            roomFullname=item.roomFullname,
        )
        yield contribution


def generate_ranking_csv_filename(event: Dict[str, Any]) -> Path:
    """Build a CSV filename for the summary of the abstracts and ranking
    from an indico event.

        Format: "<year>-<monday>-<day> <title>.csv"
    Args:
        event (Dict[str, Any]): The indico data from the event

    Returns:
        str: valid filename for the CSV file
    """
    return Path(f"{event['startDate']['date']} - {event['title']}.csv")
