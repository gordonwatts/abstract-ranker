from datetime import datetime
import re
from typing import Any, Dict, Tuple

import pytz
from joblib import Memory
from pydantic import BaseModel
import requests
from tzlocal import get_localzone

from abstract_ranker.config import CACHE_DIR

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

    def get_local_datetime(self) -> Tuple[str, str]:
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

        local_date = local_talk_time.strftime("%Y-%m-%d")
        local_time = local_talk_time.strftime("%I:%M:%S %p")
        return local_date, local_time


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
