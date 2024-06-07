from pathlib import Path
from unittest.mock import patch

from abstract_ranker.indico import load_indico_json, parse_indico_url


def test_good_load():
    expected_url = (
        "https://indico.cern.ch/export/event/1330797.json?detail=contributions"
    )
    text_json = Path("tests/data/1330797.json").read_text()

    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = text_json

        data = load_indico_json("https://indico.cern.ch/event/1330797")
        # Rest of your test code goes here
    # data = load_indico_json("https://indico.cern.ch/event/12345")


def test_parse_straight_url():
    assert parse_indico_url("https://indico.cern.ch/event/1330797") == (
        "https://indico.cern.ch",
        "1330797",
    )


def test_parse_with_slash():
    assert parse_indico_url("https://indico.cern.ch/event/1330797/") == (
        "https://indico.cern.ch",
        "1330797",
    )
