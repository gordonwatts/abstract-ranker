import json
from pathlib import Path
from unittest.mock import patch


def test_good_load(cache_dir):
    expected_url = (
        "https://indico.cern.ch/export/event/1330797.json?detail=contributions"
    )
    text_json = Path("tests/data/1330797.json").read_text()
    parsed_json = json.loads(text_json)

    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = parsed_json

        from abstract_ranker.indico import load_indico_json

        data = load_indico_json("https://indico.cern.ch/event/1330797")
        mock_get.assert_called_with(expected_url)

        assert data["title"] == "ACAT 2024"


def test_parse_straight_url():
    from abstract_ranker.indico import parse_indico_url

    assert parse_indico_url("https://indico.cern.ch/event/1330797") == (
        "https://indico.cern.ch",
        "1330797",
    )


def test_parse_with_slash():
    from abstract_ranker.indico import parse_indico_url

    assert parse_indico_url("https://indico.cern.ch/event/1330797/") == (
        "https://indico.cern.ch",
        "1330797",
    )
