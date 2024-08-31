import json
from pathlib import Path
from unittest.mock import patch

from abstract_ranker.indico import generate_ranking_csv_filename, indico_contributions


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


def test_filename():
    event = {"startDate": {"date": "2024-06-24"}, "title": "ACAT 2024"}

    assert generate_ranking_csv_filename(event).name == "2024-06-24 - ACAT 2024.csv"


def test_good_contributions_conversion(cache_dir):
    text_json = Path("tests/data/1330797.json").read_text()
    parsed_json = json.loads(text_json)

    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = parsed_json

        from abstract_ranker.indico import load_indico_json

        data = load_indico_json("https://indico.cern.ch/event/1330797")

        contributions = list(indico_contributions(data))

        assert len(contributions) == 179
        assert contributions[0].title == "ACAT 2024"
        assert any(
            c.title
            == "Rational-function interpolation from p-adic evaluations in scattering "
            "amplitude calculations"
            for c in contributions
        )
        assert contributions[0].url is not None
        assert contributions[0].url.startswith("https://indico.cern.ch/event")
