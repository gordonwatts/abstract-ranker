from datetime import datetime
import pickle

import pytest
from abstract_ranker.arxiv import (
    arxiv_contributions,
    load_arxiv_abstract,
)
from unittest.mock import patch, MagicMock


@patch("abstract_ranker.arxiv.arxiv.Search")
@patch("abstract_ranker.arxiv.arxiv.Client")
def test_load_single_topic(MockClient, MockSearch):
    import arxiv

    mock_client_instance = MockClient.return_value
    mock_client_instance.results.return_value = [
        MagicMock(spec=arxiv.Result) for _ in range(10)
    ]

    r = load_arxiv_abstract(["hep-ex"], datetime.now())

    assert len(r) == 10
    assert MockSearch.call_count == 1
    assert MockSearch.call_args[1]["query"].startswith("cat:hep-ex AND submittedDate")


@patch("abstract_ranker.arxiv.arxiv.Search")
@patch("abstract_ranker.arxiv.arxiv.Client")
def test_load_two_topics_topic(MockClient, MockSearch):
    import arxiv

    mock_client_instance = MockClient.return_value
    mock_client_instance.results.return_value = [
        MagicMock(spec=arxiv.Result) for _ in range(10)
    ]

    load_arxiv_abstract(["hep-ex", "hep-ph"], datetime.now())

    assert MockSearch.call_count == 1
    assert MockSearch.call_args[1]["query"].startswith(
        "cat:hep-ex OR cat:hep-ph AND submittedDate"
    )


@pytest.fixture()
def arxiv_data():
    with open("tests/hep-ex-10.pkl", "rb") as file:
        return pickle.load(file)


def test_contribution_conversion(arxiv_data):
    "Make sure we do the conversion correctly"
    r = list(arxiv_contributions(arxiv_data))

    assert len(r) == 10
    assert (
        r[0].title == "Axion Dark Matter eXperiment around 3.3 μeV with "
        "Dine-Fischler-Srednicki-Zhitnitsky Discovery Ability"
    )
