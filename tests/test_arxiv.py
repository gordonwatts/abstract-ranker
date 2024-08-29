from abstract_ranker.arxiv import load_arxiv_abstract
from unittest.mock import patch, MagicMock


@patch("abstract_ranker.arxiv.arxiv.Search")
@patch("abstract_ranker.arxiv.arxiv.Client")
def test_load_single_topic(MockClient, MockSearch):
    import arxiv

    mock_client_instance = MockClient.return_value
    mock_client_instance.results.return_value = [
        MagicMock(spec=arxiv.Result) for _ in range(10)
    ]

    r = load_arxiv_abstract(["hep-ex"])

    assert len(r) == 10
    assert MockSearch.call_count == 1
    assert MockSearch.call_args[1]["query"].startswith("cat:hep-ex and submittedDate")


@patch("abstract_ranker.arxiv.arxiv.Search")
@patch("abstract_ranker.arxiv.arxiv.Client")
def test_load_two_topics_topic(MockClient, MockSearch):
    import arxiv

    mock_client_instance = MockClient.return_value
    mock_client_instance.results.return_value = [
        MagicMock(spec=arxiv.Result) for _ in range(10)
    ]

    load_arxiv_abstract(["hep-ex", "hep-ph"])

    assert MockSearch.call_count == 1
    assert MockSearch.call_args[1]["query"].startswith(
        "cat:hep-ex OR cat:hep-ph and submittedDate"
    )
