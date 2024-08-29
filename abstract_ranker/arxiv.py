from datetime import datetime
from typing import List
import logging
import arxiv


def load_arxiv_abstract(topic_list: List[str]) -> List[arxiv.Result]:
    """Loads and returns basic info from the archive for a set
    of topics.

    Args:
        topic_list (List[str]): A list of topics to search for

    Returns:
        Dict: The JSON info for the arXiv abstract
    """
    assert len(topic_list) > 0, "No topics provided"

    # Search for the 10 most recent articles matching the keyword "quantum."
    today = datetime.now().strftime('%Y%m%d')
    search = arxiv.Search(
        query=f"cat:{" OR cat:".join(topic_list)} and submittedDate:[{today} TO {today}]",
        max_results=10,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    client = arxiv.Client()
    results = client.results(search)
    all_results = list(results)
    logging.info(f"Found {len(all_results)} results for topics {topic_list}")

    return all_results
