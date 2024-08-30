from datetime import datetime
from typing import Generator, List
import logging
import arxiv
from joblib import Memory

from abstract_ranker.config import CACHE_DIR
from abstract_ranker.data_model import Contribution

memory_arxiv = Memory(CACHE_DIR / "arxiv", verbose=0)


@memory_arxiv.cache
def load_arxiv_abstract(topic_list: List[str], what_day: datetime) -> List[arxiv.Result]:
    """Loads and returns basic info from the archive for a set
    of topics.

    Args:
        topic_list (List[str]): A list of topics to search for

    Returns:
        Dict: The JSON info for the arXiv abstract
    """
    assert len(topic_list) > 0, "No topics provided"

    # Search for the 10 most recent articles matching the keyword "quantum."
    the_date = what_day.strftime('%Y%m%d')
    search = arxiv.Search(
        query=f"cat:{" OR cat:".join(topic_list)} and submittedDate:[{the_date} TO {the_date}]",
        max_results=100,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    client = arxiv.Client()
    results = client.results(search)
    all_results = list(results)
    logging.info(f"Found {len(all_results)} results for topics {topic_list}")

    return all_results


def arxiv_contributions(
    event_data: List[arxiv.Result]
) -> Generator[Contribution, None, None]:

    for contrib in event_data:
        yield Contribution(
            title=contrib.title,
            abstract=contrib.summary,
            type=None,
            startDate=contrib.updated,
            endDate=contrib.updated,
            roomFullname=None,
        )
