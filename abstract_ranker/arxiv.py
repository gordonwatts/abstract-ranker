from datetime import datetime, timedelta
from typing import Generator, List
import logging
import arxiv
from joblib import Memory
from pathlib import Path

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
    # Note the capitalization: logic operand words and the specifiers are case-sensitive!!
    the_date = what_day.strftime('%Y%m%d')
    the_end_date = (what_day + timedelta(days=1)).strftime('%Y%m%d')
    query_string = (
        f"(cat:{" OR cat:".join(topic_list)}) AND submittedDate:[{the_date} TO {the_end_date}]"
        )
    logging.info(f"arXiv Query string: {query_string} for day {what_day}")

    search = arxiv.Search(
        query=query_string,
        max_results=100,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    # DO the search
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


def arxiv_ranked_filename(what_day: datetime, topics: List[str]) -> Path:
    """Generate the filename for the arXiv ranking file which includes
    of the form:

        `arxiv-<year>-<month>-<day>-<topic1>-<topic2>.csv`

    Where the topic names are alphabetically sorted.

    Args:
        what_day (datetime): The day the ranking was done
        topics (List[str]): The topics we are ranking

    Returns:
        Path to the output file (in local directory).
    """
    topic_str = "-".join(sorted(topics))
    return Path(f"arxiv-{what_day:%Y-%m-%d}-{topic_str}.csv")
