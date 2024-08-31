from typing import Generator, Tuple

from abstract_ranker.data_model import AbstractLLMResponse, Contribution
from abstract_ranker.llm_utils import query_llm

from abstract_ranker.config import interested_topics, not_interested_topics


def process_contributions(
    contributions: Generator[Contribution, None, None],
    prompt: str,
    model: str,
    use_cache: bool,
) -> Generator[Tuple[Contribution, AbstractLLMResponse], None, None]:
    """Feed each contribution to the LLM, and get back the summary information.

    Args:
        contributions (Generator[Contribution, None, None]): The contribution list.
        prompt (str): The prompt to feed the LLM.
        model (str): The name of the model to run
        use_cache (bool): If False, don't use the LLM cache.

    Yields:
        Generator[Tuple[Contribution, AbstractLLMResponse], None, None]: The summary data
                                            from the LLM.
    """

    for contrib in contributions:
        abstract_text = (
            contrib.abstract
            if not (contrib.abstract is None or len(contrib.abstract) < 10)
            else "Not given"
        )
        summary = query_llm(
            prompt,
            {
                "title": contrib.title,
                "abstract": abstract_text,
                "interested_topics": interested_topics,
                "not_interested_topics": not_interested_topics,
            },
            model,
            use_cache,
        )

        yield contrib, summary
