from typing import Generator, TypeVar
from dataclasses import dataclass
from collections import defaultdict
from abstract_ranker.data_model import Contribution

from rich.progress import Progress


@dataclass
class ContributionData:
    title: str
    abstract: str
    urls: list[str]


def convert_contribution_to_data(contribution: Contribution) -> ContributionData:
    """Convert a Contribution to a ContributionData instance with a list of URLs."""
    attachment_groups = defaultdict(list)

    for attachment in contribution.attachments:
        base_name = attachment.rsplit(".", 1)[0]
        attachment_groups[base_name].append(attachment)

    urls = []
    for base_name, attachments in attachment_groups.items():
        preferred_attachment = sorted(
            attachments, key=lambda x: (x.endswith(".pdf"), x), reverse=True
        )[0]
        urls.append(preferred_attachment)

    return ContributionData(
        title=contribution.title, abstract=contribution.abstract, urls=urls
    )


def as_a_number(interest: str) -> int:
    """Convert the interest level to a number."""

    if interest == "high":
        return 3
    elif interest == "medium":
        return 2
    elif interest == "low":
        return 1
    else:
        return 0


T = TypeVar("T")


def progress_bar(
    length: int, data: Generator[T, None, None]
) -> Generator[T, None, None]:
    """A progress bar for the indicating how close we are to being done."""
    with Progress() as progress:
        task = progress.add_task("Ranking contributions", total=length)
        for contrib in data:
            yield contrib
            progress.update(task, advance=1)
