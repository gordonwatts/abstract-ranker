from abstract_ranker.utils import convert_contribution_to_data, ContributionData
from abstract_ranker.data_model import Contribution


def test_convert_contribution_to_data_single_attachment():
    contribution = Contribution(
        title="Sample Title",
        abstract="Sample Abstract",
        type=None,
        startDate=None,
        endDate=None,
        roomFullname=None,
        url=None,
        attachments=["attachment1.pdf"],
    )
    result = convert_contribution_to_data(contribution)
    assert result == ContributionData(
        title="Sample Title", abstract="Sample Abstract", urls=["attachment1.pdf"]
    )


def test_convert_contribution_to_data_multiple_attachments_same_name():
    contribution = Contribution(
        title="Sample Title",
        abstract="Sample Abstract",
        type=None,
        startDate=None,
        endDate=None,
        roomFullname=None,
        url=None,
        attachments=["attachment1.txt", "attachment1.pdf"],
    )
    result = convert_contribution_to_data(contribution)
    assert result == ContributionData(
        title="Sample Title", abstract="Sample Abstract", urls=["attachment1.pdf"]
    )


def test_convert_contribution_to_data_multiple_different_attachments():
    contribution = Contribution(
        title="Sample Title",
        abstract="Sample Abstract",
        type=None,
        startDate=None,
        endDate=None,
        roomFullname=None,
        url=None,
        attachments=["attachment1.pdf", "attachment2.pdf"],
    )
    result = convert_contribution_to_data(contribution)
    assert result == ContributionData(
        title="Sample Title",
        abstract="Sample Abstract",
        urls=["attachment1.pdf", "attachment2.pdf"],
    )


def test_convert_contribution_to_data_no_attachments():
    contribution = Contribution(
        title="Sample Title",
        abstract="Sample Abstract",
        type=None,
        startDate=None,
        endDate=None,
        roomFullname=None,
        url=None,
        attachments=[],
    )
    result = convert_contribution_to_data(contribution)
    assert result == ContributionData(
        title="Sample Title", abstract="Sample Abstract", urls=[]
    )
