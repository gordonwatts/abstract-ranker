import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from abstract_ranker.minirag_ingester import (
    download_attachment,
    insert_into_minirag,
    process_attachments,
    run_docling,
)
from abstract_ranker.utils import ContributionData


@pytest.mark.asyncio
async def test_download_attachment(tmp_path: Path):
    attachment_url = "http://example.com/test.pdf"
    download_dir = Path(tmp_path)
    with patch("aiohttp.ClientSession.get", new_callable=AsyncMock) as mock_get:
        mock_response = AsyncMock()
        mock_response.read.return_value = b"PDF content"
        mock_response.raise_for_status = lambda: None
        mock_get.return_value = mock_response

        file_path = await download_attachment(attachment_url, download_dir)
        assert file_path.exists()
        assert file_path.name == "test.pdf"


@pytest.mark.asyncio
async def test_download_attachment_with_escape_sequences(tmp_path: Path):
    attachment_url = "http://example.com/test%20file.pdf"
    download_dir = Path(tmp_path)

    with patch("aiohttp.ClientSession.get", new_callable=AsyncMock) as mock_get:
        mock_response = AsyncMock()
        mock_response.read.return_value = b"PDF content"
        mock_response.raise_for_status = lambda: None
        mock_get.return_value = mock_response

        file_path = await download_attachment(attachment_url, download_dir)
        assert file_path.exists()
        assert file_path.name == "test file.pdf"


@pytest.mark.asyncio
async def test_download_attachment_skips_existing_file(tmp_path: Path):
    attachment_url = "http://example.com/test%20file.pdf"
    download_dir = Path(tmp_path)
    existing_file = download_dir / "test file.pdf"
    existing_file.write_text("Existing content")

    with patch("aiohttp.ClientSession.get", new_callable=AsyncMock) as mock_get:
        mock_response = AsyncMock()
        mock_response.read.return_value = b"PDF content"
        mock_response.return_value = mock_response
        mock_get.return_value = mock_response

        file_path = await download_attachment(attachment_url, download_dir)
        assert file_path.exists()
        assert file_path.name == "test file.pdf"
        assert file_path.read_text() == "Existing content"


@pytest.mark.asyncio
async def test_run_docling(tmp_path: Path):
    file_path = tmp_path / "test.pdf"
    file_path.touch()  # Create the dummy PDF file in the temp path

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value.communicate.return_value = (b"", b"")
        mock_exec.return_value.returncode = 0

        # Ensure the dummy .md file is created in the correct location
        dummy_md_file = file_path.with_suffix(".md")
        dummy_md_file.touch()

        output_file = await run_docling(file_path)
        assert output_file == dummy_md_file


@pytest.mark.asyncio
async def test_insert_into_minirag(tmp_path: Path):
    markdown_file = tmp_path / "test.md"
    markdown_file.write_text("# Sample Markdown Content")
    api_url = "http://example.com/api"
    with patch("aiohttp.ClientSession.post", new_callable=AsyncMock) as mock_post:
        mock_response = AsyncMock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        result = await insert_into_minirag("title", "abstract", markdown_file, api_url)

        # Check that the abstract and title text are passed to the call
        mock_post.assert_any_call(
            f"{api_url}/documents/text",
            headers={"Content-Type": "application/json"},
            json={
                "text": "# title\n\n## Abstract\nabstract",
                "description": "Title and abstract for test.md",
            },
        )

        assert result["success"] is True


@pytest.mark.asyncio
async def test_insert_into_minirag_file_not_found():
    markdown_file = Path("non_existent.md")
    api_url = "http://example.com/api"
    with pytest.raises(
        FileNotFoundError, match="The file non_existent.md does not exist."
    ):
        await insert_into_minirag("title", "abstract", markdown_file, api_url)


@pytest.mark.asyncio
async def test_insert_into_minirag_prevents_duplicate_insertion(tmp_path: Path):
    markdown_file = tmp_path / "test.md"
    markdown_file.write_text("# Sample Markdown Content")
    api_url = "http://example.com/api"

    with patch("aiohttp.ClientSession.post", new_callable=AsyncMock) as mock_post:
        mock_response = AsyncMock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        # First call to insert_into_minirag
        result1 = await insert_into_minirag("title", "abstract", markdown_file, api_url)
        assert result1["success"] is True
        initial_call_count = mock_post.call_count

        # Second call to insert_into_minirag with the same file
        result2 = await insert_into_minirag("title", "abstract", markdown_file, api_url)
        assert result2["success"] is True
        assert (
            mock_post.call_count == initial_call_count
        )  # Ensure no additional POST request was made


@pytest.mark.asyncio
async def test_process_attachments(tmp_path: Path):
    contributions = [
        ContributionData(
            title="Test Contribution 1",
            abstract="Abstract 1",
            urls=["http://example.com/test1.pdf", "http://example.com/test1.txt"],
        ),
        ContributionData(
            title="Test Contribution 2",
            abstract="Abstract 2",
            urls=["http://example.com/test2.pdf"],
        ),
    ]
    download_dir = Path(tmp_path)
    api_url = "http://example.com/api"
    with (
        patch(
            "abstract_ranker.minirag_ingester.download_attachment",
            new_callable=AsyncMock,
        ) as mock_download,
        patch(
            "abstract_ranker.minirag_ingester.run_docling", new_callable=AsyncMock
        ) as mock_docling,
        patch(
            "abstract_ranker.minirag_ingester.insert_into_minirag",
            new_callable=AsyncMock,
        ) as mock_insert,
    ):
        mock_download.side_effect = lambda url, dir: dir / Path(url).name
        mock_docling.side_effect = lambda file: file.with_suffix(".md")
        mock_insert.side_effect = lambda title, abstract, file, api: {"success": True}

        results = await process_attachments(contributions, download_dir, api_url)

        # Validate the structure and content of the results
        assert len(results) == 2
        assert results[0]["title"] == "Test Contribution 1"
        assert results[1]["title"] == "Test Contribution 2"
        assert all("results" in result for result in results)
        assert all(
            result["success"] is True
            for res in results
            for result in res["results"]  # type: ignore
        )


@pytest.mark.asyncio
async def test_process_attachments_concurrency_limits(tmp_path: Path):
    contributions = [
        ContributionData(
            title="Test Contribution 1",
            abstract="Abstract 1",
            urls=["http://example.com/test1.pdf"],
        ),
        ContributionData(
            title="Test Contribution 2",
            abstract="Abstract 2",
            urls=["http://example.com/test2.pdf"],
        ),
        ContributionData(
            title="Test Contribution 3",
            abstract="Abstract 3",
            urls=["http://example.com/test3.pdf"],
        ),
    ]
    download_dir = Path(tmp_path)
    api_url = "http://example.com/api"

    download_locks = []
    docling_locks = []
    ingest_locks = []

    async def mock_download_attachment(*args, **kwargs):
        lock = asyncio.Lock()
        download_locks.append(lock)
        async with lock:
            await asyncio.sleep(0.1)
        return download_dir / "mock.pdf"

    async def mock_run_docling(*args, **kwargs):
        lock = asyncio.Lock()
        docling_locks.append(lock)
        async with lock:
            await asyncio.sleep(0.1)
        return Path("mock.md")

    async def mock_insert_into_minirag(*args, **kwargs):
        lock = asyncio.Lock()
        ingest_locks.append(lock)
        async with lock:
            await asyncio.sleep(0.1)
        return {"success": True}

    with (
        patch(
            "abstract_ranker.minirag_ingester.download_attachment",
            new=mock_download_attachment,
        ),
        patch("abstract_ranker.minirag_ingester.run_docling", new=mock_run_docling),
        patch(
            "abstract_ranker.minirag_ingester.insert_into_minirag",
            new=mock_insert_into_minirag,
        ),
    ):
        await process_attachments(contributions, download_dir, api_url)

    # Verify concurrency limits
    assert len(download_locks) == 3
    assert len(docling_locks) == 3
    assert len(ingest_locks) == 3

    # Ensure only one download lock was active at a time
    assert all(not lock.locked() for lock in download_locks)

    # Ensure at most two docling locks were active at a time
    assert sum(lock.locked() for lock in docling_locks) <= 2

    # Ensure only one ingest lock was active at a time
    assert all(not lock.locked() for lock in ingest_locks)


@pytest.mark.asyncio
async def test_insert_into_minirag_utf8_file_reads_utf8_encoding() -> None:
    markdown_file = Path("./tests/data/CONNIE_ESPP2026.md")
    api_url = "http://example.com/api"

    with patch("aiohttp.ClientSession.post", new_callable=AsyncMock) as mock_post:
        mock_response = AsyncMock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        await insert_into_minirag("title", "abstract", markdown_file, api_url)


@pytest.mark.asyncio
async def test_process_attachments_propagates_exceptions(tmp_path: Path):
    contributions = [
        ContributionData(
            title="Test Contribution 1",
            abstract="Abstract 1",
            urls=["http://example.com/test1.pdf"],
        )
    ]
    download_dir = Path(tmp_path)
    api_url = "http://example.com/api"

    async def mock_run_docling(*args, **kwargs):
        raise RuntimeError("Test exception from run_docling")

    with (
        patch(
            "abstract_ranker.minirag_ingester.download_attachment",
            new_callable=AsyncMock,
        ) as mock_download,
        patch("abstract_ranker.minirag_ingester.run_docling", new=mock_run_docling),
        patch(
            "abstract_ranker.minirag_ingester.insert_into_minirag",
            new_callable=AsyncMock,
        ) as mock_insert,
    ):
        mock_download.side_effect = lambda url, dir: dir / Path(url).name
        mock_insert.side_effect = lambda title, abstract, file, api: {"success": True}

        with pytest.raises(RuntimeError, match="Test exception from run_docling"):
            await process_attachments(contributions, download_dir, api_url)


@pytest.mark.asyncio
async def test_process_attachments_skip_injection(tmp_path: Path):
    contributions = [
        ContributionData(
            title="Test Contribution 1",
            abstract="Abstract 1",
            urls=["http://example.com/test1.pdf"],
        )
    ]
    download_dir = Path(tmp_path)
    api_url = "http://example.com/api"

    with (
        patch(
            "abstract_ranker.minirag_ingester.download_attachment",
            new_callable=AsyncMock,
        ) as mock_download,
        patch(
            "abstract_ranker.minirag_ingester.run_docling", new_callable=AsyncMock
        ) as mock_docling,
        patch(
            "abstract_ranker.minirag_ingester.insert_into_minirag",
            new_callable=AsyncMock,
        ) as mock_insert,
    ):
        mock_download.side_effect = lambda url, dir: dir / Path(url).name
        mock_docling.side_effect = lambda file: file.with_suffix(".md")

        # Call process_attachments with skip_injection=True
        results = await process_attachments(
            contributions, download_dir, api_url, skip_injection=True
        )

        # Validate the structure and content of the results
        assert len(results) == 1
        assert results[0]["title"] == "Test Contribution 1"
        assert results[0]["results"] == []  # No injection results

        # Ensure insert_into_minirag was not called
        mock_insert.assert_not_called()
