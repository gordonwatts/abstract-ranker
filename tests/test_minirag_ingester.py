import pytest
from unittest.mock import AsyncMock, patch
from pathlib import Path
from abstract_ranker.minirag_ingester import (
    download_attachment,
    run_docling,
    insert_into_minirag,
    process_attachments,
)


@pytest.mark.asyncio
async def test_download_attachment(tmp_path: Path):
    attachment_url = "http://example.com/test.pdf"
    download_dir = Path(tmp_path)
    with patch("aiohttp.ClientSession.get", new_callable=AsyncMock) as mock_get:
        mock_response = AsyncMock()
        mock_response.read.return_value = b"PDF content"
        mock_get.return_value = mock_response

        file_path = await download_attachment(attachment_url, download_dir)
        assert file_path.exists()
        assert file_path.name == "test.pdf"


@pytest.mark.asyncio
async def test_run_docling():
    file_path = Path("test.pdf")
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value.communicate.return_value = (b"", b"")
        mock_exec.return_value.returncode = 0
        output_file = await run_docling(file_path)
        assert output_file == file_path.with_suffix(file_path.suffix + ".md")


@pytest.mark.asyncio
async def test_insert_into_minirag(tmp_path: Path):
    markdown_file = tmp_path / "test.md"
    markdown_file.write_text("# Sample Markdown Content")
    api_url = "http://example.com/api"
    with patch("aiohttp.ClientSession.post", new_callable=AsyncMock) as mock_post:
        mock_response = AsyncMock()
        mock_response.json.return_value = {"success": True}
        mock_post.return_value = mock_response

        result = await insert_into_minirag(markdown_file, api_url)
        assert result["success"] is True


@pytest.mark.asyncio
async def test_insert_into_minirag_file_not_found():
    markdown_file = Path("non_existent.md")
    api_url = "http://example.com/api"
    with pytest.raises(
        FileNotFoundError, match="The file non_existent.md does not exist."
    ):
        await insert_into_minirag(markdown_file, api_url)


@pytest.mark.asyncio
async def test_process_attachments(tmp_path: Path):
    attachments = ["http://example.com/test1.pdf", "http://example.com/test2.pdf"]
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
        mock_docling.side_effect = lambda file: file.with_suffix(file.suffix + ".md")
        mock_insert.side_effect = lambda file, api: {"success": True}
        results = await process_attachments(attachments, download_dir, api_url)
        assert all(result["success"] for result in results)
