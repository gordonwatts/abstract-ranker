import asyncio
from pathlib import Path
import aiohttp
import aiofiles
import subprocess
from typing import List, Dict, Union


async def download_attachment(attachment_url: str, download_dir: Path) -> Path:
    """Download an attachment from a URL asynchronously."""
    async with aiohttp.ClientSession() as session:
        response = await session.get(attachment_url)
        response.raise_for_status()
        filename = download_dir / Path(attachment_url).name
        async with aiofiles.open(filename, "wb") as file:
            await file.write(await response.read())
    return filename


async def run_docling(file_path: Path) -> Path:
    """Run the docling command on a file to generate a markdown file asynchronously."""
    output_file = file_path.with_suffix(file_path.suffix + ".md")
    process = await asyncio.create_subprocess_exec(
        "docling",
        str(file_path),
        "-o",
        str(output_file),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await process.communicate()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, "docling")
    return output_file


async def insert_into_minirag(
    markdown_file: Path, api_url: str
) -> Dict[str, Union[str, bool]]:
    """Insert the markdown file into the min-rag database via HTTP POST asynchronously."""
    if not markdown_file.exists():
        raise FileNotFoundError(f"The file {markdown_file} does not exist.")
    async with aiohttp.ClientSession() as session:
        async with aiofiles.open(markdown_file, "r") as file:
            content = await file.read()
            response = await session.post(api_url, data={"file": content})
            response.raise_for_status()
            return await response.json()


async def process_attachments(
    attachments: List[str], download_dir: Path, api_url: str
) -> List[Dict[str, Union[str, bool]]]:
    """Process a list of attachments asynchronously."""
    download_dir.mkdir(parents=True, exist_ok=True)

    async def process_single_attachment(
        attachment_url: str,
    ) -> Dict[str, Union[str, bool]]:
        try:
            file_path = await download_attachment(attachment_url, download_dir)
            markdown_file = await run_docling(file_path)
            result = await insert_into_minirag(markdown_file, api_url)
            return result
        except Exception as e:
            return {"error": str(e)}

    return await asyncio.gather(
        *(process_single_attachment(url) for url in attachments)
    )
