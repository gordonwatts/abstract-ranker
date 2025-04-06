import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Union

import aiofiles
import aiohttp
from abstract_ranker.utils import ContributionData


async def download_attachment(attachment_url: str, download_dir: Path) -> Path:
    """Download an attachment from a URL asynchronously."""
    async with aiohttp.ClientSession() as session:
        response = await session.get(attachment_url)
        response.raise_for_status()
        temp_filename = download_dir / f"{Path(attachment_url).name}-download"
        final_filename = download_dir / Path(attachment_url).name
        async with aiofiles.open(temp_filename, "wb") as file:
            await file.write(await response.read())
        temp_filename.rename(final_filename)
    return final_filename


async def run_docling(file_path: Path) -> Path:
    """Run the docling command on a file to generate a markdown file asynchronously."""
    output_file = file_path.with_suffix(file_path.suffix + ".md")
    # process = await asyncio.create_subprocess_exec(
    #     "docling",
    #     str(file_path),
    #     "-o",
    #     str(output_file),
    #     stdout=asyncio.subprocess.PIPE,
    #     stderr=asyncio.subprocess.PIPE,
    # )
    # await process.communicate()
    # assert process.returncode is not None
    # if process.returncode != 0:
    #     raise subprocess.CalledProcessError(process.returncode, "docling")
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
    return {"success": True}


async def process_attachments(
    contributions: List[ContributionData], download_dir: Path, api_url: str
) -> List[Dict[str, Union[str, List[str], bool]]]:
    """Process a list of contributions asynchronously with concurrency limits."""
    download_dir.mkdir(parents=True, exist_ok=True)

    download_semaphore = asyncio.Semaphore(1)  # Limit to 1 download at a time
    docling_semaphore = asyncio.Semaphore(2)  # Limit to 2 docling operations at a time
    ingest_semaphore = asyncio.Semaphore(1)  # Limit to 1 ingest operation at a time

    async def process_single_contribution(
        contribution: ContributionData,
    ) -> Dict[str, Union[str, List[str], bool]]:
        try:
            results = []
            for attachment_url in contribution.urls:
                async with download_semaphore:
                    file_path = await download_attachment(attachment_url, download_dir)

                async with docling_semaphore:
                    markdown_file = await run_docling(file_path)

                async with ingest_semaphore:
                    result = await insert_into_minirag(markdown_file, api_url)

                results.append(result)

            return {"title": contribution.title, "results": results}
        except Exception as e:
            return {"error": str(e)}

    return await asyncio.gather(
        *(process_single_contribution(contribution) for contribution in contributions)
    )
