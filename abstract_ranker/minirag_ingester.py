import asyncio
import logging
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import json
from pathlib import Path
from typing import Dict, List, Union

import aiofiles
import aiohttp

from abstract_ranker.utils import ContributionData


async def download_attachment(attachment_url: str, download_dir: Path) -> Path:
    """Download an attachment from a URL asynchronously."""
    # Decode URL and sanitize filename
    decoded_url = urllib.parse.unquote(attachment_url)
    sanitized_name = re.sub(r'[<>:"/\\|?*]', "", Path(decoded_url).name)
    final_filename = download_dir / sanitized_name

    # Skip download if file already exists
    if final_filename.exists():
        return final_filename

    temp_filename = download_dir / f"{sanitized_name}-download"

    async with aiohttp.ClientSession() as session:
        response = await session.get(attachment_url)
        response.raise_for_status()
        async with aiofiles.open(temp_filename, "wb") as file:
            await file.write(await response.read())
        temp_filename.rename(final_filename)

    return final_filename


async def run_docling(file_path: Path) -> Path:
    """Run the docling command on a file to generate a markdown file asynchronously."""
    # Can we skip?
    output_file = file_path.with_suffix(".md")
    if output_file.exists():
        logging.debug(f"{output_file} already exists, skipping docling execution.")
        return output_file
    logging.debug(f"Preparing to run docling on {file_path}")

    # Create a temporary PowerShell script
    temp_ps1 = tempfile.NamedTemporaryFile(delete=False, suffix=".ps1")
    try:
        ps1_content = f"""
        deactivate
        {shutil.which("powershell")} -Command "& {{
            C:\\Users\\gordo\\Code\\llm\\docling-experiments\\.venv\\Scripts\\activate.ps1
            docling '{file_path}' --output '{output_file.parent}'
        }}"
        """
        temp_ps1.write(ps1_content.encode())
        temp_ps1.close()

        # Run the PowerShell script
        process = await asyncio.create_subprocess_exec(
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            temp_ps1.name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        assert process.returncode is not None
        if process.returncode != 0:
            logging.error(
                f"Docling failed with return code {process.returncode}.\nSTDOUT: {stdout.decode()}"
                f"\nSTDERR: {stderr.decode()}"
            )
            raise subprocess.CalledProcessError(process.returncode, "powershell")
        if not output_file.exists():
            raise RuntimeError(
                f"The output file {output_file} was not created by the docling command."
            )
        logging.info(f"Successfully generated markdown file: {output_file}")
    finally:
        # Clean up the temporary PowerShell script
        Path(temp_ps1.name).unlink(missing_ok=True)

    return output_file


async def insert_into_minirag(
    title: str, abstract: str, markdown_file: Path, api_url: str
) -> Dict[str, Union[str, bool]]:
    """Insert the markdown file into the min-rag database via HTTP POST asynchronously."""
    if not markdown_file.exists():
        raise FileNotFoundError(f"The file {markdown_file} does not exist.")

    # Define the cache file path near the markdown file directory
    cache_file = markdown_file.parent / "minirag_cache.json"

    # Load the cache if it exists
    if cache_file.exists():
        with cache_file.open("r") as f:
            cache = json.load(f)
    else:
        cache = {}

    # Check if the file has already been processed
    markdown_file_name = markdown_file.name
    if markdown_file_name in cache:
        logging.info(f"Skipping insertion for {markdown_file_name}, already processed.")
        return cache[markdown_file_name]

    # Insert the title and abstract
    async with aiohttp.ClientSession() as session:
        title_abstract_text = f"# {title}\n\n## Abstract\n{abstract}"
        response = await session.post(
            f"{api_url}/documents/text",
            headers={"Content-Type": "application/json"},
            json={
                "text": title_abstract_text,
                "description": f"Title and abstract for {markdown_file_name}",
            },
        )
        result = await response.json()
        logging.debug(f"Response from inserting title and abstract: {result}")
        response.raise_for_status()

        # Next, insert the markdown file itself
        async with aiofiles.open(markdown_file, "r") as file:
            content = await file.read()
            form_data = aiohttp.FormData()
            form_data.add_field("file", content, filename=markdown_file_name)
            form_data.add_field(
                "description", f"Markdown content for {markdown_file_name}"
            )

            response = await session.post(f"{api_url}/documents/file", data=form_data)
            result = await response.json()
            logging.debug(f"Response from inserting markdown file: {result}")
            response.raise_for_status()

    # Update the cache
    cache[markdown_file_name] = result
    with cache_file.open("w") as f:
        json.dump(cache, f)

    return result


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
                    result = await insert_into_minirag(
                        contribution.title,
                        contribution.abstract,
                        markdown_file,
                        api_url,
                    )

                results.append(result)

            return {"title": contribution.title, "results": results}
        except Exception as e:
            return {"error": str(e)}

    return await asyncio.gather(
        *(process_single_contribution(contribution) for contribution in contributions)
    )
