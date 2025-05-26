import asyncio
import logging
import re
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import aiofiles
import aiohttp

from abstract_ranker.utils import ContributionData
from urllib.parse import urlparse, unquote


async def download_attachment(
    attachment_url: str, download_dir: Path, title: str
) -> Optional[Path]:
    """Download an attachment from a URL asynchronously."""
    # Extract filename from URL
    parsed_url = urlparse(attachment_url)
    filename = Path(unquote(parsed_url.path)).name
    if not filename:
        logging.warning(
            f"Filename for URL '{attachment_url}' could not be determined ('{title}')."
        )
        filename = "downloaded_file"

    # Skip zip files, tar files, gzip files, etc.
    file_suffix = Path(filename).suffix
    if file_suffix.lower() in {
        ".zip",
        ".tar",
        ".gz",
        ".tar.gz",
        ".tgz",
        ".tar.bz2",
        ".tar.xz",
        ".rar",
        ".7z",
    }:
        logging.warning(
            f"Skipping download of '{filename}' from '{attachment_url}' "
            "due to unsupported file type."
        )
        return None

    # Sanitize filename
    sanitized_name = re.sub(r'[<>:"/\\|?*$]', "", title)
    final_filename = (download_dir / sanitized_name).with_suffix(file_suffix)

    # Skip download if file already exists
    checked = False
    while not checked:
        try:
            if final_filename.exists():
                return final_filename
            checked = True
        except OSError as e:
            if "File name too long" in str(e):
                stem = final_filename.stem
                parts = stem.split(" ")
                if len(parts) > 1:
                    new_stem = " ".join(parts[:-1])
                else:
                    new_stem = stem[:-1]  # Remove last character if only one word
                final_filename = final_filename.with_name(
                    new_stem + final_filename.suffix
                )
                if len(final_filename.name) == 0:
                    raise ValueError(f"Can't build short filename for {title}")

    temp_filename = download_dir / f"{hash(sanitized_name)}-download"

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
    try:
        # Run the docling command using a subprocess
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            f'uvx docling --image-export-mode placeholder "{file_path}" --output '
            f'"{output_file.parent}"',
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
            raise subprocess.CalledProcessError(
                process.returncode, "docling {file_path}"
            )
        if not output_file.exists():
            raise RuntimeError(
                f"The output file '{output_file}' was not created by the docling command."
            )
        logging.info(f"Successfully generated markdown file: {output_file}")
    finally:
        # Clean up the temporary PowerShell script
        # Path(temp_ps1.name).unlink(missing_ok=True)
        pass

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
        logging.info(f"Skipping insertion for '{title}', already processed.")
        return cache[markdown_file_name]
    file_size_kb = markdown_file.stat().st_size / 1024
    logging.info(
        f"Starting insertion of '{title}' into minirag. File size: {file_size_kb:.2f} KB."
    )

    # Insert the title and abstract
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None)
    ) as session:
        title_abstract_text = f"# {title}\n\n## Abstract\n{abstract}"
        logging.debug(f"Inserting title and abstract for {markdown_file_name}.")
        response = await session.post(
            f"{api_url}/documents/text",
            headers={"Content-Type": "application/json"},
            json={
                "text": title_abstract_text,
                "description": f"Title and abstract for {markdown_file_name}",
            },
        )
        result = await response.json()
        logging.debug(
            f"Response from inserting title and abstract: {response.status} - {result}"
        )
        response.raise_for_status()

        # Next, insert the markdown file itself
        logging.debug(f"Inserting content for {markdown_file_name}.")
        async with aiofiles.open(markdown_file, "r", encoding="utf-8") as file:
            content = await file.read()
            form_data = aiohttp.FormData()
            form_data.add_field("file", content, filename=markdown_file_name)
            form_data.add_field(
                "description", f"Markdown content for {markdown_file_name}"
            )

            response = await session.post(f"{api_url}/documents/file", data=form_data)
            result = await response.json()
            logging.debug(
                f"Response from inserting markdown file: {response.status} - {result}"
            )
            response.raise_for_status()

    # Update the cache
    cache[markdown_file_name] = result
    with cache_file.open("w") as f:
        json.dump(cache, f)

    logging.info(f"Successfully inserted '{title}' into minirag.")

    return result


async def process_attachments(
    contributions: List[ContributionData],
    download_dir: Path,
    api_url: str,
    skip_injection: bool = False,  # New parameter to skip injection into minirag
) -> List[Dict[str, Union[str, List[str], bool]]]:
    """Process a list of contributions asynchronously with concurrency limits."""
    download_dir.mkdir(parents=True, exist_ok=True)

    download_semaphore = asyncio.Semaphore(1)  # Limit to 1 download at a time
    docling_semaphore = asyncio.Semaphore(2)  # Limit to 2 docling operations at a time
    ingest_semaphore = asyncio.Semaphore(1)  # Limit to 1 ingest operation at a time

    async def process_single_contribution(
        contribution: ContributionData,
    ) -> Dict[str, Union[str, List[str], bool]]:
        results = []
        for i, attachment_url in enumerate(contribution.urls):
            title = contribution.title
            if i > 0:
                title = f"{title} - document {i}"
            async with download_semaphore:
                file_path = await download_attachment(
                    attachment_url, download_dir, title
                )

            if file_path is not None:
                async with docling_semaphore:
                    markdown_file = await run_docling(file_path)

                if not skip_injection:  # Skip injection if the flag is set
                    async with ingest_semaphore:
                        result = await insert_into_minirag(
                            contribution.title,
                            contribution.abstract,
                            markdown_file,
                            api_url,
                        )

                    results.append(result)

        return {"title": contribution.title, "results": results}

    return await asyncio.gather(
        *(process_single_contribution(contribution) for contribution in contributions)
    )
