import csv
import logging
from pathlib import Path
from typing import Generator, Tuple

from abstract_ranker.data_model import AbstractLLMResponse, Contribution
from abstract_ranker.utils import as_a_number


def dump_to_csv_file(
    output_filename: Path,
    data: Generator[Tuple[Contribution, AbstractLLMResponse], None, None],
    progress_bar: bool,
):
    # Open the CSV file in write mode
    with output_filename.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(
            [
                "Date",
                "Time",
                "Room",
                "Title",
                "Summary",
                "Experiment",
                "Keywords",
                "Interest",
                "Type",
                "Confidence",
                "Unknown Terms",
            ]
        )

        for contrib, summary in data:
            # Write the row to the CSV file
            writer.writerow(
                [
                    (
                        contrib.startDate.strftime("%Y-%m-%d %H:%M:%S")
                        if contrib.startDate
                        else ""
                    ),
                    (
                        contrib.startDate.strftime("%Y-%m-%d %H:%M:%S")
                        if contrib.startDate
                        else ""
                    ),
                    contrib.roomFullname if contrib.roomFullname else "",
                    contrib.title,
                    summary.summary,
                    summary.experiment,
                    summary.keywords,
                    as_a_number(summary.interest),
                    contrib.type,
                    summary.confidence,
                    summary.unknown_terms,
                ]
            )
    # Print a message indicating the CSV file has been created
    logging.info(f"CSV file '{output_filename}' has been created.")
