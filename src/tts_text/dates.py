"""Building a list of Danish dates."""

from pathlib import Path
import random


DAYS = [f"{numeral}." for numeral in range(1, 32)]
MONTHS = [
    "Januar",
    "Februar",
    "Marts",
    "April",
    "Maj",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "December",
]
YEARS = [year for year in range(2000, 2030)]


def build_date_dataset(output_dir: Path | str) -> list[str]:
    """Build the date dataset.

    Args:
        output_dir: The directory to save the dataset to.

    Returns:
        A list of strings representing dates in Danish.
    """
    # Build the dataset
    dataset: list[str] = list()
    dataset.extend(
        [f"{day} {month}, {year}" for day in DAYS for month in MONTHS for year in YEARS]
    )
    dataset.extend([f"{day} {month}" for day in DAYS for month in MONTHS])
    random.shuffle(dataset)

    # Save the dataset
    dataset_path = Path(output_dir) / "dates.txt"
    with dataset_path.open("w") as f:
        f.write("\n".join(dataset))

    return dataset
