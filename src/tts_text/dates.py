"""Building a list of Danish dates."""

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


def build_date_dataset() -> list[str]:
    """Build the date dataset.

    Returns:
        A list of strings representing dates in Danish.
    """
    dataset: list[str] = list()
    dataset.extend(
        [f"{day} {month}, {year}" for day in DAYS for month in MONTHS for year in YEARS]
    )
    dataset.extend([f"{day} {month}" for day in DAYS for month in MONTHS])
    random.shuffle(dataset)
    return dataset
