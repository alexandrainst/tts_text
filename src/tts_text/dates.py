"""Building a list of Danish dates."""

from pathlib import Path
import random

from omegaconf import DictConfig


DAYS = [f"{numeral}." for numeral in range(1, 32)]
MONTHS = [
    "januar",
    "februar",
    "marts",
    "april",
    "maj",
    "juni",
    "juli",
    "august",
    "september",
    "oktober",
    "november",
    "december",
]
YEARS = [year for year in range(1970, 2030)]


def build_date_dataset(cfg: DictConfig) -> list[str]:
    """Build the date dataset.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        A list of strings representing dates in Danish.
    """
    # Build the dataset
    dataset: list[str] = list()
    dataset.extend(
        [f"{day} {month} {year}" for day in DAYS for month in MONTHS for year in YEARS]
    )
    dataset.extend([f"{day} {month}" for day in DAYS for month in MONTHS])
    random.shuffle(dataset)

    # Save the dataset
    dataset_path = Path(cfg.dirs.data) / cfg.dirs.raw / "dates.txt"
    with dataset_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(dataset))

    return dataset
