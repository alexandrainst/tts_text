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
    # Load dataset if it already exists
    dataset_path = Path(cfg.dirs.data) / cfg.dirs.raw / "dates.txt"
    if dataset_path.exists():
        with dataset_path.open("r", encoding="utf-8") as f:
            return f.read().split("\n")

    # Build the dataset
    dataset: list[str] = list()
    for year in YEARS:
        day = random.choice(DAYS)
        month = random.choice(MONTHS)
        dataset.append(f"{day} {month} {year}")
    for month in MONTHS:
        day = random.choice(DAYS)
        year = random.choice(YEARS)
        dataset.append(f"{day} {month} {year}")
    for day in DAYS:
        month = random.choice(MONTHS)
        year = random.choice(YEARS)
        dataset.append(f"{day} {month} {year}")
    dataset = list(set(dataset))
    random.shuffle(dataset)

    # Save the dataset
    with dataset_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(dataset))

    return dataset
