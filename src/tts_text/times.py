"""Building a list of Danish times."""

from pathlib import Path
import random

from omegaconf import DictConfig


MINUTES = list(range(0, 60))
HOURS = list(range(0, 24))
HOUR_PREFIXES = [
    "Fem i",
    "Fem over",
    "Ti i",
    "Ti over",
    "Kvart i",
    "Kvart over",
    "Tyve i",
    "Tyve over",
    "Halv",
]
HOUR_STRINGS = [
    "et",
    "to",
    "tre",
    "fire",
    "fem",
    "seks",
    "syv",
    "otte",
    "ni",
    "ti",
    "elleve",
    "tolv",
]


def build_time_dataset(cfg: DictConfig) -> list[str]:
    """Build the time dataset.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        A list of strings representing times in Danish.
    """
    # Load dataset if it already exists
    dataset_path = Path(cfg.dirs.data) / cfg.dirs.raw / "times.txt"
    if dataset_path.exists():
        with dataset_path.open("r", encoding="utf-8") as f:
            return f.read().split("\n")

    random.seed(cfg.random_seed)

    # Build the dataset
    dataset: list[str] = list()
    for hour in HOURS:
        minute = random.choice(MINUTES)
        dataset.append(f"{hour:02}:{minute:02}")
    for minute in MINUTES:
        hour = random.choice(HOURS)
        dataset.append(f"{hour:02}:{minute:02}")
    for hour_prefix in HOUR_PREFIXES:
        hour_string = random.choice(HOUR_STRINGS)
        dataset.append(f"{hour_prefix} {hour_string}")
    dataset = list(set(dataset))
    random.shuffle(dataset)

    # Save the dataset
    with dataset_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(dataset))

    return dataset
