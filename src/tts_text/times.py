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
    # Build the dataset
    dataset: list[str] = list()
    dataset.extend([f"{hour:02}:{minute:02}" for hour in HOURS for minute in MINUTES])
    dataset.extend(
        [
            f"{hour_prefix} {hour_str}"
            for hour_str in HOUR_STRINGS
            for hour_prefix in HOUR_PREFIXES
        ]
    )
    random.shuffle(dataset)

    # Save the dataset
    dataset_path = Path(cfg.dirs.data) / cfg.dirs.raw / "times.txt"
    with dataset_path.open("w") as f:
        f.write("\n".join(dataset))

    return dataset
