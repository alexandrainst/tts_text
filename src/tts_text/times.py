"""Building a list of Danish times."""

import random


MINUTES = list(range(0, 60))
HOURS = list(range(0, 24))
HOUR_STRINGS = [
    "Et",
    "To",
    "Tre",
    "Fire",
    "Fem",
    "Seks",
    "Syv",
    "Otte",
    "Ni",
    "Ti",
    "Elleve",
    "Tolv",
]
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


def build_time_dataset() -> list[str]:
    """Build the time dataset.

    Returns:
        A list of strings representing times in Danish.
    """
    dataset: list[str] = list()
    dataset.extend([f"{hour}:{minute}" for hour in HOURS for minute in MINUTES])
    dataset.extend(
        [
            f"{hour_prefix} {hour_str}"
            for hour_str in HOUR_STRINGS
            for hour_prefix in HOUR_PREFIXES
        ]
    )
    random.shuffle(dataset)
    return dataset
