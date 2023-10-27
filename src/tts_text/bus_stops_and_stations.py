"""Building a list of Danish bus stops and stations."""

from pathlib import Path
from omegaconf import DictConfig
import pandas as pd
import re


def build_bus_stop_and_station_dataset(cfg: DictConfig) -> list[str]:
    """Builds a list of all bus stops and stations in Denmark.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        A list of all bus stops and stations in Denmark.
    """
    # Extract the table with all bus stops and stations from the website
    table = pd.read_html("https://danskejernbaner.dk/vis.stations.oversigt.php")[0]

    # Extract the list of bus stops and stations from the table, and clean them up
    dataset = table["Stationens navn"].tolist()
    dataset = list(
        {
            re.sub(r"\(.+\)|\[.+\]", "", bus_stop_or_station).strip()
            for bus_stop_or_station in dataset
        }
    )

    # Save the dataset
    dataset_path = Path(cfg.dirs.data) / cfg.dirs.raw / "bus_stops_and_stations.txt"
    with dataset_path.open("w") as f:
        f.write("\n".join(dataset))

    return dataset
