"""Build the Danish text-to-speech dataset."""

from tts_text.dates import build_date_dataset
from tts_text.times import build_time_dataset
from tts_text.bus_stops_and_stations import build_bus_stop_and_station_dataset
import hydra
from omegaconf import DictConfig
import random
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Build the Danish text-to-speech dataset.

    Args:
        config: The Hydra configuration.
    """
    # Get the individual datasets
    raw_dir = Path(config.dirs.data) / config.dirs.raw
    date_dataset = build_date_dataset(output_dir=raw_dir)
    time_dataset = build_time_dataset(output_dir=raw_dir)
    bus_stops_and_stations = build_bus_stop_and_station_dataset(output_dir=raw_dir)

    # Combine the datasets
    dataset = date_dataset + time_dataset + bus_stops_and_stations

    # Shuffle the dataset
    random.shuffle(dataset)

    # Save the dataset
    dataset_path = Path(config.dirs.data) / config.dirs.processed
    with dataset_path.open("w") as f:
        f.write("\n".join(dataset))
    logger.info(f"Saved dataset to {dataset_path}.")
    breakpoint()


if __name__ == "__main__":
    main()
