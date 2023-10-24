"""Build the Danish text-to-speech dataset."""

from tts_text.dates import build_date_dataset
from tts_text.times import build_time_dataset
from tts_text.bus_stops_and_stations import build_bus_stop_and_station_dataset
from tts_text.utils import interleave_datasets
import hydra
from omegaconf import DictConfig
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
    sampling_probabilities = config.sampling_probabilities
    dataset_itr = interleave_datasets(
        datasets=[date_dataset, time_dataset, bus_stops_and_stations],
        sampling_probabilities=[
            sampling_probabilities.dates,
            sampling_probabilities.times,
            sampling_probabilities.bus_stops_and_stations,
        ],
    )

    # Save the dataset
    dataset_path = Path(config.dirs.data) / config.dirs.processed / "dataset.txt"
    dataset_path.unlink()
    with dataset_path.open("a") as f:
        for sample in dataset_itr:
            f.write(sample + "\n")
    logger.info(f"Saved dataset to {dataset_path}.")


if __name__ == "__main__":
    main()
