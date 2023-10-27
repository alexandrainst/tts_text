"""Build the Danish text-to-speech dataset."""

from tts_text import ALL_DATASET_BUILDERS
from tts_text.utils import interleave_datasets
import hydra
from collections import defaultdict
from omegaconf import DictConfig
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Build the Danish text-to-speech dataset.

    Args:
        config: The Hydra configuration.

    Raises:
        ValueError: If the sampling probabilities do not include all the datasets.
    """
    # Get the individual datasets
    raw_dir = Path(config.dirs.data) / config.dirs.raw
    datasets: dict[str, list[str]] = defaultdict()
    for name, builder in ALL_DATASET_BUILDERS.items():
        try:
            datasets[name] = builder(output_dir=raw_dir)
        except TypeError:
            datasets[name] = builder(cfg=config, output_dir=raw_dir)

    # Ensure that the sampling probabilities include all the datasets
    datasets_in_config = set(config.sampling_probabilities.keys()).union(
        config.include_entire_dataset
    )
    if datasets_in_config != set(datasets.keys()):
        raise ValueError(
            "All datasets must appear either in the sampling probabilities or in the "
            "`include_entire_dataset` list. The following datasets are missing from "
            f"the sampling probabilities: {set(datasets.keys()) - datasets_in_config}"
        )

    non_sampling_datasets = {
        name: datasets[name] for name in config.include_entire_dataset
    }
    sampling_datasets = {
        name: dataset
        for name, dataset in datasets.items()
        if name not in non_sampling_datasets
    }

    # Combine the datasets
    dataset_itr = interleave_datasets(
        non_sampling_datasets=list(non_sampling_datasets.values()),
        sampling_datasets=list(sampling_datasets.values()),
        sampling_probabilities=[
            config.sampling_probabilities[name] for name in sampling_datasets.keys()
        ],
    )

    # Save the dataset
    dataset_path = Path(config.dirs.data) / config.dirs.processed / "dataset.txt"
    dataset_path.unlink(missing_ok=True)
    with dataset_path.open("a") as f:
        for sample in dataset_itr:
            f.write(sample + "\n")
    logger.info(f"Saved dataset to {dataset_path}.")


if __name__ == "__main__":
    main()
