"""Build the Danish text-to-speech dataset."""

from tts_text import ALL_DATASET_BUILDERS
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

    Raises:
        ValueError: If the sampling probabilities do not include all the datasets.
    """
    # Get the individual datasets
    raw_dir = Path(config.dirs.data) / config.dirs.raw
    datasets = {
        name: builder(output_dir=raw_dir)
        for name, builder in ALL_DATASET_BUILDERS.items()
    }

    # Ensure that the sampling probabilities include all the datasets
    if set(config.sampling_probabilities.keys()) != set(datasets.keys()):
        raise ValueError(
            "The sampling probabilities must include all the datasets. Was missing "
            f"{set(datasets.keys()) - set(config.sampling_probabilities.keys())}."
        )

    # Combine the datasets
    dataset_itr = interleave_datasets(
        datasets=list(datasets.values()),
        sampling_probabilities=[
            config.sampling_probabilities[name] for name in datasets.keys()
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
