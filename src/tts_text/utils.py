"""Utility functions used by the other modules."""

from typing import Generator
import numpy as np
import random


def interleave_datasets(
    datasets: list[list[str]], sampling_probabilities: list[float]
) -> Generator[str, None, None]:
    """Interleave multiple datasets according to the given sampling probabilities.

    Args:
        datasets: The datasets to interleave.
        sampling_probabilities: The sampling probabilities for each dataset.

    Yields:
        The interleaved dataset.
    """
    while True:
        # Sample a dataset
        dataset_idx = np.random.choice(len(datasets), p=sampling_probabilities)
        dataset = datasets[dataset_idx]

        # If the dataset is empty then stop
        if len(dataset) == 0:
            return

        # Sample a sample from the dataset
        sample_idx = random.randrange(len(dataset))
        sample = dataset[sample_idx]

        # Remove the sample from the dataset
        dataset.pop(sample_idx)

        yield sample
