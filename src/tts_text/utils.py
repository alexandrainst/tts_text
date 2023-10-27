"""Utility functions used by the other modules."""

from typing import Generator
import numpy as np
import random
import itertools as it


def interleave_datasets(
    non_sampling_datasets: list[list[str]],
    sampling_datasets: list[list[str]],
    sampling_probabilities: list[float],
) -> Generator[str, None, None]:
    """Interleave multiple datasets according to the given sampling probabilities.

    Args:
        non_sampling_datasets:
            The datasets that should not be sampled. These will be shuffled together
            and included in the beginning of the interleaved dataset.
        sampling_datasets:
            The datasets that should be sampled. These will be sampled according to
            the given sampling probabilities, after the non-sampling datasets have
            been included.
        sampling_probabilities: The sampling probabilities for each dataset.

    Yields:
        The interleaved dataset.
    """
    # Start by including all datasets that shouldn't be sampled
    joined_non_sampling_datasets = list(it.chain(*non_sampling_datasets))
    random.shuffle(joined_non_sampling_datasets)
    yield from joined_non_sampling_datasets

    while True:
        # Sample a dataset
        dataset = np.random.choice(sampling_datasets, p=sampling_probabilities)

        # If the dataset is empty then stop
        if len(dataset) == 0:
            return

        # Sample a sample from the dataset
        sample_idx = random.randrange(len(dataset))
        sample = dataset[sample_idx]

        # Remove the sample from the dataset
        dataset.pop(sample_idx)

        yield sample
