"""Utility functions used by the other modules."""

from copy import deepcopy
from typing import Generator
import random
import itertools as it
from tqdm.auto import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import re


# Download the sentence splitter model
nltk.download("punkt", quiet=True)


def extract_sentences(corpus: list[str], min_sentence_length: int) -> list[str]:
    """Extract sentences from a corpus of text.

    Args:
        corpus: The corpus to extract sentences from.
        min_sentence_length: The minimum length of a sentence.

    Returns:
        The sentences in the corpus.
    """
    # Firstly split by newline, where we assume that a sentence does not span multiple
    # lines
    corpus = list(it.chain(*[example.split("\n") for example in corpus]))

    # Split dataset into sentences
    sentences = list(
        it.chain(
            *[
                sent_tokenize(text=example, language="danish")
                for example in tqdm(iterable=corpus, desc="Splitting sentences")
            ]
        )
    )

    # Remove newlines
    sentences = [sentence.replace("\n", " ") for sentence in sentences]

    # Remove sentences beginning or ending in "..."
    sentences = [
        sentence
        for sentence in sentences
        if not sentence.startswith("...") and not sentence.endswith("...")
    ]

    # Remove sentences ending in an abbreviation
    sentences = [
        sentence
        for sentence in sentences
        if re.search(r"\.[A-ZÆØÅa-zæøå]+\.$", sentence) is None
    ]

    # Remove redundant whitespace
    sentences = [re.sub(" +", " ", sentence) for sentence in sentences]

    # Remove trailing whitespace, tabs and newlines
    sentences = [sentence.strip(" \t\n") for sentence in sentences]

    # Remove too short sentences
    sentences = [
        sentence for sentence in sentences if len(sentence) > min_sentence_length
    ]

    return sentences


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

    # Make a copy of the sampling datasets to avoid mutating the original
    sampling_datasets = deepcopy(sampling_datasets)

    while len(sampling_datasets) > 0:
        # Sample a dataset
        dataset = random.choices(
            population=sampling_datasets,
            weights=sampling_probabilities,
            k=1,
        )[0]

        # If the dataset is empty then stop
        if len(dataset) == 0:
            break

        # Sample a sample from the dataset
        sample_idx = random.randrange(len(dataset))
        sample = dataset[sample_idx]

        # Remove the sample from the dataset
        dataset.pop(sample_idx)

        yield sample
