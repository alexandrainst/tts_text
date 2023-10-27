"""Module for creating a phoneme covering set."""

from dataclasses import dataclass
import json
import re
import logging
from functools import partial
from collections import Counter
from pathlib import Path
import sys
from tqdm.auto import tqdm
import multiprocessing as mp

from datasets import load_dataset, Dataset
from omegaconf import DictConfig

from .utils import extract_sentences


logger = logging.getLogger(__name__)


SPLIT_STRINGS = [
    "\t",
    "\n",
    "_NEWLINE_",
    "_START_ARTICLE_",
    "_START_SECTION_",
    "_START_PARAGRAPH_",
]


@dataclass
class PhonemeInfo:
    name: str
    examples: list[str]


PHONEME_LIST = list[PhonemeInfo]


def build_phoneme_covering_dataset(cfg: DictConfig) -> list[str]:
    """Create a phoneme covering set from the sorted wiki dataset.

    A phoneme covering set is a set of strings that contains at least one example
    word for each phoneme.

    Args:
        cfg:
            The Hydra configuration object.

    Returns:
        The phoneme covering set.
    """
    sorted_dataset = load_and_sort_wikipedia_dataset(cfg=cfg)

    # Get set of all unique phonemes
    phonemes = load_phonemes(cfg=cfg)
    all_phone_names = {phoneme.name for phoneme in phonemes}
    all_phonemes = {name: cfg.min_docs_per_phoneme for name in all_phone_names}

    # Create phoneme covering set
    dataset: list[str] = list()
    for document in tqdm(sorted_dataset, desc="Building phoneme covering set"):
        # If we have exhausted all the phonemes then stop
        if not all_phonemes:
            break

        text = document["text"]
        document_phonemes = document["phonemes"]

        # For each new phoneme found in the document, decrement the count of that
        # phoneme in the set of all phonemes. If the count reaches zero then remove
        # the phoneme from the set of all phonemes
        new_phonemes = set(all_phonemes.keys()).intersection(document_phonemes)
        for new_phoneme in new_phonemes:
            all_phonemes[new_phoneme] -= 1
            if all_phonemes[new_phoneme] == 0:
                all_phonemes.pop(new_phoneme)

        # If new phonemes were found then include the document in the dataset
        if new_phonemes:
            dataset.append(text)

    # If any phonemes were not exhausted then log a warning
    if all_phonemes:
        logger.warning(
            "Remaining phoneme counts which were still left after traversing the "
            f"corpus: {all_phonemes}"
        )

    # Save the dataset
    dataset_path = Path(cfg.dirs.data) / cfg.dirs.raw / "phoneme_covering_set.txt"
    with dataset_path.open("w") as f:
        f.write("\n".join(dataset))

    return dataset


def load_and_sort_wikipedia_dataset(cfg: DictConfig) -> Dataset:
    """Load and sort the Wikipedia dataset by phoneme occurence.

    Args:
        cfg:
            The Hydra configuration object.

    Returns:
        The sorted dataset.
    """
    # The `wiki40b` dataset is a small dataset so we can load it all into memory
    # instead of streaming it.
    dataset = load_dataset("alexandrainst/wiki40b-da", split="train")
    assert isinstance(dataset, Dataset)

    # Truncate the dataset if we're testing
    if hasattr(sys, "_called_from_test"):
        dataset = dataset.select(range(100))

    def remove_split_strings(example: dict) -> dict:
        """Removes the special Wikipedia split strings from the text."""
        doc = example["text"]
        doc = "\n".join(re.split("|".join(SPLIT_STRINGS), doc))
        example["text"] = doc
        return example

    # Remove split strings from dataset
    dataset = dataset.map(
        function=remove_split_strings,
        desc="Removing split strings from the Wikipedia dataset",
    )

    # Split dataset into sentences
    sentences = extract_sentences(
        corpus=dataset["text"], min_sentence_length=cfg.min_sentence_length
    )

    # Count phonemes in articles
    dataset = Dataset.from_dict(dict(text=sentences))
    dataset = dataset.map(
        function=partial(count_phoneme_occurences, phonemes=load_phonemes(cfg=cfg)),
        desc="Counting phonemes in the Wikipedia dataset",
        num_proc=mp.cpu_count(),
    )
    dataset = dataset.sort("unique_phonemes_count", reverse=True)

    return dataset


def count_phoneme_occurences(document: dict, phonemes: PHONEME_LIST) -> dict:
    """Count the occurences of phonemes in a document.

    Args:
        document:
            The document to count phonemes in.
        phonemes:
            The phonemes to count.

    Returns:
        The document with phoneme lists and counts added.
    """
    # Extract the words from the document
    document_text = document["text"]
    document_text = re.sub(" +", " ", document_text).strip().lower()
    words = document_text.split(" ")

    counter = Counter(words)
    phoneme_count = 0
    found_phonemes = []
    found_example_words = []
    for phoneme in phonemes:
        for example_word in phoneme.examples:
            occurences = counter[example_word]
            phoneme_count += occurences
            if occurences > 0:
                found_phonemes.append(phoneme.name)
                found_example_words.append(example_word)
    document["phoneme_count"] = phoneme_count
    document["phonemes"] = found_phonemes
    document["found_example_words"] = found_example_words
    document["unique_phonemes"] = set(found_phonemes)
    document["unique_phonemes_count"] = len(set(found_phonemes))
    return document


def load_phonemes(cfg: DictConfig) -> PHONEME_LIST:
    """Load the phoneme list.

    Args:
        cfg:
            The Hydra configuration object.

    Returns:
        The phoneme list.
    """
    phoneme_path = Path(cfg.dirs.data) / cfg.dirs.raw / cfg.dirs.phoneme_file
    with phoneme_path.open() as f:
        return [PhonemeInfo(**entry) for entry in json.load(f)]
