"""Module for creating a phoneme covering set from the sorted wiki dataset."""
import json
from pathlib import Path

from datasets import Dataset, load_from_disk
from .sort_wiki_by_phoneme_occurence import get_example_words
from omegaconf import DictConfig

import logging

logger = logging.getLogger(__name__)


def make_phoneme_covering_set(cfg: DictConfig) -> None:
    """Create a phoneme covering set from the sorted wiki dataset.

    A phoneme covering set is a set of documents that contains at least one example
    word for each phoneme.

    Args:
        cfg (DictConfig): The Hydra configuration object.
    """
    # load phoneme sorted wiki dataset
    try:
        dataset = load_from_disk(dataset_path="data/processed")
    except FileNotFoundError:
        logger.warn(
            (
                "No processed dataset found. Please run"
                "`sort_wiki_by_phoneme_occurence.py` first."
            )
        )
        return

    with open("phonemes.json") as f:
        phonemes = json.load(f)

    example_words = get_example_words(phonemes=phonemes)

    covering_set = []
    for document in dataset:
        document_phonemes = document["all_unique_phonemes"]
        new_phoneme_found = False
        for phoneme in document_phonemes:
            if phoneme in example_words["all"]:
                new_phoneme_found = True
                example_words["all"].remove(phoneme)
        if new_phoneme_found:
            covering_set.append(document)
        if len(example_words["all"]) == 0:
            break

    covering_set_dataset = Dataset.from_list(covering_set)
    final_data_path = Path(cfg.dirs.data) / "final"
    if not final_data_path.exists():
        final_data_path.mkdir(parents=True, exist_ok=True)
    covering_set_dataset.save_to_disk(final_data_path / "covering_set")
