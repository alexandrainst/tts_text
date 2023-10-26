"""Module for creating a phoneme covering set."""

import json
import re
import logging
from functools import partial
from collections import Counter
from pathlib import Path
from tqdm.auto import tqdm

from datasets import load_dataset, Dataset
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_and_wiki_by_phoneme_occurence(cfg: DictConfig) -> Dataset:
    """Load and sort the wiki dataset by phoneme occurence.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        The sorted dataset.

    Raises:
        ValueError: If `phoneme_sort_strategy` in the config is not one of "da", "en"
            or "all".
    """
    # The `wiki40b` dataset is a small dataset so we can load it all into memory
    # instead of streaming it.
    dataset = load_dataset(
        "wiki40b",
        "da",
        split="train",
        beam_runner="DirectRunner",
    )

    phoneme_path = Path(cfg.dirs.data) / cfg.dirs.raw / cfg.dirs.phoneme_file
    with phoneme_path.open() as f:
        phonemes = json.load(f)

    # Count phonemes in articles
    dataset = dataset.map(partial(count_phoneme_occurences, phonemes=phonemes, cfg=cfg))
    sort_by = cfg.phoneme_sort_strategy
    if sort_by == "da":
        dataset = dataset.sort("da_unique_phonemes_count", reverse=True)
    elif sort_by == "en":
        dataset = dataset.sort("en_unique_phonemes_count", reverse=True)
    elif sort_by == "all":
        dataset = dataset.sort("all_unique_phonemes_count", reverse=True)
    else:
        raise ValueError(f"sort_by must be one of 'da', 'en' or 'all', got {sort_by}")

    return dataset


def count_phoneme_occurences(document: dict, phonemes: dict, cfg: DictConfig) -> dict:
    """Count the occurences of phonemes in a document.

    Args:
        cfg: The Hydra configuration object.
        document: The document to count phonemes in.
        phonemes: The phonemes to count.

    Returns:
        The document with phoneme lists and counts added.
    """
    # Clean document text, wiki articles have special annotations of sections and
    # paragraphs, etc.
    document_text = document["text"]
    " ".join(
        re.split(
            "|".join(cfg.split_strings),
            document["text"],
        )
    )
    document_text = re.sub(" +", " ", document_text).strip()
    words = document_text.split(" ")

    counter = Counter(words)
    for language, phoneme_list in phonemes.items():
        phoneme_count = 0
        found_phonemes = []
        found_example_words = []
        for phoneme in phoneme_list:
            for word in phoneme["examples"]:
                occurences = counter[word]
                phoneme_count += occurences
                if occurences > 0:
                    found_phonemes.append(phoneme["name"])
                    found_example_words.append(word)
        document[f"{language}_phoneme_count"] = phoneme_count
        document[f"{language}_phonemes"] = found_phonemes
        document[f"{language}_found_example_words"] = found_example_words
        document[f"{language}_unique_phonemes"] = set(found_phonemes)
        document[f"{language}_unique_phonemes_count"] = len(set(found_phonemes))

    document["all_phoneme_count"] = (
        document["da_phoneme_count"] + document["en_phoneme_count"]
    )
    document["all_phonemes"] = document["da_phonemes"] + document["en_phonemes"]
    document["all_found_example_words"] = (
        document["da_found_example_words"] + document["en_found_example_words"]
    )
    document["all_unique_phonemes"] = set(
        document["da_unique_phonemes"] | document["en_unique_phonemes"]
    )
    document["all_unique_phonemes_count"] = len(document["all_unique_phonemes"])
    return document


def build_phoneme_covering_dataset(
    cfg: DictConfig, output_dir: Path | str
) -> list[str]:
    """Create a phoneme covering set from the sorted wiki dataset.

    A phoneme covering set is a set of strings that contains at least one example
    word for each phoneme.

    Args:
        cfg: The Hydra configuration object.
        output_dir: The directory to save the dataset to.

    Returns:
        The phoneme covering set.
    """
    sorted_dataset = load_and_wiki_by_phoneme_occurence(cfg=cfg)

    phoneme_path = Path(cfg.dirs.data) / cfg.dirs.raw / cfg.dirs.phoneme_file
    with phoneme_path.open() as f:
        phonemes = json.load(f)
    all_phone_names = {entry["name"] for entry in phonemes["da"]}.union(
        {entry["name"] for entry in phonemes["en"]}
    )
    covering_set = []
    for document in tqdm(sorted_dataset):
        if not all_phone_names:
            break
        document_phonemes = document["all_phonemes"]
        new_phonemes = set(all_phone_names).intersection(document_phonemes)
        for new_phoneme in new_phonemes:
            all_phone_names.remove(new_phoneme)
        if new_phonemes:
            covering_set.append(document)

    covering_set_dataset = Dataset.from_list(covering_set)

    def extract_paragraph_with_example_word(document: dict) -> dict:
        found_paragraphs = []
        for paragraph in re.split(
            "|".join(cfg.split_strings),
            document["text"],
        ):
            if any(
                word in document["all_found_example_words"]
                for word in paragraph.split()
            ):
                found_paragraphs.append(paragraph)

        document["extracted_text"] = " ".join(found_paragraphs)
        return document

    covering_set_dataset = covering_set_dataset.map(
        extract_paragraph_with_example_word,
    )
    dataset = [document["extracted_text"] for document in covering_set_dataset]
    dataset_path = Path(output_dir) / "phoneme_covering_set.txt"
    with dataset_path.open("w") as f:
        f.write("\n".join(dataset))

    return dataset


def get_example_words(phonemes: dict) -> dict:
    """Get example words from the phonemes dict.

    Args:
        phonemes: The phonemes dict.

    Returns:
        The example words.
    """
    # Get example words
    da_phonemes = phonemes["da"]
    en_phonemes = phonemes["en"]
    example_words: dict[str, list[str]] = {}
    example_words["da"] = [
        example for entry in da_phonemes for example in entry["examples"]
    ]
    example_words["en"] = [
        example for entry in en_phonemes for example in entry["examples"]
    ]
    example_words["all"] = example_words["da"] + example_words["en"]
    return example_words
