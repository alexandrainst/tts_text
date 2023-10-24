"""Module for sorting the wiki dataset by phoneme occurence."""
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from omegaconf import DictConfig


def sort_wiki_by_phoneme_occurence(cfg: DictConfig) -> None:
    """Sort the wiki dataset by phoneme occurence.

    Args:
        cfg (DictConfig): cfg: The Hydra configuration object.

    Raises:
        ValueError: If sort_by is not one of "da", "en" or "all".
    """
    raw_data_path = Path(cfg.dirs.data) / "raw"
    if not raw_data_path.exists():
        raw_data_path.mkdir(parents=True, exist_ok=True)

    # The `wiki40b` dataset is a small dataset so we can load it all into memory
    # instead of streaming it.
    dataset = load_dataset(
        "wiki40b",
        "da",
        split="train",
        cache_dir=raw_data_path,
        beam_runner="DirectRunner",
    )

    with open(raw_data_path / cfg.phoneme_filename) as f:
        phonemes = json.load(f)

    # Count phonemes in articles
    dataset = dataset.map(
        lambda x: count_occurences(x, phonemes),
    )
    sort_by = cfg.phoneme_sort_strategy
    if sort_by == "da":
        dataset = dataset.sort("da_unique_phonemes_count", reverse=True)
    elif sort_by == "en":
        dataset = dataset.sort("en_unique_phonemes_count", reverse=True)
    elif sort_by == "all":
        dataset = dataset.sort("all_unique_phonemes_count", reverse=True)
    else:
        raise ValueError(f"sort_by must be one of 'da', 'en' or 'all', got {sort_by}")

    processed_data_path = Path(cfg.dirs.data) / "processed"
    if not processed_data_path.exists():
        processed_data_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(processed_data_path)


def count_occurences(document: dict, phonemes: dict) -> dict:
    """Count the occurences of phonemes in a document.

    Args:
        document (dict): The document to count phonemes in.
        phonemes (dict): The phonemes to count.

    Returns:
        dict: The document with phoneme lists and counts added.
    """
    example_words = get_example_words(phonemes)

    # Clean document text, wiki articles have special annotations of sections and
    # paragraphs, etc.
    document_text = document["text"]
    document_text = " ".join(document_text.split("\n"))
    document_text = " ".join(document_text.split("\t"))
    document_text = " ".join(document_text.split("_NEWLINE_"))
    document_text = " ".join(document_text.split("_START_ARTICLE_"))
    document_text = " ".join(document_text.split("_START_SECTION_"))
    document_text = " ".join(document_text.split("_START_PARAGRAPH_"))
    document_text = re.sub(" +", " ", document_text).strip()
    words = document_text.split(" ")

    counter = Counter(words)
    for name, phoneme_list in example_words.items():
        phoneme_count = 0
        unique_phonemes_count = 0
        unique_phonemes = []
        for word in phoneme_list:
            occurences = counter[word]
            phoneme_count += occurences
            if occurences > 0:
                unique_phonemes_count += 1
                unique_phonemes.append(word)
        document[f"{name}_phoneme_count"] = phoneme_count
        document[f"{name}_unique_phonemes"] = unique_phonemes
        document[f"{name}_unique_phonemes_count"] = unique_phonemes_count
    return document


def get_example_words(phonemes: dict) -> dict:
    """Get example words from the phonemes dict.

    Args:
        phonemes (dict): The phonemes dict.

    Returns:
        dict: The example words.
    """
    # Get example words
    da_phonemes = phonemes["danish_phonemes"]
    en_phonemes = phonemes["english_phonemes"]
    example_words = {}
    example_words["da"] = [
        example for entry in da_phonemes for example in entry["examples"]
    ]
    example_words["en"] = [
        example for entry in en_phonemes for example in entry["examples"]
    ]
    example_words["all"] = example_words["da"] + example_words["en"]
    return example_words
