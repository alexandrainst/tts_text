"""Building a list of open Lex.dk articles."""

from datasets import Dataset, load_dataset
from pathlib import Path

from omegaconf import DictConfig

from .utils import extract_sentences


def build_lex_dataset(cfg: DictConfig) -> list[str]:
    """Build the Lex.dk dataset.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        A list of articles from Lex.dk.
    """
    # Load the articles
    raw_dataset = load_dataset("alexandrainst/lexdk-open", split="train")
    assert isinstance(raw_dataset, Dataset)
    articles = raw_dataset["text"]

    # Split the articles into sentences
    dataset = extract_sentences(
        corpus=articles, min_sentence_length=cfg.min_sentence_length
    )

    # Save the dataset
    dataset_path = Path(cfg.dirs.data) / cfg.dirs.raw / "lex.txt"
    with dataset_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(dataset))

    return dataset
