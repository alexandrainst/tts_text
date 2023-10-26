"""Building a list of open Lex.dk articles."""

from datasets import Dataset, load_dataset
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
import itertools as it
import re


# Download the sentence splitter model
nltk.download("punkt", quiet=True)


def build_lex_dataset(output_dir: Path | str) -> list[str]:
    """Build the Lex.dk dataset.

    Args:
        output_dir: The directory to save the dataset to.

    Returns:
        A list of articles from Lex.dk.
    """
    # Load the articles
    raw_dataset = load_dataset("alexandrainst/lexdk-open", split="train")
    assert isinstance(raw_dataset, Dataset)
    articles = raw_dataset["text"]

    # Split the articles into sentences
    dataset = list(
        it.chain(
            *[sent_tokenize(text=article, language="danish") for article in articles]
        )
    )

    # Remove newlines
    dataset = [sentence.replace("\n", " ") for sentence in dataset]

    # Remove too short sentences
    dataset = [sentence for sentence in dataset if len(sentence) > 10]

    # Remove sentences ending in "..."
    dataset = [sentence for sentence in dataset if not sentence.endswith("...")]

    # Remove sentences ending in an abbreviation
    dataset = [
        sentence
        for sentence in dataset
        if re.search(r"\.[A-ZÆØÅa-zæøå]+\.$", sentence) is None
    ]

    # Save the dataset
    dataset_path = Path(output_dir) / "lex.txt"
    with dataset_path.open("w") as f:
        f.write("\n".join(dataset))

    return dataset
