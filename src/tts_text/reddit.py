"""Building a list of reddit comments."""

from datasets import Dataset, load_dataset
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
import itertools as it
import re


# Download the sentence splitter model
nltk.download("punkt", quiet=True)


def build_reddit_dataset(output_dir: Path | str) -> list[str]:
    """Build the reddit.com dataset.

    Args:
        output_dir: The directory to save the dataset to.

    Returns:
        A list of reddit.com comments.
    """
    # Load the comments
    raw_dataset = load_dataset("alexandrainst/scandi-reddit", split="train")
    assert isinstance(raw_dataset, Dataset)
    filtered_dataset = raw_dataset.filter(
        lambda example: example["lang"] == "da"
        and example["language_confidence"] > 0.95,
        keep_in_memory=True,
    )
    comments = filtered_dataset["doc"]

    # Split the comments into sentences
    dataset = list(
        it.chain(
            *[sent_tokenize(text=article, language="danish") for article in comments]
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

    # Remove sentences with urls
    dataset = [
        sentence
        for sentence in dataset
        if re.search(r"https?://[^\s]+", sentence) is None
    ]

    # Save the dataset
    dataset_path = Path(output_dir) / "reddit.txt"
    with dataset_path.open("w") as f:
        f.write("\n".join(dataset))

    return dataset
