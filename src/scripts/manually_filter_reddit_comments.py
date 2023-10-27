"""Script for manually filtering a list of reddit comments."""

import pandas as pd
from datasets import Dataset, load_dataset
import nltk
import click
from pathlib import Path
from nltk.tokenize import sent_tokenize
import itertools as it
import re


# Download the sentence splitter model
nltk.download("punkt", quiet=True)


@click.command("Starts the process of manually filter reddit comments.")
@click.option("--output-dir", type=Path, required=True)
@click.option("--num-samples=", "-n", type=int, default=1000, help="The number of samples to annotate")
@click.option("--start-index", type=int, default=0, help="The sample index to start annotating from.")
@click.option("--username", type=str, required=True, help="The username of the person filtering the dataset.")
def filter_reddit_dataset(
    output_dir: Path | str, username: str, n: int = 1000, start_index: int = 0
):
    """Script for manually filtering a list of reddit comments.

    Args:
        output_dir: The directory to save the dataset to.
        username: The username of the person filtering the dataset.
        n: The number of sentences to manually filter.
        start_index: The index to start from.
    """
    print("Hello and welcome to the reddit comment filtering tool!")
    # Load the comments
    raw_dataset = load_dataset("alexandrainst/scandi-reddit", split="train")
    assert isinstance(raw_dataset, Dataset)

    # Pick a subset of the dataset, to speed up filtering
    raw_dataset = raw_dataset.select(range(start_index, start_index + (n * 1000)))
    filtered_dataset = raw_dataset.filter(
        lambda example: example["language"] == "da"
        and example["language_confidence"] > 0.95,
        keep_in_memory=True,
    )
    filtered_dataset = filtered_dataset.select(range(start_index, start_index + n))
    filtered_dataset = filtered_dataset.shuffle(seed=703)
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
    # Manually filter the dataset, and store which person filtered each comment
    records: list[dict[str, str | int]] = list()
    for index, sentence in enumerate(dataset[start_index : start_index + n]):
        records = prompt_user(None, records, sentence, username, start_index, index)

    filtered_answers = pd.DataFrame.from_records(records)
    # Load the previous answers, if any exist
    previous_filtering_path = Path(output_dir) / "filtered_comments.csv"
    if previous_filtering_path.exists():
        previous_filtering = pd.read_csv(previous_filtering_path)
        if not previous_filtering.equals(filtered_answers):
            print("Previous filtering found, merging with new filtering.")
            # Merge the previous answers with the new answers.
            concatenated_answers = pd.concat([previous_filtering, filtered_answers])
            filtered_answers = concatenated_answers.groupby(
                ["sentence", "index"], as_index=False, sort=False
            ).agg({"username": ", ".join, "keep": ", ".join})

    filtered_answers.to_csv(Path(output_dir) / "filtered_comments.csv", index=False)


def prompt_user(
    answer: str | None,
    records: list[dict[str, str | int]],
    sentence: str,
    username: str,
    start_index: int,
    index: int,
) -> list[dict[str, str | int]]:
    """Prompt the user for an answer.

    Args:
        answer: The answer to the question.
        records: The records of the previous answers.
        sentence: The sentence to filter.
        username: The username of the person filtering the dataset.
        start_index: The index to start from.
        index: The index of the sentence in the dataset.

    Returns:
        A list of records.
    """
    print(f"Sentence {start_index + index}: {sentence}")
    if answer is None:
        answer = input("Keep? [y/n]: ")

    if answer in ["y", "n"]:
        records.append(
            {
                "sentence": sentence,
                "username": username,
                "keep": answer,
                "index": start_index + index,
            }
        )
    else:
        print("Invalid input, must be 'y' or 'n'.")
        answer = input("Keep? [y/n]: ")
        records = prompt_user(answer, records, sentence, username, start_index, index)
    return records


if __name__ == "__main__":
    filter_reddit_dataset()
