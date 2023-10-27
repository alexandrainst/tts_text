"""Building a list of reddit comments."""

import os
import pandas as pd
from datasets import Dataset, load_dataset
from pathlib import Path
import nltk
import huggingface_hub as hf_hub


# Download the sentence splitter model
nltk.download("punkt", quiet=True)


def build_reddit_dataset(output_dir: Path | str) -> list[str]:
    """Build the reddit.com dataset.

    Args:
        output_dir: The directory to save the dataset to.

    Returns:
        A list of reddit.com comments.
    """
    # Load the manually filtered comments
    filtered_comments_path = Path(output_dir) / "filtered_comments.csv"
    if filtered_comments_path.exists():
        filtered_comments_df = pd.read_csv(filtered_comments_path)
        filtered_comments = Dataset.from_pandas(filtered_comments_df)
    else:
        try:
            filtered_comments = load_dataset(
                "alexandrainst/scandi-reddit-manually-filtered"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The filtered comments file was not found at {filtered_comments_path}"
                "nor on huggingface.co. Please run the "
                "manually_filter_reddit_comments.py script."
            )

    # Take sentences from the filtered comments, which have all answers as "y"
    filtered_comments_text = [
        comment["sentence"]
        for comment in filtered_comments
        if all(answer == "y" for answer in comment["keep"].split(" ,"))
    ]

    # Save the dataset
    dataset_path = Path(output_dir) / "reddit.txt"
    with dataset_path.open("w") as f:
        f.write("\n".join(filtered_comments_text))

    # Create a dataset repo on huggingface.co
    hf_hub.create_repo(
        repo_id="alexandrainst/scandi-reddit-manually-filtered",
        repo_type="dataset",
        token=os.environ.get("HF_HUB_TOKEN"),
        exist_ok=True,
        private=True,
    )
    filtered_comments.push_to_hub(
        repo_id="alexandrainst/scandi-reddit-manually-filtered",
        token=os.environ["HF_HUB_TOKEN"],
        private=True,
    )

    return filtered_comments_text
