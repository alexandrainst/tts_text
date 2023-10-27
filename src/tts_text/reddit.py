"""Building a list of reddit comments."""

from datasets import Dataset
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
        filtered_comments = Dataset.from_csv(path=filtered_comments_path)
    else:
        raise FileNotFoundError(
            f"The filtered comments file was not found at {filtered_comments_path}. "
            "Please run the manually_filter_reddit_comments.py script."
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
        token=hf_hub.get_token(),
        exist_ok=True,
    )
    filtered_comments.push_to_hub(
        repo_id="alexandrainst/scandi-reddit-manually-filtered",
        token=hf_hub.get_token(),
        commit_message="Manually filtered reddit.com comments.",
        create_pr=True,
    )

    return filtered_comments_text
