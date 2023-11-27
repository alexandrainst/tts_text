"""Building a list of reddit comments."""

from datasets import load_dataset
from pathlib import Path
from omegaconf import DictConfig


def build_reddit_dataset(cfg: DictConfig) -> list[str]:
    """Build the reddit.com dataset.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        A list of reddit.com comments.
    """
    try:
        filtered_comments = load_dataset(
            "alexandrainst/scandi-reddit-filtered",
            split="train",
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "The filtered comments file was not found on huggingface.co."
            "In case the 'alexandrainst/scandi-reddit-filtered' dataset has been"
            "deleted, you can recreate it by running the"
            "manually_filter_reddit_comments.py script."
        )

    # Take sentences from the filtered comments, which have all answers as "y"
    filtered_comments_text = [
        comment["sentence"]
        for comment in filtered_comments
        if all(answer == "y" for answer in comment["keep"].split(", "))
    ]
    output_dir = Path(cfg.dirs.data) / cfg.dirs.raw
    dataset_path = output_dir / "reddit.txt"
    with dataset_path.open("w") as f:
        f.write("\n".join(filtered_comments_text))

    return filtered_comments_text
