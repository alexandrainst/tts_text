"""Make a phoneme covering set of wikipedia articles.

Usage:
    python src/scripts/make_wiki_phoneme_covering.py sort_by=da
"""

import hydra
from omegaconf import DictConfig

from tts_text.make_phoneme_covering_set import make_phoneme_covering_set
from tts_text.sort_wiki_by_phoneme_occurence import sort_wiki_by_phoneme_occurence


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Make a phoneme covering set of wikipedia articles.
    Args:
        config: The Hydra config for your project.
    """
    sort_wiki_by_phoneme_occurence(cfg=config)
    make_phoneme_covering_set(cfg=config)


if __name__ == "__main__":
    main()
