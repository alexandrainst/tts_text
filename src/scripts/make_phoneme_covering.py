"""Make a phoneme covering set of wikipedia articles.

Usage:
    python src/scripts/make_phoneme_covering.py sort_by=da
"""

import hydra
from omegaconf import DictConfig
from pathlib import Path

from tts_text.phoneme_covering_set import make_phoneme_covering_set


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Make a phoneme covering set of wikipedia articles.
    Args:
        config: The Hydra config for your project.
    """
    output_path = Path(config.dirs.data) / "raw"
    make_phoneme_covering_set(cfg=config, output_path=output_path)


if __name__ == "__main__":
    main()
