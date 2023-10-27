"""General functions and fixtures related to `pytest`."""

import pytest
from typing import Generator
from omegaconf import DictConfig
from hydra import compose, initialize


initialize(config_path="../config", version_base=None)


@pytest.fixture(scope="session")
def cfg() -> Generator[DictConfig, None, None]:
    yield compose(
        config_name="config",
        overrides=[],
    )
