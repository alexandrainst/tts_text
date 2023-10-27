"""General functions and fixtures related to `pytest`."""

import pytest
from typing import Generator
from omegaconf import DictConfig
from hydra import compose, initialize
import sys


initialize(config_path="../config", version_base=None)


def pytest_configure() -> None:
    """Set a global flag when `pytest` is being run."""
    setattr(sys, "_called_from_test", True)


def pytest_unconfigure() -> None:
    """Unset the global flag when `pytest` is finished."""
    delattr(sys, "_called_from_test")


@pytest.fixture(scope="session")
def cfg() -> Generator[DictConfig, None, None]:
    yield compose(
        config_name="config",
        overrides=[],
    )
