"""
.. include:: ../../README.md
"""

import importlib.metadata
from typing import Callable
from .bus_stops_and_stations import build_bus_stop_and_station_dataset
from .dates import build_date_dataset
from .times import build_time_dataset
from .lex import build_lex_dataset
from .reddit import build_reddit_dataset
from .phoneme_covering import build_phoneme_covering_dataset
from .sundhed_dk import build_sundhed_dk_dataset
from .borger_dk import build_borger_dk_dataset

# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version(__package__)


ALL_DATASET_BUILDERS: dict[str, Callable[..., list[str]]] = dict(
    bus_stops_and_stations=build_bus_stop_and_station_dataset,
    dates=build_date_dataset,
    times=build_time_dataset,
    lex=build_lex_dataset,
    reddit=build_reddit_dataset,
    phoneme_covering=build_phoneme_covering_dataset,
    sundhed_dk=build_sundhed_dk_dataset,
    borger_dk=build_borger_dk_dataset,
)
