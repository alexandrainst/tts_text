"""Tests related to the bus stops and stations dataset."""

from tts_text.bus_stops_and_stations import build_bus_stop_and_station_dataset


def test_build_bus_stop_and_station_dataset(cfg) -> None:
    build_bus_stop_and_station_dataset(cfg=cfg)
