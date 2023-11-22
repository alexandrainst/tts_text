"""Tests related to the times dataset."""

from tts_text.times import build_time_dataset


def test_build_time_dataset(cfg) -> None:
    build_time_dataset(cfg=cfg)
