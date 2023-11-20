"""Tests related to the dates dataset."""

from tts_text.dates import build_date_dataset


def test_build_date_dataset(cfg) -> None:
    build_date_dataset(cfg=cfg)
