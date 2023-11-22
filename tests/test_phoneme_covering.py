"""Tests related to the phoneme covering dataset."""

from tts_text.phoneme_covering import build_phoneme_covering_dataset


def test_build_phoneme_covering_dataset(cfg) -> None:
    build_phoneme_covering_dataset(cfg=cfg)
