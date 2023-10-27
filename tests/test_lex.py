"""Tests related to the Lex dataset."""

from tts_text.lex import build_lex_dataset


def test_build_lex_dataset(cfg) -> None:
    build_lex_dataset(cfg=cfg)
