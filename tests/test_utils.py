"""Tests related to the utility functions."""

from typing import Generator
import pytest
from tts_text.utils import extract_sentences, interleave_datasets


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            ["Dette er en tekst. Her er noget mere tekst."],
            ["Dette er en tekst.", "Her er noget mere tekst."],
        ),
        (
            ["Hej\nNu\nEr\nDer\nLinjeskift"],
            ["Hej", "Nu", "Er", "Der", "Linjeskift"],
        ),
        (
            ["Dette er 1. dokument. Stadig 1. dokument.", "Dette er 2. dokument."],
            ["Dette er 1. dokument.", "Stadig 1. dokument.", "Dette er 2. dokument."],
        ),
    ],
)
def test_extract_sentences(text, expected) -> None:
    assert extract_sentences(corpus=text, min_sentence_length=1) == expected


class TestInterleaveDatasets:
    @pytest.fixture(scope="class")
    def dataset(self) -> list[str]:
        return ["a", "b", "c"]

    @pytest.fixture(scope="class")
    def other_dataset(self) -> list[str]:
        return ["d", "e", "f"]

    def test_interleaving_is_a_generator(self, dataset) -> None:
        assert isinstance(
            interleave_datasets(
                non_sampling_datasets=[dataset],
                sampling_datasets=[],
                sampling_probabilities=[],
            ),
            Generator,
        )

    def test_with_one_sampling_dataset(self, dataset) -> None:
        assert (
            list(
                interleave_datasets(
                    non_sampling_datasets=[],
                    sampling_datasets=[dataset],
                    sampling_probabilities=[1.0],
                )
            )
            == dataset
        )

    def test_non_sampling_dataset(self, dataset) -> None:
        assert (
            list(
                interleave_datasets(
                    non_sampling_datasets=[dataset],
                    sampling_datasets=[],
                    sampling_probabilities=[],
                )
            )
            == dataset
        )

    def test_with_two_sampling_datasets(self, dataset, other_dataset) -> None:
        assert len(
            list(
                interleave_datasets(
                    non_sampling_datasets=[],
                    sampling_datasets=[dataset, dataset],
                    sampling_probabilities=[0.5, 0.5],
                )
            )
        ) >= len(dataset)

    def test_with_two_non_sampling_datasets(self, dataset, other_dataset) -> None:
        assert (
            list(
                interleave_datasets(
                    non_sampling_datasets=[dataset, other_dataset],
                    sampling_datasets=[],
                    sampling_probabilities=[],
                )
            )
            == dataset + other_dataset
        )

    def test_with_two_sampling_datasets_and_one_non_sampling_dataset(
        self, dataset, other_dataset
    ) -> None:
        interleaved = list(
            interleave_datasets(
                non_sampling_datasets=[dataset],
                sampling_datasets=[dataset, other_dataset],
                sampling_probabilities=[0.5, 0.5],
            )
        )
        assert interleaved[: len(dataset)] == dataset
        assert set(interleaved[len(dataset) :]).issubset(set(dataset + other_dataset))
        assert len(interleaved) >= len(dataset) * 2
