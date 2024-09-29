"""Utility functions used by the other modules."""

from copy import deepcopy
from typing import Generator
import time
import random
import itertools as it
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import re
import requests as rq
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
import logging


logger = logging.getLogger(__name__)


# Download the sentence splitter model
nltk.download("punkt", quiet=True)


def extract_sentences(corpus: list[str], min_sentence_length: int) -> list[str]:
    """Extract sentences from a corpus of text.

    Args:
        corpus:
            The corpus to extract sentences from.
        min_sentence_length:
            The minimum length of a sentence.

    Returns:
        The sentences in the corpus.
    """
    # Firstly split by newline, where we assume that a sentence does not span multiple
    # lines
    corpus = list(it.chain(*[example.split("\n") for example in corpus]))

    # Split dataset into sentences
    sentences = list(
        it.chain(
            *[
                sent_tokenize(text=example, language="danish")
                for example in tqdm(iterable=corpus, desc="Splitting sentences")
            ]
        )
    )

    # Remove newlines
    sentences = [sentence.replace("\n", " ") for sentence in sentences]

    # Remove sentences beginning or ending in "..."
    sentences = [
        sentence
        for sentence in sentences
        if not sentence.startswith("...") and not sentence.endswith("...")
    ]

    # Remove sentences ending in an abbreviation
    sentences = [
        sentence
        for sentence in sentences
        if re.search(r"\.[A-ZÆØÅa-zæøå]+\.$", sentence) is None
    ]

    # Remove redundant whitespace
    sentences = [re.sub(" +", " ", sentence) for sentence in sentences]

    # Remove trailing whitespace, tabs and newlines
    sentences = [sentence.strip(" \t\n") for sentence in sentences]

    # Remove too short sentences
    sentences = [
        sentence for sentence in sentences if len(sentence) > min_sentence_length
    ]

    return sentences


def interleave_datasets(
    non_sampling_datasets: list[list[str]],
    sampling_datasets: list[list[str]],
    sampling_probabilities: list[float],
    random_seed: int,
) -> Generator[str, None, None]:
    """Interleave multiple datasets according to the given sampling probabilities.

    Args:
        non_sampling_datasets:
            The datasets that should not be sampled. These will be shuffled together
            and included in the beginning of the interleaved dataset.
        sampling_datasets:
            The datasets that should be sampled. These will be sampled according to
            the given sampling probabilities, after the non-sampling datasets have
            been included.
        sampling_probabilities:
            The sampling probabilities for each dataset.
        random_seed:
            The random seed to use.

    Yields:
        The interleaved dataset.
    """
    random.seed(random_seed)

    # Start by including all datasets that shouldn't be sampled
    joined_non_sampling_datasets = list(it.chain(*non_sampling_datasets))
    random.shuffle(joined_non_sampling_datasets)
    yield from joined_non_sampling_datasets

    # Make a copy of the sampling datasets to avoid mutating the original
    sampling_datasets = deepcopy(sampling_datasets)

    while len(sampling_datasets) > 0:
        # Sample a dataset
        dataset = random.choices(
            population=sampling_datasets,
            weights=sampling_probabilities,
            k=1,
        )[0]

        # If the dataset is empty then stop
        if len(dataset) == 0:
            break

        # Sample a sample from the dataset
        sample_idx = random.randrange(len(dataset))
        sample = dataset[sample_idx]

        # Remove the sample from the dataset
        dataset.pop(sample_idx)

        yield sample


def get_soup(
    url: str,
    dynamic: bool = False,
    retries: int | None = None,
    xpath_to_be_present: str | None = None,
) -> BeautifulSoup:
    """Get the soup of a URL.

    Args:
        url:
            The URL to get the soup of.
        dynamic:
            Whether the page is dynamically loaded.
        retries:
            The number of retries to perform if the request times out. None means
            infinite retries.
        xpath_to_be_present:
            The xpath to wait for before returning the soup. If None, we will wait 5
            seconds before returning the soup.

    Returns:
        The soup of the URL.
    """
    if not (retries is None or retries >= 0):
        raise ValueError("Number of retries must be non-negative.")

    html: str = ""
    if dynamic:
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        retries_left = 5
        while retries_left > 0 and not html:
            try:
                driver.get(url=url)
                if xpath_to_be_present:
                    wait = WebDriverWait(driver=driver, timeout=10)
                    wait.until(
                        method=EC.presence_of_element_located(
                            locator=(By.XPATH, xpath_to_be_present)
                        ),
                    )
                else:
                    time.sleep(5)
                html = driver.page_source
            except TimeoutException:
                logger.warning(f"Timed out while getting soup from {url}.")
                html = ""
            except WebDriverException:
                logger.warning(f"Could not get soup from {url}.")
                html = ""
            retries_left -= 1
    else:
        response = rq.get(url=url, timeout=10)

        # Retry if the request timed out
        retries_left = 5
        while response.status_code == 408:
            time.sleep(1)
            response = rq.get(url=url, timeout=10)
            retries_left -= 1
            if retries_left == 0:
                raise TimeoutError("The request timed out.")

        # Raise error if it was not successful
        if not str(response.status_code).startswith("2"):
            raise ConnectionError(
                f"Could not get soup from {url}. Status code: {response.status_code}"
            )

        html = response.text

    if not html:
        return BeautifulSoup("")

    soup: BeautifulSoup = BeautifulSoup("")
    if retries is None:
        while not soup.contents:
            soup = BeautifulSoup(html, "html.parser")
    elif retries > 0:
        for _ in range(retries):
            soup = BeautifulSoup(html, "html.parser")
            if soup.contents:
                break
        else:
            raise RuntimeError(f"Could not parse the URL {url}.")
    else:
        soup = BeautifulSoup(html, "html.parser")
    return soup
