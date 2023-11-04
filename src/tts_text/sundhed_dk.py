"""Scraping and preprocessing of the sundhed.dk text corpus."""

from pathlib import Path
from unicodedata import normalize
from bs4 import Tag
from omegaconf import DictConfig
from tqdm.auto import tqdm
from webdriver_manager.chrome import ChromeDriverManager
import logging
import re
from .utils import extract_sentences, get_soup


BASE_URL = "https://www.sundhed.dk"


def build_sundhed_dk_dataset(cfg: DictConfig) -> list[str]:
    """Build the sundhed.dk dataset.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        A list of articles from sundhed.dk.
    """
    # Load dataset if it already exists
    dataset_path = Path(cfg.dirs.data) / cfg.dirs.raw / "sundhed_dk.txt"
    if dataset_path.exists():
        with dataset_path.open("r", encoding="utf-8") as f:
            return f.read().split("\n")

    # Install the Chrome driver, if it isn't already installed
    logging.getLogger("WDM").setLevel(logging.WARNING)
    ChromeDriverManager().install()

    # Get the overall categories from the front page
    soup = get_soup(url=BASE_URL + "/borger/patienthaandbogen/", dynamic=True)
    category_urls = [
        BASE_URL + url_suffix.a["href"]
        for url_suffix in soup.find_all("li", class_="list-group-item")
    ]

    # Extract all articles
    all_articles: list[str] = list()
    desc = "Extracting articles from sundhed.dk"
    for category_url in tqdm(category_urls, desc=desc, leave=True):
        category_articles = extract_all_category_articles(url=category_url)
        all_articles.extend(category_articles)

    # Split the articles into sentences
    dataset = extract_sentences(
        corpus=all_articles, min_sentence_length=cfg.min_sentence_length
    )

    # Ensure sentences end with appropriate punctuation
    dataset = [
        sentence + "." if re.match(r".*[.?!]$", sentence) is None else sentence
        for sentence in dataset
    ]

    # Remove sentences not starting with a capital letter
    dataset = [sentence for sentence in dataset if sentence[0].isupper()]

    # Save the dataset
    with dataset_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(dataset))

    return dataset


def extract_all_category_articles(url: str) -> list[str]:
    """Extract all articles from a category page.

    These pages have arbitrarily nested subcategories, so this function is called
    recursively.

    Args:
        url: The URL of the category page.

    Returns:
        A list of articles from the category.
    """
    # Parse the URL
    soup = get_soup(url=url, dynamic=True)

    # Try to get the URLs of the subcategories, if any
    subcategory_urls = [
        BASE_URL + url_suffix.a["href"]
        for url_suffix in soup.find_all(name="div", class_="closed")
        if url_suffix.a is not None and url_suffix.a["href"].startswith("/")
    ]

    # If there are no subcategories then we extract all links to articles, if any
    if not subcategory_urls:
        content_div = soup.find(name="div", class_="content")
        if isinstance(content_div, Tag):
            page_url_list = content_div.find(name="ul", class_="red-arrows")
            if isinstance(page_url_list, Tag):
                subcategory_urls = [
                    BASE_URL + url_suffix.a["href"]
                    for url_suffix in page_url_list.find_all(
                        name="li", class_="topicPage"
                    )
                    if url_suffix.a is not None and url_suffix.a["href"].startswith("/")
                ]

    # If there are no subcategories or links to articles then we assume that the page
    # is an article, and we extract it
    if not subcategory_urls:
        if isinstance(soup.article, Tag):
            title = soup.article.h1.text if isinstance(soup.article.h1, Tag) else ""

            # Get the author of the article
            author: str = ""
            metadata_div = soup.article.find(name="div", class_="meta-data")
            if isinstance(metadata_div, Tag):
                author_elt = metadata_div.find(name="span", itemprop="author")
                author = author_elt.text if isinstance(author_elt, Tag) else ""

            # Get all the facts from the fact boxes
            facts: list[str] = list()
            fact_box_elts = soup.article.find_all(name="div", class_="faktabox")
            for fact_box_elt in fact_box_elts:
                fact_box_facts = fact_box_elt.find_all(name="li")
                facts.extend(
                    [
                        normalize("NFKC", fact_box_fact.text.strip())
                        for fact_box_fact in fact_box_facts
                    ]
                )

            # Get the content of the article
            content_elt = (
                soup.section.find(
                    "p", attrs={"data-sdk-core-htmlcompile": "page.PageContent"}
                )
                if isinstance(soup.section, Tag)
                else None
            )
            content = (
                normalize("NFKC", content_elt.text).strip()
                if isinstance(content_elt, Tag)
                else ""
            )

            article_str = f"{title}\n{author}\n\n{content}\n\n\nFakta:\n\n" + "\n".join(
                facts
            )
            return [article_str]

    # If it wasn't and article then we recursively extract all articles from the
    # subcategories
    desc = f"Extracting articles from {url}"
    category_articles: list[str] = list()
    for subcategory_url in tqdm(subcategory_urls, desc=desc, leave=False):
        subcategory_articles = extract_all_category_articles(url=subcategory_url)
        category_articles.extend(subcategory_articles)
    return category_articles
