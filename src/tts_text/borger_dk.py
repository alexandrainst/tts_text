"""Scraping and preprocessing of the sundhed.dk text corpus."""

from pathlib import Path
from unicodedata import normalize
from bs4 import Tag, BeautifulSoup
from omegaconf import DictConfig
from tqdm.auto import tqdm
from webdriver_manager.chrome import ChromeDriverManager
import logging
import re
from .utils import extract_sentences, get_soup


BASE_URL = "https://www.borger.dk"
SUBSITES_TO_IGNORE = [
    "/kampagnesider/",  # Its own special subsite, with a question-answer-type
    # navigation, which leads to articles on the main site
]


def build_borger_dk_dataset(cfg: DictConfig) -> list[str]:
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
    # retry if it fails
    soup: BeautifulSoup = BeautifulSoup("")
    while not soup.contents:
        soup = get_soup(url=BASE_URL, dynamic=True)

    category_urls = [
        BASE_URL + url_suffix.contents[1].attrs["href"]
        for url_suffix in soup.find_all("li", class_="col-12")
    ]

    # Extract all articles
    all_articles: list[str] = list()
    found_urls: list[str] = list()
    desc = "Extracting articles from sundhed.dk"
    for category_url in tqdm(category_urls, desc=desc, leave=True):
        # Borger.dk is structured in to categories, which are structured in to
        # subcategories, which are structured in to articles. Every article contains
        # a text, and a link to another article. The links go both up, down and
        # sideways in the hierarchy, so we need to restrict ourselves to one category
        # at a time, and only follow links downwards, otherwise the recursion will
        # never end. This is why we use the category variable.
        category = category_url.split("/")[-1]
        category_articles, found_urls = extract_all_category_articles(
            cfg=cfg, url=category_url, category=category, found_urls=found_urls
        )
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


def extract_all_category_articles(
    cfg: DictConfig, url: str, category: str, parsed_urls: list[str] = [], found_urls=[]
) -> tuple[list[str], list[str]]:
    """Extract all articles from a category page.

    These pages have arbitrarily nested subcategories, so this function is called
    recursively.

    Args:
        url: The URL of the category page.
        parsed_urls: A list of URLs that have already been parsed.

    Returns:
        A list of articles from the category.
    """
    # Often connection fails when we try to access a page, so we retry a few times
    # but we give up fairly fast due to time constraints.
    soup: BeautifulSoup = BeautifulSoup("")
    for _ in range(cfg.scraping_retry_connection_limit):
        soup = get_soup(url=url, dynamic=True)
        if soup.contents:
            break

    if not soup.contents:
        return [], found_urls

    # Get links in collapsable sidebar
    subcategory_urls: list[str] = []
    for collapsed_div in soup.find_all(name="div", class_="collapse"):
        if isinstance(collapsed_div, Tag):
            # Find all the good links in the sidebar.
            subcategory_urls.extend(
                [
                    BASE_URL + url_suffix.attrs["href"]
                    for url_suffix in collapsed_div.find_all(
                        name="a", class_="nav-link"
                    )
                    if (
                        url_suffix.attrs is not None  # Its a link
                        and category
                        in url_suffix.attrs["href"]  # Only links to the same category
                        and url_suffix.attrs["href"]
                        not in url  # We are not going up in the hierarchy
                        and not any(
                            subsite in url_suffix.attrs["href"]
                            for subsite in SUBSITES_TO_IGNORE
                        )  # Alternative navigation hierarchies
                        and BASE_URL
                        not in url_suffix.attrs["href"]  # Links which contain
                        # the base url are not articles, but are javascript-based
                        # navigation or links to files hosted on borger.dk
                        and BASE_URL + url_suffix.attrs["href"]
                        not in found_urls  # We have not already found this link
                    )
                ]
            )

    # If there are no new subcategories, then we assume that the page has an article
    # and we extract it. Any article is accompanied by a link to another
    # article, so we open these and extract them as well
    if not subcategory_urls:
        subcategory_articles: list[str] = list()
        articles_to_return: list[str] = list()
        for content_div in soup.find_all(name="div", class_="content-text"):
            # Extract the article
            article_str = ""
            for paragraph in content_div.find_all(name="p"):
                article_str += paragraph.text + " "

            # Clean the final article text
            article_str = normalize("NFKC", article_str).replace("–", "-").strip()
            article_str = re.sub(
                r"[^a-zA-Z0-9æøåéÉÆØÅ.,\-?!:\n\/()\[\] ]%\"\'", "", article_str
            )

            # Some articles are only a link to another article, so we ignore them
            if article_str:
                articles_to_return.append(article_str)

            # Find all the good links to nested articles.
            for list_link in content_div.find_all(name="ul", class_="list--links"):
                # TODO: make list comprehension into function
                subcategory_urls.extend(
                    [
                        BASE_URL + url_suffix.attrs["href"]
                        for url_suffix in list_link.find_all(name="a")
                        if (
                            url_suffix.attrs is not None  # Its a link
                            and category
                            in url_suffix.attrs[
                                "href"
                            ]  # Only links to the same category
                            and url_suffix.attrs["href"]
                            not in url  # We are not going up in the hierarchy
                            and not any(
                                subsite in url_suffix.attrs["href"]
                                for subsite in SUBSITES_TO_IGNORE
                            )  # Alternative navigation hierarchies
                            and BASE_URL
                            not in url_suffix.attrs["href"]  # Links which contain
                            # the base url are not articles, but are javascript-based
                            # navigation or links to files hosted on borger.dk
                            and BASE_URL + url_suffix.attrs["href"]
                            not in found_urls  # We have not already found this link
                        )
                    ]
                )

            # Follow the good links to nested articles
            desc = f"Extracting articles from {url}"
            parsed_urls.append(url)
            found_urls.extend(subcategory_urls + [url])
            for subcategory_url in tqdm(subcategory_urls, desc=desc, leave=False):
                if subcategory_url in parsed_urls:
                    continue
                extracted_articles, found_urls = extract_all_category_articles(
                    cfg=cfg,
                    url=subcategory_url,
                    category=category,
                    parsed_urls=parsed_urls,
                    found_urls=found_urls,
                )
                subcategory_articles.extend(extracted_articles)

        # We have now followed all the good links in the articles hence
        # subcategory_articles should now only contain articles. We add
        # these to the articles we have already found in this article

        return articles_to_return + subcategory_articles, found_urls

    # If it wasn't and article then we recursively extract all articles from the
    # subcategories
    desc = f"Extracting articles from {url}"
    category_articles: list[str] = list()
    parsed_urls.append(url)
    found_urls.extend(subcategory_urls + [url])
    for subcategory_url in tqdm(subcategory_urls, desc=desc, leave=False):
        if subcategory_url in parsed_urls:
            continue
        subcategory_articles, found_urls = extract_all_category_articles(
            cfg=cfg,
            url=subcategory_url,
            category=category,
            parsed_urls=parsed_urls,
            found_urls=found_urls,
        )
        category_articles.extend(subcategory_articles)
    return category_articles, found_urls
